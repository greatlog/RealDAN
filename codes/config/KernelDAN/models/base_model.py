import logging
import os
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel, DistributedDataParallel

from archs import build_loss, build_network, build_scheduler
from utils.registry import MODEL_REGISTRY

logger = logging.getLogger("base")


@MODEL_REGISTRY.register()
class BaseModel:
    def __init__(self, opt):

        self.opt = opt

        if opt["dist"]:
            self.rank = torch.distributed.get_rank()
            self.world_size = torch.distributed.get_world_size()
        else:
            self.rank = 0  # non dist training

        self.device = torch.device(f"cuda:{torch.cuda.current_device()}")
        self.is_train = opt["is_train"]
        self.log_dict = OrderedDict()

        self.data_names = []
        self.networks = {}

        self.optimizers = {}
        self.schedulers = {}

    def setup_train(self, train_opt):
        # define losses
        loss_opt = train_opt["losses"]
        self.losses = self.build_losses(loss_opt)

        # build optmizers
        optimizer_opts = train_opt["optimizers"]
        self.optimizers = self.build_optimizers(optimizer_opts)

        # set schedulers
        scheduler_opts = train_opt["schedulers"]
        self.schedulers = self.build_schedulers(scheduler_opts)

        # set to training state
        self.set_network_state(self.networks.keys(), "train")

    def feed_data(self, data):
        pass

    def optimize_parameters(self):
        pass

    def get_current_visuals(self):
        pass

    def get_current_losses(self):
        pass

    def print_network(self):
        pass

    def save(self, label):
        pass

    def load(self):
        pass

    def build_network(self, net_opt):

        net = build_network(net_opt)
        
        if isinstance(net, nn.Module):
            net = self.model_to_device(net)

            if net_opt.get("pretrain"):
                pretrain = net_opt.pop("pretrain")
                self.load_network(net, pretrain["path"], pretrain["strict_load"])

            self.print_network(net)
        return net

    def build_losses(self, loss_opt):
        losses = {}

        defined_loss_names = list(loss_opt.keys())
        assert set(defined_loss_names).issubset(set(self.loss_names))

        for name in defined_loss_names:
            loss_conf = loss_opt.get(name)
            if loss_conf["weight"] > 0:
                self.loss_weights[name] = loss_conf.pop("weight")
                losses[name] = build_loss(loss_conf).to(self.device)

        return losses

    def build_optimizers(self, optim_opts):
        optimizers = {}

        if "default" in optim_opts.keys():
            default_optim = optim_opts.pop("default")

        defined_optimizer_names = list(optim_opts.keys())
        assert set(defined_optimizer_names).issubset(self.networks.keys())

        for name in defined_optimizer_names:
            optim_opt = optim_opts[name]
            if optim_opt is None:
                optim_opt = default_optim.copy()

            params = []
            for v in self.networks[name].parameters():
                if v.requires_grad:
                    params.append(v)

            optim_type = optim_opt.pop("type")
            optimizer = getattr(torch.optim, optim_type)(params=params, **optim_opt)
            optimizers[name] = optimizer

        return optimizers

    def build_schedulers(self, scheduler_opts):
        """Set up scheduler."""
        schedulers = {}
        if "default" in scheduler_opts.keys():
            default_opt = scheduler_opts.pop("default")

        for name in self.optimizers.keys():
            scheduler_opt = scheduler_opts[name]
            if scheduler_opt is None:
                scheduler_opt = default_opt.copy()

            schedulers[name] = build_scheduler(
                self.optimizers[name], scheduler_opt
            )

        return schedulers

    def model_to_device(self, net):
        """Model to device. It also warps models with DistributedDataParallel
        or DataParallel.
        Args:
            net (nn.Module)
        """
        net = net.to(self.device)
        if self.opt["dist"]:
            net = DistributedDataParallel(net, device_ids=[torch.cuda.current_device()])
        else:
            net = DataParallel(net)
        return net

    def print_network(self, net):
        # Generator
        s, n = self.get_network_description(net)
        if isinstance(net, nn.DataParallel) or isinstance(net, DistributedDataParallel):
            net_struc_str = "{} - {}".format(
                net.__class__.__name__, net.module.__class__.__name__
            )
        else:
            net_struc_str = "{}".format(net.__class__.__name__)
        if self.rank <= 0:
            logger.info(
                "Network structure: {}, with parameters: {:,d}".format(
                    net_struc_str, n
                )
            )
            logger.info(s)

    def set_optimizer(self, names, operation):
        for name in names:
            getattr(self.optimizers[name], operation)()

    def set_requires_grad(self, names, requires_grad):
        for name in names:
            if isinstance(self.networks[name], nn.Module):
                for v in self.networks[name].parameters():
                    v.requires_grad = requires_grad

    def set_network_state(self, names, state):
        for name in names:
            if isinstance(self.networks[name], nn.Module):
                getattr(self.networks[name], state)()

    def clip_grad_norm(self, names, norm):
        for name in names:
            nn.utils.clip_grad_norm_(self.networks[name].parameters(), max_norm=norm)

    def _set_lr(self, lr_groups_l):
        """set learning rate for warmup,
        lr_groups_l: list for lr_groups. each for a optimizer"""
        for optimizer, lr_groups in zip(self.optimizers, lr_groups_l):
            for param_group, lr in zip(optimizer.param_groups, lr_groups):
                param_group["lr"] = lr

    def _get_init_lr(self):
        # get the initial lr, which is set by the scheduler
        init_lr_groups_l = []
        for optimizer in self.optimizers:
            init_lr_groups_l.append([v["initial_lr"] for v in optimizer.param_groups])
        return init_lr_groups_l

    def update_learning_rate(self, cur_iter, warmup_iter=-1):
        for _, scheduler in self.schedulers.items():
            scheduler.step()
        #### set up warm up learning rate
        if cur_iter < warmup_iter:
            # get initial lr for each group
            init_lr_g_l = self._get_init_lr()
            # modify warming-up learning rates
            warm_up_lr_l = []
            for init_lr_g in init_lr_g_l:
                warm_up_lr_l.append([v / warmup_iter * cur_iter for v in init_lr_g])
            # set learning rate
            self._set_lr(warm_up_lr_l)

    def get_current_learning_rate(self):
        # return self.schedulers[0].get_lr()[0]
        return list(self.optimizers.values())[0].param_groups[0]["lr"]

    def get_network_description(self, network):
        """Get the string and total parameters of the network"""
        if isinstance(network, nn.DataParallel) or isinstance(
            network, DistributedDataParallel
        ):
            network = network.module
        s = str(network)
        n = sum(map(lambda x: x.numel(), network.parameters()))
        return s, n

    def save_network(self, network, network_label, iter_label):
        save_filename = "{}_{}.pth".format(iter_label, network_label)
        save_path = os.path.join(self.opt["path"]["models"], save_filename)
        if isinstance(network, nn.DataParallel) or isinstance(
            network, DistributedDataParallel
        ):
            network = network.module
        state_dict = network.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        torch.save(state_dict, save_path)

    def save(self, iter_label):
        for name in self.optimizers.keys():
            self.save_network(self.networks[name], name, iter_label)

    def load_network(self, network, load_path, strict=True):
        if load_path is not None:
            if isinstance(network, nn.DataParallel) or isinstance(
                network, DistributedDataParallel
            ):
                network = network.module
            load_net = torch.load(load_path)
            load_net_clean = OrderedDict()  # remove unnecessary 'module.'
            for k, v in load_net.items():
                if k.startswith("module."):
                    load_net_clean[k[7:]] = v
                else:
                    load_net_clean[k] = v
            network.load_state_dict(load_net_clean, strict=strict)

    def save_training_state(self, epoch, iter_step):
        """Saves training state during training, which will be used for resuming"""
        state = {"epoch": epoch, "iter": iter_step, "schedulers": {}, "optimizers": {}}
        for k, s in self.schedulers.items():
            state["schedulers"][k] = s.state_dict()
        for k, o in self.optimizers.items():
            state["optimizers"][k] = o.state_dict()
        save_filename = "{}.state".format(iter_step)
        save_path = os.path.join(self.opt["path"]["training_state"], save_filename)
        torch.save(state, save_path)

    def resume_training(self, resume_state):
        """Resume the optimizers and schedulers for training"""
        resume_optimizers = resume_state["optimizers"]
        resume_schedulers = resume_state["schedulers"]
        assert len(resume_optimizers) == len(
            self.optimizers
        ), "Wrong lengths of optimizers"
        assert len(resume_schedulers) == len(
            self.schedulers
        ), "Wrong lengths of schedulers"
        for name, o in resume_optimizers.items():
            self.optimizers[name].load_state_dict(o)
        for name, s in resume_schedulers.items():
            self.schedulers[name].load_state_dict(s)

    def reduce_loss_dict(self, loss_dict):
        """reduce loss dict.
        In distributed training, it averages the losses among different GPUs .
        Args:
            loss_dict (OrderedDict): Loss dict.
        """
        with torch.no_grad():
            if self.opt["dist"]:
                keys = []
                losses = []
                for name, value in loss_dict.items():
                    keys.append(name)
                    losses.append(value)
                losses = torch.stack(losses, 0)
                torch.distributed.reduce(losses, dst=0)
                if self.rank == 0:
                    losses /= self.world_size
                loss_dict = {key: loss for key, loss in zip(keys, losses)}

            log_dict = OrderedDict()
            for name, value in loss_dict.items():
                log_dict[name] = value.mean().item()

            return log_dict

    def get_current_log(self):
        return self.log_dict

    def get_bare_model(self, net):
        """Get bare model, especially under wrapping with
        DistributedDataParallel or DataParallel.
        """
        if isinstance(net, (DataParallel, DistributedDataParallel)):
            net = net.module
        return net

    def model_ema(self, net1, net2, decay=0.999):
        net_g = self.get_bare_model(net1)
        net_g_ema = self.get_bare_model(net2)

        net_g_params = dict(net_g.named_parameters())
        net_g_ema_params = dict(net_g_ema.named_parameters())

        for k in net_g_ema_params.keys():
            net_g_ema_params[k].data.mul_(decay).add_(net_g_params[k].data, alpha=1 - decay)
