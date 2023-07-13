import logging
from collections import OrderedDict

import torch
import torch.nn as nn

from utils.registry import MODEL_REGISTRY

from .base_model import BaseModel

logger = logging.getLogger("base")


@MODEL_REGISTRY.register()
class DegSRModel(BaseModel):
    def __init__(self, opt):
        super().__init__(opt)

        self.data_names = ["lr", "hr"]

        self.network_names = ["netDeg1", "netDeg2", "netSR", "netD", "netSR_ema"]
        self.networks = {}

        self.loss_names = ["sr_adv", "sr_pix", "sr_percep", "deg_reg"]
        self.loss_weights = {}
        self.losses = {}
        self.optimizers = {}

        # define networks and load pretrained models
        nets_opt = opt["networks"]
        defined_network_names = list(nets_opt.keys())
        assert set(defined_network_names).issubset(set(self.network_names))

        for name in defined_network_names:
            setattr(self, name, self.build_network(nets_opt[name]))
            self.networks[name] = getattr(self, name)

        if self.is_train:
            train_opt = opt["train"]
            # setup loss, optimizers, schedulers
            self.setup_train(train_opt)

            # ema_decay
            self.ema_decay = 0 if train_opt["ema_decay"] is None else train_opt["ema_decay"]
            if self.ema_decay > 0:
                self.netSR_ema = self.build_network(opt["networks"]["netSR"])
                self.networks["netSR_ema"] = self.netSR_ema
                self.model_ema(self.netSR, self.netSR_ema, 0)

            # grad_norm
            if train_opt["grad_norm"] is not None:
                self.grad_norm = train_opt["grad_norm"]
            
            # queue
            self.queue_size = train_opt["queue_size"]

    def feed_data(self, data):
        self.gt = data["src"].to(self.device)
    
    def make_deg_label(self, deg_outs):

        self.lr = deg_outs.pop("lr")
        self.hr = deg_outs.pop("hr")

        gt_deg = []
        for k, v in deg_outs.items():
            gt_deg.append(v.view(v.shape[0], -1))

        self.gt_deg = torch.cat(gt_deg, dim=1)

    def forward(self):

        deg_outs1 = self.netDeg1(self.gt)
        deg_outs2 = self.netDeg2(deg_outs1["lr"], self.gt)

        self.deg_outs = deg_outs1
        for k, v in deg_outs2.items():
            if not (k in ["lr", "hr"]):
                if k in self.deg_outs.keys():
                    self.deg_outs[k] = torch.stack(
                        [self.deg_outs[k], v], -1
                    )
                else:
                    self.deg_outs[k] = v
                    
        self.deg_outs["hr"] = deg_outs2["hr"]
        self.deg_outs["lr"] = deg_outs2["lr"]
                
        self.dequeue_and_enqueue()
        self.make_deg_label(self.deg_outs)

        srs, degs = self.netSR(self.lr)
        self.sr = srs[-1]
        self.deg = degs[-1]

    def optimize_parameters(self, step):

        self.forward()
        loss_dict = OrderedDict()

        l_sr = 0
        if self.losses.get("sr_pix"):
            sr_pix = self.losses["sr_pix"](self.hr, self.sr)
            loss_dict["sr_pix"] = sr_pix
            l_sr += self.loss_weights["sr_pix"] * sr_pix
        
        if self.losses.get("deg_reg"):
            reg = self.losses["deg_reg"](self.deg, self.gt_deg)
            loss_dict["deg_reg"] = reg
            l_sr += self.loss_weights["deg_reg"] * reg

        if self.losses.get("sr_adv"):
            self.set_requires_grad(["netD"], False)
            sr_adv_g = self.calculate_gan_loss_G(
                self.netD, self.losses["sr_adv"], self.hr, self.sr
            )
            loss_dict["sr_adv_g"] = sr_adv_g
            l_sr += self.loss_weights["sr_adv"] * sr_adv_g

        if self.losses.get("sr_percep"):
            sr_percep, sr_style = self.losses["sr_percep"](self.hr, self.sr)
            loss_dict["sr_percep"] = sr_percep
            if sr_style is not None:
                loss_dict["sr_style"] = sr_style
                l_sr += self.loss_weights["sr_percep"] * sr_style
            l_sr += self.loss_weights["sr_percep"] * sr_percep

        self.set_optimizer(names=["netSR"], operation="zero_grad")
        l_sr.backward()
        if hasattr(self, "grad_norm"):
            self.clip_grad_norm(["netSR"], norm=self.grad_norm)
        self.set_optimizer(names=["netSR"], operation="step")

        if self.ema_decay > 0:
            self.model_ema(self.netSR, self.netSR_ema, self.ema_decay)

        if self.losses.get("sr_adv"):
            self.set_requires_grad(["netD"], True)
            sr_adv_d = self.calculate_gan_loss_D(
                self.netD, self.losses["sr_adv"], self.hr, self.sr
            )
            loss_dict["sr_adv_d"] = sr_adv_d

            self.optimizers["netD"].zero_grad()
            sr_adv_d.backward()
            self.optimizers["netD"].step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

    def calculate_gan_loss_D(self, netD, criterion, real, fake):

        d_pred_fake = netD(fake.detach().clone())
        d_pred_real = netD(real)

        loss_real = criterion(d_pred_real, True, is_disc=True)
        loss_fake = criterion(d_pred_fake, False, is_disc=True)

        return loss_real + loss_fake

    def calculate_gan_loss_G(self, netD, criterion, real, fake):

        d_pred_fake = netD(fake)
        loss_real = criterion(d_pred_fake, True, is_disc=False)

        return loss_real

    def test(self, data):
        self.real_lr = data["src"].to(self.device)

        model = self.netSR
        if hasattr(self, "netSR_ema"):
            model = self.netSR_ema

        model.eval()
        with torch.no_grad():
            srs, degs = model(self.real_lr)
            self.sr = srs[-1]
        model.train()

    def get_current_visuals(self, need_GT=True):
        out_dict = OrderedDict()
        out_dict["lr"] = self.real_lr.detach()[0].float().cpu()
        out_dict["sr"] = self.sr.detach()[0].float().cpu()
        return out_dict
    
    def save(self, iter_label):
        for name in self.optimizers.keys():
            self.save_network(self.networks[name], name, iter_label)
        if hasattr(self, "netSR_ema"):
            self.save_network(self.networks["netSR_ema"], "netSR_ema", iter_label)
    
    @torch.no_grad()
    def dequeue_and_enqueue(self):
        """It is the training pair pool for increasing the diversity in a batch.

        Batch processing limits the diversity of synthetic degradations in a batch. For example, samples in a
        batch could not have different resize scaling factors. Therefore, we employ this training pair pool
        to increase the degradation diversity in a batch.
        """
        # initialize
        b, c, h, w = self.gt.size()
        if not hasattr(self, 'queues'):
            self.queues = {}
            assert self.queue_size % b == 0, f'queue size {self.queue_size} should be divisible by batch size {b}'
            for k, v in self.deg_outs.items():
                if v is not None:
                    if len(v.shape) == 1:
                        self.queues[k] = torch.zeros(self.queue_size).to(self.device)
                    else:
                        self.queues[k] = torch.zeros(self.queue_size, *v.shape[1:]).to(self.device)
            self.queue_ptr = 0
        if self.queue_ptr == self.queue_size:  # the pool is full
            # do dequeue and enqueue
            # shuffle
            idx = torch.randperm(self.queue_size)
            for k in self.queues.keys():
                self.queues[k] = self.queues[k][idx]
            # get first b samples
            deg_outs = {}
            for k in self.queues.keys():
                deg_outs[k] = self.queues[k][:b].clone()
            # update the queue
            for k in self.queues.keys():
                self.queues[k][:b] = self.deg_outs[k].clone()
            
            self.deg_outs = deg_outs
        else:
            # only do enqueue
            for k in self.queues.keys():
                self.queues[k][self.queue_ptr:self.queue_ptr + b] = self.deg_outs[k].clone()
            self.queue_ptr = self.queue_ptr + b