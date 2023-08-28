This is an official implementation of [RealDAN](https://arxiv.org/abs/2308.08816)

If this repo works for you, please cite our paper
```bitex
@article{luo2023end,
  title={End-to-end Alternating Optimization for Real-World Blind Super Resolution},
  author={Luo, Zhengxiong and Huang, Yan and Li, Shang and Wang, Liang and Tan, Tieniu},
  journal={International Journal of Computer Vision (IJCV)},
  year={2023}
}
```

This repo is buid on the basis of [BasicSR](https://github.com/XPixelGroup/BasicSR)

## Model Weights
Download the checkpoints of RealDAN from [BaiduYun](https://pan.baidu.com/s/1tNT6G-6vh6fCnZrvXLvBBw?pwd=ig96)(password: ig96).

Put the downloaded checkpoints into [checkpoints](./checkpoints)

The model weights and datasets are also available at huggingface[https://huggingface.co/lzxlog/RealDAN] now.

## Inference

For inference on Real-World images

```bash
cd codes/config/RealDAN
python3 inference.py \
--opt options/test/dan_edsr_gan_real.yml \
--input_dir=/dir/of/input/images \
--output_dir=/dir/of/saved/outputs
```

For inference on blurry images

```bash
cd codes/config/KernelDAN
python3 inference.py \
--opt options/test/x4.yml \
--input_dir=/dir/of/input/images \
--output_dir=/dir/of/saved/outputs
```

## Evaluation

For evaluation on DIV2K-Real, please download the [dataset](https://pan.baidu.com/s/1tNT6G-6vh6fCnZrvXLvBBw?pwd=ig96) to your own path, and run

```bash
cd codes/config/RealDAN
python3 test.py \
--opt options/test/dan_edsr_gan_syn.yml
```

and

```bash
cd codes/config/RealDAN
python3 test.py \
--opt options/test/dan_edsr_syn.yml
```


For evaluation on DIV2KRK, please download the [dataset](http://www.wisdom.weizmann.ac.il/~vision/kernelgan/DIV2KRK_public.zip) to your own path, and run

```bash
cd codes/config/KernelDAN
python3 test.py \
--opt options/test/x2.yml 
```

and

```bash
cd codes/config/KernelDAN
python3 test.py \
--opt options/test/x4.yml 
```
