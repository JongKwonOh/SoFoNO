# [NeuroComputing'25] 
# <img src="https://github.com/JongKwonOh/sofono-page/blob/main/src/assets/logo.png" alt="Logo" width="28"> [SoFoNO](https://jongkwonoh.github.io/sofono-page/) : Arbitrary-Scale Image Super-Resolution via Sobolev Fourier Neural Operator

This repository contains the official implementation for SoFoNO introduced in the following [paper](https://www.sciencedirect.com/science/article/pii/S0925231225026165):

## Software Environment

This project was implemented and tested with the following software configuration:

- **Python**: 3.9
- **PyTorch**: 2.6.0
- **Torchvision**: 0.21.0
- **CUDA**: 12.4
- **cuDNN**: 9.1.0.70
- **NumPy**: 2.0.2
- **SciPy**: 1.13.1
- **OpenCV**: 4.11.0.86

## Hardware
- **GPU**: NVIDIA A100 80GB

## Download Dataset
To download and prepare the DIV2K dataset for training and validation, follow these steps:

1. **Create a data directory and navigate into it:**
   ```bash
   mkdir data
   cd data
2. **Download the DIV2K train and validation datasets:**
   ```bash 
   wget http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip
   wget http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip
3. **Unzip the downloaded files:**
   ```bash
   unzip DIV2K_train_HR.zip
   unzip DIV2K_valid_HR.zip

## Train
`python train.py --config configs/train_edsr-SoFoNO.yaml`
If you want to change SoFoNO's argument, please modify the yaml file.

```yaml
model:
  name: sofono
  args:
    encoder_spec:
      name: edsr-baseline
      args:
        no_upsampling: true
    width: 256
    T : 3
    ranges : [-1, 1]
    local_branch : Conv
    init_s : 0.0
```

## Test
Download a DIV2K pre-trained model.

Model|Download
:-:|:-:
SoFoNO|[Google Drive](https://drive.google.com/drive/folders/1OUeO5mKuWb_TXkRzyjyUdIAxevp15-cK)

`python test.py --config configs/test_SoFoNO.yaml --mcell True`
You should input the test data and model information into the yaml file.
```yaml
test_dataset:
  dataset:
    name: image-folder
    args:
      root_path: ./data/DIV2K_valid_HR
  wrapper:
      name: sr-implicit-downsampled-fast
      args:
        scale_min: 4
        scale_max: 4
        ranges : [-1, 1]
  batch_size: 1
eval_type: div2k-4
eval_bsize: 500

model_path: ./SoFoNO.pth
```

## Demo
`python demo.py --input input.png --model ./SoFoNO.pth --scale 2 --output output.png`

<!--## Citation
If you find our work useful in your research, please consider citing:
```
--- 
```-->

## Reference
- For PyTorch implementation details, see [Ketkar et al., 2021](https://pytorch.org/).

## Acknowledgements
This code is built on [LIIF](https://github.com/yinboc/liif), [LTE](https://github.com/jaewon-lee-b/lte) and [SRNO](https://github.com/2y7c3/Super-Resolution-Neural-Operator)
