# Two-path Target-aware Contrastive Regression for Action Quality Assessment
This repository contains the PyTorch implementation for Two-path Target-aware Contrastive Regression.

## Code for Two-path Target-aware Contrastive Regression (T²CR)
### Requirement
- Python 3.7.9
- Pytorch 1.7.1
- torchvision 0.8.2
- timm 0.3.4
- torch_videovision
```
pip install git+https://github.com/hassony2/torch_videovision
```

### Data Preperation
- The FineDiving dataset [[Xu et al.]](https://github.com/xujinglin/FineDiving)
- The MTL-AQA dataset [[Parmar and Morris]](https://github.com/ParitoshParmar/MTL-AQA)
- Download AQA-7 Dataset:
```
mkdir AQA-Seven & cd AQA-Seven
wget http://rtis.oit.unlv.edu/datasets/AQA-7.zip
unzip AQA-7.zip
```

### Pretrain Model
The Kinetics pretrained I3D downloaded from the reposity [kinetics_i3d_pytorch](https://github.com/hassony2/kinetics_i3d_pytorch/blob/master/model/model_rgb.pth)
```
model_rgb.pth
```

### Experimental Setting
```
MTL_T2CR.yaml
FineDiving_T2CR.yaml
Seven_T2CR.yaml
```

### Training and Evaluation
```
# train a model
bash ./scripts/train.sh 0,1 MTL T²CR

# resume the training process
bash ./scripts/train.sh 0,1 MTL T²CR --resume

# test a trained model
bash ./scripts/test.sh 0,1 MTL T²CR ./experiments/T2CR/MTL/T²CR/last.pth
# last.pth is obtained by train.sh and saved at "experiments/T2CR/MTL/T²CR/"
```

### More descriptions coming soon!
