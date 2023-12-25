# Data-Balanced Transformer for Accelerated LNP Screening in mRNA Delivery

## Getting Started

### Installation

Set up conda environment and clone the github repo. [Uni-Core](https://github.com/dptech-corp/Uni-Core/releases) is needed, please install it first. `unicore-0.0.1+ CU117torch2.0.0-CP39-CP39-linux_x86_64-whl` is recommended. 

```
# create a new environment
$ conda create --name LNPs python=3.9 -y
$ conda activate LNPs

# install requirements
$ pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1
$ pip install -r requirements.txt
#Download pretrained weights and finetune weights
$ git clone https://github.com/wklix/LNPs.git
$ cd ./LNPs
$ wget https://github.com/wklix/LNPs/releases/download/v1.0/mol_pre_all_h_220816.pt
$ wget https://github.com/wklix/LNPs/releases/download/v1.0/mol_pre_no_h_220816.pt
$ mv *.pt weights/
$ wget https://github.com/wklix/LNPs/releases/download/v1.0/LDS.zip
$ wget https://github.com/wklix/LNPs/releases/download/v1.0/FDS.zip
$ unzip LDS.zip -d ./dataset/Random
$ unzip FDS.zip -d ./dataset/Scaffold
```
### Training

You can choose between two data splitting methods: random and scaffold. The corresponding data is located in  `./dataset`. Our dataset is from [AGILE](https://github.com/bowang-lab/AGILE). If you want to select the random data splittingmethod, please go to line 19 in `./tasks/split.py` and comment out line 20. On the other hand, if you want to choose the scaffold data splitting method, please go to line 20 in `./tasks/split.py` and comment out line 19.

```
19 self.split_method = split_method    #random
20 self.split_method = '5fold_scaffold'  #scaffold
```

We have employed two data balancing : label distribution smoothing (LDS) and molecular feature distribution smoothing (FDS). 

If you want to use LDS, please add the 'weight' variable at line 132 in `./tasks/trainer.py`. Otherwise, remove the 'weight' variable. It is already included by default.

```
132 loss = torch.mean(((outputs - net_target) ** 2)*weight) 
```

If you want to use FDS, please follow these steps:

1. In `./models/unimol.py`, uncomment lines 164-166.

   ```
   164 start_smooth= 1
   165 if self.training and epoch >= start_smooth:
   166 	cls_repr = self.FDS.smooth(cls_repr1, labels, epoch) 
   ```
   
2. In `./tasks/trainer.py`', uncomment lines 165-166.

   ```
   165 model.FDS.update_last_epoch_stats(epoch)
   166 model.FDS.update_running_stats(encodings, labels1, epoch)
   ```

After the above selections have been made, you can proceed with model training:

```
# ./dataset/Random/train.csv is the path of dataset
$ python train_model.py ./dataset/Random/train.csv
```

### Test

Run `test.py` to observe the predictive results on the test set. We provide models trained under random and scaffold data splitting. The models can be found at `./dataset/Random/LDS` and `./dataset/Scaffold/FDS`.

```
$ python test.py
```

## Acknowledgement

- AGILE: [https://github.com/bowang-lab/AGILE](https://github.com/bowang-lab/AGILE)
- Unimol: [https://github.com/dptech-corp/Uni-Mol/tree/main](https://github.com/dptech-corp/Uni-Mol/tree/main)
- imbalanced-regression: [https://github.com/YyzHarry/imbalanced-regression](https://github.com/YyzHarry/imbalanced-regression)
