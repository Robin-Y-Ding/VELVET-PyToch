# VELVET-PyToch

In this repository, we provide a PyToch implementation of the paper, "VELVET: a noVel Ensemble Learning approach to automatically locate VulnErable sTatements", to compensate the drawbacks of the original implementation with the outdated TensorFlow. 

With this PyTorch implementation, it is 
1. Easier for PyTorch users to adapt and customize for further development on top of VELVET.
2. Enabling the potential integration of the pre-trained Transformer checkpoints, such as CodeBERT.

## Environment Setup

```sh
git clone https://github.com/Robin-Y-Ding/VELVET-PyToch.git;
cd VELVET-PyToch;
conda create -n velvet-pytorch Python=3.9.7;
conda activate velvet-pytorch;
pip install -r requirements.txt;
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia;
```

## Train VELVET

Download the popular vulnerability detection dataset with statement-level labels, [BigVul](https://dl.acm.org/doi/10.1145/3379597.3387501), pre-processed by [LineVul](https://github.com/davidhin/linevd/tree/main). If gdown does not work, please use the browser to directly access and download the link.

```sh
gdown https://drive.google.com/uc?id=1HTNrPo0w5JApBSRdEJzm7GBVcMfclIte;
gdown https://drive.google.com/uc?id=1lVpbadbKPGkwH3PSyUGqDMlZhxkkriDj;
gdown https://drive.google.com/uc?id=11wNxHMMDSTUYgv58Nyo5cPWqmspJdrsv;
```

Run the script with the following command:

```sh
python run_velvet.py \
    --model_name=velvet_model.bin \
    --output_dir=./saved_models \
    --do_train \
    --do_test \
    --train_data_file=<PATH_TO_processed_train.csv> \
    --eval_data_file=<PATH_TO_processed_val.csv> \
    --test_data_file=<PATH_TO_processed_test.csv> \
    --joern_output_dir=<PATH_TO_joern_output> \
    --epochs 10 \
    --encoder_block_size 512 \
    --train_batch_size 64 \
    --eval_batch_size 64 \
    --learning_rate 5e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 123456  2>&1 | tee train_velvet.log
```



# Citation

### VELVET

```
@inproceedings{ding2022velvet,
author = {Y. Ding and S. Suneja and Y. Zheng and J. Laredo and A. Morari and G. Kaiser and B. Ray},
booktitle = {2022 IEEE International Conference on Software Analysis, Evolution and Reengineering (SANER)},
title = {VELVET: a noVel Ensemble Learning approach to automatically locate VulnErable sTatements},
year = {2022},
issn = {1534-5351},
pages = {959-970},
keywords = {location awareness;codes;neural networks;static analysis;software;data models;security},
doi = {10.1109/SANER53432.2022.00114},
url = {https://doi.ieeecomputersociety.org/10.1109/SANER53432.2022.00114},
publisher = {IEEE Computer Society},
address = {Los Alamitos, CA, USA},
month = {mar}
}

```

# Acknowledgment

This repository is initialized from [Michael Fu's re-implementation of VELVET](https://github.com/optimatch/optimatch), which serves as a baseline in his paper, [Optimatch](https://arxiv.org/abs/2306.06109). Thanks, Michael, for the excellent re-implementation of VELVET!