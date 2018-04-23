# [PYTORCH] Character-level Convolutional Networks for Text Classification

## Introduction

Here is my pytorch implementation of the model described in the paper "Character-level Convolutional Networks for Text Classification" [paper](https://arxiv.org/abs/1509.01626). 

## Datasets:

Statistics of datasets I used for experiments. These datasets could be download from [link](https://drive.google.com/drive/u/0/folders/0Bz8a_Dbh9Qhbfll6bVpmNUtUcFdjYmF2SEpmZUZUcVNiMUw1TWN6RDV3a0JHT3kxLVhVR2M)

| Dataset                | Classes | Train samples | Test samples |
|------------------------|:---------:|:---------------:|:--------------:|
| AG’s News              |    4    |    120 000    |     7 600    |
| Sogou News             |    5    |    450 000    |    60 000    |
| DBPedia                |    14   |    560 000    |    70 000    |
| Yelp Review Polarity   |    2    |    560 000    |    38 000    |
| Yelp Review Full       |    5    |    650 000    |    50 000    |
| Yahoo! Answers         |    10   |   1 400 000   |    60 000    |
| Amazon Review Full     |    5    |   3 000 000   |    650 000   |
| Amazon Review Polarity |    2    |   3 600 000   |    400 000   |

## Setting:

I almost keep default setting as described in the paper. For optimizer and learning rate, there are 2 settings I use:

- **SGD** optimizer with initial learning rate of 0.01. The learning rate is halved every 3 epochs.
- **Adam** optimizer with initial learning rate of 0.001.

Additionally, in the original model, 1 epoch is seen as a loop over batch_size*num_batch records (128*5000 or 128*10000 or 128*30000), so it means that there are records used more than once for 1 epoch. In my model, 1 epoch is a complete loop over the whole dataset, where each record is used exactly once.

## Training

If you want to train a model with common dataset and default parameters, please run the following commands:
- **python train.py -d dataset_name**, for example, python train.py -d dbpedia
If you want to train a model with common dataset and your preference parameters: For example, optimizer and learning rate, you could run:
- **python train.py -d dataset_name -p optimizer_name -l learning_rate**: For example, python train.py -d dbpedia -p sgd -l 0.01
If you want to train a model with your own dataset, please specify path to input and output folder:
- **python train.py -i path/to/input/folder -o path/to/output/folder**

## Test

For testing a trained model with your test file, please run the following command:
- **python inference.py -i path/to/test/file -p path/to/trained/model -o path/to/output/file**

You could find some trained models I have trained in [link](https://drive.google.com/open?id=1zzC4r0nn8yInWjCbVrVZPFYyOWJQizqh)

## Experiments:

Results for test set are presented as follows:  A(B)/C/D:
- **A** is accuracy reproduced here.
- **B** is accuracy reported in the paper.
- **C** is the optimizer used. **S** is SGD with initial learning of 0.01, while **A** is Adam with initial learning rate of 0.001.
- **D** is the epoch when maximum accuracy observed.

Each experiment is run over 10 epochs.

| Model scale |       ag_news    |     sogu_news    |     db_pedia     |   yelp_polarity  |    yelp_review   |   yahoo_answer   | amazon_review | amazon_polarity |
|:-------------:|:------------------:|:------------------:|:------------------:|:------------------:|:------------------:|:------------------:|:------------------:|
|    Small    | 88.20(84.35)/A/4 | 94.95(91.35)/A/9 | 97.58(98.02)/A/8 |                  |                  | 67.55(70.16)/S/9 |                  |                  |
|    Large    | 88.17(87.18)/A/2 | 95.48(95.12)/A/6 | 97.65(98.27)/A/7 | 94.21(94.11)/A/5 | 60.55(60.38)/S/5 |                  |                  |                  |

You could find detail log of each experiment containing loss, accuracy and confusion matrix at the end of each epoch in **output/datasetname_scale/logs.txt**, for example output/ag_news_small/logs.txt
