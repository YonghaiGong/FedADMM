# [ICDE2022] This code is the implementation of FedADMM
[FedADMM: A Robust Federated Deep Learning Framework with Adaptivity to System Heterogeneity](https://ieeexplore.ieee.org/abstract/document/9835545)

Author: Yonghai Gong, Yichuan Li, and  Nikolaos M. Freris

The code is modified from https://github.com/AshwinRJ/Federated-Learning-PyTorch

## Requirments

Install all the packages from requirments.txt

* python = 3.8.5
* pytorch = 1.9.0
* torchvision = 0.10.0
* numpy = 1.19.2
* tensorboardx = 2.2
* matplotlib = 3.3.2

## Datasets and Models

| Model | \# of para | Dataset | Target Accuracy |
| --- | --- | --- | --- |
| CNN 1 | 1,663,370 | MNIST | 97% |
| CNN 1 | 1,663,370 | FMNIST | 80% |
| CNN 2 | 1,105,098 | CIFAR-10 | 45% |

## Dataset Distribution

*   `IID:` data are evenly distributed to clients
*   `non-IID:` first arrange the training data by label and then distribute them into shards. Each client is assigned two shards randomly.

## Options

The default values for various parameters parsed to the experiment are given in options.py. Details are given on some of those parameters:

*   `--num_users`: number of users: m
*   `--frac:` the fraction of clients: C
*   `--epochs:` number of global rounds of training
*   `--local_ep:` the number of local epochs: E
*   `--local_bs:` local batch size: B
*   `--eta:` server step size
*   `--lr:` local learning rate
*   `--rho:` coefficient of the proximal parameter:
*   `--iid:` 1 for IID, and 0 for non-IID
*   `--model:` name of the model
*   `--dataset:` name of the dataset


## Publications
If you find this repository useful for your research, please cite the following paper:
```
@inproceedings{fedadmm,
  title={Fed{ADMM}: A Robust Federated Deep Learning Framework with Adaptivity to System Heterogeneity},
  author = {Gong, Yonghai and Li, Yichuan and Freris, Nikolaos M.},
  booktitle = {ICDE},
  pages = {2575-2587},
  year = {2022}
}
```

(If you have any questions, feel free to ask me via gongyh@mail.ustc.edu.cn)
