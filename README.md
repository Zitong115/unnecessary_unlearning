This is the code for ***avoiding unnecessary unlearning***.

# Requirements

To run this code, please install packages according to ''requirements.txt'' in this repository.

# File organization

the folder is organized as

    .
    ├─config
    ├─dataset
    ├─log
    ├─MIA
    ├─models
    └─src

Folder ''config'' contains a sample config file for removing 100 samples from the CIFAR-10 dataset.

Folder ''dataset'' contains datasets downloaded from common resources and it is organized as follows.

    dataset
    ├── cifar-100-python
    │   ├── file.txt~
    │   ├── meta
    │   ├── test
    │   └── train
    ├── cifar-10-batches-py
    │   ├── batches.meta
    │   ├── data_batch_1
    │   ├── data_batch_2
    │   ├── data_batch_3
    │   ├── data_batch_4
    │   ├── data_batch_5
    │   ├── readme.html
    │   └── test_batch
    └── MNIST
        ├── processed
        └── raw

Folders "log", "MIA" and "models" are used to save logs and models generated during the experiment. You may change the path in the config file, i.e. fields "dataset_path", "model_save_path", "log_path", "attack_path", to change the location where these results are stored.

# How to use the code

To run this code, you may use the following cmd in the "src" folder.

    python main.py --config ../config/filter_rfmodel_CIFAR10_100.yml

You can choose to run different kinds of experiments by setting different parameters in the config file. Here is explanation about some parameters.

- log_appendix: The appendix of the log file. It can be any string for identification.
- dataset: The name of the dataset you are to use. Choose from {"MNIST","CIFAR10","CIFAR100"}
- model: The name of the model you are to use. For dataset MNIST, it's "2-layer-CNN". For "CIFAR10", it's "ResNet-18".
- model_pretrained: Whether to use a pretrained ResNet-18 or not.
- original_training_exp: Whether to train an original model or not.
- unlearning_request_filter_exp: Whether to experiment on reduced unlearning dataset or not.
- filter_method: Methods to reduce data removal requests. "rfmodel" is our proposed method, while "clustering", "confidence", and "curvature" are alternative methods entailed in our paper.
- score_thres_dict: The thresholds for the chosen filter_method. For "rfmodel", there is no such threshold. The threshold value has the following mapping: {-1: avg(s)+std(s), -2:avg(s), -3:max(avg(s)-std(s), 0.001)}. The indication of avg(s), std(s) please refer to our paper.
- retraining_exp: Whether or not to run the retraining experiment. 
- unlearning_data_selection: Unlearning scenarios. Choosing from {"Random", "Byclass"}.
- unlearning_proportion: The number of removal requests.
- distribution_mining_exp: Whether or not to calculate the score for alternative methods of the given dataset. You should run this experiment before you reduce the number of data removal requests using alternative methods.
- SISA_exp: Whether to run the case study experiment or not.
- Hessian_unlearning_exp: Whetehr to run the CR approach or not.