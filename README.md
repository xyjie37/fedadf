# FedADF
An algorithm for tackling catastrophic forgetting in dynamic data heterogeneity federated learning
## Data Prepare
### Cifar100
#### Distribution of Dilikere：
Change alpha and task first and then run:  

`python dataset/split_data_cifar100.py`
#### Distribution of Block:
Change task first and then run:  

`python dataset/split_block_cifar100.py`  
Operate similarly on Fmnist and Cifar10 dataset.Miniimagenet requires an extra bit of preparation first,you need to run `data_prepare.py` first and then operate similarily.
## Run Scripts
Choose Resnet18 on Cifar100 and Mimiimagenet,CNN on fmnist and Cifar10.You could run through:  

`python main_fedad.py --dataset cifar100 --model resnet18 --num_classes 100 --epochs 100 --lr 0.1 --num_users 20 --frac 0.5 --local_ep 5 --local_bs 50 --results_save run0 --wd 0.0 --datasetpath /root/FedAnc/dataset/cifar100-dir-0.1-task-3 --task_num 3`
