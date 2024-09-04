# fedadf
an algorithm for addressing catastrophic forgetting in dynamic data heterogeneity federated learning
## data prepare
### Cifar100
#### Distribution of Dilikere：
you could change alpha and task first and then run:
`python dataset/split_data_cifar100.py`
#### Distribution of Block:
you could change task first and then run:
`python dataset/split_block_cifar100.py`  
Operate similarly on Fmnist and Cifar10 dataset
## run scripts
python main_fedad.py --dataset cifar100 --model resnet --num_classes 100 --epochs 100 --lr 0.1 --num_users 20 --frac 0.5 --local_ep 5 --local_bs 50 --results_save run0 --wd 0.0 --datasetpath /root/FedAnc/dataset/cifar100-dir-0.1-task-3 --task_num 3
