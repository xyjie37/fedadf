# fedadf
an algorithm for addressing catastrophic forgetting in dynamic data heterogeneity federated learning
## data prepare
### Fmnist
cd dataset/
'''python split_data_fmnist.py'''
## run scripts
python main_fedad.py --dataset cifar100 --model resnet --num_classes 100 --epochs 100 --lr 0.1 --num_users 20 --frac 0.5 --local_ep 5 --local_bs 50 --results_save run0 --wd 0.0 --datasetpath /root/FedAnc/dataset/cifar100-dir-0.1-task-3 --task_num 3
