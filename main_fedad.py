#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import pickle
import numpy as np
import pandas as pd
import torch

from create_anchor import creat_anchor, agg_func, proto_aggregation
from utils.options import args_parser
from utils.train_utils import get_data, get_model
from utils.My_dataset import AnchorDataset
from models.Update import LocalUpdate, LocalUpdateFedAD
from models.test import test_img, test_img_local, test_img_local_all
import os
import gc 

import pdb

# dataset_path = 'Fmnist-client100-dir0.1'

if __name__ == '__main__':
    # parse args
    args = args_parser()
    dataset_path = args.datasetpath
    task_num = args.task_num
    
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')


    base_dir = './save/{}/{}_num{}_C{}_le{}_bs{}_round{}_m{}_lr{}/{}/'.format(
        dataset_path, args.model, args.num_users, args.frac, args.local_ep, args.local_bs, args.epochs, args.momentum, args.lr, args.results_save)
    #ase_dir = '/root/results'
    algo_dir = 'fedAD'
    
    if not os.path.exists(os.path.join(base_dir, algo_dir)):
        os.makedirs(os.path.join(base_dir, algo_dir), exist_ok=True)

    # build a global model
    net_glob = get_model(args)
    net_glob.train()

    global_anchor = None
    if(args.dataset == 'fmnist'):
        global_anchor = creat_anchor(10, 32)
    elif(args.dataset == 'cifar10'):
        global_anchor = creat_anchor(10, 256)
    elif(args.dataset == 'cifar100'):
        global_anchor = creat_anchor(100, 512)
    elif(args.dataset == 'miniimagenet'):
        global_anchor = creat_anchor(100, 512)
        # global_anchor = torch.load("cifar100_anchor.pth").detach()
    
    global_anchor = global_anchor.to(args.device)


    # build local models
    net_local_list = []
    for user_idx in range(args.num_users):
        net_local_list.append(copy.deepcopy(net_glob))
    
    # training
    results_save_path = os.path.join(base_dir, algo_dir, 'results.csv')

    loss_train = []
    net_best = None
    best_loss = None
    best_acc = None
    best_epoch = None

    lr = args.lr
    results = []
    
    for iter in range(args.epochs):
        
        w_glob = None
        loss_locals = []
        local_protos = {}
        # Client Sampling
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        # print("Round {}, lr: {:.6f}, {}".format(iter, lr, idxs_users))
        task=(iter//10)%task_num#每过10个轮次进行任务切换
        print('Current task: ', task)
        # Local Updates
        print(idxs_users)
        for idx in idxs_users:
            #数据集名字，序号
            local = LocalUpdateFedAD(args=args, anchor=global_anchor, dataset=dataset_path, idxs=idx, task = task)
            net_local = copy.deepcopy(net_glob)
            w_local, loss, reps = local.train(net=net_local.to(args.device), teacher_net=net_glob, lr=lr)
            agg_protos = agg_func(reps)
            local_protos[idx] = agg_protos
            loss_locals.append(loss)

            if w_glob is None:
                w_glob = copy.deepcopy(w_local)
            else:
                for k in w_glob.keys():
                    w_glob[k] += w_local[k]
        
        # Aggregation
        for k in w_glob.keys():
            w_glob[k] = torch.div(w_glob[k], m)
        
        new_anchor = proto_aggregation(local_protos)
        for i in range(args.num_classes):
            if i in new_anchor:
                global_anchor[i] = 0.05 * new_anchor[i].to(args.device) + 0.95 * global_anchor[i]

        embedded_list = []
        label_list = []
        # print(len(global_anchor))
        for i in range (len(global_anchor)):
            mean = global_anchor.detach().cpu().numpy()[i]
            temp = np.random.normal(loc=mean, scale=0.1, size=(1000, len(mean)))
            label = ([i]*1000)
            embedded_list.append(temp)
            label_list = label_list + label
        embedded_list = np.vstack(embedded_list)
        label_list = np.array(label_list)
        # print(embedded_list.shape)
        # print(label_list.shape)
        train_set = AnchorDataset(embedded_list, label_list)
        anchor_dataloader = torch.utils.data.DataLoader(dataset=train_set, batch_size=1000, shuffle=True, drop_last=True)

        net_glob.load_state_dict(w_glob, strict=True)
        head_params = [p for name, p in net_glob.named_parameters() if 'linear' in name]

        optimizer = torch.optim.SGD(head_params, lr=0.002)
        loss_func = torch.nn.CrossEntropyLoss()


        for i in range(args.head_epoch):
            for batch_idx, (x, y) in enumerate(anchor_dataloader):
                x, y = x.to(args.device), y.to(args.device)
                logits = net_glob.only_liner(x)
                loss = loss_func(logits, y)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        optimizer.zero_grad()

        del anchor_dataloader
        del train_set
        del embedded_list
        del label_list

        w_glob = net_glob.state_dict()
        
        # Broadcast
        update_keys = list(w_glob.keys())
        w_glob = {k: v for k, v in w_glob.items() if k in update_keys}
        for user_idx in range(args.num_users):
            net_local_list[user_idx].load_state_dict(w_glob, strict=False)
        # net_glob.load_state_dict(w_glob, strict=False)


        # print loss
        loss_avg = sum(loss_locals) / len(loss_locals)
        loss_train.append(loss_avg)

        gc.collect()
        torch.cuda.empty_cache()

        if (iter + 1) % args.test_freq == 0:
            acc_test, acc_test_var, loss_test = test_img_local_all(net_local_list, args, dataset_test=dataset_path, task=task, return_all=False)
            # loss_test = 0.0
            # acc_test = 0.0
            print('Round {:3d}, Average loss {:.3f}, Test loss {:.3f}, Test accuracy: {:.2f}'.format(
                iter, loss_avg, loss_test, acc_test))
            # print('Round {:3d}'.format(iter))
            all_acc, all_loss = test_img(net_glob, datatest=dataset_path, args=args)

            print('All Test Data: Average loss: {:.4f}, Accuracy: {:.2f}% '.format(
                all_loss, all_acc))

            if best_acc is None or all_acc > best_acc:
                # net_best = copy.deepcopy(net_glob)
                best_acc = all_acc
                best_epoch = iter
                
                best_save_path = os.path.join(base_dir, algo_dir, 'best_model.pt')
                
                torch.save(net_glob.state_dict(), best_save_path)
                

            results.append(np.array([iter, task, loss_avg, loss_test, acc_test, all_acc, best_acc]))
            final_results = np.array(results)
            final_results = pd.DataFrame(final_results, columns=['epoch','task', 'loss_avg', 'loss_test', 'acc_test','all_acc', 'best_acc'])
            final_results.to_csv(results_save_path, index=False)
        gc.collect()
        torch.cuda.empty_cache()
    print('Best model, iter: {}, acc: {}'.format(best_epoch, best_acc))
