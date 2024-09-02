import copy

from torch import nn
import torch
import torch.nn.functional as F
import copy
import numpy as np
import os, random
import torch.nn as nn

'''
class Anchor_Loss2(nn.Module):

    def __init__(self, anchor):
        super(Anchor_Loss2, self).__init__()
        self.reject_threshold = 0
        self.anchors = anchor.data
        self.num_class = len(self.anchors)
        # self.update_anchors("rep_mnist_anchor.pth")

    def forward(self, x, y):
        
        loss_mse = nn.MSELoss()
        uniq_l, uniq_c = y.unique(return_counts=True)
        anchors = copy.deepcopy(x.data)
        loss = 0.0
        for i, label in enumerate(uniq_l):
            if uniq_c[i] <= self.reject_threshold:
                continue
            # anchors[y == label, :] = self.anchors[label, :]
            cur_data = anchors[y == label, :]

            anchor = torch.ones_like(cur_data)
            anchor[:] = self.anchors[label, :]

            loss1 = loss_mse(cur_data, anchor)

            loss2 = 0.0
            for j in range(self.num_class):
                if j != label:
                    anchor[:] = self.anchors[j, :]
                    loss2 += loss_mse(cur_data, anchor)
            
            cur_loss = loss1 - loss2

            loss += cur_loss

        # loss1 = loss_mse(anchors, x)

        # l2_loss = torch.cdist(anchors, x).mean()

        return loss

    def update_anchors(self, path):
        self.anchors = torch.load(path)


class Anchor_Loss3():

    def __init__(self, anchor):
        super(Anchor_Loss2, self).__init__()
        self.reject_threshold = 0
        self.anchors = anchor.data
        self.num_class = len(self.anchors)
        self.loss_mse = nn.MSELoss()

    def get_loss(self, x, y):
        
        uniq_l, uniq_c = y.unique(return_counts=True)
        anchors = torch.ones_like(x)
        loss = 0.0
        for i, label in enumerate(uniq_l):
            if uniq_c[i] <= self.reject_threshold:
                continue
            # anchors[y == label, :] = self.anchors[label, :]
            anchors[y == label, :] = self.anchors[label, :]

        loss1 = self.loss_mse(x, anchors)

            # loss2 = 0.0
            # for j in range(self.num_class):
            #     if j != label:
            #         anchor[:] = self.anchors[j, :]
            #         loss2 += loss_mse(cur_data, anchor)
            
            # cur_loss = loss1 - loss2
           

        return loss1

    def update_anchors(self, path):
        self.anchors = torch.load(path)
'''
class FaAnchor_Loss(nn.Module):

    def __init__(self, anchor):
        super(FaAnchor_Loss, self).__init__()
        self.reject_threshold = 0
        self.anchors = anchor.data
        self.num_class = len(self.anchors)

    def forward(self, x, y):
        loss_mse = nn.MSELoss()
        uniq_l, uniq_c = y.unique(return_counts=True)
        anchors = copy.deepcopy(x.data)
        loss = 0.0
        for i, label in enumerate(uniq_l):
            if uniq_c[i] <= self.reject_threshold:
                continue
            cur_data = anchors[y == label, :]

            anchor = torch.ones_like(cur_data)
            anchor[:] = self.anchors[label, :]

            centre_dis = cur_data - anchor  # compute distance between input and anchors
            pow_ = torch.pow(centre_dis, 2)  # square
            sum_1 = torch.sum(pow_, dim=1)  # sum all distance
            dis_ = torch.div(sum_1, uniq_c[i].float())  # mean by class
            cur_loss = dis_.mean()  # mean loss

            loss += cur_loss

        return loss

    def update_anchors(self, path):
        self.anchors = torch.load(path)

class contrastive_Loss(nn.Module):

    def __init__(self, anchor):
        super(contrastive_Loss, self).__init__()
        self.reject_threshold = 0
        self.anchors = anchor.data
        self.num_class = len(self.anchors)
        self.loss_mse = nn.MSELoss()

    def get_loss(self, x, y):
        uniq_l, uniq_c = y.unique(return_counts=True)
        anchors = torch.ones_like(x)
        loss = 0.0
        for i, label in enumerate(uniq_l):
            if uniq_c[i] <= self.reject_threshold:
                continue
            anchors[y == label, :] = self.anchors[label, :]

            centre = self.anchors[label, :]
            counter = torch.histc(y, bins=self.num_class, min=0, max=self.num_class-1)
            count = counter[y.long()]
            scores_matrix = torch.mm(x, centre.T)
            exp_scores_matrix = torch.exp(scores_matrix)
            total = torch.sum(exp_scores_matrix, dim=1)
            loss_vector = torch.zeros_like(total)

            for index, value in enumerate(y):
                loss_vector[index] = -torch.log(exp_scores_matrix[index][value.long()]/total[index])

            loss += torch.mean(loss_vector)

        return loss

    def update_anchors(self, path):
        self.anchors = torch.load(path)