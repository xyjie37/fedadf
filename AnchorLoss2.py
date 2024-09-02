import copy

from torch import nn
import torch
import torch.nn.functional as F


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
