#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import math
import pdb
import copy
from torch.optim import Optimizer

from AnchorLossfa import FaAnchor_Loss
from AnchorLoss2 import Anchor_Loss2
from utils.data_utils import load_train_data, load_test_data
from torch.utils.data import DataLoader
import numpy as np
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)



class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label

    
class LocalUpdate(object):
    def __init__(self, args, dataset=None, idxs=None, task=0, pretrain=False):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        # self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)#读取数据集
        self.ldr_train = load_train_data(dataset, idxs, task, batch_size=self.args.local_bs)
        self.pretrain = pretrain

    def train(self, net, lr, idx=-1, local_eps=None):
        net.train()

        # train and update
        
        # For ablation study
        optimizer = torch.optim.SGD(net.parameters(), lr=lr,
                                    momentum=self.args.momentum,
                                    weight_decay=self.args.wd)

        epoch_loss = []
        
        if local_eps is None:
            if self.pretrain:
                local_eps = self.args.local_ep_pretrain
            else:
                local_eps = self.args.local_ep
        for iter in range(local_eps):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                logits = net(images)
                loss = self.loss_func(logits, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                batch_loss.append(loss.item())

            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)
    
class LocalUpdateBabu(object):
    def __init__(self, args, dataset=None, idxs=None, pretrain=False, task = 0):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        # self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)#读取数据集
        self.ldr_train = load_train_data(dataset, idxs, task, batch_size=self.args.local_bs)
        self.pretrain = pretrain


    def train(self, net, body_lr, idx=-1, local_eps=None):
        net.train()

        # train and update
        body_params = [p for name, p in net.named_parameters() if 'linear' not in name]
        
        optimizer = torch.optim.SGD(body_params, lr = body_lr, momentum=self.args.momentum, weight_decay=self.args.wd)

        epoch_loss = []
        
        if local_eps is None:
            if self.pretrain:
                local_eps = self.args.local_ep_pretrain
            else:
                local_eps = self.args.local_ep
        for iter in range(local_eps):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                logits = net(images)

                loss = self.loss_func(logits, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                batch_loss.append(loss.item())

            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)


    
    
def balanced_softmax_loss(labels, logits, sample_per_class, reduction="mean"):
    """Compute the Balanced Softmax Loss between `logits` and the ground truth `labels`.
    Args:
    labels: A int tensor of size [batch].
    logits: A float tensor of size [batch, no_of_classes].
    sample_per_class: A int tensor of size [no of classes].
    reduction: string. One of "none", "mean", "sum"
    Returns:
    loss: A float tensor. Balanced Softmax Loss.
    """
    spc = sample_per_class.type_as(logits)
    spc = spc.unsqueeze(0).expand(logits.shape[0], -1)
    logits = logits + spc.log()
    loss = F.cross_entropy(input=logits, target=labels, reduction=reduction)
    return loss
    



class LocalUpdateFedProx(object):
    def __init__(self, args, dataset=None, idxs=None, pretrain=False, task = 0):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.ldr_train = load_train_data(dataset, idxs, task, batch_size=self.args.local_bs)
        self.pretrain = pretrain

    def train(self, net, body_lr, head_lr):
        net.train()
        g_net = copy.deepcopy(net)
        
        body_params = [p for name, p in net.named_parameters() if 'linear' not in name]
        head_params = [p for name, p in net.named_parameters() if 'linear' in name]
        
        optimizer = torch.optim.SGD([{'params': body_params, 'lr': body_lr},
                                     {'params': head_params, 'lr': head_lr}],
                                    momentum=self.args.momentum,
                                    weight_decay=self.args.wd)

        epoch_loss = []
        
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                logits = net(images)

                loss = self.loss_func(logits, labels)
                
                # for fedprox
                fed_prox_reg = 0.0
                for l_param, g_param in zip(net.parameters(), g_net.parameters()):
                    fed_prox_reg += (0.1 / 2 * torch.norm((l_param - g_param)) ** 2)
                loss += fed_prox_reg
                
                loss.backward()
                optimizer.step()

                batch_loss.append(loss.item())

            epoch_loss.append(sum(batch_loss)/len(batch_loss))
            
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss) 
    
    



class NTD_Loss(nn.Module):
    def __init__(self, num_classes, tau=3, beta=1):
        super(NTD_Loss, self).__init__()
        self.CE = nn.CrossEntropyLoss()
        self.MSE = nn.MSELoss()
        self.KLDiv = nn.KLDivLoss(reduction="batchmean")
        self.tau = tau
        self.beta = beta
        self.num_classes = num_classes 
    def forward(self, logits, targets, dg_logits):
        ce_loss = self.CE(logits, targets)
        ntd_loss = self._ntd_loss(logits, dg_logits, targets)

        loss = ce_loss + self.beta * ntd_loss

        return loss

    def _ntd_loss(self, logits, dg_logits, targets):
        T = self.tau  
        dg_probs = F.softmax(dg_logits / T, dim=1)
        student_probs = F.softmax(logits / T, dim=1)
        kl_div_loss = self.KLDiv(F.log_softmax(logits / T, dim=1), dg_probs)
        kl_div_loss /= self.num_classes

        return kl_div_loss
    
class LocalUpdateFedAnchor(object):
    def __init__(self, args, anchor=None, dataset=None, idxs=None, task=0, pretrain=False):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.an_loss = Anchor_Loss2(anchor=anchor.to(args.device))
        self.selected_clients = []
        # self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)#读取数据集
        self.ldr_train = load_train_data(dataset, idxs, task, batch_size=self.args.local_bs)
        self.pretrain = pretrain

    def train(self, net, lr, idx=-1, local_eps=None):
        net.train()

        # train and update
        body_params = [p for name, p in net.named_parameters() if 'linear' not in name]
        head_params = [p for name, p in net.named_parameters() if 'linear' in name]

        for para in head_params:
            para.requried_grad = False


        # For ablation study
        optimizer = torch.optim.SGD(body_params, lr=lr,
                                    momentum=self.args.momentum,
                                    weight_decay=self.args.wd)

        epoch_loss = []

        if local_eps is None:
            if self.pretrain:
                local_eps = self.args.local_ep_pretrain
            else:
                local_eps = self.args.local_ep
        for iter in range(local_eps):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                features = net.extract_features(images)
                logits = net.only_liner(features)
                loss1 = self.loss_func(logits, labels)
                loss2 = self.an_loss(features, labels)
                loss = loss1 + 0.2 * loss2
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))

        agg_protos_label = {}
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                features = net.extract_features(images)
                uniq_l = labels.unique()
                for i, label in enumerate(uniq_l):
                    if label.item() in agg_protos_label:
                        torch.cat((agg_protos_label[label.item()], features.cpu()[labels == label.item(), :]), 0)
                    else:
                        agg_protos_label[label.item()] = features.cpu()[labels == label.item(), :]

        return net.state_dict(), sum(epoch_loss) / len(epoch_loss), agg_protos_label
def refine_as_not_true(logits, targets, num_classes):
    nt_positions = torch.arange(0, num_classes).to(logits.device)
    nt_positions = nt_positions.repeat(logits.size(0), 1)
    nt_positions = nt_positions[nt_positions[:, :] != targets.view(-1, 1)]
    nt_positions = nt_positions.view(-1, num_classes - 1)

    logits = torch.gather(logits, 1, nt_positions)

    return logits   


class LocalUpdateNTD(object):
    def __init__(self, args, dataset=None, task=0, idxs=None):
        self.args = args
        self.dataset = dataset
        self.idxs = idxs
        self.loss_func = NTD_Loss(num_classes=args.num_classes, tau=args.tau, beta=args.beta)
        self.ldr_train = load_train_data(dataset, idxs, task, batch_size=self.args.local_bs)

    def train(self, net, lr=None):
        self.optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)

        net.train()
        
        epoch_loss = []
        num_updates = 0
        
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                logits = net(images)
                dg_logits = net(images).detach()
                loss = self.loss_func(logits, labels, dg_logits)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                num_updates += 1
                batch_loss.append(loss.item())
            
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
        
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)


class LocalUpdateFA(nn.Module):
    def __init__(self, args, anchor=None, dataset=None, idxs=None, task=0, pretrain=False):
        super(LocalUpdateFA, self).__init__()
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
#         self.an_loss = FaAnchor_Loss(anchor=anchor.to(args.device))
        num_classes = args.num_classes if hasattr(args, 'num_classes') else num_classes
        self.selected_clients = []
        self.ldr_train = load_train_data(dataset, idxs, task, batch_size=self.args.local_bs)
        self.pretrain = pretrain
        self.anchor = anchor

    def train(self, net, lr, idx=-1, local_eps=None):
        net.train()
        local_eps = self.args.local_ep
        # 动态学习率调整
        if self.args.wd != 0:
            lr = lr * (self.args.wd** idx)

        # 初始优化器只优化除分类器外的参数
        # 将生成器转换为列表，以便可以使用len()函数
        all_params = list(net.parameters())
        linear_params = list(net.linear.parameters()) if hasattr(net, 'linear') else []
        body_params = [p for name, p in net.named_parameters() if 'linear' not in name]
        
        optimizer = torch.optim.SGD(body_params, lr=lr,
                                momentum=self.args.momentum,
                                weight_decay=self.args.wd)

        # 分类器的独立优化
        head_params = [p for p in all_params if id(p) in [id(param) for param in net.linear.parameters()]]
        optimizer_c = torch.optim.Adam(head_params, lr=self.args.head_lr) 

        epoch_loss = []
        for epoch in range(local_eps):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                features = net.extract_features(images)
                logits = net.only_liner(features)
                loss1 = self.loss_func(logits, labels)
                loss2 = classifier_calibration_loss(net.linear.weight, self.anchor, self.args.num_classes)
                loss = loss1 + 0.5 * loss2
                #loss = loss1
                # 反向传播和优化整个网络
                loss.backward()
                optimizer.step()
                # 反向传播和优化分类器
#                 optimizer_c.zero_grad()
#                 # 重新计算分类器的损失，这里使用.detach()来避免计算features的梯度
#                 loss2 = self.an_loss(features.detach(), labels)
#                 loss2.requires_grad_()
#                 loss2.backward()
#                 optimizer_c.step()

                optimizer.zero_grad()

                batch_loss.append(loss.item())

            epoch_loss.append(sum(batch_loss) / len(batch_loss))
        agg_protos_label = {}
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                features = net.extract_features(images)
                uniq_l = labels.unique()
                for i, label in enumerate(uniq_l):
                    if label.item() in agg_protos_label:
                        agg_protos_label[label.item()] = torch.cat((agg_protos_label[label.item()], features.cpu()[labels == label.item(), :]), 0)
                    else:
                        agg_protos_label[label.item()] = features.cpu()[labels == label.item(), :]

        return net.state_dict(), sum(epoch_loss) / len(epoch_loss), agg_protos_label

def classifier_calibration_loss(classifier_params, feature_anchors, num_classes):
    """
    计算分类器校准损失。
    
    :param classifier_params: 客户端的分类器参数 (phi_i)
    :param feature_anchors: 特征锚点 (a_c)
    :param num_classes: 类别总数 (C)
    :return: 分类器校准损失
    """
    logits = torch.matmul(classifier_params, feature_anchors.T)  # 计算分类器输出
    exp_logits = torch.exp(logits)  # 对输出进行指数运算
    log_prob = torch.log(exp_logits / torch.sum(exp_logits, dim=1, keepdim=True))  # 计算对数概率
    loss = -torch.mean(torch.sum(log_prob, dim=1))  # 计算平均损失
    return loss

class AnchorDistillationLoss(nn.Module):
    def __init__(self, student_outputs, teacher_outputs, anchors, temperature=1.0, device='cuda'):
        """
        初始化AnchorDistillationLoss模块。
        参数:
        - student_outputs: 学生模型输出的logits，维度 [batch_size, C]
        - teacher_outputs: 教师模型输出的概率分布，维度 [batch_size, C]
        - anchors: 特征锚点的集合，维度 [C, anchor_feature_dim]
        - temperature: 温度参数，用于调整softmax分布的平滑程度
        - device: 设备，默认为 'cuda'
        """
        super(AnchorDistillationLoss, self).__init__()
        self.anchors = nn.Parameter(anchors, requires_grad=False)  # 锚点不应梯度下降
        self.temperature = temperature
        self.student_outputs = student_outputs
        self.teacher_outputs = teacher_outputs
        self.device = device

        # 确保 anchors 的维度是 [num_classes, num_classes]
        num_classes = student_outputs.size(1)
        if anchors.size(1) != num_classes:
            # 如果 anchors 的第二维度与 num_classes 不匹配，则进行调整
            # 使用线性变换来调整 anchors 的维度
            self.anchors = self.adjust_anchors(anchors, num_classes)

    def adjust_anchors(self, anchors, num_classes):
        """
        调整 anchors 的维度以匹配 num_classes。
        """
        # 假设我们使用一个简单的线性变换来调整 anchors 的大小
        linear_transform = nn.Linear(anchors.size(1), num_classes).to(self.device)
        adjusted_anchors = linear_transform(anchors)
        return nn.Parameter(adjusted_anchors, requires_grad=False)

    def forward(self):
        """
        计算蒸馏损失。
        返回:
        - loss: 计算得到的蒸馏损失
        """
        # 将锚点的形状调整为 [1, C, C]，以便可以批次运算
        anchors_expanded = self.anchors.unsqueeze(0)
        
        # 计算学生模型的softmax概率
        student_probs = F.softmax(self.student_outputs / self.temperature, dim=1)
        
        # 计算学生模型在锚点上的特征表示
        student_features = torch.matmul(student_probs, anchors_expanded.squeeze(0))
        
        # 确保 student_features 的形状与 teacher_outputs 匹配
        if student_features.size(1) != self.teacher_outputs.size(1):
            # 使用线性变换来调整 student_features 的形状
            linear_transform = nn.Linear(student_features.size(1), self.teacher_outputs.size(1)).to(self.device)
            student_features = linear_transform(student_features)

        # 使用teacher_outputs作为真实分布，student_features作为预测分布
        # 计算蒸馏损失
        loss = -torch.sum(self.teacher_outputs * F.log_softmax(student_features / self.temperature, dim=1), dim=1)
        
        # 取所有样本损失的平均值作为批次的蒸馏损失
        loss = torch.mean(loss)
        
        return loss

class LocalUpdateFedAD(object):
    def __init__(self, args, anchor=None, dataset=None, idxs=None, task=0, pretrain=False):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.ldr_train = load_train_data(dataset, idxs, task, batch_size=self.args.local_bs)
        self.pretrain = pretrain
        self.anchor = anchor.to(self.args.device) 
        self.num_classes = args.num_classes
        
        # 初始化 Anchor_Contrastive_AlignmentLoss

    def train(self, net, teacher_net, lr, idx=-1, local_eps=None):
        net.train()
        teacher_net.eval()  # 确保教师模型处于评估模式
        num_classes, feat_dim = self.anchor.size()
        
        # 调整教师模型的输出维度
        if self.args.dataset == 'fmnist':
            input_tensor = torch.randn(1, 1, 28, 28).to(self.args.device)
        else:
            input_tensor = torch.randn(1, 3, 32, 32).to(self.args.device)
        with torch.no_grad():
            teacher_output_size = teacher_net(input_tensor).size(1)
        self.teacher_output_adjuster = nn.Linear(teacher_output_size, 100).to(self.args.device)

        # train and update
        body_params = [p for name, p in net.named_parameters() if 'linear' not in name]
        head_params = [p for name, p in net.named_parameters() if 'linear' in name]

        for para in head_params:
            para.requires_grad = False

        optimizer = torch.optim.SGD(body_params, lr=lr,
                                    momentum=self.args.momentum,
                                    weight_decay=self.args.wd)

        epoch_loss = []

        if local_eps is None:
            if self.pretrain:
                local_eps = self.args.local_ep_pretrain
            else:
                local_eps = self.args.local_ep

        for iter in range(local_eps):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                features = net.extract_features(images)
                logits = net.only_liner(features)  # 使用 liner 层作为分类器
                
                loss1 = self.loss_func(logits, labels)
                
                # 获取教师模型的输出并调整其维度
                with torch.no_grad():
                    teacher_outputs = teacher_net(images)
                    adjusted_teacher_outputs = self.teacher_output_adjuster(teacher_outputs)

                # 计算蒸馏损失
                distillation_loss = AnchorDistillationLoss(logits, adjusted_teacher_outputs, self.anchor, temperature=1.0)()
                #loss2 = alpha * distillation_loss + (1 - alpha) * calibration_loss
                loss = loss1 + 0.2 * distillation_loss
                # loss = loss1 + distillation_loss
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))

        agg_protos_label = {}
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                features = net.extract_features(images)
                uniq_l = labels.unique()
                for i, label in enumerate(uniq_l):
                    if label.item() in agg_protos_label:
                        agg_protos_label[label.item()] = torch.cat((agg_protos_label[label.item()], features.cpu()[labels == label.item(), :]), 0)
                    else:
                        agg_protos_label[label.item()] = features.cpu()[labels == label.item(), :]

        return net.state_dict(), sum(epoch_loss) / len(epoch_loss), agg_protos_label


class LocalUpdateMLP(object):
    def __init__(self, args, dataset=None, idxs=None, task=0, pretrain=False):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.ldr_train = load_train_data(dataset, idxs, task, batch_size=self.args.local_bs)
        self.pretrain = pretrain
        self.prototypes = None  # 存储本地原型
        self.num_classes = args.num_classes  

    def update_prototypes(self, net, dataloader):
        """
        更新本地原型。
        :param net: 模型的网络部分，用于提取特征
        :param dataloader: 用于获取数据的Dataloader
        """
        net.eval()  # 切换到评估模式以提取特征
        features = []
        labels = []
        with torch.no_grad():  # 禁用梯度计算
            for images, label in dataloader:
                images = images.to(self.args.device)
                feature = net(images)  # 特征提取
                features.append(feature.cpu())  # 收集特征
                labels.append(label.cpu())

        # 将数据转换为张量
        features = torch.cat(features, dim=0)
        labels = torch.cat(labels, dim=0)
        
        # 计算每个类别的原型
        unique_labels = torch.unique(labels)
        self.prototypes = torch.zeros(self.num_classes, features.size(1))  # 固定长度的原型列表
        for i, lbl in enumerate(unique_labels):
            idx = (labels == lbl).nonzero().view(-1)
            if len(idx) > 0:
                self.prototypes[lbl] = torch.mean(features[idx], dim=0)
    def update_semantic_prototypes(self, net, dataloader, num_clusters):
        """
        使用k-means聚类算法更新语义原型。
        :param net: 模型的网络部分，用于提取特征
        :param dataloader: 用于获取数据的Dataloader
        :param num_clusters: 要确定的语义原型数量
        :return: 计算得到的语义原型张量
        """
        net.eval()  # 切换到评估模式以提取特征
        features_all = []
        labels_all = []
        with torch.no_grad():  # 禁用梯度计算
            for images, labels in dataloader:
                images = images.to(self.args.device)
                features = net(images)  # 特征提取
                features_all.append(features.cpu())  # 收集特征
                labels_all.append(labels.cpu())

        # 将数据转换为numpy数组以进行聚类
        features_all = np.concatenate(features_all, axis=0)
        labels_all = np.concatenate(labels_all, axis=0)

        # 使用k-means算法进行聚类
        kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(features_all)

        # 根据聚类结果初始化语义原型
        semantic_prototypes = torch.zeros(num_clusters, features_all.shape[1])
        for i in range(num_clusters):
            # 选择属于当前聚类的所有样本
            cluster_members = (kmeans.labels_ == i).nonzero()[0]
            if len(cluster_members) > 0:
                # 计算并更新语义原型
                semantic_prototypes[i] = torch.from_numpy(np.mean(features_all[cluster_members], axis=0))

        return semantic_prototypes
    def prototype_regularization_loss(self, features, prototypes):
        """
        计算原型正则化损失。
        
        :param features: 模型输出的特征向量。
        :param prototypes: 类原型。
        :return: 原型正则化损失。
        """
        features = features.unsqueeze(1)  # [N, C] -> [N, 1, C]
        prototypes = prototypes.unsqueeze(0)  # [C, D] -> [1, C, D]
        dist = torch.norm(features - prototypes, dim=2)  # 计算特征和原型之间的距离
        min_dist, _ = torch.min(dist, dim=1)  # 找到最近的原型 [N, 1]
        reg_loss = torch.mean(min_dist)  # 计算原型正则化损失
        return reg_loss
    def semantic_prototype_regularization_loss(self, features, semantic_prototype):
        """
        计算语义原型正则化损失。

        :param features: 模型输出的特征向量。
        :param semantic_prototype: 语义原型。
        :return: 语义原型正则化损失。
        """
        features = features.unsqueeze(1)  # [N, C] -> [N, 1, C]
        semantic_prototype = semantic_prototype.unsqueeze(0)  # [C, D] -> [1, C, D]
        dist = torch.norm(features - semantic_prototype, dim=2)  # 计算特征和语义原型之间的距离
        min_dist, _ = torch.min(dist, dim=1)  # 找到最近的语义原型 [N, 1]
        semantic_reg_loss = torch.mean(min_dist)  # 计算语义原型正则化损失
        return semantic_reg_loss
    def federated_inter_task_regularization_loss(self, old_prototypes, new_prototypes, alpha=0.5):
        """
        计算跨任务正则化损失。
        
        :param old_prototypes: 旧任务的原型。
        :param new_prototypes: 新任务的原型。
        :param alpha: 正则化系数。
        :return: 跨任务正则化损失。
        """
        # 确保旧原型和新原型的尺寸一致
        assert old_prototypes.size() == new_prototypes.size(), "Prototype sizes do not match!"
        
        # 计算原型之间的距离
        dist = torch.norm(old_prototypes - new_prototypes, dim=1)
        
        # 使用KL散度作为损失函数
        kl_divergence = torch.nn.functional.kl_div(
            torch.log(1e-8 + torch.exp(dist)),
            torch.zeros_like(dist), 
            reduction='sum'
        )
        
        # 返回加权的跨任务正则化损失
        return alpha * kl_divergence / dist.size(0)

    def train(self, net, lr, now_epoch, prototype, semantic_prototype, idx=-1, local_eps=None):
        net.train()
        optimizer = torch.optim.SGD(net.parameters(), lr=lr,
                                    momentum=self.args.momentum,
                                    weight_decay=self.args.wd)

        epoch_loss = []
        reg_loss_sum = 0
        if local_eps is None:
            local_eps = self.args.local_ep
        self.update_prototypes(net, self.ldr_train)
        for iter in range(local_eps):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                logits = net(images)
                loss = self.loss_func(logits, labels)
                if now_epoch > 0:
                    # 计算原型正则化损失
                    features = net(images)  # 重新获取特征，因为原型正则化需要
                    reg_loss = self.prototype_regularization_loss(features, prototype)
                    reg_loss_sum += reg_loss.item()  # 累积原型正则化损失
                    semantic_reg_loss = self.semantic_prototype_regularization_loss(features, semantic_prototype)
                    inter_task_reg_loss = self.federated_inter_task_regularization_loss(
                        self.prototypes, 
                        prototype
                    )
                    # 总损失是交叉熵损失和原型正则化损失的和
                    total_loss = loss + 0.5 * reg_loss + 0.1 * semantic_reg_loss + 0.1 * inter_task_reg_loss
                else:
                    total_loss = loss
                
                # 更新模型参数
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                # 累积批次损失
                batch_loss.append(total_loss.item())

            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        # 训练完成后更新原型和语义原型
        #self.update_prototypes(net, self.ldr_train)
        semantic_prototypes = self.update_semantic_prototypes(net, self.ldr_train, num_clusters=self.num_classes // 2) # 假设我们选择类别数量一半的聚类中心
        self.semantic_prototypes = semantic_prototypes

        return net.state_dict(), sum(epoch_loss) / len(epoch_loss), self.prototypes, self.semantic_prototypes
