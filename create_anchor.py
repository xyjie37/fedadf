import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


def creat_anchor(class_num=10, dim=32):
    random_vector = torch.randn(class_num, dim, requires_grad=True)
    optimizer = optim.SGD([random_vector], lr=0.01)
    # 训练迭代次数
    num_epochs = 1000

    for epoch in range(num_epochs):
        # 计算余弦相似度
        similarity_matrix = F.cosine_similarity(random_vector.unsqueeze(1), random_vector.unsqueeze(0), dim=2)
        euclidean_distances = torch.cdist(random_vector, random_vector)

        mask = ~torch.eye(class_num, dtype=bool)
        cosine_loss = similarity_matrix[mask].view(class_num, class_num-1)
        l2_loss = euclidean_distances[mask].view(class_num, class_num-1)
        var_loss = torch.var(random_vector, dim=1)

        loss = var_loss.sum() + cosine_loss.sum() - 0.1 * l2_loss.sum()  # 后续还可以加上对每一行向量的标准差的限制值，减少某个值过大
        # loss = var_loss.sum() + cosine_loss.sum()
        # 打印损失
        if (epoch + 1)%100 == 0:
            print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}')
        # 清零梯度，进行反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    anchor = torch.clone(random_vector).detach()
    return anchor

def agg_func(protos):
    """
    Returns the average of the weights.
    """

    for [label, proto_list] in protos.items():
        # if len(proto_list) > 1:
            # proto = 0 * proto_list[0].data
            # for i in proto_list:
            #     proto += i.data
        protos[label] = proto_list.mean(dim=0)
        # else:
        #     protos[label] = proto_list[0]

    return protos


def proto_aggregation(local_protos_list):
    agg_protos_label = dict()
    for idx in local_protos_list:
        local_protos = local_protos_list[idx]
        for label in local_protos.keys():
            if label in agg_protos_label:
                # agg_protos_label[label].append(local_protos[label])
                agg_protos_label[label] = torch.cat((agg_protos_label[label], torch.unsqueeze(local_protos[label], 0)), dim = 0)
            else:
                agg_protos_label[label] = torch.unsqueeze(local_protos[label], 0)

    for k in agg_protos_label.keys():
        agg_protos_label[k] = torch.mean(agg_protos_label[k], dim=0)


    return agg_protos_label

if __name__ == "__main__":
    # 创建一个10行32列的随机向量，并启用梯度
    random_vector = torch.randn(100, 256, requires_grad=True, dtype=torch.double)
    print(random_vector)
    # 定义优化器，例如使用随机梯度下降 (SGD)
    optimizer = optim.SGD([random_vector], lr=0.001)

    # 训练迭代次数
    num_epochs = 10
    class_num = 100

    for epoch in range(num_epochs):
        # 计算余弦相似度
        similarity_matrix = F.cosine_similarity(random_vector.unsqueeze(1), random_vector.unsqueeze(0), dim=2)
        euclidean_distances = torch.cdist(random_vector, random_vector)

        # mask = ~torch.eye(class_num, dtype=bool)
        mask = torch.triu(torch.ones(class_num,class_num, dtype=torch.long),diagonal=1)
        cosine_loss = similarity_matrix[mask]
        l2_loss = euclidean_distances[mask]
        var_loss = torch.var(random_vector, dim=1)

        loss = var_loss.mean() + cosine_loss.mean() - 0.00001 * l2_loss.mean()

        # 打印损失
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}')

        # 清零梯度，进行反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(random_vector)
