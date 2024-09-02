import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
import random
import os


# # 设置CIFAR-10数据集的根目录
data_root = "./"  # 请将路径修改为你希望保存数据的目录

# 定义数据转换
transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                    std=[0.229, 0.224, 0.225])])

# 加载CIFAR-10数据集
cifar_dataset = datasets.CIFAR10(root=data_root, train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(
        cifar_dataset, batch_size=len(cifar_dataset.data), shuffle=False)

for _, train_data in enumerate(trainloader, 0):
    all_data, all_targets = train_data

# 获取CIFAR-10数据集的类别列表
classes = cifar_dataset.classes

# 创建保存新数据集的目录
new_dataset_root = os.path.join(data_root, "cifar10_set")
os.makedirs(new_dataset_root, exist_ok=True)

# 为每个类别随机选择20个样本并保存
num_samples_per_class = 20
total_idxs = []
for label in range(len(classes)):
    print(label)
    # 获取指定类别的所有样本索引
    # class_indices = [i for i, label in enumerate(cifar_dataset.targets) if cifar_dataset.classes[label] == class_name]
    class_indices = [i for i, cur_label in enumerate(all_targets) if cur_label == label]
    # 随机选择指定数量的样本
    selected_indices = random.sample(class_indices, num_samples_per_class)

    total_idxs = total_idxs + selected_indices

data = all_data[total_idxs]
label = all_targets[total_idxs]
X_train = torch.Tensor(data).type(torch.float32)
y_train = torch.LongTensor(label)

set_data = [(x, y) for x, y in zip(X_train, y_train)]

torch.save(set_data, new_dataset_root+"/cifar10set.pt")

print("新数据集已保存到:", new_dataset_root)

# subset_dataset = torch.load(new_dataset_root+"/cifar10set.pt")
#
# trainloader = torch.utils.data.DataLoader(
#         subset_dataset, batch_size=len(subset_dataset), shuffle=True)
#
# for img, label in trainloader:
#     print(label.unique())