import os.path
import pandas as pd
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import pickle
from data_util import split_data, save_file
from torch.utils.data import Dataset

class MiniImageNetDataset(Dataset):
    def __init__(self, data_file, transform=None):
        with open(data_file, 'rb') as f:
            data = pickle.load(f)
            self.data = data['data']
            self.labels = data['labels']
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
num_client = 20
num_task = 7
num_classes = 100
alpha = 0.1
np.random.seed(2266)
#数据集存储地址
datasetroot_dir = "/root/mini_imagenet_cifar_format"
#生成数据集存储地址
basedir = "./miniimagenet-dir-{}-task-{}".format(alpha, num_task)
if not os.path.exists(basedir):
    os.mkdir(basedir)

# Define transforms to apply to the data
transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean=[0.507, 0.487, 0.441],
                                                    std=[0.267, 0.256, 0.276])])


train_dataset = MiniImageNetDataset(data_file=datasetroot_dir + '/train.pkl', transform=transform)
test_dataset = MiniImageNetDataset(data_file=datasetroot_dir + '/test.pkl', transform=transform)


trainloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=len(train_dataset.data), shuffle=False)
testloader = torch.utils.data.DataLoader(
    test_dataset, batch_size=len(test_dataset.data), shuffle=False)

for _, train_data in enumerate(trainloader, 0):
    train_dataset.data, train_dataset.targets = train_data
for _, test_data in enumerate(testloader, 0):
    test_dataset.data, test_dataset.targets = test_data

total_image = []
total_label = []

total_image.extend(train_dataset.data.cpu().detach().numpy())
total_image.extend(test_dataset.data.cpu().detach().numpy())
total_image = np.array(total_image)

total_label.extend(train_dataset.targets.cpu().detach().numpy())
total_label.extend(test_dataset.targets.cpu().detach().numpy())
total_label = np.array(total_label)

image_per_client = [[] for _ in range(num_client)]
label_per_client = [[] for _ in range(num_client)]
statistic = [[] for _ in range(num_client)]



#记录索引的字典 key = client编号  value = []索引list
dataidx_map = {}
#每一个数据的索引
idxs = np.array(range(len(total_label)))
#每一个类数据的索引
idx_for_each_class = []
for i in range(num_classes):
    idx_for_each_class.append(idxs[total_label == i])

#对每类数据操作
for i in range(num_classes):
    num_images = len(idx_for_each_class[i])
    num_per_client = num_images / num_client
    per_client_image_number = [int(num_per_client) for _ in range(num_client)]
    idx = 0
    for client, num_sample in enumerate(per_client_image_number):
        if client not in dataidx_map.keys():
            dataidx_map[client] = idx_for_each_class[i][idx:idx + num_sample]
        else:
            dataidx_map[client] = np.append(dataidx_map[client], idx_for_each_class[i][idx:idx + num_sample], axis=0)
        idx += num_sample

# 遍历每个客户端,得到每个客户端的索引
df = pd.DataFrame(columns=[str(i) for i in range(100)])
for client in range(num_client):
    idxs = dataidx_map[client]
    image_per_client[client] = total_image[idxs]
    label_per_client[client] = total_label[idxs]
    row = [0 for i in range(100)]
    for i in np.unique(label_per_client[client]):
        statistic[client].append((int(i), int(sum(label_per_client[client] == i))))
        row[i] = int(sum(label_per_client[client] == i))
    df.loc[len(df)] = row
df.to_csv(basedir + "/client-statics.csv")

for client in range(num_client):
    print(f"Client {client}\t Size of data: {len(image_per_client[client])}\t Labels: ", np.unique(label_per_client[client]))
    print(f"\t\t Samples of labels: ", [i for i in statistic[client]])
    print("=" * 50)




K = num_classes
least_samples = len(image_per_client[0])//10
if num_task == 10:
    least_samples = len(image_per_client[0])//20
print("least samples:" , least_samples)
N = least_samples


col = [str(i) for i in range(100)]
col.append('client')
col.append('task')
df = pd.DataFrame(columns=col)
#将每类数据按照迪利克雷分布分配给每个任务
for client_id in range(num_client):
    client_images = image_per_client[client_id]
    client_dataset_label = label_per_client[client_id]
    X = [[] for _ in range(num_task)]
    Y = [[] for _ in range(num_task)]
    client_idx_map = {}
    min_size = 0


    while min_size < least_samples:
        idx_batch = [[] for _ in range(num_task)]
        for k in range(K):
            idx_k = np.where(client_dataset_label == k)[0]
            np.random.shuffle(idx_k)
            proportions = np.random.dirichlet(np.repeat(alpha, num_task))
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
            min_size = min([len(idx_j) for idx_j in idx_batch])

    for j in range(num_task):
        client_idx_map[j] = idx_batch[j]

    for task in range(num_task):
        row = [0 for i in range(102)]
        row[100] = client_id
        row[101] = task
        idxs = client_idx_map[task]
        Y[task] = client_dataset_label[idxs]
        X[task] = client_images[idxs]

        info = []
        for i in np.unique(Y[task]):
            info.append((int(i), int(sum(Y[task]==i))))
            row[i] = int(sum(Y[task]==i))
        df.loc[len(df)] = row

        print(f"Client {client_id}  Task {task}\t Size of data: {len(X[task])}\t Labels: ", np.unique(Y[task]))
        print(f"\t\t Samples of labels: ", [i for i in info])
        print("-" * 50)
    print("=" * 50 + "\n\n")


    # 保存数据
    train_data, test_data = split_data(X, Y)

    if not os.path.exists(basedir + "/train"):
        os.mkdir(basedir + "/train")
    if not os.path.exists(basedir + "/test"):
        os.mkdir(basedir + "/test")

    train_path = basedir + "/train/client-" + str(client_id) + "-task-"
    test_path = basedir + "/test/client-" + str(client_id) + "-task-"
    save_file(train_path, test_path, train_data, test_data)
df.to_csv(basedir + "/task-statics.csv")


path = basedir + '/test/'
all_test_data = {}
for client_id in range(num_client):
    for task in range(num_task):
        file = path + 'client-' + str(client_id) + '-task-' + str(task) + '.npz'
        with open(file, 'rb') as f:
            data = np.load(f, allow_pickle=True)['data'].tolist()
            if 'x' not in all_test_data.keys():
                all_test_data['x'] = data['x']
                all_test_data['y'] = data['y']
            else:
                all_test_data['x'] = np.concatenate((all_test_data['x'], data['x']))
                all_test_data['y'] = np.concatenate((all_test_data['y'], data['y']))

test_path = basedir + "/test/test-data"

with open(test_path + '.npz', 'wb') as f:
    np.savez_compressed(f, data=(all_test_data))
