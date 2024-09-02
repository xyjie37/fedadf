import os
import shutil
from sklearn.model_selection import train_test_split
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms

parent_dir = '/root/miniimagenet'
base_dir = '/root/mini_imagenet'  


if not os.path.exists(base_dir):
    os.makedirs(base_dir)


train_size = 0.8
test_size = 0.2

for folder_name in os.listdir(parent_dir):
    folder_path = os.path.join(parent_dir, folder_name)
    
    if os.path.isdir(folder_path):
        images = [img for img in os.listdir(folder_path) if img.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
 
        train_images, test_images = train_test_split(images, train_size=train_size, test_size=test_size, random_state=42)

        train_dir = os.path.join(base_dir, 'train', folder_name)
        test_dir = os.path.join(base_dir, 'test', folder_name)

        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)
        
        for img in train_images:
            src = os.path.join(folder_path, img)
            dst = os.path.join(train_dir, img)
            shutil.copy(src, dst)
            
        for img in test_images:
            src = os.path.join(folder_path, img)
            dst = os.path.join(test_dir, img)
            shutil.copy(src, dst)

print("训练集和测试集划分完成，并按照ImageFolder格式保存。")
import os
import numpy as np
from PIL import Image
import pickle
from tqdm import tqdm  # 引入tqdm

def save_dataset_in_cifar_format(base_folder, output_folder):
    print('1')
    subfolders = {'train': 'train', 'test': 'test'}
    data = {'train': [], 'test': []}
    labels = {'train': [], 'test': []}
    print('2')
    class_names = []
    class_index = 0
    print('start')
    # 遍历训练集和测试集
    for subfolder in subfolders.values():
        path = os.path.join(base_folder, subfolder)
        folders = sorted([f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))])
        print('0')
        for folder in tqdm(folders, desc=f"Processing {subfolder}"):
            if folder not in class_names:
                class_names.append(folder)
                class_index = class_names.index(folder)
                print('1')
            folder_path = os.path.join(path, folder)
            img_names = os.listdir(folder_path)
            for img_name in tqdm(img_names, desc=f"Images in {folder}"):
                img_path = os.path.join(folder_path, img_name)
                img = Image.open(img_path)
                img_data = np.array(img)
                print('2')
                data[subfolder].append(img_data)
                labels[subfolder].append(class_index)
    
    # 转换为numpy数组
    for subfolder in subfolders.values():
        data[subfolder] = np.array(data[subfolder])
        labels[subfolder] = np.array(labels[subfolder])
        print('3')
    
    # 保存数据和标签
    for subfolder in subfolders.values():
        output_path = os.path.join(output_folder, f"{subfolder}.pkl")
        with open(output_path, 'wb') as f:
            pickle.dump({'data': data[subfolder], 'labels': labels[subfolder]}, f)
        print('4')
    # 保存meta信息
    meta_path = os.path.join(output_folder, "meta.pkl")
    with open(meta_path, 'wb') as f:
        pickle.dump({'class_names': class_names}, f)

# 使用函数
base_folder = '/root/mini_imagenet'
output_folder = '/root/autodl_tmp/mini_imagenet_cifar_format'
save_dataset_in_cifar_format(base_folder, output_folder)
import os
import numpy as np
from PIL import Image
import pickle
from tqdm import tqdm  # 引入tqdm
def adjust_and_reshape_image(img):
    """调整图像尺寸至32x32，并确保为RGB模式"""
    img = img.resize((32, 32), Image.ANTIALIAS)  # 调整尺寸
    return np.array(img.convert('RGB'))  # 确保为RGB并转换为数组
def save_dataset_in_cifar_format(base_folder, output_folder):
    subfolders = {'train': 'train', 'test': 'test'}
    data = {'train': [], 'test': []}
    labels = {'train': [], 'test': []}
    class_names = []
    class_index = 0
    # 遍历训练集和测试集
    for subfolder in subfolders.values():
        path = os.path.join(base_folder, subfolder)
        folders = sorted([f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))])
        for folder in tqdm(folders, desc=f"Processing {subfolder}"):
            if folder not in class_names:
                class_names.append(folder)
                class_index = class_names.index(folder)
            folder_path = os.path.join(path, folder)
            img_names = os.listdir(folder_path)
            for img_name in tqdm(img_names, desc=f"Images in {folder}"):
                img_path = os.path.join(folder_path, img_name)
                if os.path.isfile(img_path):  # 检查是否为文件
                    img = Image.open(img_path)
                    img_data = adjust_and_reshape_image(img)
                    data[subfolder].append(img_data)
                    labels[subfolder].append(class_index)
    # 转换为numpy数组
    for subfolder in subfolders.values():
        data[subfolder] = np.array(data[subfolder])  # 现在这里应该不会报错了
        labels[subfolder] = np.array(labels[subfolder])
    
    # 保存数据和标签
    for subfolder in subfolders.values():
        output_path = os.path.join(output_folder, f"{subfolder}.pkl")
        with open(output_path, 'wb') as f:
            pickle.dump({'data': data[subfolder], 'labels': labels[subfolder]}, f)
    # 保存meta信息
    meta_path = os.path.join(output_folder, "meta.pkl")
    with open(meta_path, 'wb') as f:
        pickle.dump({'class_names': class_names}, f)

# 使用函数
base_folder = '/root/mini_imagenet'
output_folder = '/root/autodl-tmp/mini_imagenet_cifar_format'
save_dataset_in_cifar_format(base_folder, output_folder)
