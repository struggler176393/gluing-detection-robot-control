from PIL import Image
import numpy as np
import cv2

import sys
sys.path.append('./')

from predict_code.pointcloud_vis import NPImgTocolorPointMap
import matplotlib.pyplot as plt
from torchvision import transforms as T
import torch
import pandas as pd
import os
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def create_df(path, expand_scale):
    name = []
    for i in range(expand_scale):
        for dirname, _, filenames in os.walk(path):
            for filename in filenames:
                name.append(filename.split('.')[0].replace('depth', ''))

    return pd.DataFrame({'id': name}, index=np.arange(0, len(name)))



def plt_depth_range(depth_img):
    depth_test = depth_img.reshape(-1)
    depth_test = depth_test[~np.isnan(depth_test)]


    # 绘制直方图分布
    plt.hist(depth_test, bins=30, color='blue', alpha=0.7)

    # 添加标题和轴标签
    plt.title('Histogram')
    plt.xlabel('Value')
    plt.ylabel('Frequency')

    # 显示图像
    plt.show()



def augment_data(color_img, depth_img, mask):
    
    # 随机翻转
    if np.random.random() < 0.5:
        random_number = np.random.choice([-1, 0, 1])
        color_img = cv2.flip(color_img, random_number)
        depth_img = cv2.flip(depth_img, random_number)
        mask = cv2.flip(mask, random_number)
    
    # 随机旋转
    if np.random.random() < 0.5:
        angle = np.random.uniform(-10, 10)
        rows, cols = color_img.shape[:2]
        M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
        color_img = cv2.warpAffine(color_img, M, (cols, rows))
        depth_img = cv2.warpAffine(depth_img, M, (cols, rows))
        mask = cv2.warpAffine(mask, M, (cols, rows))
    
    # 随机平移
    if np.random.random() < 0.5:
        tx = np.random.randint(-100, 100)
        ty = np.random.randint(-100, 100)
        M = np.float32([[1, 0, tx], [0, 1, ty]])
        rows, cols = color_img.shape[:2]
        color_img = cv2.warpAffine(color_img, M, (cols, rows))
        depth_img = cv2.warpAffine(depth_img, M, (cols, rows))
        mask = cv2.warpAffine(mask, M, (cols, rows))
    
    # 随机缩放
    if np.random.random() < 0.5:
        scale = np.random.uniform(0.8, 1.2)
        color_img = cv2.resize(color_img, None, fx=scale, fy=scale)
        depth_img = cv2.resize(depth_img, None, fx=scale, fy=scale)
        mask = cv2.resize(mask, None, fx=scale, fy=scale)
    
    # 随机裁剪
    if np.random.random() < 0.5:
        cut_scale = 0.8

        dx = int(color_img.shape[1]*cut_scale)
        dy = int(color_img.shape[0]*cut_scale)
        x = np.random.randint(0, color_img.shape[1] - dx)
        y = np.random.randint(0, color_img.shape[0] - dy)
        color_img = color_img[y:y+dy, x:x+dx, :]
        depth_img = depth_img[y:y+dy, x:x+dx]
        mask = mask[y:y+dy, x:x+dx]
    
    # 随机噪声
    if np.random.random() < 0.5:
        noise = np.random.normal(0, 0.22, color_img.shape).astype(np.uint8)
        color_img = cv2.add(color_img, noise)
    
    # 高斯模糊
    if np.random.random() < 0.2:
        ksize = np.random.choice([3, 5, 7])
        color_img = cv2.GaussianBlur(color_img, (ksize, ksize), 0)
    

    

    return color_img, depth_img, mask


def preprocess_data(color_img, depth_img, mask):

    width = 1280
    height = 720
    color_img = cv2.resize(color_img, (width, height))
    depth_img = cv2.resize(depth_img, (width, height))
    


    ## 深度图像
    # 将NAN转换为0
    depth_img[np.isnan(depth_img)] = 0
    depth_img[depth_img == 0] = np.nan
    depth_img = np.array(depth_img,dtype=np.float32)
    # 对深度图像进行值截断
    min_value = 0
    max_value = 1
    depth_img = np.clip(depth_img, min_value, max_value)
    # 归一化
    depth_img = cv2.normalize(depth_img, None, 0, 1, cv2.NORM_MINMAX, cv2.CV_8U)
    # 扩充第三个维度
    depth_img = np.expand_dims(depth_img,2)
    
    ## 彩色图像
    # 归一化
    color_img = color_img.astype(np.float32)
    color_img /= 255.0
    
    # 转换色彩空间
    color_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)

    




    ## torch 张量化
    depth_torch = T.ToTensor()(depth_img)
    color_torch = T.ToTensor()(color_img)
    

    rgbd_torch = torch.cat((color_torch,depth_torch),dim=0)


    ## mask
    mask = cv2.resize(mask, (width, height))
    mask = mask/100
    mask_torch = torch.from_numpy(mask).long()

    return rgbd_torch, mask_torch

    



# Dataset class
class Glue_Dataset(Dataset):

    def __init__(self, color_img_path, depth_img_path, mask_path, data_list):
        self.color_img_path = color_img_path
        self.depth_img_path = depth_img_path
        self.mask_path = mask_path
        self.data_list = data_list


    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        # print( self.data_list[idx])

        color_img = cv2.imread(self.color_img_path + 'color' + self.data_list[idx] + '.png')
        depth_img = cv2.imread(self.depth_img_path + 'depth'  + self.data_list[idx] + '.tiff',-1)
        mask = cv2.imread(self.mask_path + 'color' + self.data_list[idx] + '.png', cv2.IMREAD_GRAYSCALE)

        color_img, depth_img, mask = augment_data(color_img, depth_img, mask)
        rgbd_torch, mask_torch = preprocess_data(color_img, depth_img, mask)

        return rgbd_torch, mask_torch

# Dataset class
class Evaluation_Glue_Dataset(Dataset):

    def __init__(self, color_img_path, depth_img_path, data_list):
        self.color_img_path = color_img_path
        self.depth_img_path = depth_img_path
        self.data_list = data_list


    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        # print( self.data_list[idx])

        color_img = cv2.imread(self.color_img_path + 'color' + self.data_list[idx] + '.png')
        depth_img = cv2.imread(self.depth_img_path + 'depth'  + self.data_list[idx] + '.tiff',-1)

        rgbd_torch = preprocess_data(color_img, depth_img)

        return rgbd_torch


# set image path
DATA_PATH = 'gluing_dataset'
COLOR_IMAGE_PATH = DATA_PATH + '/color/'
DEPTH_IMAGE_PATH = DATA_PATH + '/depth/'
MASK_PATH = DATA_PATH + '/mask/'
COLORDEPTH_PATH = DATA_PATH + '/colored_depth/'

# 数据集扩充倍数
expand_scale = 3
df = create_df(COLORDEPTH_PATH, expand_scale)

list_trainval, list_test = train_test_split(df['id'].values, test_size=0.2, random_state=6)
list_train, list_val = train_test_split(list_trainval, test_size=0.2, random_state=6)

train_set = Glue_Dataset(COLOR_IMAGE_PATH, DEPTH_IMAGE_PATH, MASK_PATH, list_train)
val_set = Glue_Dataset(COLOR_IMAGE_PATH, DEPTH_IMAGE_PATH, MASK_PATH, list_val)

batch_size= 1

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True, drop_last=True)



















for i, data in enumerate(tqdm(train_loader)):
    rgbd_torch, mask_torch = data  # img:[B,C,H,W], mask:[B,H,W]




# color_img = cv2.imread('gluing_dataset/color/color1.png')  # (1080, 1440, 3)
# depth_img = np.array(Image.open('gluing_dataset/depth/depth156.tiff'))  # (1080, 1440)
# mask = cv2.imread('gluing_dataset/mask/color1.png', 0)  # (1080, 1440)

# # 进行数据增强
# color_img, depth_img, mask = augment_data(color_img, depth_img, mask)
# # NPImgTocolorPointMap(color_img, depth_img, ifvalid=False, ifvis=False)

# # 进行数据预处理
# rgbd_torch, mask_torch = preprocess_data(color_img, depth_img, mask)




# print(rgbd_torch.shape)
# # 显示增强后的图像
# cv2.imshow('Augmented Color Image', color_img)
# # cv2.imshow('Augmented Depth Image', augmented_depth_img)
# cv2.imshow('Augmented Mask', mask)
# cv2.waitKey(0)
# cv2.destroyAllWindows()





