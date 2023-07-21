import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
from PIL import Image
import cv2
import albumentations as A
import os
from tqdm import tqdm
import open3d as o3d

import sys
sys.path.append('./')

from predict_code.unet_RGBD_model import ViT_UNet
from predict_code.utils import *
from predict_code.pointcloud_vis import NPImgTocolorPointMap
from predict_code.trajectory_planning import cluster,trajectory_xyz_cal



def predict_preprocess_data(color_img, depth_img):

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


    return rgbd_torch



def predict_seam(color_img, depth_img, model, vis=True, device="cuda"):
    # Set GPU number
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"


    rgbd_torch = predict_preprocess_data(color_img, depth_img)

    
    model.eval()

    model.to(device);
    rgbd_torch = rgbd_torch.to(device)
    with torch.no_grad():
        image = rgbd_torch.unsqueeze(0)

        output = model(image)
        mask = torch.argmax(output, dim=1)
        mask = mask.cpu().squeeze(0)

    mask_np = mask.numpy().astype(np.uint8)
    mask_np = cv2.resize(mask_np, (1440, 1080))

    color_img[mask_np==1]=[0,0,255]

    data = NPImgTocolorPointMap(color_img, depth_img, ifvalid=True, ifvis=vis)


    
    return data


def select_seam_points(data, ifvis=True):
    rgb = np.array([1, 0, 0])  # 目标RGB颜色

    # 取出RGB为[1, 0, 0]的点云数据
    seam_points = data[np.all(data[:, 3:6] == rgb, axis=1)]
    if ifvis:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(seam_points[:,:3])
        pcd.colors = o3d.utility.Vector3dVector(seam_points[:,3:])

        o3d.visualization.draw_geometries([pcd])
    return seam_points





# import pretrained model
model = ViT_UNet(img_size=(720, 1280), in_channel = 4)
model_path = "checkpoint/Final_ViT_UNet.pth"
model.load_state_dict(torch.load(model_path)['model_state_dict'])


if __name__ == "__main__":
    idx = 301
    color_img = cv2.imread('gluing_dataset/color/color'+ str(idx) +'.png')
    depth_img = np.array(Image.open('gluing_dataset/depth/depth'+ str(idx) +'.tiff'))
    # depth_img = cv2.imread('gluing_dataset/depth/depth'+ str(idx) +'.tiff',-1)

    data = predict_seam(color_img, depth_img, model, vis=True, device="cuda")
    seam_points = select_seam_points(data, ifvis=False)
    trajectory_xyz_cal(cluster(seam_points[:,:3]))
 




# cv2.imshow('1',color_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()








