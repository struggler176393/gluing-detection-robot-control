# -*- coding: utf-8 -*-
import json
import cv2
import numpy as np
import os
import shutil

def cvt_one(json_path, save_path):
    # load img and json
    data = json.load(open(json_path,encoding='gbk'))

    img_h = 1080
    img_w = 1440
    img = np.zeros([img_h,img_w])

    color_bg = (0)
    color_seam = (100)
    points_bg = [(0, 0), (0, img_h), (img_w, img_h), (img_w, 0)]
    img = cv2.fillPoly(img, [np.array(points_bg)], color_bg)

    # draw roi
    for i in range(len(data['shapes'])):
        # data['shapes']
        points = data['shapes'][i]['points']


        # name = data['objects'][i]['name']
        # points_x = data['frames'][0]['polygon'][i]['x']
        # points_y = data['frames'][0]['polygon'][i]['y']
        # points = np.vstack([points_x,points_y])
        # points = [point for point in points.T]
        # color =  data['shapes'][i]['fill_color']
        # data['shapes'][i]['fill_color'] = label_color[name]  # 修改json文件中的填充颜色为我们设定的颜色
        # if label_color:
        #     img = cv2.fillPoly(img, [np.array(points, dtype=int)], label_color[name])
        # else:
        #     img = cv2.fillPoly(img, [np.array(points, dtype=int)], (color[0], color[1], color[2]))
        
        img = cv2.fillPoly(img, [np.array(points, dtype=int)], color_seam)
    cv2.imwrite(save_path, img)


if __name__ == '__main__':
    save_dir ='gluing_dataset/mask'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    file_dir = 'gluing_dataset/json'
    all_files = os.listdir(file_dir)
    json_files = list(filter(lambda x: '.json' in x, all_files))  # 读取jpg
    # json_files = list(filter(lambda x:  in x, files))  # 读取json

    # label_color = {'seam':int(1)}


    for i in range(len(json_files)):
        json_path = file_dir + '/' + json_files[i]
        print(json_path)
        save_path = save_dir + '/' + json_files[i].replace('.json', '.png')
        cvt_one(json_path, save_path)
