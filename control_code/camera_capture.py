import PyRVC as RVC
import os
import numpy as np
import cv2
from PIL import Image
from time import time 
import sys
sys.path.append('./')
from predict_code.predict import predict_seam, model



def TryCreateDir(d):
    if not os.path.exists(d):
        os.makedirs(d)
        return 1
    else:
        return 0



def DepthMapToPointMap(x, file_path):
    # im = Image.open('Data/dp1.tiff')
    im = Image.open(file_path)
    # 将图像转化成numpy数组
    dp = np.array(im)



    # convert depthmap to pointmap
    # print(x.GetIntrinsicParameters())
    ret, intrinsic_matrix, distortion = x.GetIntrinsicParameters()
    intrinsic_matrix = np.array(intrinsic_matrix).reshape((3, 3))
    k1, k2, k3, p1, p2 = distortion
    distortion_cv = np.array([k1, k2, p1, p2, k3])

    print(intrinsic_matrix,distortion_cv)

    width = 1440
    height = 1080
    X = range(width)
    Y = range(height)
    XY = np.array(np.meshgrid(X, Y), dtype=float).reshape((2, -1)).T
    undistorted_XY = cv2.undistortPoints(
        XY, intrinsic_matrix, distortion_cv).reshape((height, width, 2))
    convert_pm = np.array([undistorted_XY[:, :, 0] * dp,
                            undistorted_XY[:, :, 1] * dp, dp]).reshape((3, -1)).T
    
    np.savetxt("Data/point_map1.xyz", convert_pm)
    
    return convert_pm


# 取中心线段作为胶缝的点云重建
def center_seam_NPImgTocolorPointMap(color, depth, ifvalid,add_mask):
    # im = Image.open('Data/dp1.tiff')
    color = np.array(color)
    color = cv2.cvtColor(color,cv2.COLOR_RGB2BGR)
    color = color.reshape(-1,3)
    
    # 将图像转化成numpy数组
    depth = np.array(depth)




    intrinsic_matrix = np.array([[2.40189575e+03, 0.00000000e+00, 7.41044678e+02],
                                  [0.00000000e+00, 2.40110034e+03, 5.95356873e+02],
                                  [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
    
    
    distortion_cv = np.array([-3.33325416e-02,  3.83851230e-02,  5.59388369e-04,  4.35525930e-04,  6.31685734e-01])
    width = 1440
    height = 1080
    X = range(width)
    Y = range(height)
    XY = np.array(np.meshgrid(X, Y), dtype=float).reshape((2, -1)).T
    undistorted_XY = cv2.undistortPoints(
        XY, intrinsic_matrix, distortion_cv).reshape((height, width, 2))
    convert_pm = np.array([undistorted_XY[:, :, 0] * depth,
                            undistorted_XY[:, :, 1] * depth, depth]).reshape((3, -1)).T
    
    if add_mask:
        mask_img = np.zeros([1080,1440])
        mask_img[530:550,470:970] = 1
        mask_img = mask_img.reshape(-1)
        color[mask_img==1]=[255,0,0]
    
    if ifvalid:
        valid_color_img = color[~np.isnan(convert_pm[:, 2])]/255
        valid_pcd_data = convert_pm[~np.isnan(convert_pm[:, 2])]

        print(valid_color_img.shape)
        print(valid_pcd_data.shape)

        data = np.hstack([valid_pcd_data,valid_color_img])
    else:
        data = np.hstack([convert_pm,color/255])
    # np.savetxt("Data/point_map1.xyz", convert_pm)
    
    return data


# 取中心线段作为胶缝
def color_pointcloud_capture(ifvalid=True,add_mask = False):
# 准备工作
# ------------------------------------------------------------------#
    RVC.SystemInit()
    opt = RVC.SystemListDeviceTypeEnum.All
    ret, devices = RVC.SystemListDevices(opt)
    print("RVC X Camera devices number:%d" % len(devices))
    if len(devices) == 0:
        print("Can not find any RVC X Camera!")
        RVC.SystemShutdown()
        return 1
    print("devices size = %d" % len(devices))


    x = RVC.X1.Create(devices[0], RVC.CameraID_Left)
    if x.IsValid() == True:
        print("RVC X Camera is valid!")
    else:
        print("RVC X Camera is not valid!")
        RVC.X1.Destroy(x)
        RVC.SystemShutdown()
        return 1

    ret1 = x.Open()
    if ret1 and x.IsOpen() == True:
        print("RVC X Camera is opened!")
    else:
        print("RVC X Camera is not opened!")
        RVC.X1.Destroy(x)
        RVC.SystemShutdown()
        return 1
    

# ------------------------------------------------------------------#


    cap_opt = RVC.X1_CaptureOptions()
    cap_opt.transform_to_camera = True

    ret2 = x.Capture(cap_opt)


    if ret2 == True:
        print("RVC X Camera capture successed!")

        # Get image data and image size.
        img = x.GetImage()
        width = img.GetSize().cols
        height = img.GetSize().rows

        # Check the camera color information.
        print("width=%d, height=%d" % (width, height))
        if img.GetType() == RVC.ImageTypeEnum.Mono8:
            print("This is mono camera")
        else:
            print("This is color camera")

        # Convert image to array and save it.
        img = np.array(img, copy=False)
        cv2.imwrite("Data/a.png", img)
        # print("Save image successed!")
        
        # get depth map (m) and save it
        dp = x.GetDepthMap()
        # dp.SaveDepthMap("Data/dp1.tiff", True)
        convert_pm = center_seam_NPImgTocolorPointMap(img, dp, ifvalid,add_mask)

        print(convert_pm.shape)

        # convert_pm = DepthMapToPointMap(x, "Data/dp1.tiff")
        
        print("Save point map successed!")



    else:
        print("RVC X Camera capture failed!")
        x.Close()
        RVC.X1.Destroy(x)
        RVC.SystemShutdown()
        return 1
    


# ------------------------------------------------------------------#

    x.Close()
    RVC.X1.Destroy(x)
    RVC.SystemShutdown()

    return img, dp, convert_pm




def capture_predict():
# 准备工作
# ------------------------------------------------------------------#
    RVC.SystemInit()
    opt = RVC.SystemListDeviceTypeEnum.All
    ret, devices = RVC.SystemListDevices(opt)
    print("RVC X Camera devices number:%d" % len(devices))
    if len(devices) == 0:
        print("Can not find any RVC X Camera!")
        RVC.SystemShutdown()
        return 1
    print("devices size = %d" % len(devices))


    x = RVC.X1.Create(devices[0], RVC.CameraID_Left)
    if x.IsValid() == True:
        print("RVC X Camera is valid!")
    else:
        print("RVC X Camera is not valid!")
        RVC.X1.Destroy(x)
        RVC.SystemShutdown()
        return 1

    ret1 = x.Open()
    if ret1 and x.IsOpen() == True:
        print("RVC X Camera is opened!")
    else:
        print("RVC X Camera is not opened!")
        RVC.X1.Destroy(x)
        RVC.SystemShutdown()
        return 1
    

# ------------------------------------------------------------------#


    cap_opt = RVC.X1_CaptureOptions()
    cap_opt.transform_to_camera = True

    ret2 = x.Capture(cap_opt)


    if ret2 == True:
        print("RVC X Camera capture successed!")

        # Get image data and image size.
        img = x.GetImage()
        width = img.GetSize().cols
        height = img.GetSize().rows

        # Check the camera color information.
        print("width=%d, height=%d" % (width, height))
        if img.GetType() == RVC.ImageTypeEnum.Mono8:
            print("This is mono camera")
        else:
            print("This is color camera")


        img = np.array(img, copy=False)

        dp = np.array(x.GetDepthMap())
        # dp.SaveDepthMap("Data/dp1.tiff", True)
        data = predict_seam(img, dp, model, vis=False, device="cuda")


        # convert_pm = DepthMapToPointMap(x, "Data/dp1.tiff")
        
        print("Save point map successed!")



    else:
        print("RVC X Camera capture failed!")
        x.Close()
        RVC.X1.Destroy(x)
        RVC.SystemShutdown()
        return 1
    


# ------------------------------------------------------------------#

    x.Close()
    RVC.X1.Destroy(x)
    RVC.SystemShutdown()

    return img, dp, data



if __name__ == "__main__":
    # App()
    # color_pointcloud_capture()
    capture_predict()
