import PyRVC as RVC
import os
import numpy as np
import cv2
from PIL import Image
import time

from pynput import keyboard
from pynput.keyboard import Key, KeyCode
import threading


global object_captured
object_captured = False


def App():
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


    save_address = "gluing_dataset/"
    i = 341



    while True:
        global object_captured

        


        if object_captured:
            
                ret2 = x.Capture(cap_opt)
                if ret2 == True:

                    img = x.GetImage()


                    img = np.array(img, copy=False)
                    cv2.imwrite(save_address+"color/color"+str(i)+".png", img)

                    dp = x.GetDepthMap()
                    dp.SaveDepthMap(save_address+"depth/depth"+str(i)+".tiff", True)


                    depth_image = cv2.imread(save_address+'depth/depth'+str(i)+'.tiff', cv2.IMREAD_ANYDEPTH)
                    min_value = 0
                    max_value = 1
                    depth_image = np.clip(depth_image, min_value, max_value)


                    normalized_depth_image = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                    colormap = cv2.COLORMAP_JET
                    pseudo_color_depth_image = cv2.applyColorMap(normalized_depth_image, colormap)
                    cv2.imwrite(save_address+'colored_depth/depth'+str(i)+'.png', pseudo_color_depth_image)

                    
                    print(i)



                    i = i+1
                    
                    object_captured = False


    


# ------------------------------------------------------------------#

    x.Close()
    RVC.X1.Destroy(x)
    RVC.SystemShutdown()

    return 0



def on_press(key):
    global object_captured
    
    if key == Key.enter:
        object_captured = True
        print("capture!")




def thread_job1():
    key_listener = keyboard.Listener(
            on_press=on_press
        )
    key_listener.start()






if __name__ == "__main__":
    add_thread1 = threading.Thread(target = App)
    add_thread1.start()

    add_thread2 = threading.Thread(target = thread_job1)
    add_thread2.start()
