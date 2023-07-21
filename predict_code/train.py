import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import os
from tqdm import tqdm
from unet_RGBD_model import ViT_UNet
from dataset_transform import (train_loader, val_loader)
from utils import *


# Set GPU number
os.environ["CUDA_VISIBLE_DEVICES"] = "0"



def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def fit(start_epoch,epochs, model, train_loader, val_loader, criterion, optimizer, scheduler, device="cuda"):
    torch.cuda.empty_cache()
    train_losses = []
    test_losses = []
    val_iou = [];
    val_acc = []
    train_iou = [];
    train_acc = []
    lrs = []
    min_loss = np.inf
    decrease = 1;
    not_improve = 0

    model.to(device)
    fit_time = time.time()
    if start_epoch == epochs:
        print("The model has trained for {} epochs".format(epochs))
        return False
    else:
        for e in range(start_epoch,epochs):
            since = time.time()
            running_loss = 0
            iou_score = 0
            accuracy = 0
            # training loop
            model.train()
            for i, data in enumerate(tqdm(train_loader)):
                # training phase

                img, mask_img = data
                img = img.to(device)
                mask = mask_img.to(device)

                # image_tiles = image_tiles.view(-1, c, h, w)

                # forward
                output = model(img)
                loss = criterion(output, mask)
                # evaluation metrics
                iou_score += mIoU(output, mask)
                accuracy += pixel_accuracy(output, mask)
                # backward
                loss.backward()
                optimizer.step()  # update weight
                optimizer.zero_grad()  # reset gradient

                # step the learning rate
                lrs.append(get_lr(optimizer))
                scheduler.step()

                running_loss += loss.item()

            else:
                model.eval()
                test_loss = 0
                test_accuracy = 0
                val_iou_score = 0
                # validation loop
                with torch.no_grad():
                    for i, data in enumerate(tqdm(val_loader)):
                        # reshape to 9 patches from single image, delete batch size
                        img, mask_img = data
                        img = img.to(device)
                        mask = mask_img.to(device)

                        # image_tiles = image_tiles.view(-1, c, h, w)

                        # forward
                        output = model(img)

                        # evaluation metrics
                        val_iou_score += mIoU(output, mask)
                        test_accuracy += pixel_accuracy(output, mask)
                        # loss
                        loss = criterion(output, mask)
                        test_loss += loss.item()

                # calculate mean for each batch
                train_losses.append(running_loss / len(train_loader))
                test_losses.append(test_loss / len(val_loader))

                

                if (test_loss / len(val_loader)) > min_loss:
                    not_improve += 1
                    min_loss = (test_loss / len(val_loader))
                    print(f'Loss Not Decrease for {not_improve} time')
                    if not_improve == 70:
                        print('Loss not decrease for 70 times, Stop Training')
                        break

                # iou
                val_iou.append(val_iou_score / len(val_loader))
                train_iou.append(iou_score / len(train_loader))
                train_acc.append(accuracy / len(train_loader))
                val_acc.append(test_accuracy / len(val_loader))
                print("Epoch:{}/{}..".format(e + 1, epochs),
                    "Train Loss: {:.3f}..".format(running_loss / len(train_loader)),
                    "Val Loss: {:.3f}..".format(test_loss / len(val_loader)),
                    "Train mIoU:{:.3f}..".format(iou_score / len(train_loader)),
                    "Val mIoU: {:.3f}..".format(val_iou_score / len(val_loader)),
                    "Train Acc:{:.3f}..".format(accuracy / len(train_loader)),
                    "Val Acc:{:.3f}..".format(test_accuracy / len(val_loader)),
                    "Time: {:.2f}m".format((time.time() - since) / 60))
                
                if min_loss > (test_loss / len(val_loader)):
                    print('Loss Decreasing.. {:.3f} >> {:.3f} '.format(min_loss, (test_loss / len(val_loader))))
                    min_loss = (test_loss / len(val_loader))
                    decrease += 1
                    best_state = {
                        'best_epoch': e+1,
                        'epochs': epochs,
                        'train_acc': train_acc[-1],
                        'val_acc': val_acc[-1],
                        'train_iou': train_iou[-1],
                        'val_iou': val_iou[-1],
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                    }
                    # if decrease % 10 == 0:
                    #     print('saving model...')
                    #     state = {
                    #         'epoch': e+1,
                    #         'train_acc': train_acc[-1],
                    #         'val_acc': val_acc[-1],
                    #         'train_iou': train_iou[-1],
                    #         'val_iou': val_iou[-1],
                    #         'model_state_dict': model.state_dict(),
                    #         'optimizer_state_dict': optimizer.state_dict(),
                    #     }
                    #     torch.save(state, 'checkpoint/ViT_UNet_mIoU-{:.3f}.pth'.format(val_iou_score / len(val_loader)))
            print(e)
            # if e % 2 ==0:
            #     print('saving model...')
            #     torch.save(best_state, 'checkpoint/1.pth')
        print('saving model...')
        torch.save(best_state, 'checkpoint/Final_ViT_UNet.pth')
        
        history = {'train_loss': train_losses, 'val_loss': test_losses,
                'train_miou': train_iou, 'val_miou': val_iou,
                'train_acc': train_acc, 'val_acc': val_acc,
                'lrs': lrs}
        print('Total time: {:.2f} m'.format((time.time() - fit_time) / 60))
        return history

# def weights_init(m):
#     classname = m.__class__.__name__
#     if classname.find('Conv2d') != -1:
#         torch.nn.init.xavier_normal_(m.weight.data)   # 权重值正态分布
#         torch.nn.init.constant_(m.bias.data, 0.0)     # 偏置为0
#     elif classname.find('Linear') != -1:
#         torch.nn.init.xavier_normal_(m.weight.data)
#         torch.nn.init.constant_(m.bias.data, 0.0)


max_lr = 1e-3
epochs = 35
weight_decay = 1e-4
model = ViT_UNet(img_size=(720, 1280), in_channel = 4)

try:
    checkpoint = torch.load('checkpoint/Final_ViT_UNet.pth')
    # checkpoint = torch.load('checkpoint/1.pth')
    start_epoch = checkpoint['best_epoch']
    model.load_state_dict(checkpoint['model_state_dict'])
    print('Use pretrained model')
except:
    print('No existing model, starting training from scratch...')
    start_epoch = 0
    # model = model.apply(weights_init)


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=max_lr, weight_decay=weight_decay)
sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs=epochs,
                                            steps_per_epoch=len(train_loader))

history = fit(start_epoch,epochs, model, train_loader, val_loader, criterion, optimizer, sched)








def plot_loss(history):
    plt.plot(history['val_loss'], label='val', marker='o')
    plt.plot(history['train_loss'], label='train', marker='o')
    plt.title('Loss per epoch');
    plt.ylabel('loss');
    plt.xlabel('epoch')
    plt.legend(), plt.grid()
    plt.show()


def plot_score(history):
    plt.plot(history['train_miou'], label='train_mIoU', marker='*')
    plt.plot(history['val_miou'], label='val_mIoU', marker='*')
    plt.title('Score per epoch');
    plt.ylabel('mean IoU')
    plt.xlabel('epoch')
    plt.legend(), plt.grid()
    plt.show()


def plot_acc(history):
    plt.plot(history['train_acc'], label='train_accuracy', marker='*')
    plt.plot(history['val_acc'], label='val_accuracy', marker='*')
    plt.title('Accuracy per epoch');
    plt.ylabel('Accuracy')
    plt.xlabel('epoch')
    plt.legend(), plt.grid()
    plt.show()

if history != False:
    plot_loss(history)
    plot_score(history)
    plot_acc(history)
