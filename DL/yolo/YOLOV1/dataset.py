# -*- coding:utf-8 -*-

from __future__ import print_function
import xml.etree.ElementTree as ET
# import skimage.io as io
import matplotlib.pyplot as plt
from torch.nn import functional as F
import torch.utils.data
import numpy as np
import torch
import json
import cv2
import os
from PIL import Image
import random

def glob_format(path,base_name = False):
    print('--------pid:%d start--------------' % (os.getpid()))
    fmt_list = ('.jpg', '.jpeg', '.png',".xml")
    fs = []
    if not os.path.exists(path):return fs
    for root, dirs, files in os.walk(path):
        for file in files:
            item = os.path.join(root, file)
            # item = unicode(item, encoding='utf8')
            fmt = os.path.splitext(item)[-1]
            if fmt.lower() not in fmt_list:
                # os.remove(item)
                continue
            if base_name:fs.append(file)  # fs.append(os.path.splitext(file)[0])
            else:fs.append(item)
    print('--------pid:%d end--------------' % (os.getpid()))
    return fs

class PennFudanDataset(torch.utils.data.Dataset):
    """
    # download the Penn-Fudan dataset
    wget https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip
    # extract it in the current folder
    unzip PennFudanPed.zip
    """

    def __init__(self, root,num_class=1,num_boxes=2, transforms=None):
        self.root = root
        self.num_class = num_class
        self.num_boxes = num_boxes
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        # 确保imgs与masks相对应
        self.imgs = list(sorted(os.listdir(os.path.join(root, "PNGImages"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "PedMasks"))))

    def __getitem__(self, idx):
        # load images ad masks
        img_path = os.path.join(self.root, "PNGImages", self.imgs[idx])
        mask_path = os.path.join(self.root, "PedMasks", self.masks[idx])
        img = Image.open(img_path).convert("RGB")
        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance ，因为每种颜色对应不同的实例
        # with 0 being background
        mask = Image.open(mask_path)

        mask = np.array(mask)
        # instances are encoded as different colors
        # 实例被编码为不同的颜色（0为背景，1为对象1,2为对象2,3为对象3，...）
        obj_ids = np.unique(mask)  # array([0, 1, 2], dtype=uint8),mask有2个对象分别为1,2
        # first id is the background, so remove it
        # first id是背景，所以删除它
        obj_ids = obj_ids[1:]  # array([1, 2], dtype=uint8)

        # split the color-encoded mask into a set
        # of binary masks ,0,1二值图像
        # 将颜色编码的掩码分成一组二进制掩码 SegmentationObject-->mask
        masks = mask == obj_ids[:, None, None]  # shape (2, 536, 559)，2个mask
        # obj_ids[:, None, None] None为增加对应的维度，shape为 [2, 1, 1]

        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):  # mask反算对应的bbox
            pos = np.where(masks[i])  # 返回像素值为1 的索引，pos[0]对应行(y)，pos[1]对应列(x)
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        # return img, target

        # 转成[n,7,7,50] 便于批量处理
        strides_h = 32
        strides_w = 32 # 网络缩减的尺度(每个网格大小)
        h,w = img.shape[1:] # 缩放到统一尺度后图像大小
        # 网格大小 如 7x7
        grid_ceil_h = h//strides_h
        grid_ceil_w = w//strides_w

        boxes = target["boxes"] # 格式 x1,y1,x2,y2
        labels = target["labels"]

        # x1,y1,x2,y2->x0,y0,w,h
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        # [x0,y0,w,h]
        x0 = (x1 + x2) / 2.
        y0 = (y1 + y2) / 2.
        w_b = (x2 - x1) / w  # 0~1
        h_b = (y2 - y1) / h  # 0~1

        # 判断box落在哪个grid ceil
        # 取格网左上角坐标
        # grid_ceil = (y0 / strides).int() * w_f + (x0 / strides).int()
        grid_ceil = ((x0 / strides_w).int(), (y0 / strides_h).int())

        # normal 0~1
        # gt_box 中心点坐标-对应grid cell左上角的坐标/ 格网大小使得范围变成0到1
        x0 = (x0 - grid_ceil[0].float() * strides_w) / strides_w
        y0 = (y0 - grid_ceil[1].float() * strides_h) / strides_h

        # grid_ceil = grid_ceil[1] * w_f + grid_ceil[0]

        result = torch.zeros([grid_ceil_h,grid_ceil_w,self.num_boxes*(5+self.num_class)],dtype=img.dtype,device=img.device)

        #
        for i,(y,x) in enumerate(zip(grid_ceil[1],grid_ceil[0])):
            result[y,x,[0,0+5+self.num_class]]=x0[i]
            result[y,x,[1,1+5+self.num_class]]=y0[i]
            result[y,x,[2,2+5+self.num_class]]=w_b[i]
            result[y,x,[3,3+5+self.num_class]]=h_b[i]
            result[y,x,[4,4+5+self.num_class]]=1 # 置信度
            result[y,x,[5,5+5+self.num_class]]=labels[i].float()

        return img,result

    def __len__(self):
        return len(self.imgs)

class PennFudanDataset2(torch.utils.data.Dataset):
    """
    # download the Penn-Fudan dataset
    wget https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip
    # extract it in the current folder
    unzip PennFudanPed.zip
    """

    def __init__(self, root,num_class=1,num_boxes=2, transforms=None):
        self.root = root
        self.num_class = num_class
        self.num_boxes = num_boxes
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        # 确保imgs与masks相对应
        self.imgs = list(sorted(os.listdir(os.path.join(root, "PNGImages"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "PedMasks"))))

    def __getitem__(self, idx):
        # load images ad masks
        img_path = os.path.join(self.root, "PNGImages", self.imgs[idx])
        mask_path = os.path.join(self.root, "PedMasks", self.masks[idx])
        img = Image.open(img_path).convert("RGB")
        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance ，因为每种颜色对应不同的实例
        # with 0 being background
        mask = Image.open(mask_path)

        mask = np.array(mask)
        # instances are encoded as different colors
        # 实例被编码为不同的颜色（0为背景，1为对象1,2为对象2,3为对象3，...）
        obj_ids = np.unique(mask)  # array([0, 1, 2], dtype=uint8),mask有2个对象分别为1,2
        # first id is the background, so remove it
        # first id是背景，所以删除它
        obj_ids = obj_ids[1:]  # array([1, 2], dtype=uint8)

        # split the color-encoded mask into a set
        # of binary masks ,0,1二值图像
        # 将颜色编码的掩码分成一组二进制掩码 SegmentationObject-->mask
        masks = mask == obj_ids[:, None, None]  # shape (2, 536, 559)，2个mask
        # obj_ids[:, None, None] None为增加对应的维度，shape为 [2, 1, 1]

        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):  # mask反算对应的bbox
            pos = np.where(masks[i])  # 返回像素值为1 的索引，pos[0]对应行(y)，pos[1]对应列(x)
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)

def resize_boxes(boxes, original_size, new_size):
    ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(new_size, original_size))
    ratio_height, ratio_width = ratios
    xmin, ymin, xmax, ymax = boxes.unbind(1)
    xmin = xmin * ratio_width
    xmax = xmax * ratio_width
    ymin = ymin * ratio_height
    ymax = ymax * ratio_height
    return torch.stack((xmin, ymin, xmax, ymax), dim=1)

class Resize_fixed(object):
    def __init__(self,img_size=448,image_mean=None,image_std=None,training=True,multi_scale=True): # 416
        # if not isinstance(img_size,(tuple,list)):
        #     img_size=[img_size]

        self.img_size=img_size
        self.training=training
        if image_mean is None:
            image_mean = [0.485, 0.456, 0.406]
        if image_std is None:
            image_std = [0.229, 0.224, 0.225]

        self.image_mean=image_mean
        self.image_std=image_std

        if multi_scale:
            s = img_size / 32
            nb=8
            self.multi_scale = ((np.linspace(0.5, 2, nb) * s).round().astype(np.int) * 32)

            # self.multi_scale=[320,352,384,416,448,480,512,544,576,608]

    def normalize(self, image):
        dtype, device = image.dtype, image.device
        mean = torch.as_tensor(self.image_mean, dtype=dtype, device=device)
        std = torch.as_tensor(self.image_std, dtype=dtype, device=device)
        return (image - mean[:, None, None]) / std[:, None, None]

    def pad_img(self,A,box=None, mode='constant', value=0):
        h, w = A.size()[-2:]
        if h >= w:
            diff = h - w
            pad_list = [diff // 2, diff - diff // 2, 0, 0]
            if box is not None:
                box=[[b[0]+diff // 2,b[1],b[2]+diff // 2,b[3]] for b in box]
                box=torch.as_tensor(box)
        else:
            diff = w - h
            pad_list = [0, 0, diff // 2, diff - diff // 2]
            if box is not None:
                box = [[b[0], b[1]+diff // 2, b[2], b[3]+diff // 2] for b in box]
                box = torch.as_tensor(box)

        A_pad = F.pad(A, pad_list, mode=mode, value=value)

        return A_pad,box,h,w

    def resize(self, image, target):

        if self.training:
            size = random.choice(self.multi_scale)
        else:
            # FIXME assume for now that testing uses the largest scale
            size = self.img_size

        if target is None:
            bbox=None
        else:
            bbox = target["boxes"]
        image,bbox,h,w=self.pad_img(image,box=bbox)

        # resize
        image = F.interpolate(
            image[None],size=(size,size), mode='bilinear', align_corners=False)[0]

        if target is None:
            target={}
            target["scale_factor"] = torch.as_tensor([h,w,size], device=image.device)
            return image, target

        bbox = resize_boxes(bbox, (h,h) if h>w else (w,w), image.shape[-2:])
        target["boxes"] = bbox

        return image, target

    def __call__(self, image, target):
        # normalizer
        image = self.normalize(image)
        image, target = self.resize(image, target)

        return image, target

class InferDataset(torch.utils.data.Dataset):
    def __init__(self, root, num_class=1, num_boxes=2, transforms=None):
        self.root = root
        self.num_class = num_class
        self.num_boxes = num_boxes
        self.transforms = transforms
        self.paths = glob_format(root)
    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path).convert("RGB")
        origin_img = np.asarray(img)

        target = {} # 虚假的
        boxes = torch.as_tensor([[0,0,0,0]], dtype=torch.float32)
        # there is only one class
        labels = torch.ones((1,), dtype=torch.int64)
        target["boxes"] = boxes
        target["labels"] = labels

        if self.transforms is not None:
            img,target = self.transforms(img,target)

        return img,path,torch.as_tensor(origin_img,dtype=img.dtype,device=img.device)


if __name__=="__main__":
    import transforms
    from draw import draw_rect
    root = r"C:\Users\MI\Documents\GitHub\PennFudanPed"
    dataset = PennFudanDataset(root, transforms=transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(0.5),
            Resize_fixed(224,training=False)
            # transforms.ColorJitter(0.05, 0.05, 0.05, 0.05)
        ]
    ))

    # """
    for image, target in dataset:
        image = np.asarray(image).transpose([1, 2, 0])
        image =np.clip(image*255,0,255)
        for k, v in target.items():
            target[k] = v.numpy()
        draw_rect(image,target)
    """

    data_loader = torch.utils.data.DataLoader(dataset,batch_size=5)
    for image, target in data_loader:
        print()
    # """