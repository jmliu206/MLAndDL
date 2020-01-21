import torch
from torch import nn
import torch.utils.data
import torchvision
import numpy as np
import pickle,os
from dataset import *
from loss import *
from network import *
import transforms
# from torchvision import transforms
import cv2
from boxes import batched_nms

_GRAY = (218, 227, 218)
_GREEN = (18, 127, 15)
_RED = (20, 50, 255)

classes=["__background__","person"]

def apply_nms(prediction,conf_thres=0.8,nms_thres=0.4,filter_labels=[],device="cpu"):
    # for idx,prediction in enumerate(detections):
    # 1.先按scores过滤分数低的,过滤掉分数小于conf_thres
    ms = prediction["scores"] > conf_thres
    if torch.sum(ms) == 0:
        return None
    else:
        last_scores = []
        last_labels = []
        last_boxes = []

        # 2.类别一样的按nms过滤，如果Iou大于nms_thres,保留分数最大的,否则都保留
        # 按阈值过滤
        scores = prediction["scores"][ms]
        labels = prediction["labels"][ms]
        boxes = prediction["boxes"][ms]
        unique_labels = labels.unique()
        for c in unique_labels:
            if c in filter_labels:continue

            # Get the detections with the particular class
            temp = labels == c
            _scores = scores[temp]
            _labels = labels[temp]
            _boxes = boxes[temp]
            if len(_labels) > 1:
                # Sort the detections by maximum objectness confidence
                # _, conf_sort_index = torch.sort(_scores, descending=True)
                # _scores=_scores[conf_sort_index]
                # _boxes=_boxes[conf_sort_index]

                # """
                # keep=py_cpu_nms(_boxes.cpu().numpy(),_scores.cpu().numpy(),nms_thres)
                # keep = nms(_boxes, _scores, nms_thres)
                keep = batched_nms(_boxes, _scores,_labels, nms_thres)
                last_scores.extend(_scores[keep])
                last_labels.extend(_labels[keep])
                last_boxes.extend(_boxes[keep])

            else:
                last_scores.extend(_scores)
                last_labels.extend(_labels)
                last_boxes.extend(_boxes)

        return {"scores": last_scores, "labels": last_labels, "boxes": last_boxes}

def vis_class(img, pos, class_str, font_scale=0.35):
    """Visualizes the class."""
    # temp_GREEN=np.clip(np.asarray(_GREEN)*label,0,255).astype(np.uint8).tolist()

    x0, y0 = int(pos[0]), int(pos[1])
    # Compute text size.
    txt = class_str
    font = cv2.FONT_HERSHEY_SIMPLEX
    ((txt_w, txt_h), _) = cv2.getTextSize(txt, font, font_scale, 1)
    # Place text background.
    back_tl = x0, y0 - int(1.2 * txt_h)
    back_br = x0 + txt_w, y0
    cv2.rectangle(img, back_tl, back_br, _GREEN, -1) # _GREEN
    # Show text.
    txt_tl = x0, y0 - int(0.2 * txt_h)
    cv2.putText(img, txt, txt_tl, font, font_scale, _GRAY, lineType=cv2.LINE_AA)
    cv2.rectangle(img,(pos[0],pos[1]),(pos[2],pos[3]),_GREEN,2) # _GREEN
    return img

def draw_rect(image,pred,scale_factor):
    labels = pred["labels"]
    bboxs = pred["boxes"]
    scores = pred["scores"]
    h,w,size=scale_factor

    for label,bbox,score in zip(labels,bboxs,scores):
        label=label.cpu().numpy()
        bbox=bbox.cpu().numpy()#.astype(np.int16)
        score=score.cpu().numpy()
        class_str="%s:%.3f"%(classes[int(label)],score) # 跳过背景
        # pos=list(map(lambda x:int(x/scale_factor),bbox))
        if h>=w:
            bbox = bbox*h / size
            diff = h - w
            bbox[0]-= diff // 2
            bbox[2]-= diff // 2
        else:
            bbox = bbox * w / size
            diff = w - h
            bbox[1] -= diff // 2
            bbox[3] -= diff // 2

        pos = list(map(int, bbox))

        image=vis_class(image,pos,class_str,0.5)
    return image

class YOLOV1Infer(nn.Module):
    def __init__(self,root:str):
        super(YOLOV1Infer,self).__init__()
        self.batch_size = 1
        # root = ""
        seed = 100
        resize = 416
        self.num_classes = 1  # 不包含背景
        self.num_anchors = 2  # 2个box
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        torch.manual_seed(seed)
        kwargs = {'num_workers': 5, 'pin_memory': True} if self.use_cuda else {}
        self.save_model = "./models/model.pt"

        trainDataset = InferDataset(root, transforms=transforms.Compose(
            [
                # transforms.Resize((resize, resize)),
                # transforms.ToTensor(),
                # transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
                transforms.ToTensor(),
                Resize_fixed(resize, training=False)
            ]
        ))

        self.data_loader = torch.utils.data.DataLoader(
            trainDataset,
            batch_size=self.batch_size, shuffle=False,
            **kwargs
        )

        # self.network = YOLOV1Net(self.num_classes, "resnet18", 512, True, 0.5)
        self.network = YOLOV1Net(self.num_classes, "resnet50", 2048, True, 0.5)

        if self.use_cuda:
            self.network.to(self.device)
        self.loss_func = YOLOv1Loss(self.device, self.num_anchors, self.num_classes)

        self.network.load_state_dict(torch.load(self.save_model))

        self.conf_thres = 0.7
        self.nms_thres = 0.4
        self.image_mean = [0.485, 0.456, 0.406]
        self.image_std = [0.229, 0.224, 0.225]

    def renormalize(self, image):
        dtype, device = image.dtype, image.device
        mean = torch.as_tensor(self.image_mean, dtype=dtype, device=device)
        std = torch.as_tensor(self.image_std, dtype=dtype, device=device)
        # return (image - mean[:, None, None]) / std[:, None, None]
        return (image *std[None,:, None, None])+mean[None,:, None, None]

    def forward(self):
        self.network.eval()
        with torch.no_grad():
            for idx, (data,path,origin_img) in enumerate(self.data_loader):
                if self.use_cuda:
                    data = data.to(self.device)

                output = self.network(data)
                detections = self.loss_func(output)

                if len(detections) > 0:
                    _detections = apply_nms(detections[0], self.conf_thres, self.nms_thres, device=self.device, filter_labels=[])
                if _detections is None: continue

                image = origin_img[0].cpu().numpy()
                image = image.astype(np.uint8)#.transpose([1, 2, 0])
                # print("===========",image.shape,"====================")
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                scale_factor = [image.shape[0],image.shape[1],data.size(2)]
                image = draw_rect(image, _detections, scale_factor)

                # save
                newPath = path[0].replace("PNGImages","result")
                if not os.path.exists(os.path.dirname(newPath)):os.makedirs(os.path.dirname(newPath))
                cv2.imwrite(newPath,image)

                # cv2.imshow("test", image)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()

if __name__=="__main__":
    model = YOLOV1Infer("../valid/PNGImages")
    model()