from torch import nn
import torch
from torch.nn import functional as F
import random
import sys

class YOLOv1Loss(nn.Module):
    def __init__(self,device="cpu",num_anchors=2,
                 num_classes=20, # 不包括背景
                 strides = 32,
                 threshold_conf=0.05,threshold_cls=0.5):
        super(YOLOv1Loss,self).__init__()
        # self.training = training
        self.device = device
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        self.strides = strides
        self.threshold_conf = threshold_conf
        self.threshold_cls = threshold_cls

    def forward(self,preds,targets=None):
        if targets is None:
            return self.predict(preds)
        else:
            return self.compute_loss(preds,targets)

    def compute_loss(self,preds,targets):
        """
        :param preds: [n,7,7,12]
        :param targets: [n,7,7,12]
        :return:
        """
        preds = preds.contiguous().view(-1,self.num_anchors*(5+self.num_classes))
        targets = targets.contiguous().view(-1,self.num_anchors*(5+self.num_classes))
        index = targets[...,4]==1
        no_index = targets[...,4]!=1
        has_obj = preds[index]
        no_obj = preds[no_index]
        targ_obj = targets[index]

        # 置信度loss(负样本置信度为0，)
        loss_conf = F.binary_cross_entropy(has_obj[...,4],torch.ones_like(has_obj[...,4]),reduction="sum") # 对应目标
        loss_conf += F.binary_cross_entropy(has_obj[...,4+5+self.num_classes],torch.ones_like(has_obj[...,4]),reduction="sum") # 对应目标

        loss_no_conf = F.binary_cross_entropy(no_obj[...,4],torch.zeros_like(no_obj[...,4]),reduction="sum") # 对应背景
        loss_no_conf += F.binary_cross_entropy(no_obj[...,4+5+self.num_classes],torch.zeros_like(no_obj[...,4]),reduction="sum") # 对应背景

        # boxes loss
        # loss_box = F.mse_loss(has_obj[...,:4],targ_obj[...,:4],reduction="sum")
        # loss_box += F.mse_loss(has_obj[...,5+self.num_classes:4+5+self.num_classes],targ_obj[...,:4],reduction="sum")

        loss_box = F.smooth_l1_loss(has_obj[...,:4],targ_obj[...,:4],reduction="sum")
        loss_box += F.smooth_l1_loss(has_obj[...,5+self.num_classes:4+5+self.num_classes],targ_obj[...,5+self.num_classes:4+5+self.num_classes],reduction="sum")


        # classify loss
        loss_clf = F.binary_cross_entropy(has_obj[...,5],targ_obj[...,5],reduction="sum")
        loss_clf += F.binary_cross_entropy(has_obj[...,5+5+self.num_classes],targ_obj[...,5],reduction="sum")

        loss_no_clf = F.binary_cross_entropy(no_obj[...,5],torch.zeros_like(no_obj[...,5]),reduction="sum")
        loss_no_clf += F.binary_cross_entropy(no_obj[...,5+5+self.num_classes],torch.zeros_like(no_obj[...,5]),reduction="sum")

        losses = {
            "loss_conf": loss_conf,
            "loss_no_conf": loss_no_conf * 0.05,
            "loss_box": loss_box * 50.,
            "loss_clf": loss_clf,
            "loss_no_clf": loss_no_clf*0.05,
            # "iou_loss": iou_loss
        }

        return losses

    def predict(self, preds):
        fh,fw = preds.shape[1:-1]
        preds = preds.contiguous().view(-1, self.num_anchors,5 + self.num_classes)
        # 选择置信度最高的对应box(多个box时)
        new_preds = torch.zeros_like(preds)[:,0,:]
        for i,p in enumerate(preds):
            # conf
            if p[0,4]*p[0,5]>p[1,4]*p[1,5]:
                new_preds[i]=preds[i,0,:]
            else:
                new_preds[i] = preds[i,1,:]

        preds = new_preds
        pred_box = preds[:, :4]
        pred_conf = preds[:, 4]
        pred_cls = preds[:, 5]  # *pred_conf # # 推理时做 p_cls*confidence

        # 转成x1,y1,x2,y2
        pred_box = self.reverse_normalize((fh,fw), pred_box)
        # pred_box = clip_boxes_to_image(pred_box, input_img[0].size()[-2:])  # 裁剪到图像内
        # # 过滤尺寸很小的框
        # keep = remove_small_boxes(pred_box.round(), self.min_size)
        # pred_box = pred_box[keep]
        # pred_cls = pred_cls[keep]
        # confidence = pred_conf[keep]#.squeeze(1)

        confidence = pred_conf

        # condition = ((pred_cls * confidence).max(dim=1)[0] > self.threshold_conf) & (
        #         confidence.squeeze(1) > self.threshold_conf)

        condition = ((pred_cls * confidence).max(dim=0)[0] > self.threshold_conf) & (
                confidence > self.threshold_conf)

        keep = torch.nonzero(condition).squeeze(1)

        # keep = torch.nonzero(confidence > self.threshold_conf).squeeze(1)
        pred_box = pred_box[keep]
        pred_cls = pred_cls[keep]
        confidence = confidence[keep]

        # labels and scores
        # scores, labels = torch.softmax(pred_cls, -1).max(dim=1)
        # scores, labels = pred_cls.max(dim=1)

        scores, labels = pred_cls,torch.ones_like(pred_cls)

        # 过滤分类分数低的
        # keep = torch.nonzero(scores > self.threshold_cls).squeeze(1)
        keep = torch.nonzero(scores > self.threshold_cls)
        pred_box, scores, labels, confidence = pred_box[keep], scores[keep], labels[keep], confidence[keep]

        return [{"boxes": pred_box, "scores": scores, "labels": labels, "confidence": confidence}]

    def reverse_normalize(self,featureShape, boxes):
        # [x0,y0,w,h]-->normalize 0~1--->[x1,y1,x2,y2]
        # h, w = input_img.size()[-2:]
        h_f, w_f = featureShape
        strides_h = self.strides # h // h_f
        strides_w = self.strides # w // w_f
        h,w = strides_h*h_f,strides_w*w_f

        # to 格网(x,y) 格式
        temp = torch.arange(0, len(boxes))
        grid_y = temp // w_f
        grid_x = temp - grid_y * w_f

        x0 = boxes[:, 0] * strides_w + (grid_x * strides_w).float().to(self.device)
        y0 = boxes[:, 1] * strides_h + (grid_y * strides_h).float().to(self.device)
        w_b = boxes[:, 2] * w
        h_b = boxes[:, 3] * h

        x1 = x0 - w_b / 2.
        y1 = y0 - h_b / 2.
        x2 = x0 + w_b / 2.
        y2 = y0 + h_b / 2.

        return torch.stack((x1, y1, x2, y2), dim=0).t()