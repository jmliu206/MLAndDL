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
import time
from torch.utils.tensorboard import SummaryWriter

class YOLOV1(nn.Module):
    def __init__(self,isTrain=True):
        super(YOLOV1,self).__init__()
        self.isTrain=isTrain
        self.batch_size=16
        self.test_batch_size=1
        self.epochs = 200
        self.print_freq = 1
        self.num_classes = 1 # 不包含背景
        self.num_anchors = 2 # 2个box
        self.learning_rate = 2e-3
        seed = int(time.time()*1000)
        resize = 416
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        torch.manual_seed(seed)
        kwargs = {'num_workers': 5, 'pin_memory': True} if self.use_cuda else {}
        self.save_model = "./models/model.pt"
        summaryPath = "./models/yolov1_resnet50_416/"
        if not os.path.exists(os.path.basename(os.path.dirname(self.save_model))):
            os.makedirs(os.path.basename(os.path.dirname(self.save_model)))

        root = "../PennFudanPed"
        # root = r"C:\Users\MI\Documents\GitHub\PennFudanPed"
        if self.isTrain:
            trainDataset = PennFudanDataset(root, transforms=transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.RandomHorizontalFlip(0.5),
                    Resize_fixed(resize, training=False)
                    # transforms.ColorJitter(0.05, 0.05, 0.05, 0.05)
                ]
            ))

            self.train_loader = torch.utils.data.DataLoader(
                trainDataset,
                batch_size=self.batch_size,shuffle=True,
                **kwargs
            )
        else:
            testDataset = PennFudanDataset2(root, transforms=transforms.Compose(
                [
                    transforms.ToTensor(),
                    # transforms.RandomHorizontalFlip(0.5),
                    Resize_fixed(resize, training=False)
                    # transforms.ColorJitter(0.05, 0.05, 0.05, 0.05)
                ]
            ))

            self.test_loader = torch.utils.data.DataLoader(
                testDataset,
                batch_size=self.test_batch_size, shuffle=False,
                **kwargs
            )

        # self.network = YOLOV1Net(self.num_classes,"resnet18",512,True,0.5)
        self.network = YOLOV1Net(self.num_classes,"resnet50",2048,True,0.5)

        if self.use_cuda:
            self.network.to(self.device)

        self.loss_func = YOLOv1Loss(self.device,self.num_anchors,self.num_classes)

        if self.isTrain:
            # optimizer
            base_params = list(
                map(id, self.network.backbone.parameters())
            )
            logits_params = filter(lambda p: id(p) not in base_params, self.network.parameters())

            params = [
                {"params": logits_params, "lr": self.learning_rate},  # 1e-3
                {"params": self.network.backbone.parameters(), "lr": self.learning_rate / 10},  # 1e-4
            ]

            # self.optimizer = torch.optim.SGD(params,  momentum=self.sgd_momentum) # lr=self.learning_rate,
            self.optimizer = torch.optim.Adam(params, weight_decay=4e-05)
            self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=20, gamma=0.1)

        # 加载预训练模型
        if os.path.exists(self.save_model):
            self.network.load_state_dict(torch.load(self.save_model))

        self.writer = SummaryWriter(summaryPath)

    def forward(self):
        if self.isTrain:
            for epoch in range(self.epochs):
                self.__train(epoch)
                # update the learning rate
                self.lr_scheduler.step()
                torch.save(self.network.state_dict(), self.save_model)
                # torch.save(self.network.state_dict(), self.save_model+"_"+str(epoch))
        else:
            self.__test()


    def __train(self, epoch):
        self.network.train()
        num_trains = len(self.train_loader.dataset)
        for idx, (data, target) in enumerate(self.train_loader):
            if self.use_cuda:
                data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            output = self.network(data)

            loss_dict = self.loss_func(output,target)

            losses = sum(loss for loss in loss_dict.values())

            self.optimizer.zero_grad()
            losses.backward()
            self.optimizer.step()

            # 记录到TensorBoard
            self.writer.add_scalar('total_loss',losses.item(),epoch * num_trains//self.batch_size + idx)
            for key, loss in loss_dict.items():
                self.writer.add_scalar(key, loss.item(), epoch * num_trains//self.batch_size + idx)

            if idx % self.print_freq == 0:
                ss = "epoch:{}-({}/{}) ".format(epoch, idx*self.batch_size, num_trains)
                ss += "total:{:.3f}\t".format(losses.item())
                for key, loss in loss_dict.items():
                    ss += "{}:{:.3f}\t".format(key, loss.item())

                print(ss)

    def __test(self):
        self.network.eval()
        with torch.no_grad():
            for idx, (data, target) in enumerate(self.test_loader):
                if self.use_cuda:
                    data = data.to(self.device)

                output = self.network(data)
                preds = self.loss_func(output)[0]

                print("\npred:\t",preds["boxes"])
                print("\ngt_box:\t",target["boxes"])

                if idx>10:break

if __name__=="__main__":
    model = YOLOV1(isTrain=True)
    model()