import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models as tvmodels
from efficientnet_pytorch import EfficientNet
import math
from utils import YTMTBlock_ADD_negative,YTMTBlock_ADD_symmetric,YTMTBlock_Concat_negative,YTMTBlock_Concat_symmetric
import numpy as np
import os,cv2
import matplotlib.pyplot as plt

class convZ(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(convZ, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True), )
    def forward(self, input):
        return self.conv(input)

class Interection_ConvBlock(nn.Module): # pattern
    def __init__(self, in_channels=512, out_channels=256,Interection=True,pattern= 'CS'):
        super(Interection_ConvBlock, self).__init__()
        #self.up_l =  nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2)
        self.conv_l = ResBlock(in_channels, out_channels)

        #self.up_r = nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2)
        self.conv_r = ResBlock(in_channels, out_channels)
        self.Interection = Interection

        if pattern == 'AN':
            self.ytmt_norm = YTMTBlock_ADD_negative()
        elif pattern == 'AS':
            self.ytmt_norm = YTMTBlock_ADD_symmetric()
        elif pattern == 'CN':
            self.ytmt_norm = YTMTBlock_Concat_negative(out_channels, BN=True)
        else:
            self.ytmt_norm = YTMTBlock_Concat_symmetric(out_channels, BN=True)
    def forward(self, input_l, input_r):
        #up_l = self.up_l(input_l)
        #merge_l = torch.cat([up_l,], dim=1)
        out_l = self.conv_l(input_l)

        #up_r = self.up_r(input_r)
        #merge_r = torch.cat([up_r, pre_fea], dim=1)
        out_r = self.conv_r(input_r)

        if self.Interection:
            out_l, out_r = self.ytmt_norm(out_l, out_r)
        return out_l, out_r

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ConstantNormalize(nn.Module):
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        super(ConstantNormalize, self).__init__()
        mean = torch.Tensor(mean).view([1, 3, 1, 1])
        std = torch.Tensor(std).view([1, 3, 1, 1])
        # https://discuss.pytorch.org/t/keeping-constant-value-in-module-on-correct-device/10129
        self.register_buffer('mean', mean)
        self.register_buffer('std', std)

    def forward(self, x):
        return (x - self.mean) / (self.std + 1e-5)


class Conv1x1(nn.Sequential):
    def __init__(self, in_planes, out_planes=16, has_se=False, se_reduction=None):
        if has_se:
            if se_reduction is None:
                # se_reduction= int(math.sqrt(in_planes))
                se_reduction = 2
            super(Conv1x1, self).__init__(SELayer(in_planes, se_reduction),
                                          nn.Conv2d(in_planes, out_planes, 1, bias=False),
                                          nn.BatchNorm2d(out_planes),
                                          nn.ReLU()
                                          )
        else:
            super(Conv1x1, self).__init__(nn.Conv2d(in_planes, out_planes, 1, bias=False),
                                          nn.BatchNorm2d(out_planes),
                                          nn.ReLU()
                                          )




# https://pytorch.org/docs/stable/_modules/torchvision/models/resnet.html#resnext50_32x4d
class ResBlock(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(ResBlock, self).__init__()
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, padding=1, bias=False)
        #self.bn1 = nn.BatchNorm2d(out_planes)
        self.bn1 = nn.BatchNorm2d(out_planes // 2)
        self.IN1 = nn.InstanceNorm2d(out_planes // 2)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_planes, out_planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        #out = self.bn1(out)
        out1, out2 = torch.chunk(out, 2, dim=1)
        out1 = self.IN1(out1)
        out2 = self.bn1(out2)

        out = torch.cat([out1, out2], dim=1)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        #out = self.relu(out)

        return out


class ResBlock1(nn.Module):
    '''expand + depthwise + pointwise'''
    def __init__(self, in_planes, out_planes, expansion=1, stride=1):
        super(ResBlock1, self).__init__()
        self.stride = stride

        planes = expansion * in_planes
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1,
                               stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, groups=planes,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, out_planes, kernel_size=1,
                               stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes)

        self.shortcut = nn.Sequential()
        # if stride == 1 and in_planes != out_planes:
        #     self.shortcut = nn.Sequential(
        #         nn.Conv2d(in_planes, out_planes, kernel_size=1,
        #                   stride=1, padding=0, bias=False),
        #         nn.BatchNorm2d(out_planes),
        #     )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = out + x #self.shortcut(x) if self.stride==1 else out
        return out


ml_features = []


def feature_hook(module, fea_in, fea_out):
    #     print("hooker working")
    # module_name.append(module.__class__)
    # features_in_hook.append(fea_in)
    global ml_features
    ml_features.append(fea_out)
    return None



class SHADOW(nn.Module):
    # decompose net
    def __init__(self,
                 backbone='efficientnet-b3',
                 proj_planes=16,
                 pred_planes=32,
                 use_pretrained=True,
                 fix_backbone=False,
                 has_se=False,
                 num_of_layers=8
                 ,pattern= 'CS',
                 Interection=True):

        super(SHADOW, self).__init__()

        # load backbone
        if use_pretrained:
            self.feat_net = EfficientNet.from_pretrained(backbone) # backbone : 'efficientnet-b3'
            # https://github.com/lukemelas/EfficientNet-PyTorch/blob/master/efficientnet_pytorch/model.py
        else:
            self.feat_net = EfficientNet.from_name(backbone)
        print("Total number of paramerters in EfficientNet networks is {} ".format(
            sum(x.numel() for x in self.feat_net.parameters())))
        print("Total number of requires_grad paramerters in EfficientNet networks is {} ".format(
            sum(p.numel() for p in self.feat_net.parameters() if p.requires_grad)))

        # remove classification head to get correct param count
        self.feat_net._avg_pooling = None
        self.feat_net._dropout = None
        self.feat_net._fc = None
        self.pattern = pattern

        # register hook to extract multi-level features
        in_planes = []
        feat_layer_ids = list(range(0, len(self.feat_net._blocks), 2))
        for idx in feat_layer_ids:
            self.feat_net._blocks[idx].register_forward_hook(hook=feature_hook)
            in_planes.append(self.feat_net._blocks[idx]._bn2.num_features)

        if fix_backbone:
            for param in self.feat_net.parameters():
                param.requires_grad = False

        self.norm = ConstantNormalize()

        # 1*1 projection conv
        proj_convs = [Conv1x1(ip, proj_planes, has_se=has_se) for ip in in_planes]
        self.proj_convs = nn.ModuleList(proj_convs)

        if backbone == 'efficientnet-b0':
            channel_factor = 8
        elif backbone == 'efficientnet-b1':
            channel_factor = 12
        elif backbone == 'efficientnet-b2':
            channel_factor = 12
        elif backbone == 'efficientnet-b3':
            channel_factor = 13
        self.temp_conv = Conv1x1(proj_planes * channel_factor,proj_planes * channel_factor, has_se=has_se)

        # two stream feature
        self.stem_conv1 = Conv1x1(proj_planes * len(in_planes), pred_planes, has_se=has_se)  # 1*1
        self.stem_conv2 = Conv1x1(proj_planes * len(in_planes), pred_planes, has_se=has_se)

        self.Interection_convs1 = Interection_ConvBlock(in_channels=pred_planes,out_channels=pred_planes
                                                        ,pattern = self.pattern,Interection=Interection)
        self.Interection_convs2 = Interection_ConvBlock(in_channels=pred_planes, out_channels=pred_planes,
                                                        pattern=self.pattern,Interection=Interection)
        self.Interection_convs3 = Interection_ConvBlock(in_channels=pred_planes, out_channels=pred_planes,
                                                        pattern=self.pattern,Interection=Interection)
        self.Interection_convs4 = Interection_ConvBlock(in_channels=pred_planes, out_channels=pred_planes,
                                                        pattern=self.pattern,Interection=Interection)

        # prediction
        pred_layers1 = []
        pred_layers1.append(nn.Conv2d(pred_planes, 1, 1))
        self.pred_conv1 = nn.Sequential(*pred_layers1)

        pred_layers2 = []
        pred_layers2.append(nn.Conv2d(pred_planes, 1, 1))
        self.pred_conv2 = nn.Sequential(*pred_layers2)


        for m in self.modules():
            if isinstance(m, nn.ReLU):
                m.inplace = False

    def forward(self, x):
        global ml_features

        b, c, h, w = x.size()
        ml_features = []

        _ = self.feat_net.extract_features(self.norm(x))
        h_f, w_f = ml_features[0].size()[2:]
        proj_features = []
        for i in range(len(ml_features)):
            cur_proj_feature = self.proj_convs[i](ml_features[i])
            cur_proj_feature_up = F.interpolate(cur_proj_feature, size=(h_f, w_f), mode='bilinear')
            proj_features.append(cur_proj_feature_up)
        cat_feature = torch.cat(proj_features, dim=1)
        cat_feature = self.temp_conv(cat_feature)

        out_l = self.stem_conv1(cat_feature)  # stem_feat [N,32,h_f, w_f ]
        out_r = self.stem_conv2(cat_feature)

        out_l, out_r = self.Interection_convs1(out_l, out_r)
        out_l, out_r = self.Interection_convs2(out_l, out_r)
        out_l, out_r = self.Interection_convs3(out_l, out_r)
        out_l, out_r = self.Interection_convs4(out_l, out_r)

        S_out = F.interpolate(self.pred_conv1(out_l), size=(h, w), mode='bilinear')
        NS_out = F.interpolate(self.pred_conv1(out_r), size=(h, w), mode='bilinear')
        return S_out,NS_out

if __name__ == "__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = SHADOW(backbone='efficientnet-b3',num_of_layers=4).to(device)
    print(model.pred_conv1)
    input = torch.randn(1, 3, 416, 416).to(device)
    S_out,NS_out= model(input)
    print('-'*50)
    print(S_out.shape)
    print('#generator parameters:', sum(param.numel() for param in model.parameters()))
    from thop import profile
    flops, params = profile(model, inputs=(input,))
    print(flops, params, '----', flops / 1000000000, params / 1000000)

    from torch.autograd import Variable
    import time
    import argparse
    parser = argparse.ArgumentParser(description='Runing time')
    parser.add_argument('--gpu', type=int, default=0, help='GPU')
    parser.add_argument('--iter_size', type=int, default=100, help='TOTAL_ITER')
    args = parser.parse_args()

    # SAVE_FOLDER = 'time'
    GPU = args.gpu
    TOTAL_ITER = args.iter_size



    img = torch.rand((1, 3, 416, 416))
    if GPU >= 0:
        with torch.cuda.device(GPU):
            img = img.cuda()
    img = Variable(img, requires_grad=False)

    # prepare model
    # name, (model, forward) = model_forward[MODEL_ID]
    if GPU >= 0:
        with torch.cuda.device(GPU):
            model = SHADOW(backbone='efficientnet-b3',num_of_layers=4,proj_planes=16,
                 pred_planes=32,).cuda()
    print('#generator parameters:', sum(param.numel() for param in model.parameters()))
    model.eval()

    # Warm up
    for _ in range(100):
        x1,x2 = model(img)

    # Test
    # print('Test {} ...'.format(name))

    t = time.time()
    for _ in range(TOTAL_ITER):
        x1,x2 = model(img)
    mean_time = (time.time() - t) / TOTAL_ITER

    print('\tmean time: {}'.format(mean_time))

