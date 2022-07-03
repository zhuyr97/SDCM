import os,cv2
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np

def feature_save(tensor,name):
    # tensor = torchvision.utils.make_grid(tensor.transpose(1,0))
    # tensor = torch.mean(tensor,dim=1).repeat(3,1,1)
    if not os.path.exists(str(name)):
        os.makedirs(str(name))
    for i in range(tensor.shape[1]):
        inp = tensor[:,i,:,:].detach().cpu().numpy().transpose(1,2,0)
        inp = np.clip(inp,0,1)
        inp = (inp-np.min(inp))/(np.max(inp)-np.min(inp))
        inp =np.squeeze(inp)
        plt.figure()
        plt.imshow(inp)
        plt.savefig(str(name) + '/' + str(i) + '.png')
        #cv2.imwrite(str(name)+'/'+str(i)+'.png',inp*255.0)
def feature_save1(tensor,name):
    # tensor = torchvision.utils.make_grid(tensor.transpose(1,0))
    # tensor = torch.mean(tensor,dim=1).repeat(3,1,1)
    if not os.path.exists(str(name)):
        os.makedirs(str(name))
    for i in range(tensor.shape[1]):
        inp = tensor[:,i,:,:].detach().cpu().numpy().transpose(1,2,0)
        inp = np.clip(np.abs(inp),0,1)
        inp = (inp-np.min(inp))/(np.max(inp)-np.min(inp))
        inp = np.squeeze(inp)
        #cv2.imwrite(str(name)+'/'+str(i)+'.png',inp*255.0)
        plt.figure()
        plt.imshow(inp)
        plt.savefig(str(name) + '/' + str(i) + '.png')

class ResBlock0(nn.Module):
    def __init__(self, in_planes, out_planes,BN=True):
        super(ResBlock0, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.BN = BN
        self.identity_conv = nn.Conv2d(in_planes, out_planes, kernel_size=3, padding=1, bias=False)
    def forward(self, x):
        identity = self.identity_conv(x)
        out = self.conv1(x)
        if self.BN:
            out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        if self.BN:
            out = self.bn2(out)
        out += identity
        #out = self.relu(out)
        return out
class ResBlock1(nn.Module):
    def __init__(self, in_planes, out_planes,BN=True):
        super(ResBlock1, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.BN = BN
        self.identity_conv = nn.Conv2d(in_planes, out_planes, kernel_size=3, padding=1, bias=False)
    def forward(self, x):
        out = self.conv1(x)
        if self.BN:
            out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        if self.BN:
            out = self.bn2(out)
        out = self.relu(out)
        return out


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True,BN=True):
        super(ResBlock, self).__init__()
        feature = in_channels
        self.BN = BN
        self.conv1 = nn.Conv2d(in_channels, feature, kernel_size=3, padding=1, bias=bias)
        self.bn1 = nn.BatchNorm2d(feature)
        self.relu1 = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.conv2_1 = nn.Conv2d(feature, feature, kernel_size=3, padding=1, bias=bias)
        #self.conv2_2 = nn.Conv2d(feature, feature, kernel_size=3, padding=3, bias=bias,dilation=3)
        self.conv2_3 = nn.Conv2d(feature, feature, kernel_size=3, padding=5, bias=bias,dilation=5)
        self.conv3 = nn.Conv2d((feature*2), out_channels, kernel_size=1, padding=0, bias=bias)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.identity_conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
    def forward(self, x):
        identity = self.identity_conv(x)

        residual = self.relu1(self.conv1(x))
        if self.BN:
            residual = self.bn1(residual)
        residual1 = self.relu1(self.conv2_1(residual))
        #residual2 = self.relu1(self.conv2_2(residual))
        residual3 = self.relu1(self.conv2_3(residual))
        residual = torch.cat((residual1, residual3), dim=1)
        out = self.conv3(residual)
        if self.BN:
            out = self.bn2(out)

        out += identity
        return out

class YTMTBlock_ADD_negative(nn.Module):
    def __init__(self):
        super(YTMTBlock_ADD_negative, self).__init__()
        self.relu = nn.ReLU()

    def forward(self, input_l, input_r):
        out_lp, out_ln = self.relu(input_l), input_l - self.relu(input_l)#feature_save(out_lp,'out_lp')
        out_rp, out_rn = self.relu(input_r), input_r - self.relu(input_r)

        out_l = out_lp + out_rn
        out_r = out_rp + out_ln
        return out_l, out_r

class YTMTBlock_ADD_symmetric(nn.Module):
    def __init__(self):
        super(YTMTBlock_ADD_symmetric, self).__init__()
        self.relu = nn.ReLU()

    def forward(self, input_l, input_r):
        out_lp, out_ln = self.relu(input_l), self.relu(input_l)-input_l
        out_rp, out_rn = self.relu(input_r), self.relu(input_r)- input_r

        out_l = out_lp + out_rn
        out_r = out_rp + out_ln
        return out_l, out_r

class YTMTBlock_Concat_negative(nn.Module):
    def __init__(self, channels,BN=True):
        super(YTMTBlock_Concat_negative, self).__init__()
        self.relu = nn.ReLU()
        self.BN = BN
        self.fusion_l = nn.Conv2d(channels * 2, channels, kernel_size=1)
        self.fusion_r = nn.Conv2d(channels * 2, channels, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, input_l, input_r):
        out_lp, out_ln = self.relu(input_l), input_l - self.relu(input_l)
        out_rp, out_rn = self.relu(input_r), input_r - self.relu(input_r)
        out_l = self.fusion_l(torch.cat([out_lp, out_rn], dim=1))
        if self.BN:
            out_l= self.bn1(out_l)
        out_r = self.fusion_r(torch.cat([out_rp, out_ln], dim=1))
        if self.BN:
            out_r= self.bn2(out_r)
        return out_l, out_r

class YTMTBlock_Concat_positive(nn.Module):
    def __init__(self, channels,BN=True):
        super(YTMTBlock_Concat_positive, self).__init__()
        self.relu = nn.ReLU()
        self.BN = BN
        self.fusion_l = nn.Conv2d(channels * 2, channels, kernel_size=1)
        self.fusion_r = nn.Conv2d(channels * 2, channels, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, input_l, input_r):
        out_lp, out_ln = self.relu(input_l), self.relu(input_l)
        out_rp, out_rn = self.relu(input_r),  self.relu(input_r)
        out_l = self.fusion_l(torch.cat([out_lp, out_rn], dim=1))
        if self.BN:
            out_l= self.bn1(out_l)
        out_r = self.fusion_r(torch.cat([out_rp, out_ln], dim=1))
        if self.BN:
            out_r= self.bn2(out_r)
        return out_l, out_r

class YTMTBlock_Concat_symmetric(nn.Module):
    def __init__(self, channels, BN=True):
        super(YTMTBlock_Concat_symmetric, self).__init__()
        self.relu = nn.ReLU()
        self.BN = BN
        self.fusion_l = nn.Conv2d(channels * 2, channels, kernel_size=1)
        self.fusion_r = nn.Conv2d(channels * 2, channels, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, input_l, input_r):
        out_lp, out_ln = self.relu(input_l), self.relu(input_l)-input_l
        out_rp, out_rn = self.relu(input_r),  self.relu(input_r)- input_r
        out_l = self.fusion_l(torch.cat([out_lp, out_rn], dim=1))
        if self.BN:
            out_l = self.bn1(out_l)
        out_r = self.fusion_r(torch.cat([out_rp, out_ln], dim=1))
        if self.BN:
            out_r = self.bn2(out_r)
        return out_l, out_r


class YTMTBlock_Concat_symmetricA(nn.Module):
    def __init__(self, channels, BN=True):
        super(YTMTBlock_Concat_symmetricA, self).__init__()
        self.relu = nn.ReLU()
        self.BN = BN
        self.fusion_l = nn.Conv2d(channels * 2, channels, kernel_size=1)
        self.fusion_r = nn.Conv2d(channels * 2, channels, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, input_l, input_r):
        out_lp, out_ln = self.relu(input_l), self.relu(input_l)-input_l
        out_rp, out_rn = self.relu(input_r),  self.relu(input_r)- input_r
        out_l = self.fusion_l(torch.cat([out_lp, out_rn], dim=1))
        if self.BN:
            out_l = self.bn1(out_l)
        out_r = self.fusion_r(torch.cat([out_rp, out_ln], dim=1))
        if self.BN:
            out_r = self.bn2(out_r)
        return out_l, out_r

class Conv1x1(nn.Sequential):
    def __init__(self, in_planes, out_planes=16, BN=True):
        self.BN =BN
        if self.BN:
            super(Conv1x1, self).__init__(nn.Conv2d(in_planes, out_planes, 1, bias=False),
                                          nn.BatchNorm2d(out_planes),
                                          nn.ReLU()
                                          )
        else:
            super(Conv1x1, self).__init__(nn.Conv2d(in_planes, out_planes, 1, bias=False),
                                          nn.ReLU()
                                          )

def wbce(pred, gt):
    pos = torch.eq(gt, 1).float()
    neg = torch.eq(gt, 0).float()
    num_pos = torch.sum(pos)
    num_neg = torch.sum(neg)
    num_total = num_pos + num_neg
    alpha_pos = num_neg / num_total
    alpha_neg = num_pos / num_total
    weights = alpha_pos * pos + alpha_neg * neg
    return nn.functional.binary_cross_entropy_with_logits(pred, gt, weights)

class MyWcploss(nn.Module):
    def __init__(self):
        super(MyWcploss, self).__init__()
    def forward(self, pred, gt):
        eposion = 1e-10
        sigmoid_pred = torch.sigmoid(pred)
        count_pos = torch.sum(gt)*1.0+eposion
        count_neg = torch.sum(1.-gt)*1.0
        beta = count_neg/count_pos
        beta_back = count_pos / (count_pos + count_neg)

        bce1 = nn.BCEWithLogitsLoss(pos_weight=beta)
        loss = beta_back*bce1(pred, gt)
        return loss
class MyWcploss1(nn.Module):
    def __init__(self):
        super(MyWcploss1, self).__init__()
    def forward(self, pred, gt):
        eposion = 1e-10
        #gt = torch.ones_like(gt) - gt
        sigmoid_pred = torch.sigmoid(pred)
        count_pos = torch.sum(gt)*1.0+eposion
        count_neg = torch.sum(1.-gt)*1.0
        beta = count_neg/count_pos
        beta_back = count_pos / (count_pos + count_neg)

        bce1 = nn.BCEWithLogitsLoss(pos_weight=beta)
        loss = beta_back*bce1(pred, gt)
        return loss

