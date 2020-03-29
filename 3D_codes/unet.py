# import packages
import torch
import torch.nn as nn
import torch.utils.data
import torchvision
import torchvision.transforms
import os
import time
import numpy as np
import skimage.transform
import matplotlib.pyplot as plt
import SimpleITK as sitk
import pandas as pd
import sklearn.model_selection as model_selection
from tqdm import tqdm
import skimage
import pickle


def get_activation(activation_type):
    activation_type = activation_type.lower()
    if hasattr(nn, activation_type):
        return getattr(nn, activation_type)()
    else:
        return nn.ReLU()


def _make_nConv(in_channels, out_channels, nb_Conv, activation='ReLU', n_group = 1):
    layers = []
    layers.append(ConvBatchNorm(in_channels, out_channels, activation, n_group))

    for _ in range(nb_Conv - 1):
        layers.append(ConvBatchNorm(out_channels, out_channels, activation, n_group))
    return nn.Sequential(*layers)


class ConvBatchNorm(nn.Module):
    """(convolution => [BN] => ReLU)"""

    def __init__(self, in_channels, out_channels, activation='ReLU', n_group = 1):
        super(ConvBatchNorm, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, groups = n_group,
                              kernel_size=3, padding=1)
        self.norm = nn.BatchNorm3d(out_channels)
        self.activation = get_activation(activation)

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        return self.activation(out)


class DownBlock(nn.Module):
    """Downscaling with maxpool convolution"""

    def __init__(self, in_channels, out_channels, nb_Conv, activation='ReLU',n_group = 1):
        super(DownBlock, self).__init__()
        self.maxpool = nn.MaxPool3d(2)
        self.nConvs = _make_nConv(in_channels, out_channels, nb_Conv, activation, n_group = n_group)

    def forward(self, x):
        out = self.maxpool(x)
        return self.nConvs(out)


class UpBlock(nn.Module):
    """Upscaling then conv"""

    def __init__(self, in_channels, out_channels, nb_Conv, activation='ReLU'):
        super(UpBlock, self).__init__()

        self.up = nn.Upsample(scale_factor=2)
        self.nConvs = _make_nConv(in_channels, out_channels, nb_Conv, activation)

    def forward(self, x, skip_x):
        out = self.up(x)
        x = torch.cat([out, skip_x], dim=1)  # dim 1 is the channel dimension
        return self.nConvs(x)


class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels_,n_group = 1):
        super(Encoder, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels_ # list of channels out of each block

        self.inc = ConvBatchNorm(in_channels, out_channels_[0],n_group = n_group)
        self.down1 = DownBlock(out_channels_[0], out_channels_[1], nb_Conv=2,n_group = n_group)
        self.down2 = DownBlock(out_channels_[1], out_channels_[2], nb_Conv=2,n_group = n_group)
        self.down3 = DownBlock(out_channels_[2], out_channels_[3], nb_Conv=2,n_group = n_group)
        self.down4 = DownBlock(out_channels_[3], out_channels_[4], nb_Conv=2,n_group = n_group)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        return x1, x2, x3, x4, x5


class Decoder(nn.Module):
  def __init__(self, in_channels, out_channels, n_classes):
    super(Decoder, self).__init__()
    self.in_channels = in_channels # must be a list of the nb of channels of inputs x1, ..., x5
    self.out_channels = out_channels #list of the nb of channels that each up conv must produce

    self.up1 = UpBlock(in_channels[-1] + in_channels[-2], out_channels[0], nb_Conv=2)
    self.up2 = UpBlock(out_channels[0] + in_channels[-3], out_channels[1], nb_Conv=2)
    self.up3 = UpBlock(out_channels[1] + in_channels[-4], out_channels[2], nb_Conv=2)
    self.up4 = UpBlock(out_channels[2] + in_channels[-5], out_channels[3], nb_Conv=2)
    self.outc = nn.Conv3d(out_channels[3], n_classes, kernel_size=3, stride=1, padding=1)
    self.last_activation = get_activation('Softmax')

  def forward(self, x1, x2, x3, x4, x5):
    x = self.up1(x5, x4)
    x = self.up2(x, x3)
    x = self.up3(x, x2)
    x = self.up4(x, x1)
    logits = self.last_activation(self.outc(x))
    return logits

class Hybrid_Unet(nn.Module):

  def __init__(self, n_channels, n_classes, encoder_channels, decoder_channels):
    super(Hybrid_Unet, self).__init__()
    self.n_channels = n_channels
    self.n_classes = n_classes
    self.encoders = nn.ModuleList()
    for k in range(n_channels):
      self.encoders.append(Encoder(1, encoder_channels))
    self.decoder = Decoder([4*a for a in encoder_channels], decoder_channels, n_classes)

  def forward(self, x):
    X1 = []
    X2 = []
    X3 = []
    X4 = []
    X5 = []
    k = 0
    for mod in self.encoders:
      x1, x2, x3, x4, x5 = mod(torch.unsqueeze(x[:,k,...],1))
      X1.append(x1)
      X2.append(x2)
      X3.append(x3)
      X4.append(x4)
      X5.append(x5)
      k+=1
    y1 = torch.cat(X1, dim=1)
    y2 = torch.cat(X2, dim=1)
    y3 = torch.cat(X3, dim=1)
    y4 = torch.cat(X4, dim=1)
    y5 = torch.cat(X5, dim=1)
    return self.decoder(y1, y2, y3, y4, y5)



class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, n_group = 1):
        '''
        n_channels : number of channels of the input.
        By default 4, because we have 4 modalities
        n_labels : number of channels of the ouput.
        By default 4 (3 labels + 1 for the background)
        '''
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        # Question here
        self.inc = ConvBatchNorm(n_channels, 64, n_group = n_group)
        self.down1 = DownBlock(64, 128, nb_Conv=2, n_group = n_group)
        self.down2 = DownBlock(128, 256, nb_Conv=2, n_group = n_group)
        self.down3 = DownBlock(256, 512, nb_Conv=2, n_group = n_group)
        self.down4 = DownBlock(512, 512, nb_Conv=2, n_group = n_group)
        self.up1 = UpBlock(1024, 256, nb_Conv=2)
        self.up2 = UpBlock(512, 128, nb_Conv=2)
        self.up3 = UpBlock(256, 64, nb_Conv=2)
        self.up4 = UpBlock(128, 64, nb_Conv=2)
        self.outc = nn.Conv3d(64, n_classes, kernel_size=3, stride=1, padding=1)
        self.last_activation = get_activation('Softmax')

    def forward(self, x):
        # Question here
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.last_activation(self.outc(x))
        return logits

def pre_normalize(img):
    # normalize one modality (3d)
    idx = np.where(img > 0)
    non_zero = img[idx[0], idx[1], idx[2]]
    img_m = np.mean(non_zero)
    img_std = np.std(non_zero)
    if img_std>0:
        img_new = (img - img_m) / img_std
    else:
        img_new = img - img_m
    return img_new

class TorchDataset(torch.utils.data.Dataset):

    def __init__(self, files_list, img_path, label_path, transform=None):
        super(TorchDataset, self).__init__()
        self.files = files_list
        self.images_path = img_path
        self.labels_path = label_path
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):

        patient = self.files[idx]

        irm, mask = self.load(patient)
        sample = (irm, mask)

        if self.transform:
            irm, mask = self.transform(sample)

        return (irm, mask, patient)

    def load(self, ID):
        if type(ID)==str:# concatenate all the modality together as input
            patient_path = os.path.join(self.images_path, ID)
            irm = sitk.GetArrayFromImage(sitk.ReadImage(patient_path))
            for i in range(4):
                irm[i] = pre_normalize(irm[i])
            mask_path = os.path.join(self.labels_path, ID)
        elif len(ID) == 2:# use only one modality as input
            patient_path = os.path.join(self.images_path, ID[0])
            irm = sitk.GetArrayFromImage(sitk.ReadImage(patient_path))[ID[1]]
            irm = np.expand_dims(pre_normalize(irm),axis=0)
            mask_path = os.path.join(self.labels_path, ID[0])

        mask = sitk.GetArrayFromImage(sitk.ReadImage(mask_path))
        mask[mask == 4] = 3
        label = 4
        mask = mask.astype(np.int16)
        # turn mask into (4,155,240,240) so that every layer represent one class
        mask = np.rollaxis(np.eye(label, dtype=np.uint8)[mask], -1, 0)
        # irm (1,155,240,240)
        # mask (4,155,240,240)
        return irm, mask

def flip_scale_crop_train(sample):
    irm, mask = sample
    irm_out = []
    mask_out = []

    n_modality = irm.shape[0]
    n_maskc = mask.shape[0]
    # flip at 0.5 proba
    d = np.random.randint(3) + 1
    num = np.random.random() > 0.5
    if num:
        irm = np.flip(irm, axis=d)
        mask = np.flip(mask, axis=d)

    # scale
    factor = np.random.random() / 10 + 0.6
    for k in range(n_modality):
        irm_out.append(skimage.transform.rescale(irm[k], factor))
    for k in range(n_maskc):
        mask_out.append(np.round(skimage.transform.rescale(mask[k].astype(float), factor)))
    irm_out = np.stack(irm_out, axis=0)
    mask_out = np.stack(mask_out, axis=0)

    # crop image
    _, a, b, c = irm_out.shape
    irm_ = irm_out[:, (-40 + a // 2):(40 + a // 2), (-64 + b // 2):(64 + b // 2), (-64 + c // 2):(64 + c // 2)]
    mask_ = mask_out[:, (-40 + a // 2):(40 + a // 2), (-64 + b // 2):(64 + b // 2), (-64 + c // 2):(64 + c // 2)]

    return (irm_, mask_)


def flip_scale_crop_test(sample):
    irm, mask = sample
    irm_out = []
    mask_out = []
    n_modality = irm.shape[0]
    n_maskc = mask.shape[0]

    # scale
    factor = 0.6
    for k in range(n_modality):
        irm_out.append(skimage.transform.rescale(irm[k], factor))
    for k in range(n_maskc):
        mask_out.append(np.round(skimage.transform.rescale(mask[k].astype(float), factor)))
    irm_out = np.stack(irm_out, axis=0)
    mask_out = np.stack(mask_out, axis=0)

    # crop image
    _, a, b, c = irm_out.shape
    irm_ = irm_out[:, (-40 + a // 2):(40 + a // 2), (-64 + b // 2):(64 + b // 2), (-64 + c // 2):(64 + c // 2)]
    mask_ = mask_out[:, (-40 + a // 2):(40 + a // 2), (-64 + b // 2):(64 + b // 2), (-64 + c // 2):(64 + c // 2)]

    return (irm_, mask_)

