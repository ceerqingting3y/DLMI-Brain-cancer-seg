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

def dice_loss(input, target):
    smooth = 0.01
    target = target.float()
    input = input.float()
    input_flat = input.contiguous().view(-1)
    target_flat = target.contiguous().view(-1)
    intersection = (input_flat * target_flat).sum()
    return 1 - ((2. * intersection + smooth) /
                (input_flat.pow(2).sum() + target_flat.pow(2).sum() + smooth))

def mean_dice_loss(input, target):
    channels = list(range(target.shape[1]))
    loss = 0
    for channel in channels:
        dice = dice_loss(input[:, channel, ...],
                         target[:, channel, ...])
        loss += dice

    return loss / len(channels)

def to_var(x, device):
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    x = x.to(device,dtype=torch.float)
    return x

def to_numpy(x):
    if not (isinstance(x, np.ndarray) or x is None):
        if x.is_cuda:
            x = x.data.cpu()
        x = x.numpy()
    return x

def save_checkpoint(state, save_path):

    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    epoch = state['epoch']
    filename = save_path + '/' + \
        'model.{:02d}.pth.tar'.format(epoch)
    torch.save(state, filename)


def print_summary(epoch, i, nb_batch, loss, batch_time,
                  average_loss, average_time, mode):
    '''
        mode = Train or Test
    '''
    summary = '[' + str(mode) + '] Epoch: [{0}][{1}/{2}]\t'.format(
        epoch, i, nb_batch)

    string = ''
    string += ('Dice Loss {:.4f} ').format(loss)
    string += ('(Average {:.4f}) \t').format(average_loss)
    string += ('Batch Time {:.4f} ').format(batch_time)
    string += ('(Average {:.4f}) \t').format(average_time)

    summary += string

    print(summary)

def metrics(mask_, gt_):
    '''
    Taking to binary array of same shape as input
    This function compute the confusion matrix and use it to calculate
    Dice metrics, Sensitivity and Specificity
    Input : mask_, gt_ numpy array of identic shape (only 1 and 0)
    Output : List of 3 scores
    '''
    lnot = np.logical_not
    land = np.logical_and

    true_positive = np.sum(land((mask_), (gt_)))
    false_positive = np.sum(land((mask_), lnot(gt_)))
    false_negative = np.sum(land(lnot(mask_), (gt_)))
    true_negative = np.sum(land(lnot(mask_), lnot(gt_)))

    M = np.array([[true_negative, false_negative],
                  [false_positive, true_positive]]).astype(np.float64)
    metrics = {}
    metrics['Sensitivity'] = M[1, 1] / (M[0, 1] + M[1, 1] + 1e-5)
    metrics['Specificity'] = M[0, 0] / (M[0, 0] + M[1, 0] + 1e-5)
    metrics['Dice'] = 2 * M[1, 1] / (M[1, 1] * 2 + M[1, 0] + M[0, 1] + 1e-5)
    # metrics may be NaN if denominator is zero! use np.nanmean() while
    # computing average to ignore NaNs.

    return [metrics['Dice'], metrics['Sensitivity'], metrics['Specificity']]


def calculate_regions(mask):
    wt = (mask == 1) + (mask == 3) + (mask == 2)
    tc = (mask == 1) + (mask == 3)
    et = mask == 3
    return wt, tc, et


def evalAllmetric(mask_, gt_):
    '''
    This functions takes as input two numpy array with labels between
    0, 1, 2 and 4 and calculate the metrics as defined in BraTS data challenge
    mask_ and gt_ should be array of int
    '''
    ref_wt, ref_tc, ref_et = calculate_regions(mask_)
    wt, tc, et = calculate_regions(gt_)

    wt_metrics = metrics(ref_wt, wt)
    tc_metrics = metrics(ref_tc, tc)
    et_metrics = metrics(ref_et, et)

    # pd.DataFrame({'wt': wt_metrics, 'tc': tc_metrics, 'et': et_metrics},
    # index=['Dice', 'Sensitivity', 'Specificity'])
    return wt_metrics + tc_metrics + et_metrics

def train_model(model, loader, criterion, optimizer, epoch, device, fre=100, sample_size = 100,rand = False,batch_size=1):
    t0 = time.time()
    T = 0.
    total_loss = 0.

    for i, sample in enumerate(loader, 1):
        (irms, masks, patients) = sample

        irms = to_var(irms.float(), device)
        masks = to_var(masks.float(), device)
        pred_masks = model(irms)
        dice_loss = criterion(pred_masks, masks)
        optimizer.zero_grad()
        dice_loss.backward()
        optimizer.step()
        batch_time = time.time() - t0
        T += batch_size * batch_time
        total_loss += batch_size * dice_loss
        average_loss = total_loss / (i * batch_size)
        average_time = T / (i * batch_size)

        t0 = time.time()

        if i % fre == 0:
            print_summary(epoch, i, len(loader), dice_loss, batch_time,
                          average_loss, average_time, "Train")
        if i==sample_size and rand:
            break

    print_summary(epoch, i, len(loader), dice_loss, batch_time,
                  average_loss, average_time, "Train")
    return average_loss

def validate_model(model, loader, criterion, epoch,device,fre=100, sample_size = 100,rand = False,batch_size=1):
    t0 = time.time()
    T = 0.
    total_loss = 0.

    for i, sample in enumerate(loader, 1):
        (irms, masks, patients) = sample
        irms = to_var(irms.float(), device)
        masks = to_var(masks.float(), device)
        pred_masks = model(irms)
        dice_loss = criterion(pred_masks, masks)

        batch_time = time.time() - t0
        T += batch_size * batch_time
        total_loss += batch_size * dice_loss
        average_loss = total_loss / (i * batch_size)
        average_time = T / (i * batch_size)

        t0 = time.time()

        if i % fre == 0:
            print_summary(epoch, i, len(loader), dice_loss, batch_time,
                          average_loss, average_time, "Validation")
        if i==sample_size and rand:
            break
    print_summary(epoch, i, len(loader), dice_loss, batch_time,
                          average_loss, average_time, "Validation")
    return average_loss

def predict(loader, model, device, batch_size=1):
    preds = {}
    metricss = []
    for i, sample in tqdm(enumerate(loader, 1)):
        # Take variable and put them to GPU
        (irm, mask, patient) = sample
        irm = to_var(irm.float(), device)
        mask = to_numpy(mask)
        pred_mask = to_numpy(model(irm))
        ref = np.argmax(mask[0], axis=0)
        mask_out = np.argmax(pred_mask[0], axis=0)

        met = evalAllmetric(ref, mask_out)
        metricss.append(met)

    return preds, metricss