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
import sys

from unet import UNet,TorchDataset,flip_scale_crop_test,flip_scale_crop_train
from utils import mean_dice_loss,to_var,to_numpy,save_checkpoint,evalAllmetric
from utils import train_model,validate_model,predict

import pandas as pd


def predict_ensemble(loader, models, device, batch_size=1):
    preds = {}
    metricss = []
    for i, sample in tqdm(enumerate(loader, 1)):
        # Take variable and put them to GPU
        (irm, mask, patient) = sample
        irm = to_var(irm.float(), device)
        mask = to_numpy(mask)
        pred_masks = np.zeros_like(mask)
        for p in range(len(models)):
            model = models[p]
            irm_ = irm[:,p,:,:,:].view(1,1,80,128,128)
            pred_mask = to_numpy(model(irm_))
            pred_masks += pred_mask

        ref = np.argmax(mask[0], axis=0)
        mask_out = np.argmax(pred_masks[0], axis=0)

        met = evalAllmetric(ref, mask_out)
        metricss.append(met)

    return preds, metricss


train_df = pd.read_csv('train_files.csv')
test_df = pd.read_csv('test_files.csv')
tr_files = train_df['files'].tolist()
test_files = test_df['files'].tolist()
np.random.shuffle(tr_files)
val_files = tr_files[:int(len(tr_files)*0.1)]
train_files = tr_files[int(len(tr_files)*0.1):]

learning_rate = 1e-4
image_size = (128, 128, 80)

n_modality = 4
n_labels = 4

epochs = 10
batch_size = 1

print_frequency = 100
save_frequency = 1
save_model = True
tumor_percentage = 0.5


save_path = 'save/'
loss_function = 'dice_loss'

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

path_img = "../Task01_BrainTumour/imagesTr"
labels_img = "../Task01_BrainTumour/labelsTr"

data_tr = TorchDataset(train_files,path_img,labels_img,transform=flip_scale_crop_train)
train_loader = torch.utils.data.DataLoader(data_tr,
                                           batch_size=batch_size, shuffle=True,
                                           drop_last=True)

data_val = TorchDataset(val_files,path_img,labels_img,transform=flip_scale_crop_test)
val_loader = torch.utils.data.DataLoader(data_val,
                                           batch_size=batch_size, shuffle=True,
                                           drop_last=True)

data_ts = TorchDataset(test_files,path_img,labels_img,transform=flip_scale_crop_test)
test_loader = torch.utils.data.DataLoader(data_ts,
                                           batch_size=batch_size,
                                           drop_last=True)


model_name = 'model.10.pth.tar'
models = []
for i in range(4):
    session_name = '3D_unet_vote_fusion_mod_new_' +str(i)+'_epoch_'+str(epochs)
    model_path = save_path + 'fusion_models/' + session_name + '/'
    model = UNet(n_channels=1, n_classes=n_labels)
    model.to(device)
    state_dict = torch.load(model_path + model_name)['state_dict']
    model.load_state_dict(state_dict)
    models.append(model)

torch.cuda.empty_cache()
with torch.no_grad():
    for model in models:
        model.eval()
    preds, metrics = predict_ensemble(test_loader, models, device,batch_size=1)
metrics_ar = np.stack(metrics, axis=0)
metrics_m = np.mean(metrics_ar, axis=0)
# metrics_std = np.std(metrics_ar, axis=0)

model_path_save = save_path + 'fusion_models/'
idxs = ['wt', 'tc', 'et']
cols = ['Dice', 'Sensitivity', 'Specificity']
mm = pd.DataFrame(metrics_m.reshape(3, -1), index=idxs)
mm.columns = cols
# save mean test scores
mm.to_csv(model_path_save + 'fusion_vote_test_evaluate_mean.csv')
# save all the test scores

df_ar = pd.DataFrame(metrics_ar)
df_ar.columns = [a+'_'+b for a in idxs for b in cols]
df_ar.to_csv(model_path_save +'fusion_vote_test_evaluate_all.csv')
