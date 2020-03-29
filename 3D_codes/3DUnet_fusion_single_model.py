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
# single modality model training
modality_num = int(sys.argv[1])

train_df = pd.read_csv('train_files.csv')
test_df = pd.read_csv('test_files.csv')
train_files = train_df['files'].tolist()
test_files = test_df['files'].tolist()
tr_files = [[a,modality_num] for a in train_files]
ts_files = [[a,modality_num] for a in test_files]
np.random.shuffle(tr_files)
val_files = tr_files[:int(len(tr_files)*0.1)]
tr_files = tr_files[int(len(tr_files)*0.1):]

learning_rate = 1e-4
image_size = (128, 128, 80)

n_modality = 1
n_labels = 4

epochs = 10
batch_size = 1

print_frequency = 100
save_frequency = 1
save_model = True
tumor_percentage = 0.5

#patience = 20

save_path = 'save/'
loss_function = 'dice_loss'

session_name = '3D_unet_vote_fusion_mod_new_' +str(modality_num)+'_epoch_'+str(epochs)
model_path = save_path + 'fusion_models/' + session_name + '/'
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

path_img = "../Task01_BrainTumour/imagesTr"
labels_img = "../Task01_BrainTumour/labelsTr"

data_tr = TorchDataset(tr_files,path_img,labels_img,transform=flip_scale_crop_train)
train_loader = torch.utils.data.DataLoader(data_tr,
                                           batch_size=batch_size, shuffle=True,
                                           drop_last=True)

data_val = TorchDataset(val_files,path_img,labels_img,transform=flip_scale_crop_test)
val_loader = torch.utils.data.DataLoader(data_val,
                                           batch_size=batch_size, shuffle=True,
                                           drop_last=True)

# test data no shuffle
data_ts = TorchDataset(ts_files,path_img,labels_img,transform=flip_scale_crop_test)
test_loader = torch.utils.data.DataLoader(data_ts,
                                           batch_size=batch_size,
                                           drop_last=True)

train = False
start_from_trained = False
start_file = 'save/fusion_models/3D_unet_vote_fusion_mod_2_epoch_25/model.16.pth.tar'
start_epoch = 17
if train:
    # train model
    model = UNet(n_channels=n_modality, n_classes=n_labels)
    model.to(device)
    if start_from_trained:
        state_dict = torch.load(start_file)['state_dict']
        model.load_state_dict(state_dict)

    criterion = mean_dice_loss
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_losses = []
    val_losses = []
    #early_stopping = EarlyStopping(patience=patience, verbose=True)

    for epoch in range(1,epochs+1):
        print('\nEpoch [{}/{}] '.format(epoch, epochs))
        print(session_name)

        # train for one epoch
        model.train()
        print('[Training]')
        tr_loss = train_model(model, train_loader, criterion, optimizer, epoch, device, fre=100, sample_size = 400,rand = False,batch_size=1)
        train_losses.append(tr_loss)

        # validation for one epoch
        print('[Validation]')
        with torch.no_grad():
            model.eval()
            val_loss = validate_model(model, val_loader, criterion,epoch, device, fre=100, sample_size = 100,rand = False, batch_size=1)
            val_losses.append(val_loss)

        # save weights
        if save_model and epoch % save_frequency == 0:
            save_checkpoint({'epoch': epoch,
                             'state_dict': model.state_dict(),
                             'optimizer': optimizer.state_dict(),
                             'train_loss': tr_loss,
                             'valid_loss': val_loss}, model_path)

    losses = {'train_loss':train_losses,'val_loss':val_losses}
    with open(model_path+'losses.pickle', 'wb') as handle:
        pickle.dump(losses, handle, protocol=pickle.HIGHEST_PROTOCOL)

else:
    losses = {}
    losses['train_loss'] = []
    losses['valid_loss'] = []
    for i in range(1, 11):
        check = torch.load(model_path + 'model.' + str(i).zfill(2) + '.pth.tar')
        losses['train_loss'].append(check['train_loss'])
        losses['valid_loss'].append(check['valid_loss'])

    plt.figure()
    plt.plot(np.arange(1, 11), losses['train_loss'], label='Train loss')
    plt.plot(np.arange(1, 11), losses['valid_loss'], label='Valid loss')
    plt.legend()
    plt.savefig(model_path + 'loss_plot.png')

    model_name = 'model.10.pth.tar'
    model = UNet(n_channels=n_modality, n_classes=n_labels)
    model.to(device)
    state_dict = torch.load(model_path + model_name)['state_dict']
    model.load_state_dict(state_dict)

    torch.cuda.empty_cache()
    with torch.no_grad():
        model.eval()
        preds, metrics = predict(test_loader, model, device,batch_size=1)
    metrics_ar = np.stack(metrics, axis=0)
    metrics_m = np.mean(metrics_ar, axis=0)
    # metrics_std = np.std(metrics_ar, axis=0)

    idxs = ['wt', 'tc', 'et']
    cols = ['Dice', 'Sensitivity', 'Specificity']
    mm = pd.DataFrame(metrics_m.reshape(3, -1), index=idxs)
    mm.columns = cols
    # save mean test scores
    mm.to_csv(model_path + 'test_evaluate_mean.csv')
    # save all the test scores

    df_ar = pd.DataFrame(metrics_ar)
    df_ar.columns = [a+'_'+b for a in idxs for b in cols]
    df_ar.to_csv(model_path+'test_evaluate_all.csv')



