import torch.utils.data
import time
import numpy as np
import skimage.transform
import skimage

from unet import UNet,TorchDataset,flip_scale_crop_test,flip_scale_crop_train
from utils import mean_dice_loss,to_var,to_numpy,save_checkpoint,evalAllmetric
from utils import train_model,validate_model,predict

# split training data into train:test = 3:1
from os import listdir
from os.path import isfile, join
import pickle
import matplotlib.pyplot as plt
import pandas as pd
path = '../Task01_BrainTumour/imagesTr/'
files = [f for f in listdir(path) if isfile(join(path, f)) and not f.startswith('.')]
files = sorted(files)
seed = 10
np.random.seed(seed)
np.random.shuffle(files)
test_files = files[:121]
tr_files = files[121:]
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

session_name = '3D_unet_concat_new_' +str(epochs)
model_path = save_path + 'baseline_models/' + session_name + '/'
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

train_fg = False
if train_fg:
    # train model
    model = UNet(n_channels=n_modality, n_classes=n_labels)
    model.to(device)
    criterion = mean_dice_loss
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_losses = []
    val_losses = []
    # early_stopping = EarlyStopping(patience=patience, verbose=True)

    for epoch in range(1,epochs+1):
        print('\nEpoch [{}/{}] '.format(epoch, epochs))
        # print(session_name)

        # train for one epoch
        model.train()
        print('[Training]')
        tr_loss = train_model(model, train_loader, criterion, optimizer, epoch, device, fre=100, sample_size = 100,rand = False,batch_size=1)
        train_losses.append(tr_loss)

        # validation for one epoch
        print('[Validation]')
        with torch.no_grad():
            model.eval()
            val_loss = validate_model(model, val_loader, criterion,epoch, device, fre=100, sample_size = 100,rand = False,batch_size=1)
            val_losses.append(val_loss)

        if save_model and epoch % save_frequency == 0:
            save_checkpoint({'epoch': epoch,
                             'state_dict': model.state_dict(),
                             'optimizer': optimizer.state_dict(),
                             'train_loss': tr_loss,
                             'valid_loss': val_loss}, model_path)
    import pickle
    losses = {'train_loss': train_losses, 'val_loss': val_losses}
    with open(model_path + 'losses.pickle', 'wb') as handle:
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

    # mstd = np.concatenate([metrics_m,metrics_std])
    mm = pd.DataFrame(metrics_m.reshape(3, -1), index=['wt', 'tc', 'et'])
    mm.columns = ['Dice', 'Sensitivity', 'Specificity']
    mm.to_csv(model_path + 'test_evaluate.csv')