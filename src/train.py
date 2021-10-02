from glob import glob
from os.path import join

import matplotlib.pyplot as plt
import torch
from monai.data import DataLoader, Dataset
from monai.losses import DiceLoss
from monai.networks.layers.factories import Norm
from monai.networks.nets import UNet
from monai.transforms import (Compose, EnsureChannelFirstD, EnsureTypeD,
                              LoadImageD, OrientationD)
from monai.transforms.utility.dictionary import AddChannelD, LambdaD
from monai.utils.misc import first, set_determinism
from torch.optim import Adam

set_determinism(8127301)
BATCH_SZ = 4
lr = 1e-3
device = torch.device('cuda')
channels = (8, 16, 32, 64)
strides = (1, 2, 2)
num_res_units = 0
num_epochs = 200
step = 1


base_dataset_dir = join('.', '2D3DReconstruction')
label_dir = join(base_dataset_dir, '3D')
lat_dir = join(base_dataset_dir, 'lat')
pa_dir = join(base_dataset_dir, 'pa')

label_filepaths = sorted(glob(join(label_dir, '*.nii.gz')))
lat_filepaths = sorted(glob(join(lat_dir, '*_lat.jpg')))  # had to convert  tiff to jpg
pa_filepaths = sorted(glob(join(pa_dir, '*_pa.jpg')))

print(len(label_filepaths), len(lat_filepaths), len(pa_filepaths))
print(lat_filepaths[0], pa_filepaths[0], label_filepaths[0])

split_idx = 100
train_pa_filepaths = pa_filepaths[:100]
train_lat_filepaths = lat_filepaths[:100]
train_label_filepaths = label_filepaths[:100]

val_pa_filepaths = pa_filepaths[100:]
val_lat_filepaths = lat_filepaths[100:]
val_label_filepaths = label_filepaths[100:]

keys = ['pa', 'lat', '3d']


def obtain_dict(pa_filepaths, lat_filepaths, label_filepaths):

    img_label_dict = []
    for pa, lat, label in zip(pa_filepaths, lat_filepaths, label_filepaths):
        img_label_dict.append({keys[0]: pa, keys[1]: lat, keys[2]: label})

    return img_label_dict


train_img_label_dict = obtain_dict(train_pa_filepaths, train_lat_filepaths,
                                   train_label_filepaths)
val_img_label_dict = obtain_dict(val_pa_filepaths, val_lat_filepaths,
                                 val_label_filepaths)

transforms_train = Compose([
    LoadImageD(keys),
    EnsureChannelFirstD(keys),
    OrientationD(keys=keys[2], axcodes='RAS'),
    EnsureTypeD(keys),
    LambdaD(keys=[keys[0], keys[1]], func=lambda x: x.expand(1, 128, 128, 128))  # we are going to try to learn from one of the images first
])

trainds = Dataset(train_img_label_dict, transforms_train)
trainloader = DataLoader(trainds, batch_size=BATCH_SZ, shuffle=True)

valds = Dataset(val_img_label_dict, transforms_train)
valloader = DataLoader(trainds, batch_size=BATCH_SZ, shuffle=True)
for data in trainds:
    print(data[keys[0]].shape, data[keys[1]].shape, data[keys[2]].shape)
    plt.subplot(1, 3, 1)
    plt.imshow(data[keys[0]][0, 64], cmap='gray')
    plt.subplot(1, 3, 2)
    plt.imshow(data[keys[1]][0, 64], cmap='gray')
    plt.subplot(1, 3, 3)
    plt.imshow(data[keys[2]][0, 64], cmap='gray')
    plt.show()
    plt.tight_layout()
    plt.savefig('sample-data.jpg')

    break


model = UNet(3, 1, 1, channels, strides, num_res_units=num_res_units, norm=Norm.BATCH).to(device)
loss_function = DiceLoss(include_background=False, sigmoid=True)
optimizer = Adam(model.parameters(), lr)
# test the workflow

batch = first(trainloader)
image = batch[keys[0]].to(device)
# image.expand(BATCH_SZ, 1, 128, 128, 128)
label = batch[keys[1]].to(device)
print(f'batch image shape {image.shape} label shape {label.shape}')
predicted_label = model(image)
dice_loss = loss_function(predicted_label, label)
print(f'Dice Loss {dice_loss.item():.3f}')
plt.imshow(predicted_label[0, 0, 64].detach().cpu(), cmap='gray')
plt.savefig('sample-prediction.jpg')
