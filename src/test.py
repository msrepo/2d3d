from glob import glob
from os.path import join

import matplotlib.pyplot as plt
import torch
from ignite.engine import (create_supervised_evaluator,
                           create_supervised_trainer)
from ignite.engine.engine import Engine
from ignite.engine.events import Events
from ignite.handlers import ModelCheckpoint
from ignite.metrics import Loss
from ignite.utils import setup_logger
from monai.data import DataLoader, Dataset, NiftiSaver
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.networks.layers.factories import Norm
from monai.networks.nets import UNet
from monai.transforms import (Compose, EnsureChannelFirstD, EnsureTypeD,
                              LoadImageD, OrientationD)
from monai.transforms.utility.dictionary import AddChannelD, LambdaD
from monai.utils.enums import CommonKeys
from monai.utils.misc import first, set_determinism

set_determinism(8127301)
BATCH_SZ = 4
device = torch.device('cuda')
channels = (8, 16, 32, 64)
strides = (1, 2, 2)
num_res_units = 0
step = 1

model_path = glob(join('.', 'TrainedModels', 'Unet*.pt'))[0]
base_dataset_dir = join('.', '2D3DReconstruction')
label_dir = join(base_dataset_dir, '3D')
lat_dir = join(base_dataset_dir, 'lat')
pa_dir = join(base_dataset_dir, 'pa')

label_filepaths = sorted(glob(join(label_dir, '*.nii.gz')))
lat_filepaths = sorted(glob(join(lat_dir, '*_lat.jpg')))  # had to convert  tiff to jpg
pa_filepaths = sorted(glob(join(pa_dir, '*_pa.jpg')))


split_idx = 100


val_pa_filepaths = pa_filepaths[100:]
val_lat_filepaths = lat_filepaths[100:]
val_label_filepaths = label_filepaths[100:]

keys = ['pa', 'lat', '3d']


def obtain_dict(pa_filepaths, lat_filepaths, label_filepaths):

    img_label_dict = []
    for pa, lat, label in zip(pa_filepaths, lat_filepaths, label_filepaths):
        img_label_dict.append({keys[0]: pa, keys[1]: lat, keys[2]: label})

    return img_label_dict


val_img_label_dict = obtain_dict(val_pa_filepaths, val_lat_filepaths,
                                 val_label_filepaths)

transforms_train = Compose([
    LoadImageD(keys),
    EnsureChannelFirstD(keys),
    OrientationD(keys=keys[2], axcodes='RAS'),
    EnsureTypeD(keys),
    LambdaD(keys=[keys[0], keys[1]], func=lambda x: x.expand(1, 128, 128, 128))  # we are going to try to learn from single image first
])


valds = Dataset(val_img_label_dict, transforms_train)
valloader = DataLoader(valds, batch_size=BATCH_SZ, shuffle=True)


model = UNet(3, 1, 1, channels, strides, num_res_units=num_res_units, norm=Norm.BATCH).to(device)
print(f'loading model from path {model_path}')
checkpoint = torch.load(model_path)
model.load_state_dict(checkpoint)


model.eval()

result_path = join('.', 'results')
saver_2d3d = segsaver = NiftiSaver(result_path)


def run():
    for i, data in enumerate(valloader):
        print(data.keys())
        with torch.no_grad():

            image = data[keys[0]].to(device)
            label = data[keys[2]].to(device)
            print(f'batch image shape {image.shape} label shape {label.shape}')
            predicted_label = model(image)
            saver_2d3d.save_batch(predicted_label.detach(), data['3d_meta_dict'])


run()
