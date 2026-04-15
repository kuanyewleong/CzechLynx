import timm
import os
import torch
from itertools import chain
from datetime import datetime
import pandas as pd

from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim import AdamW
from timm.data.transforms_factory import create_transform
# from timm.data import create_transform

from wildlife_tools.train.trainer import set_seed, BasicTrainer
from wildlife_tools.train import ArcFaceLoss
from wildlife_tools.train.callbacks import EpochCheckpoint, EpochLog, EpochCallbacks
from wildlife_tools.data import WildlifeDataset


lr = 0.0001
epochs = 10
device = 'cuda'
num_workers = 4
batch_size = 128
accumulation_steps = 4
output_folder = f'output/{datetime.now().strftime("run_%Y%m%d_%H%M%S")}'
root = 'data'

os.makedirs(output_folder, exist_ok=True)

# load model
name = 'hf-hub:BVRA/MegaDescriptor-T-224'
model = timm.create_model(name, num_classes=0, pretrained=True)
embedding_size = model.feature_info[-1]["num_chs"]

# load data
metadata = pd.read_csv(os.path.join(root, "CzechLynxDataset-Metadata-Real.csv"))
database_metadata = metadata[metadata['split-geo_aware'] == 'train']

transform_database = create_transform(input_size = 224, is_training = True, auto_augment = "rand-m10-n2-mstd1")
dataset_database = WildlifeDataset(root=root, metadata=database_metadata, transform=transform_database, col_label="unique_name")

# loss function
objective = ArcFaceLoss(
    num_classes=dataset_database.num_classes,
    embedding_size=embedding_size,
    margin=0.25,
    scale=16,
)

# define optimizer and scheduler
optimizer = AdamW(params=chain(model.parameters(), objective.parameters()), lr=lr)
scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 1e-3)
callbacks = EpochCallbacks([EpochCheckpoint(output_folder, save_step=1), EpochLog(output_folder)])

# start training
set_seed(0)
trainer =  BasicTrainer(
    dataset=dataset_database,
    model=model,
    objective=objective,
    optimizer=optimizer,
    scheduler=scheduler,
    device=device,
    epochs=epochs,
    num_workers=num_workers,
    batch_size=batch_size,
    accumulation_steps=accumulation_steps,
    epoch_callback=callbacks,
)
trainer.train()
torch.save(model.state_dict(), os.path.join(output_folder, "checkpoint-final.pth"))