import os
from itertools import chain
from datetime import datetime

import pandas as pd
import timm
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from timm.data.transforms_factory import create_transform

from wildlife_tools.data import WildlifeDataset
from wildlife_tools.train import ArcFaceLoss
from wildlife_tools.train.callbacks import EpochCheckpoint, EpochLog, EpochCallbacks
from wildlife_tools.train.trainer import BasicTrainer, set_seed


# -------------------------------
# Config
# -------------------------------
lr = 1e-4
epochs = 50
device = 'cuda' if torch.cuda.is_available() else 'cpu'
num_workers = 4
batch_size = 16
accumulation_steps = 2
val_fraction = 0.15
random_seed = 42
split_column = 'split-time_closed'
label_column = 'unique_name'
input_size = 384
root = 'data'
output_folder = f'output/{datetime.now().strftime("run_%Y%m%d_%H%M%S")}'
os.makedirs(output_folder, exist_ok=True)


# -------------------------------
# Helpers
# -------------------------------
def stratified_identity_holdout(df: pd.DataFrame, label_col: str, frac: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Hold out validation images per identity while keeping every identity in train.
    For identities with >= 3 images, keep at least 1 image in val and >= 2 in train.
    For identities with 2 images, keep both in train.
    For singleton identities, keep in train only.
    """
    train_parts = []
    val_parts = []

    for _, group in df.groupby(label_col, sort=False):
        group = group.sample(frac=1.0, random_state=seed)
        n = len(group)

        if n >= 3:
            n_val = max(1, int(round(n * frac)))
            n_val = min(n_val, n - 2)
            val_parts.append(group.iloc[:n_val])
            train_parts.append(group.iloc[n_val:])
        else:
            train_parts.append(group)

    train_df = pd.concat(train_parts, axis=0).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    val_df = pd.concat(val_parts, axis=0).sample(frac=1.0, random_state=seed).reset_index(drop=True) if val_parts else pd.DataFrame(columns=df.columns)
    return train_df, val_df


def summarize_split(name: str, df: pd.DataFrame, label_col: str) -> None:
    print(f'{name}: {len(df)} images | {df[label_col].nunique()} identities')


# -------------------------------
# Load model
# -------------------------------
name = 'hf-hub:BVRA/MegaDescriptor-L-384'
model = timm.create_model(name, num_classes=0, pretrained=True)
embedding_size = model.feature_info[-1]['num_chs']


# -------------------------------
# Load metadata and define splits
# -------------------------------
metadata = pd.read_csv(os.path.join(root, 'CzechLynxDataset-Metadata-Real.csv'))

if split_column not in metadata.columns:
    raise ValueError(f'Missing split column: {split_column}')
if label_column not in metadata.columns:
    raise ValueError(f'Missing label column: {label_column}')

train_pool = metadata[metadata[split_column] == 'train'].copy().reset_index(drop=True)
test_metadata = metadata[metadata[split_column] == 'test'].copy().reset_index(drop=True)

train_metadata, val_metadata = stratified_identity_holdout(
    train_pool,
    label_col=label_column,
    frac=val_fraction,
    seed=random_seed,
)

if len(val_metadata) == 0:
    raise RuntimeError('Validation split is empty. Increase training-set size or adjust holdout logic.')

summarize_split('Train', train_metadata, label_column)
summarize_split('Validation', val_metadata, label_column)
summarize_split('Test', test_metadata, label_column)

train_metadata.to_csv(os.path.join(output_folder, 'train_split.csv'), index=False)
val_metadata.to_csv(os.path.join(output_folder, 'val_split.csv'), index=False)
test_metadata.to_csv(os.path.join(output_folder, 'test_split.csv'), index=False)


# -------------------------------
# Transforms / datasets
# -------------------------------
transform_train = create_transform(
    input_size=input_size,
    is_training=True,
    auto_augment='rand-m10-n2-mstd1',
)

dataset_train = WildlifeDataset(
    root=root,
    metadata=train_metadata,
    transform=transform_train,
    col_label=label_column,
)


# -------------------------------
# Loss / optimizer / scheduler
# -------------------------------
objective = ArcFaceLoss(
    num_classes=dataset_train.num_classes,
    embedding_size=embedding_size,
    margin=0.25,
    scale=16,
)

optimizer = AdamW(
    params=chain(model.parameters(), objective.parameters()),
    lr=lr,
    weight_decay=1e-4,
)

scheduler = CosineAnnealingLR(
    optimizer,
    T_max=epochs,
    eta_min=lr * 1e-3,
)

callbacks = EpochCallbacks([
    EpochCheckpoint(output_folder, save_step=1),
    EpochLog(output_folder),
])


# -------------------------------
# Train
# -------------------------------
set_seed(random_seed)
trainer = BasicTrainer(
    dataset=dataset_train,
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

final_ckpt = os.path.join(output_folder, 'checkpoint-final.pth')
torch.save(model.state_dict(), final_ckpt)
print(f'Saved final checkpoint to: {final_ckpt}')
print(f'Saved train/val/test CSV splits to: {output_folder}')
