import os
import json
import pickle
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
import torchvision.transforms as T
from wildlife_datasets.datasets import WildlifeDataset
from wildlife_tools.features import DeepFeatures
from wildlife_tools.features.local import AlikedExtractor
from wildlife_tools.similarity import CosineSimilarity
from wildlife_tools.similarity.calibration import IsotonicCalibration
from wildlife_tools.similarity.pairwise.lightglue import MatchLightGlue
from wildlife_tools.similarity.wildfusion import SimilarityPipeline, WildFusion

import matplotlib.pyplot as plt


# -------------------------------
# Config
# -------------------------------
root = 'data'
output_folder = 'output/run_20260416_140125'  # change to your latest training run
split_column = 'split-time_closed'
label_column = 'unique_name'
model_name = 'hf-hub:BVRA/MegaDescriptor-L-384'
checkpoint_path = os.path.join(output_folder, 'checkpoint-final.pth')
train_split_csv = os.path.join(output_folder, 'train_split.csv')
val_split_csv = os.path.join(output_folder, 'val_split.csv')
test_split_csv = os.path.join(output_folder, 'test_split.csv')
cache_dir = Path(output_folder) / 'cache'
cache_dir.mkdir(parents=True, exist_ok=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
num_examples = 8
top_k = 3
wildfusion_B = 10
aliked_batch_size = 16
mega_batch_size = 16


# -------------------------------
# Helpers
# -------------------------------
def get_topk_acc(dataset_query, dataset_database, similarity, top_k=3):
    acc = []
    pred_idx = similarity.argsort(axis=1)
    for query_idx in range(len(dataset_query)):
        database_idx = pred_idx[query_idx, -top_k:][::-1]
        query_label = dataset_query.df[dataset_query.col_label].iloc[query_idx]
        database_labels = [dataset_database.df[dataset_database.col_label].iloc[di] for di in database_idx]
        acc.append(query_label in database_labels)
    return float(np.mean(acc))


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def stable_df_hash(df: pd.DataFrame, cols=None) -> str:
    if cols is None:
        cols = list(df.columns)
    cols = [c for c in cols if c in df.columns]
    if len(cols) == 0:
        return sha256_text('empty-df')

    df_small = df[cols].copy()
    for c in df_small.columns:
        df_small[c] = df_small[c].astype(str)

    row_hash = pd.util.hash_pandas_object(df_small, index=True).values
    return hashlib.sha256(row_hash.tobytes()).hexdigest()


def file_signature(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        return {'exists': False, 'path': str(p)}
    stat = p.stat()
    return {
        'exists': True,
        'path': str(p.resolve()),
        'size': stat.st_size,
        'mtime_ns': stat.st_mtime_ns,
    }


def make_cache_key(payload: dict) -> str:
    payload_str = json.dumps(payload, sort_keys=True, default=str)
    return sha256_text(payload_str)[:24]


def save_pickle(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(path: Path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def build_protocol_splits(metadata_path: str, split_col: str):
    if os.path.isfile(train_split_csv) and os.path.isfile(val_split_csv) and os.path.isfile(test_split_csv):
        print('[INFO] Loading explicit train/val/test splits from output folder.')
        train_df = pd.read_csv(train_split_csv)
        val_df = pd.read_csv(val_split_csv)
        test_df = pd.read_csv(test_split_csv)
        return train_df, val_df, test_df

    print('[WARN] Split CSV files not found. Falling back to metadata split only.')
    metadata = pd.read_csv(metadata_path)
    train_df = metadata[metadata[split_col] == 'train'].copy().reset_index(drop=True)
    test_df = metadata[metadata[split_col] == 'test'].copy().reset_index(drop=True)

    val_parts = []
    train_parts = []
    for _, group in train_df.groupby(label_column, sort=False):
        group = group.sample(frac=1.0, random_state=42)
        if len(group) >= 3:
            val_parts.append(group.iloc[:1])
            train_parts.append(group.iloc[1:])
        else:
            train_parts.append(group)

    train_df = pd.concat(train_parts, axis=0).reset_index(drop=True)
    if val_parts:
        val_df = pd.concat(val_parts, axis=0).reset_index(drop=True)
    else:
        val_df = train_df.iloc[: min(100, len(train_df))].copy().reset_index(drop=True)
    return train_df, val_df, test_df


def evaluate_and_print(name: str, sim: np.ndarray, dataset_query, dataset_database, max_k: int = 5):
    print(f'\n{name}')
    for k in range(1, max_k + 1):
        acc = get_topk_acc(dataset_query, dataset_database, sim, top_k=k)
        print(f'  Top-{k}: {acc:.4f}')


def save_retrieval_examples(similarity: np.ndarray, dataset_query, dataset_database, output_dir: str, prefix: str, k: int = 3):
    os.makedirs(output_dir, exist_ok=True)
    pred_idx = similarity.argsort(axis=1)
    n = min(num_examples, len(dataset_query))
    sample_indices = np.random.RandomState(42).choice(np.arange(len(dataset_query)), n, replace=False)

    for query_idx in sample_indices:
        database_idx = pred_idx[query_idx, -k:][::-1]
        score = similarity[query_idx, database_idx]

        fig, ax = plt.subplots(1, 1 + k, figsize=((1 + k) * 3, 3))
        query_data = dataset_query[query_idx]
        ax[0].imshow(query_data[0])
        ax[0].set_title(f'Q: {query_data[1]}')
        ax[0].axis('off')

        for i, idx in enumerate(database_idx):
            database_data = dataset_database[idx]
            ax[i + 1].imshow(database_data[0])
            ax[i + 1].set_title(f'{database_data[1]}\n{score[i]:.3f}')
            ax[i + 1].axis('off')

        plt.tight_layout()
        save_path = os.path.join(output_dir, f'{prefix}_query_{query_idx:05d}.png')
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)


def compute_or_load_similarity(cache_name: str, compute_fn):
    path = cache_dir / f'{cache_name}.npz'
    if path.exists():
        print(f'[CACHE HIT] Loading similarity matrix from: {path}')
        return np.load(path)['similarity']
    print(f'[CACHE MISS] Computing similarity: {cache_name}')
    sim = np.asarray(compute_fn(), dtype=np.float32)
    np.savez_compressed(path, similarity=sim)
    print(f'[CACHE SAVE] Similarity matrix saved to: {path}')
    return sim


# -------------------------------
# Transforms
# -------------------------------
transform_display = T.Compose([
    T.Resize([384, 384]),
])

transform_mega = T.Compose([
    T.Resize([384, 384]),
    T.ToTensor(),
    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])

transform_aliked = T.Compose([
    T.Resize([512, 512]),
    T.ToTensor(),
])


# -------------------------------
# Dataset loading
# -------------------------------
metadata_path = os.path.join(root, 'CzechLynxDataset-Metadata-Real.csv')
train_metadata, val_metadata, test_metadata = build_protocol_splits(metadata_path, split_column)

for split_name, df in [('Train gallery', train_metadata), ('Calibration', val_metadata), ('Test query', test_metadata)]:
    print(f'{split_name}: {len(df)} images | {df[label_column].nunique()} identities')

dataset_train_gallery = WildlifeDataset(root, train_metadata, load_label=True, col_label=label_column)
dataset_calibration = WildlifeDataset(root, val_metadata, load_label=True, col_label=label_column)
dataset_query = WildlifeDataset(root, test_metadata, load_label=True, col_label=label_column)

n_query = len(dataset_query)


# -------------------------------
# Model loading
# -------------------------------
model = timm.create_model(model_name, num_classes=0, pretrained=True)
if os.path.isfile(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    model.load_state_dict(checkpoint, strict=True)
    print(f'[INFO] Loaded checkpoint: {checkpoint_path}')
else:
    raise FileNotFoundError(f'Checkpoint not found: {checkpoint_path}')

model = model.to(device)
model.eval()


# -------------------------------
# Similarity pipelines
# Standalone pipelines are uncalibrated.
# Fusion pipelines keep calibration and can reuse cached calibrators.
# -------------------------------
matcher_aliked = SimilarityPipeline(
    matcher=MatchLightGlue(features='aliked', device=device, batch_size=aliked_batch_size),
    extractor=AlikedExtractor(),
    transform=transform_aliked,
    calibration=None,
)

matcher_mega = SimilarityPipeline(
    matcher=CosineSimilarity(),
    extractor=DeepFeatures(model=model, device=device, batch_size=mega_batch_size),
    transform=transform_mega,
    calibration=None,
)

matcher_aliked_fusion = SimilarityPipeline(
    matcher=MatchLightGlue(features='aliked', device=device, batch_size=aliked_batch_size),
    extractor=AlikedExtractor(),
    transform=transform_aliked,
    calibration=IsotonicCalibration(),
)

matcher_mega_fusion = SimilarityPipeline(
    matcher=CosineSimilarity(),
    extractor=DeepFeatures(model=model, device=device, batch_size=mega_batch_size),
    transform=transform_mega,
    calibration=IsotonicCalibration(),
)

wildfusion = WildFusion(
    calibrated_pipelines=[matcher_aliked_fusion, matcher_mega_fusion],
    priority_pipeline=matcher_mega_fusion,
)


# -------------------------------
# Cache keys
# -------------------------------
common_cache_info = {
    'metadata_csv': file_signature(metadata_path),
    'checkpoint': file_signature(checkpoint_path),
    'train_csv': file_signature(train_split_csv),
    'val_csv': file_signature(val_split_csv),
    'test_csv': file_signature(test_split_csv),
    'model_name': model_name,
    'device': device,
    'gallery_hash': stable_df_hash(train_metadata, cols=['path', label_column, split_column]),
    'query_hash': stable_df_hash(test_metadata, cols=['path', label_column, split_column]),
    'calibration_hash': stable_df_hash(val_metadata, cols=['path', label_column, split_column]),
}

calib_cache_key = make_cache_key({
    **common_cache_info,
    'stage': 'calibration',
    'calibration_type': 'IsotonicCalibration',
    'aliked_transform': 'Resize512_ToTensor',
    'mega_transform': 'Resize384_ToTensor_NormalizeImageNet',
    'wildfusion_priority': 'matcher_mega_fusion',
})

aliked_calib_path = cache_dir / f'aliked_isotonic_{calib_cache_key}.pkl'
mega_calib_path = cache_dir / f'mega_isotonic_{calib_cache_key}.pkl'


# -------------------------------
# Calibration: val vs gallery, not self-vs-self
# Only affects fusion pipelines.
# -------------------------------
if aliked_calib_path.exists() and mega_calib_path.exists():
    print(f'[CACHE HIT] Loading calibration from:\n  {aliked_calib_path}\n  {mega_calib_path}')
    matcher_aliked_fusion.calibration = load_pickle(aliked_calib_path)
    matcher_mega_fusion.calibration = load_pickle(mega_calib_path)
else:
    print('[CACHE MISS] Fitting WildFusion calibration on val vs train-gallery ...')
    wildfusion.fit_calibration(dataset_calibration, dataset_train_gallery)
    save_pickle(matcher_aliked_fusion.calibration, aliked_calib_path)
    save_pickle(matcher_mega_fusion.calibration, mega_calib_path)
    print(f'[CACHE SAVE] Calibration saved to:\n  {aliked_calib_path}\n  {mega_calib_path}')


# -------------------------------
# Similarity computation with ablations
# -------------------------------
mega_cache_key = make_cache_key({
    **common_cache_info,
    'stage': 'similarity_mega',
    'calibrated': False,
})

aliked_cache_key = make_cache_key({
    **common_cache_info,
    'stage': 'similarity_aliked',
    'calibrated': False,
})

wildfusion_cache_key = make_cache_key({
    **common_cache_info,
    'stage': 'similarity_wildfusion',
    'calibration_cache_key': calib_cache_key,
    'B': wildfusion_B,
    'wildfusion_priority': 'matcher_mega_fusion',
})

similarity_mega = compute_or_load_similarity(
    f'similarity_mega_{mega_cache_key}',
    lambda: matcher_mega(dataset_query, dataset_train_gallery),
)

similarity_aliked = compute_or_load_similarity(
    f'similarity_aliked_{aliked_cache_key}',
    lambda: matcher_aliked(dataset_query, dataset_train_gallery),
)

similarity_wildfusion = compute_or_load_similarity(
    f'similarity_wildfusion_{wildfusion_cache_key}',
    lambda: wildfusion(dataset_query, dataset_train_gallery, B=wildfusion_B),
)


# -------------------------------
# Evaluation
# -------------------------------
dataset_train_gallery_display = WildlifeDataset(
    root,
    train_metadata,
    transform=transform_display,
    load_label=True,
    col_label=label_column,
)

dataset_query_display = WildlifeDataset(
    root,
    test_metadata,
    transform=transform_display,
    load_label=True,
    col_label=label_column,
)

evaluate_and_print('MegaDescriptor only', similarity_mega, dataset_query, dataset_train_gallery)
evaluate_and_print('ALIKED + LightGlue only', similarity_aliked, dataset_query, dataset_train_gallery)
evaluate_and_print('WildFusion', similarity_wildfusion, dataset_query, dataset_train_gallery)

pred_idx = similarity_wildfusion.argsort(axis=1)[:, -1]
pred_scores = similarity_wildfusion[np.arange(n_query), pred_idx]
print(f'\nWildFusion similarity matrix: {similarity_wildfusion.shape}')
print(f'Top predicted (index, score): {list(zip(pred_idx[:10], np.round(pred_scores[:10], 3)))}')

retrieval_dir = os.path.join(output_folder, 'retrieval_figures')
save_retrieval_examples(similarity_mega, dataset_query_display, dataset_train_gallery_display, retrieval_dir, prefix='mega', k=top_k)
save_retrieval_examples(similarity_aliked, dataset_query_display, dataset_train_gallery_display, retrieval_dir, prefix='aliked', k=top_k)
save_retrieval_examples(similarity_wildfusion, dataset_query_display, dataset_train_gallery_display, retrieval_dir, prefix='wildfusion', k=top_k)

print(f'\nSaved retrieval examples to: {retrieval_dir}')
