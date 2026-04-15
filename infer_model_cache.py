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
from wildlife_tools.similarity import CosineSimilarity
from wildlife_tools.similarity.wildfusion import SimilarityPipeline, WildFusion
from wildlife_tools.similarity.pairwise.lightglue import MatchLightGlue
from wildlife_tools.features.local import AlikedExtractor
from wildlife_tools.similarity.calibration import IsotonicCalibration

import matplotlib.pyplot as plt
from shapely.geometry import Point
from matplotlib.lines import Line2D


def get_acc(dataset_query, dataset_database, similarity, top_k=3):
    acc = []
    pred_idx = similarity.argsort(axis=1)
    for query_idx in range(len(dataset_query)):
        database_idx = pred_idx[query_idx, -top_k:][::-1]

        query_label = dataset_query.df[dataset_query.col_label].iloc[query_idx]
        database_labels = [dataset_database.df[dataset_database.col_label].iloc[di] for di in database_idx]

        acc.append(query_label in database_labels)
    return np.mean(acc)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def stable_df_hash(df: pd.DataFrame, cols=None) -> str:
    """
    Build a stable hash of selected dataframe columns.
    This is used to invalidate caches when dataset content changes.
    """
    if cols is None:
        cols = list(df.columns)

    cols = [c for c in cols if c in df.columns]
    if len(cols) == 0:
        return sha256_text("empty-df")

    df_small = df[cols].copy()

    # Convert everything to string to keep hashing stable across dtypes
    for c in df_small.columns:
        df_small[c] = df_small[c].astype(str)

    row_hash = pd.util.hash_pandas_object(df_small, index=True).values
    return hashlib.sha256(row_hash.tobytes()).hexdigest()


def file_signature(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        return {"exists": False}
    stat = p.stat()
    return {
        "exists": True,
        "path": str(p.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def make_cache_key(payload: dict) -> str:
    payload_str = json.dumps(payload, sort_keys=True, default=str)
    return sha256_text(payload_str)[:24]


def save_pickle(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


root = "data"
output_folder = "output/run_20260415_150322"
cache_dir = Path(output_folder) / "cache"
cache_dir.mkdir(parents=True, exist_ok=True)

transform_display = T.Compose([
    T.Resize([224, 224]),
])

transform = T.Compose([
    *transform_display.transforms,
    T.ToTensor(),
    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
])

transforms_aliked = T.Compose([
    T.Resize([512, 512]),
    T.ToTensor()
])

# ------------------------------------------------------------------
# Dataset loading
# ------------------------------------------------------------------
metadata_path = os.path.join(root, "CzechLynxDataset-Metadata-Real.csv")
metadata = pd.read_csv(metadata_path)

# IMPORTANT: fix the random subset so cache can be reused across runs
metadata = metadata.sample(int(len(metadata) * 0.5), random_state=42).reset_index(drop=True)

dataset = WildlifeDataset(root, metadata, load_label=True, col_label="unique_name")
dataset_database = dataset.get_subset(dataset.metadata["split-time_closed"] == "train")
dataset_query = dataset.get_subset(dataset.metadata["split-time_closed"] == "test")
dataset_calibration = WildlifeDataset(
    root,
    df=dataset_database.metadata[:100].copy(),
    load_label=True,
    col_label="unique_name"
)

n_query = len(dataset_query)

# ------------------------------------------------------------------
# Model loading
# ------------------------------------------------------------------
name = "hf-hub:BVRA/MegaDescriptor-T-224"
checkpoint_path = os.path.join(output_folder, "checkpoint-final.pth")
model = timm.create_model(name, num_classes=0, pretrained=True)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Loading fine-tuned weights
if os.path.isfile(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint)

model = model.to(device)
model.eval()

# ------------------------------------------------------------------
# Similarity pipelines
# ------------------------------------------------------------------
matcher_aliked = SimilarityPipeline(
    matcher=MatchLightGlue(features="aliked", device=device, batch_size=16),
    extractor=AlikedExtractor(),
    transform=transforms_aliked,
    calibration=IsotonicCalibration()
)

matcher_mega = SimilarityPipeline(
    matcher=CosineSimilarity(),
    extractor=DeepFeatures(model=model, device=device, batch_size=16),
    transform=transform,
    calibration=IsotonicCalibration()
)

wildfusion = WildFusion(
    calibrated_pipelines=[matcher_aliked, matcher_mega],
    priority_pipeline=matcher_mega
)

# ------------------------------------------------------------------
# Cache keys
# ------------------------------------------------------------------
common_cache_info = {
    "metadata_csv": file_signature(metadata_path),
    "checkpoint": file_signature(checkpoint_path),
    "model_name": name,
    "device": device,
    "sample_fraction": 0.5,
    "sample_random_state": 42,
    "database_hash": stable_df_hash(
        dataset_database.metadata,
        cols=["path", "unique_name", "split-time_closed"]
    ),
    "query_hash": stable_df_hash(
        dataset_query.metadata,
        cols=["path", "unique_name", "split-time_closed"]
    ),
    "calibration_hash": stable_df_hash(
        dataset_calibration.metadata,
        cols=["path", "unique_name", "split-time_closed"]
    ),
}

calib_cache_key = make_cache_key({
    **common_cache_info,
    "stage": "calibration",
    "calibration_type": "IsotonicCalibration",
    "aliked_transform": "Resize512_ToTensor",
    "mega_transform": "Resize224_ToTensor_NormalizeImageNet",
})

aliked_calib_path = cache_dir / f"aliked_isotonic_{calib_cache_key}.pkl"
mega_calib_path = cache_dir / f"mega_isotonic_{calib_cache_key}.pkl"

# ------------------------------------------------------------------
# Calibration: compute once, then load from disk
# ------------------------------------------------------------------
if aliked_calib_path.exists() and mega_calib_path.exists():
    print(f"[CACHE HIT] Loading calibration from:\n  {aliked_calib_path}\n  {mega_calib_path}")
    matcher_aliked.calibration = load_pickle(aliked_calib_path)
    matcher_mega.calibration = load_pickle(mega_calib_path)
else:
    print("[CACHE MISS] Fitting WildFusion calibration...")
    wildfusion.fit_calibration(dataset_calibration, dataset_calibration)

    save_pickle(matcher_aliked.calibration, aliked_calib_path)
    save_pickle(matcher_mega.calibration, mega_calib_path)
    print(f"[CACHE SAVE] Calibration saved to:\n  {aliked_calib_path}\n  {mega_calib_path}")

similarity_cache_key = make_cache_key({
    **common_cache_info,
    "stage": "similarity",
    "calibration_cache_key": calib_cache_key,
    "B": 10,
    "wildfusion_priority": "matcher_mega",
    "similarity_dtype": "float32",
})

similarity_cache_path = cache_dir / f"similarity_{similarity_cache_key}.npz"

# ------------------------------------------------------------------
# Similarity: compute once, then load from disk
# ------------------------------------------------------------------
if similarity_cache_path.exists():
    print(f"[CACHE HIT] Loading similarity matrix from: {similarity_cache_path}")
    similarity = np.load(similarity_cache_path)["similarity"]
else:
    print("[CACHE MISS] Computing WildFusion similarity...")
    similarity = wildfusion(dataset_query, dataset_database, B=10)
    similarity = np.asarray(similarity, dtype=np.float32)

    np.savez_compressed(similarity_cache_path, similarity=similarity)
    print(f"[CACHE SAVE] Similarity matrix saved to: {similarity_cache_path}")

pred_idx = similarity.argsort(axis=1)[:, -1]
pred_scores = similarity[np.arange(n_query), pred_idx]
print(f"Similarity matrix: {similarity.shape}")
print(f"Top predicted (index, score): {list(zip(pred_idx, np.round(pred_scores, 3)))[:10]}")

# ------------------------------------------------------------------
# Visualization / evaluation dataset
# ------------------------------------------------------------------
num_examples = 8
top_k = 3

dataset = WildlifeDataset(root, metadata, transform=transform_display, load_label=True, col_label="unique_name")
dataset_database = dataset.get_subset(dataset.metadata["split-time_closed"] == "train")
dataset_query = dataset.get_subset(dataset.metadata["split-time_closed"] == "test")

print("Accuracy:")
for k in range(1, top_k + 1):
    acc = get_acc(dataset_query, dataset_database, similarity, top_k=k)
    print(f"  Top {k}: {acc:.3f}")

pred_idx = similarity.argsort(axis=1)

output_dir = "retrieval_figures"
os.makedirs(output_dir, exist_ok=True)

for n, query_idx in enumerate(np.random.choice(np.arange(len(dataset_query)), num_examples, replace=False)):
    database_idx = pred_idx[query_idx, -top_k:][::-1]
    score = similarity[query_idx, database_idx]

    fig, ax = plt.subplots(1, 1 + top_k, figsize=((1 + top_k) * 3, 3))

    query_data = dataset_query[query_idx]
    ax[0].imshow(query_data[0])
    ax[0].set_title(query_data[1])
    ax[0].axis("off")

    for i, idx in enumerate(database_idx):
        database_data = dataset_database[idx]
        ax[i + 1].imshow(database_data[0])
        ax[i + 1].set_title(f"{database_data[1]}: {score[i]:.3f}")
        ax[i + 1].axis("off")

    plt.tight_layout()

    save_path = os.path.join(output_dir, f"query_{query_idx:05d}.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)