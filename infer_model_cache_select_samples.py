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


# ============================================================
# Utility functions
# ============================================================

def get_acc(dataset_query, dataset_database, similarity, top_k=3):
    acc = []
    pred_idx = similarity.argsort(axis=1)
    for query_idx in range(len(dataset_query)):
        database_idx = pred_idx[query_idx, -top_k:][::-1]

        query_label = dataset_query.df[dataset_query.col_label].iloc[query_idx]
        database_labels = [
            dataset_database.df[dataset_database.col_label].iloc[di]
            for di in database_idx
        ]
        acc.append(query_label in database_labels)
    return np.mean(acc)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def stable_df_hash(df: pd.DataFrame, cols=None) -> str:
    if cols is None:
        cols = list(df.columns)

    cols = [c for c in cols if c in df.columns]
    if len(cols) == 0:
        return sha256_text("empty-df")

    df_small = df[cols].copy()
    for c in df_small.columns:
        df_small[c] = df_small[c].astype(str)

    row_hash = pd.util.hash_pandas_object(df_small, index=True).values
    return hashlib.sha256(row_hash.tobytes()).hexdigest()


def file_signature(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        return {"exists": False, "path": str(p)}
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


def get_label_by_index(dataset_obj, idx):
    return dataset_obj.df[dataset_obj.col_label].iloc[idx]



# ============================================================
# Config
# ============================================================

root = "data"
output_folder = "output/run_20260415_150609"
cache_dir = Path(output_folder) / "cache"
cache_dir.mkdir(parents=True, exist_ok=True)

metadata_path = os.path.join(root, "CzechLynxDataset-Metadata-Real.csv")
checkpoint_path = os.path.join(output_folder, "checkpoint-final.pth")

sample_fraction = 0.5
sample_random_state = 42
calibration_subset_size = 100
wildfusion_B = 10

name = "hf-hub:BVRA/MegaDescriptor-T-224"
device = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# Transforms
# ============================================================

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


# ============================================================
# Dataset loading
# ============================================================

metadata = pd.read_csv(metadata_path)

# Fixed seed so caches remain reusable across runs
metadata = metadata.sample(
    int(len(metadata) * sample_fraction),
    random_state=sample_random_state
).reset_index(drop=True)

dataset_no_transform = WildlifeDataset(
    root,
    metadata,
    load_label=True,
    col_label="unique_name"
)

dataset_database_no_transform = dataset_no_transform.get_subset(
    dataset_no_transform.metadata["split-time_closed"] == "train"
)
dataset_query_no_transform = dataset_no_transform.get_subset(
    dataset_no_transform.metadata["split-time_closed"] == "test"
)

dataset_calibration = WildlifeDataset(
    root,
    df=dataset_database_no_transform.metadata[:calibration_subset_size].copy(),
    load_label=True,
    col_label="unique_name"
)

n_query = len(dataset_query_no_transform)
n_database = len(dataset_database_no_transform)

print(f"Dataset loaded:")
print(f"  Train size: {n_database}")
print(f"  Test size : {n_query}")
print(f"  Calibration subset size: {len(dataset_calibration)}")


# ============================================================
# Model loading
# ============================================================

model = timm.create_model(name, num_classes=0, pretrained=True)

if os.path.isfile(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint)
    print(f"Loaded fine-tuned checkpoint: {checkpoint_path}")
else:
    print("Checkpoint not found. Using pretrained backbone only.")

model = model.to(device)
model.eval()


# ============================================================
# Pipelines
# ============================================================

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


# ============================================================
# Cache keys
# ============================================================

common_cache_info = {
    "metadata_csv": file_signature(metadata_path),
    "checkpoint": file_signature(checkpoint_path),
    "model_name": name,
    "device": device,
    "sample_fraction": sample_fraction,
    "sample_random_state": sample_random_state,
    "calibration_subset_size": calibration_subset_size,
    "database_hash": stable_df_hash(
        dataset_database_no_transform.metadata,
        cols=["path", "unique_name", "split-time_closed"]
    ),
    "query_hash": stable_df_hash(
        dataset_query_no_transform.metadata,
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


# ============================================================
# Calibration cache
# ============================================================

if aliked_calib_path.exists() and mega_calib_path.exists():
    print("[CACHE HIT] Loading calibration objects")
    matcher_aliked.calibration = load_pickle(aliked_calib_path)
    matcher_mega.calibration = load_pickle(mega_calib_path)
else:
    print("[CACHE MISS] Fitting calibration")
    wildfusion.fit_calibration(dataset_calibration, dataset_calibration)

    save_pickle(matcher_aliked.calibration, aliked_calib_path)
    save_pickle(matcher_mega.calibration, mega_calib_path)
    print(f"[CACHE SAVE] {aliked_calib_path}")
    print(f"[CACHE SAVE] {mega_calib_path}")


# ============================================================
# Similarity cache: test vs train
# ============================================================

similarity_cache_key = make_cache_key({
    **common_cache_info,
    "stage": "similarity_test_vs_train",
    "calibration_cache_key": calib_cache_key,
    "B": wildfusion_B,
    "wildfusion_priority": "matcher_mega",
    "similarity_dtype": "float32",
})

similarity_cache_path = cache_dir / f"similarity_{similarity_cache_key}.npz"

if similarity_cache_path.exists():
    print(f"[CACHE HIT] Loading test-vs-train similarity: {similarity_cache_path}")
    similarity = np.load(similarity_cache_path)["similarity"]
else:
    print("[CACHE MISS] Computing test-vs-train similarity")
    similarity = wildfusion(dataset_query_no_transform, dataset_database_no_transform, B=wildfusion_B)
    similarity = np.asarray(similarity, dtype=np.float32)
    np.savez_compressed(similarity_cache_path, similarity=similarity)
    print(f"[CACHE SAVE] {similarity_cache_path}")

pred_idx = similarity.argsort(axis=1)[:, -1]
pred_scores = similarity[np.arange(n_query), pred_idx]
print(f"Similarity matrix (test vs train): {similarity.shape}")
print(f"Top predicted (index, score): {list(zip(pred_idx, np.round(pred_scores, 3)))[:10]}")


# ============================================================
# Similarity cache: test vs test
# ============================================================

similarity_test_test_cache_key = make_cache_key({
    **common_cache_info,
    "stage": "similarity_test_vs_test",
    "calibration_cache_key": calib_cache_key,
    "B": wildfusion_B,
    "wildfusion_priority": "matcher_mega",
    "similarity_dtype": "float32",
    "matrix_type": "query_vs_query",
})

similarity_test_test_cache_path = cache_dir / f"similarity_test_test_{similarity_test_test_cache_key}.npz"

if similarity_test_test_cache_path.exists():
    print(f"[CACHE HIT] Loading test-vs-test similarity: {similarity_test_test_cache_path}")
    similarity_test_test = np.load(similarity_test_test_cache_path)["similarity"]
else:
    print("[CACHE MISS] Computing test-vs-test similarity")
    similarity_test_test = wildfusion(dataset_query_no_transform, dataset_query_no_transform, B=wildfusion_B)
    similarity_test_test = np.asarray(similarity_test_test, dtype=np.float32)
    np.savez_compressed(similarity_test_test_cache_path, similarity=similarity_test_test)
    print(f"[CACHE SAVE] {similarity_test_test_cache_path}")


# ============================================================
# Display dataset for visualization
# ============================================================

dataset_display = WildlifeDataset(
    root,
    metadata,
    transform=transform_display,
    load_label=True,
    col_label="unique_name"
)

dataset_database = dataset_display.get_subset(
    dataset_display.metadata["split-time_closed"] == "train"
)
dataset_query = dataset_display.get_subset(
    dataset_display.metadata["split-time_closed"] == "test"
)


# ============================================================
# Retrieval accuracy (test vs train)
# ============================================================

top_k = 3

print("Accuracy:")
for k in range(1, top_k + 1):
    acc = get_acc(dataset_query, dataset_database, similarity, top_k=k)
    print(f"  Top {k}: {acc:.3f}")


def visualize_candidate_vs_three(
    dataset_test,
    similarity_test_test,
    candidate_idx,
    compare_indices,
    save_path,
):
    if len(compare_indices) != 3:
        raise ValueError("compare_indices must contain exactly 3 indices.")

    candidate_img, candidate_label = dataset_test[candidate_idx]
    compare_labels = [get_label_by_index(dataset_test, i) for i in compare_indices]

    if len(set(compare_labels)) != 1:
        raise ValueError(
            f"The 3 comparison images do not share the same ID. "
            f"Found labels: {compare_labels}"
        )

    scores = similarity_test_test[candidate_idx, compare_indices]

    fig, ax = plt.subplots(1, 4, figsize=(12, 3))

    ax[0].imshow(candidate_img)
    ax[0].set_title(f"Candidate\n{candidate_label}")
    ax[0].axis("off")

    for j, idx in enumerate(compare_indices):
        img, label = dataset_test[idx]
        ax[j + 1].imshow(img)
        ax[j + 1].set_title(f"{label}: {scores[j]:.3f}")
        ax[j + 1].axis("off")

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def sample_candidate_and_comparisons_by_two_ids(
    dataset_obj,
    candidate_label,
    comparison_label,
    n_compare=3,
    random_seed=42,
):
    labels = dataset_obj.df[dataset_obj.col_label].values
    rng = np.random.default_rng(random_seed)

    candidate_pool = np.where(labels == candidate_label)[0]
    if len(candidate_pool) < 1:
        raise ValueError(f"No test images found for candidate ID '{candidate_label}'.")

    candidate_idx = int(rng.choice(candidate_pool, size=1, replace=False)[0])

    comparison_pool = np.where(labels == comparison_label)[0]
    if candidate_label == comparison_label:
        comparison_pool = np.array([i for i in comparison_pool if i != candidate_idx])

    if len(comparison_pool) < n_compare:
        raise ValueError(
            f"Not enough comparison images for ID '{comparison_label}'. "
            f"Found {len(comparison_pool)}, need at least {n_compare}."
        )

    compare_indices = rng.choice(comparison_pool, size=n_compare, replace=False).tolist()
    return candidate_idx, [int(x) for x in compare_indices]


candidate_id = "lynx_200"
comparison_id = "lynx_200"

selection_random_seed = 42


# ============================================================
# Automated selection from test set
# ============================================================

candidate_idx, compare_indices = sample_candidate_and_comparisons_by_two_ids(
    dataset_obj=dataset_query,
    candidate_label=candidate_id,
    comparison_label=comparison_id,
    n_compare=3,
    random_seed=selection_random_seed,
)

candidate_label = get_label_by_index(dataset_query, candidate_idx)
candidate_path = dataset_query.metadata.iloc[candidate_idx]["path"]

compare_labels = [get_label_by_index(dataset_query, i) for i in compare_indices]
compare_paths = [dataset_query.metadata.iloc[i]["path"] for i in compare_indices]

print("\nAutomated comparison selection (test vs test):")
print(f"  Comparison ID : {comparison_id}")
print(f"  Candidate idx : {candidate_idx}")
print(f"  Candidate path: {candidate_path}")
print(f"  Candidate label: {candidate_label}")

for p, idx, lbl in zip(compare_paths, compare_indices, compare_labels):
    print(f"  Compare image: {p} | idx={idx} | label={lbl}")

output_dir = "retrieval_figures"
os.makedirs(output_dir, exist_ok=True)

save_path = os.path.join(output_dir, f"candidate_comparison_{comparison_id}.png")

visualize_candidate_vs_three(
    dataset_test=dataset_query,
    similarity_test_test=similarity_test_test,
    candidate_idx=candidate_idx,
    compare_indices=compare_indices,
    save_path=save_path,
)

comparison_scores = similarity_test_test[candidate_idx, compare_indices]
print("\nComparison scores:")
for p, s in zip(compare_paths, comparison_scores):
    print(f"  {p}: {s:.4f}")

print(f"\nSaved comparison figure to: {save_path}")