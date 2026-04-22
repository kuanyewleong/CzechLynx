import os
import json
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import torch
import torchvision.transforms as T
from PIL import Image, ImageOps

from wildlife_datasets.datasets import WildlifeDataset


# -------------------------------
# Config
# -------------------------------
root = "data"
output_folder = "output/run_20260416_140125"   # change if needed
split_column = "split-time_closed"
label_column = "unique_name"
model_name = "hf-hub:BVRA/MegaDescriptor-L-384"

checkpoint_path = os.path.join(output_folder, "checkpoint-final.pth")
train_split_csv = os.path.join(output_folder, "train_split.csv")
val_split_csv = os.path.join(output_folder, "val_split.csv")
test_split_csv = os.path.join(output_folder, "test_split.csv")
metadata_path = os.path.join(root, "CzechLynxDataset-Metadata-Real.csv")

cache_dir = Path(output_folder) / "cache"
cache_dir.mkdir(parents=True, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
wildfusion_B = 10

# -------------------------------------------------
# FIXED DEMO POOLS
# Replace these with your own chosen dataset indices
# -------------------------------------------------
DEMO_QUERY_INDICES = [
    8902, 8985, 9573, 9574, 8822, 9326, 7173, 7174, 7214, 7289
]

DEMO_GALLERY_INDICES = [
    3480, 4608, 36, 101, 162, 933, 798, 4878, 2, 100
]
# the above indices are from ID: 'lynx_050', 'lynx_051', 'lynx_052', 'lynx_053', 'lynx_054'

# -------------------------------
# Helpers adapted from your script
# -------------------------------
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


def build_protocol_splits(metadata_path: str, split_col: str):
    if os.path.isfile(train_split_csv) and os.path.isfile(val_split_csv) and os.path.isfile(test_split_csv):
        train_df = pd.read_csv(train_split_csv)
        val_df = pd.read_csv(val_split_csv)
        test_df = pd.read_csv(test_split_csv)
        return train_df, val_df, test_df

    metadata = pd.read_csv(metadata_path)
    train_df = metadata[metadata[split_col] == "train"].copy().reset_index(drop=True)
    test_df = metadata[metadata[split_col] == "test"].copy().reset_index(drop=True)

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


def compute_wildfusion_cache_path(train_metadata: pd.DataFrame, val_metadata: pd.DataFrame, test_metadata: pd.DataFrame) -> Path:
    common_cache_info = {
        "metadata_csv": file_signature(metadata_path),
        "checkpoint": file_signature(checkpoint_path),
        "train_csv": file_signature(train_split_csv),
        "val_csv": file_signature(val_split_csv),
        "test_csv": file_signature(test_split_csv),
        "model_name": model_name,
        "device": device,
        "gallery_hash": stable_df_hash(train_metadata, cols=["path", label_column, split_column]),
        "query_hash": stable_df_hash(test_metadata, cols=["path", label_column, split_column]),
        "calibration_hash": stable_df_hash(val_metadata, cols=["path", label_column, split_column]),
    }

    calib_cache_key = make_cache_key({
        **common_cache_info,
        "stage": "calibration",
        "calibration_type": "IsotonicCalibration",
        "aliked_transform": "Resize512_ToTensor",
        "mega_transform": "Resize384_ToTensor_NormalizeImageNet",
        "wildfusion_priority": "matcher_mega_fusion",
    })

    wildfusion_cache_key = make_cache_key({
        **common_cache_info,
        "stage": "similarity_wildfusion",
        "calibration_cache_key": calib_cache_key,
        "B": wildfusion_B,
        "wildfusion_priority": "matcher_mega_fusion",
    })

    return cache_dir / f"similarity_wildfusion_{wildfusion_cache_key}.npz"


def add_candidate_border(img: Image.Image, border_size=12):
    return ImageOps.expand(img.convert("RGB"), border=border_size, fill="red")


def pil_to_image(x):
    if isinstance(x, Image.Image):
        return x.convert("RGB")
    return Image.fromarray(np.array(x)).convert("RGB")


def format_option_label(df: pd.DataFrame, idx: int) -> str:
    row = df.iloc[idx]
    return f"{row[label_column]} | {Path(str(row['path'])).name} | idx={idx}"


def validate_demo_indices(indices, df_len, pool_name):
    bad = [i for i in indices if i < 0 or i >= df_len]
    if bad:
        raise ValueError(f"Invalid {pool_name} indices: {bad}")


# -------------------------------
# Display transform
# -------------------------------
transform_display = T.Compose([
    T.Resize([384, 384]),
])


# -------------------------------
# Cached loading
# -------------------------------
@st.cache_data
def load_metadata_and_similarity():
    train_metadata, val_metadata, test_metadata = build_protocol_splits(metadata_path, split_column)

    validate_demo_indices(DEMO_QUERY_INDICES, len(test_metadata), "query")
    validate_demo_indices(DEMO_GALLERY_INDICES, len(train_metadata), "gallery")

    sim_path = compute_wildfusion_cache_path(train_metadata, val_metadata, test_metadata)
    if not sim_path.exists():
        raise FileNotFoundError(
            f"Cached WildFusion similarity file not found:\n{sim_path}\n\n"
            "Please confirm output_folder/model_name/device/B match the run that generated the cache."
        )

    similarity_wildfusion = np.load(sim_path)["similarity"]
    return train_metadata, val_metadata, test_metadata, similarity_wildfusion, str(sim_path)


@st.cache_resource
def load_display_datasets():
    train_metadata, _, test_metadata, _, _ = load_metadata_and_similarity()

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

    return dataset_query_display, dataset_train_gallery_display


# -------------------------------
# App
# -------------------------------
st.set_page_config(page_title="Czech Lynx ReID Demo", layout="wide")
st.title("Lynx Matching")
st.write(
    "Pick one lynx picture, then choose three other pictures to compare with it. "
    "The red picture is the main one. Bigger similarity means the pictures are more alike."
)

try:
    train_metadata, val_metadata, test_metadata, similarity_wildfusion, sim_path = load_metadata_and_similarity()
    dataset_query_display, dataset_train_gallery_display = load_display_datasets()
except Exception as e:
    st.error(str(e))
    st.stop()

with st.expander("Demo info", expanded=False):
    st.write(f"Fixed query demo pool size: {len(DEMO_QUERY_INDICES)}")
    st.write(f"Fixed gallery demo pool size: {len(DEMO_GALLERY_INDICES)}")
    st.write(f"Similarity matrix shape: {similarity_wildfusion.shape}")
    st.write(f"Cache file: {sim_path}")

query_option_map = {
    format_option_label(test_metadata, idx): idx for idx in DEMO_QUERY_INDICES
}
gallery_option_map = {
    format_option_label(train_metadata, idx): idx for idx in DEMO_GALLERY_INDICES
}

st.subheader("Pick your pictures")

left, right = st.columns([1, 1])

with left:
    query_choice = st.selectbox(
        "Pick the main lynx picture",
        options=list(query_option_map.keys())
    )

with right:
    gallery_choices = st.multiselect(
        "Pick 3 pictures to compare",
        options=list(gallery_option_map.keys()),
        default=list(gallery_option_map.keys())[:3],
        max_selections=3,
    )

if len(gallery_choices) != 3:
    st.info("Please select exactly 3 comparison images.")
    st.stop()

q_idx = query_option_map[query_choice]
g_indices = [gallery_option_map[x] for x in gallery_choices]

query_img, _ = dataset_query_display[q_idx]
gallery_img_1, _ = dataset_train_gallery_display[g_indices[0]]
gallery_img_2, _ = dataset_train_gallery_display[g_indices[1]]
gallery_img_3, _ = dataset_train_gallery_display[g_indices[2]]

query_img = pil_to_image(query_img)
gallery_img_1 = pil_to_image(gallery_img_1)
gallery_img_2 = pil_to_image(gallery_img_2)
gallery_img_3 = pil_to_image(gallery_img_3)

score_1 = float(similarity_wildfusion[q_idx, g_indices[0]])
score_2 = float(similarity_wildfusion[q_idx, g_indices[1]])
score_3 = float(similarity_wildfusion[q_idx, g_indices[2]])

query_row = test_metadata.iloc[q_idx]
gallery_row_1 = train_metadata.iloc[g_indices[0]]
gallery_row_2 = train_metadata.iloc[g_indices[1]]
gallery_row_3 = train_metadata.iloc[g_indices[2]]

st.markdown("---")
st.subheader("Result")

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown("### Main Candidate A")
    st.image(add_candidate_border(query_img), use_container_width=True)
    st.markdown(
        "<div style='text-align:center; color:red; font-weight:700;'>This is the image to compare</div>",
        unsafe_allow_html=True,
    )
    st.caption(f"Lynx ID: {query_row[label_column]}")
    st.caption(f"File: {Path(str(query_row['path'])).name}")

with c2:
    st.markdown("### Image B")
    st.image(gallery_img_1, use_container_width=True)
    st.metric("Similarity", f"{score_1:.4f}")
    st.caption(f"Lynx ID: {gallery_row_1[label_column]}")
    st.caption(f"File: {Path(str(gallery_row_1['path'])).name}")

with c3:
    st.markdown("### Image C")
    st.image(gallery_img_2, use_container_width=True)
    st.metric("Similarity", f"{score_2:.4f}")
    st.caption(f"Lynx ID: {gallery_row_2[label_column]}")
    st.caption(f"File: {Path(str(gallery_row_2['path'])).name}")

with c4:
    st.markdown("### Image D")
    st.image(gallery_img_3, use_container_width=True)
    st.metric("Similarity", f"{score_3:.4f}")
    st.caption(f"Lynx ID: {gallery_row_3[label_column]}")
    st.caption(f"File: {Path(str(gallery_row_3['path'])).name}")

st.markdown("---")
st.subheader("Which one is most similar?")

ranking = sorted(
    [
        ("B", score_1, gallery_row_1),
        ("C", score_2, gallery_row_2),
        ("D", score_3, gallery_row_3),
    ],
    key=lambda x: x[1],
    reverse=True,
)

for rank, (name, score, row) in enumerate(ranking, start=1):
    st.write(f"{rank}. Image {name} — similarity = {score:.4f} — Lynx ID: {row[label_column]}")

best_name, best_score, _ = ranking[0]
st.success(f"Best match among the three: Image {best_name} (score = {best_score:.4f})")