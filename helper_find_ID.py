import os
import pandas as pd

root = "data"
output_folder = "output/run_20260416_140125"
split_column = "split-time_closed"
label_column = "unique_name"

train_split_csv = os.path.join(output_folder, "train_split.csv")
val_split_csv = os.path.join(output_folder, "val_split.csv")
test_split_csv = os.path.join(output_folder, "test_split.csv")
metadata_path = os.path.join(root, "CzechLynxDataset-Metadata-Real.csv")


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


train_metadata, val_metadata, test_metadata = build_protocol_splits(metadata_path, split_column)

shared_ids = sorted(set(test_metadata[label_column]) & set(train_metadata[label_column]))
print(f"Shared identities: {len(shared_ids)}")

# Pick 5 IDs manually after inspection, or keep the first 5 for now
demo_shared_ids = shared_ids[50:55]

demo_query_indices = []
demo_gallery_indices = []

for lynx_id in demo_shared_ids:
    q_rows = test_metadata[test_metadata[label_column] == lynx_id]
    g_rows = train_metadata[train_metadata[label_column] == lynx_id]

    # take 2 from each side
    q_take = q_rows.head(2)
    g_take = g_rows.head(2)

    demo_query_indices.extend(q_take.index.tolist())
    demo_gallery_indices.extend(g_take.index.tolist())

print("DEMO_SHARED_IDS =", demo_shared_ids)
print("DEMO_QUERY_INDICES =", demo_query_indices)
print("DEMO_GALLERY_INDICES =", demo_gallery_indices)

print("\nChosen query images:")
print(test_metadata.loc[demo_query_indices, [label_column, "path"]])

print("\nChosen gallery images:")
print(train_metadata.loc[demo_gallery_indices, [label_column, "path"]])