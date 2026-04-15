import os
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


root = 'data'
output_folder = 'output/run_20260415_150322'

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


# Loading the dataset
metadata = pd.read_csv(os.path.join(root, "CzechLynxDataset-Metadata-Real.csv"))
metadata = metadata.sample(int(len(metadata) * 0.5))

dataset = WildlifeDataset(root, metadata, load_label=True, col_label="unique_name")
dataset_database = dataset.get_subset(dataset.metadata['split-time_closed'] == 'train')
dataset_query = dataset.get_subset(dataset.metadata['split-time_closed'] == 'test')
dataset_calibration = WildlifeDataset(root, df=dataset_database.metadata[:100], load_label=True, col_label="unique_name") 

n_query = len(dataset_query)


# Loading the models
name = 'hf-hub:BVRA/MegaDescriptor-T-224'
checkpoint_path = os.path.join(output_folder, "checkpoint-final.pth")
model = timm.create_model(name, num_classes=0, pretrained=True)
device = 'cuda'

# Loading fine-tuned weights
if os.path.isfile(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint)

matcher_aliked = SimilarityPipeline(
    matcher = MatchLightGlue(features='aliked', device=device, batch_size=16),
    extractor = AlikedExtractor(),
    transform = transforms_aliked,
    calibration = IsotonicCalibration()
)

matcher_mega = SimilarityPipeline(
    matcher = CosineSimilarity(),
    extractor = DeepFeatures(model=model, device=device, batch_size=16),
    transform = transform,
    calibration = IsotonicCalibration()
)


# Calibrating the WildFusion
wildfusion = WildFusion(calibrated_pipelines = [matcher_aliked, matcher_mega], priority_pipeline = matcher_mega)
wildfusion.fit_calibration(dataset_calibration, dataset_calibration)

# Compute WildFusion similarity
similarity = wildfusion(dataset_query, dataset_database, B=10)
pred_idx = similarity.argsort(axis=1)[:,-1]
pred_scores = similarity[range(n_query), pred_idx]
print(f"Similarity matrix: {similarity.shape}")
print(f"Top predicted (index, score): {list(zip(pred_idx, np.round(pred_scores,3)))[:10]}")


num_examples = 8
top_k = 3

dataset = WildlifeDataset(root, metadata, transform=transform_display, load_label=True, col_label="unique_name")
dataset_database = dataset.get_subset(dataset.metadata['split-time_closed'] == 'train')
dataset_query = dataset.get_subset(dataset.metadata['split-time_closed'] == 'test')

print("Accuracy:")
for k in range(1, top_k+1):
    acc = get_acc(dataset_query, dataset_database, similarity, top_k=k)
    print(f"  Top {k}: {acc:.3f}")

pred_idx = similarity.argsort(axis=1)
for query_idx in np.random.choice(np.arange(len(dataset_query)), num_examples, False):
    database_idx = pred_idx[query_idx, -top_k:][::-1]
    score = similarity[query_idx, database_idx]
    fig, ax = plt.subplots(1, 1+top_k, figsize=((1+top_k)*3, 3))
    
    query_data = dataset_query[query_idx]
    ax[0].imshow(query_data[0])
    ax[0].set_title(query_data[1])

    for i, idx in enumerate(database_idx):
        database_data = dataset_database[idx]
        ax[i+1].imshow(database_data[0])
        ax[i+1].set_title(database_data[1] + f": {score[i]:.3f}")

    plt.show()