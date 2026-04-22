import cv2
import json
import os
import numpy as np
from PIL import Image
import pandas as pd
from pycocotools import mask as mask_utils
from pycocotools import mask as rle_tools
import matplotlib.pyplot as plt

def load_image(row, root_path):
    """
    Load an image, its mask, and pose keypoints from a dataset row.

    Args:
        row (pd.Series): A row from the dataset containing 'path', 'mask', and optionally 'pose'.
        root_path (str): The root directory where images are stored.

    Returns:
        image (PIL.Image.Image): Loaded image.
        mask (np.ndarray): Decoded mask with shape (H, W, 1) as uint8.
        pose (dict or None): Pose keypoints if available, else None.
    """
    # Load pose if it exists and is a string
    pose = json.loads(row['pose']) if isinstance(row.get('pose'), str) else None

    # Load image
    image_path = os.path.join(root_path, row['path'])
    image = Image.open(image_path).convert("RGB")  # Ensure image is in RGB

    # Decode mask
    mask_data = json.loads(row['mask'])
    mask = mask_utils.decode(mask_data).astype(np.uint8)
    mask = np.expand_dims(np.asfortranarray(mask), axis=-1)  # Ensure shape (H, W, 1)

    return image, mask, pose


root_path = "data"
metadata_real = pd.read_csv(os.path.join(root_path, "CzechLynxDataset-Metadata-Real.csv"))
metadata_synthetic = pd.read_csv(os.path.join(root_path, "CzechLynxDataset-Metadata-Synthetic.csv"))

examples = 16
seed = 42


# Select random row indices
np.random.seed(seed) # comment this out if you want different samples each time
idxs = np.random.permutation(len(metadata_real))[:examples]

# Create subplots
fig, ax = plt.subplots(2, examples // 2, figsize=((examples // 2) * 5, 2 * 5))
ax = ax.flatten()

# Plot images with masks
for i, idx in enumerate(idxs):
    row = metadata_real.iloc[idx]
    image, mask, pose = load_image(row, root_path)
    
    ax[i].imshow(image)
    ax[i].imshow(mask, alpha=0.4)
    
    ax[i].axis('off')

plt.tight_layout()
# plt.show()

# Save the figure to a file.
plt.savefig("lynx_data_samples.png", bbox_inches='tight', dpi=150)

# Close the plot to free up memory
plt.close(fig)