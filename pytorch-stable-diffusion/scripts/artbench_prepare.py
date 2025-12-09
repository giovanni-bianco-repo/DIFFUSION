"""
Script to prepare ArtBench images and captions for LoRA training.

What it does (fully automatic):
- Downloads ArtBench-10 from Kaggle via kagglehub (if not already cached)
- Reads the CIFAR-style binary data (artbench-10-batches-py)
- Exports N images per style to ./img
- Creates ./captions.txt with style-aware captions

Usage:
  python artbench_prepare.py

Dependencies (install first if needed):
  pip install kagglehub torchvision pillow
"""

import os
import sys
from typing import Dict, Any, Tuple, Optional

import shutil

# =========================
# Configuration (edit here)
# =========================

# Kaggle dataset identifier for ArtBench-10
KAGGLE_DATASET = "alexanderliao/artbench10"

# Number of images per style/class to export
N_PER_CLASS = 50

# Where to save output (relative to this script)
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
IMG_DIR = os.path.join(_THIS_DIR, "img")
CAPTIONS_FILE = os.path.join(_THIS_DIR, "captions.txt")


# ===============
# Dependency check
# ===============

try:
    import kagglehub
except ImportError:
    print(
        "ERROR: kagglehub is not installed.\n"
        "Install it with:\n"
        "  pip install kagglehub\n"
    )
    sys.exit(1)

try:
    from PIL import Image
except ImportError:
    print(
        "ERROR: Pillow (PIL) is not installed.\n"
        "Install it with:\n"
        "  pip install pillow\n"
    )
    sys.exit(1)

try:
    import numpy as np
    from torchvision.datasets.vision import VisionDataset
except ImportError:
    print(
        "ERROR: torchvision (and numpy) are required.\n"
        "Install them with:\n"
        "  pip install torchvision numpy\n"
    )
    sys.exit(1)


# ============================
# Minimal ArtBench10 dataset
# (CIFAR-style binary loader)
# ============================

import pickle


class ArtBench10(VisionDataset):
    """
    ArtBench-10 dataset in CIFAR-style binary format.

    This is essentially the official ArtBench10 class from:
      https://github.com/liaopeiyuan/artbench
    but with no downloading logic used here (we read from Kaggle files).
    """

    base_folder = "artbench-10-batches-py"
    # We don't use url / filename here; kagglehub handles the download.
    url = "https://artbench.eecs.berkeley.edu/files/artbench-10-python.tar.gz"
    filename = "artbench-10-python.tar.gz"
    tgz_md5 = "9df1e998ee026aae36ec60ca7b44960e"

    train_list = [
        ["data_batch_1", "c2e02a78dcea81fe6fead5f1540e542f"],
        ["data_batch_2", "1102a4dcf41d4dd63e20c10691193448"],
        ["data_batch_3", "177fc43579af15ecc80eb506953ec26f"],
        ["data_batch_4", "566b2a02ccfbafa026fbb2bcec856ff6"],
        ["data_batch_5", "faa6a572469542010a1c8a2a9a7bf436"],
    ]

    test_list = [
        ["test_batch", "fa44530c8b8158467e00899609c19e52"],
    ]

    meta = {
        "filename": "meta",
        "key": "styles",
        "md5": "5bdcafa7398aa6b75d569baaec5cd4aa",
    }

    def __init__(
        self,
        root: str,
        train: bool = True,
        transform=None,
        target_transform=None,
    ) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.train = train

        if self.train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

        self.data: Any = []
        self.targets = []

        base = os.path.join(self.root, self.base_folder)
        if not os.path.isdir(base):
            raise RuntimeError(
                f"Could not find '{self.base_folder}' under root: {self.root}\n"
                f"Contents of root: {os.listdir(self.root)}"
            )

        # load numpy arrays from CIFAR-style batches
        for file_name, _checksum in downloaded_list:
            file_path = os.path.join(base, file_name)
            if not os.path.exists(file_path):
                raise RuntimeError(f"Missing batch file: {file_path}")
            with open(file_path, "rb") as f:
                entry = pickle.load(f, encoding="latin1")
                self.data.append(entry["data"])
                # labels are stored as a list of ints
                if "labels" in entry:
                    self.targets.extend(entry["labels"])
                else:
                    self.targets.extend(entry["fine_labels"])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # N x H x W x C

        # Load class names from meta file
        meta_path = os.path.join(base, self.meta["filename"])
        if not os.path.exists(meta_path):
            raise RuntimeError(f"Missing meta file: {meta_path}")
        with open(meta_path, "rb") as infile:
            meta_data = pickle.load(infile, encoding="latin1")
            self.classes = meta_data[self.meta["key"]]
        self.class_to_idx = {name: i for i, name in enumerate(self.classes)}

    def __getitem__(self, index: int) -> Tuple[Image.Image, int]:
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)  # convert numpy array to PIL Image

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return len(self.data)


# =======================
# Kaggle download helpers
# =======================


def download_artbench_via_kaggle() -> str:
    """
    Download ArtBench-10 via kagglehub and return the local root path.
    """
    print(f"Downloading (or reusing cached) Kaggle dataset: {KAGGLE_DATASET} ...")
    dataset_path = kagglehub.dataset_download(KAGGLE_DATASET)
    print(f"Kaggle dataset path: {dataset_path}")
    return dataset_path


def find_artbench_root(kaggle_path: str) -> str:
    """
    Search under kaggle_path for the directory that contains 'artbench-10-batches-py'.
    Returns the path that should be used as ArtBench10(root=...).
    """
    # Case 1: base folder directly inside kaggle_path
    if os.path.isdir(os.path.join(kaggle_path, ArtBench10.base_folder)):
        return kaggle_path

    # Case 2: one level below or deeper – walk the tree
    for root, dirs, _files in os.walk(kaggle_path):
        if ArtBench10.base_folder in dirs:
            return root

    raise RuntimeError(
        f"Could not find '{ArtBench10.base_folder}' inside downloaded Kaggle path: {kaggle_path}"
    )


# ==========================
# Export images + captions
# ==========================


def export_images_and_captions(
    artbench_root: str,
    img_dir: str,
    captions_file: str,
    n_per_class: int,
) -> None:
    """
    Export N images per style from ArtBench10 (train split) to img_dir and
    write corresponding captions to captions_file.
    """
    os.makedirs(img_dir, exist_ok=True)

    print(f"Loading ArtBench10 from root: {artbench_root}")
    dataset = ArtBench10(root=artbench_root, train=True)

    num_classes = len(dataset.classes)
    print(f"Found {num_classes} styles: {dataset.classes}")

    counts: Dict[int, int] = {i: 0 for i in range(num_classes)}
    total_needed = num_classes * n_per_class
    captions = []

    print(
        f"Exporting up to {n_per_class} images per style (total <= {total_needed})..."
    )
    for idx in range(len(dataset)):
        img, label = dataset[idx]

        if counts[label] >= n_per_class:
            # Already have enough for this style
            continue

        style_name = dataset.classes[label]
        idx_in_class = counts[label]

        # Flattened filename; you can change this pattern if you want
        filename = f"{style_name}_{idx_in_class:05d}.png".replace(" ", "_")
        out_path = os.path.join(img_dir, filename)

        img.save(out_path)
        counts[label] += 1

        caption = f"{filename}: a painting in {style_name} style"
        captions.append(caption)

        # Early stop if we've collected all
        if sum(counts.values()) >= total_needed:
            break

    # Write captions file
    with open(captions_file, "w", encoding="utf-8") as f:
        for line in captions:
            f.write(line + "\n")

    print("\nDone!")
    print(f"  - Exported {sum(counts.values())} images into: {img_dir}")
    print(f"  - Wrote captions for {len(captions)} images into: {captions_file}")
    print("  - Images per style:")
    for i, style in enumerate(dataset.classes):
        print(f"      {style}: {counts[i]} images")


# =========
# Main
# =========


def main() -> None:
    print("=== ArtBench-10 LoRA Prep (Kaggle, fully automatic) ===\n")
    print(f"Kaggle dataset   : {KAGGLE_DATASET}")
    print(f"Output img dir   : {IMG_DIR}")
    print(f"Captions file    : {CAPTIONS_FILE}")
    print(f"Images per style : {N_PER_CLASS}")
    print()

    kaggle_path = download_artbench_via_kaggle()
    artbench_root = find_artbench_root(kaggle_path)

    export_images_and_captions(
        artbench_root=artbench_root,
        img_dir=IMG_DIR,
        captions_file=CAPTIONS_FILE,
        n_per_class=N_PER_CLASS,
    )


if __name__ == "__main__":
    main()
