import numpy as np
import pandas as pd
import torch
import cv2
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from .config import (
    WM38Data,
    WM811KData,
    WM811K_LABEL_MAP,
    device
)

# -------------------------------------------------------------------
# MixedWM38
# -------------------------------------------------------------------
def loadWM38():
    data = np.load(WM38Data)

    # Load images and labels
    raw_labels = data['arr_1']

    # Add 9th class — no defect
    no_defect = (raw_labels.sum(axis=1, keepdims=True) == 0).astype(np.float32)
    labels = np.concatenate([raw_labels, no_defect], axis=1)

    # Normalize and add channel dimension
    images = data['arr_0'].astype('float32') / 2.0
    images = np.expand_dims(images, axis=1)

    print(f"WM38 Images shape: {images.shape}")
    print(f"WM38 Labels shape: {labels.shape}")

    return images, labels


def getWM38Loaders(images, labels, batch_size=64, test_split=0.2, random_state=42):
    # Convert to tensors
    image_tensor = torch.tensor(images)
    label_tensor = torch.tensor(labels).float()

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        image_tensor, label_tensor,
        test_size=test_split,
        random_state=random_state
    )

    # DataLoaders
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    print(f"Train size: {len(train_dataset)} Test size: {len(test_dataset)}")
    print(f"Train batches: {len(train_loader)} Test batches: {len(test_loader)}")

    return train_loader, test_loader, X_train, X_test, y_train, y_test


# -------------------------------------------------------------------
# WM-811K
# -------------------------------------------------------------------
def loadWM811K():
    wm811K = pd.read_pickle(WM811KData)

    # Drop nulls and unlabeled rows
    wm811K = wm811K.dropna(subset=['waferMap'])
    wm811K2 = wm811K.copy()
    wm811K2 = wm811K2.drop(['waferIndex'], axis=1)
    wm811K2.failureType = wm811K2.failureType.apply(
        lambda x: x if isinstance(x, str) and len(x) > 0 else float("NaN")
    )
    wm811K2 = wm811K2[wm811K2['failureType'].notna()]

    print(f"WM811K samples after cleaning: {len(wm811K2)}")
    print(wm811K2['failureType'].value_counts())

    return wm811K2


def preprocessWM811K(wm811K2):
    images_811k = []
    labels_811k = []

    for i in range(len(wm811K2)):
        img = cv2.resize(
            np.array(wm811K2['waferMap'].iloc[i], dtype='float32'),
            (52, 52)
        )
        img = img / 2.0

        hot_encoding = np.zeros(9, dtype='float32')
        hot_encoding[WM811K_LABEL_MAP[wm811K2['failureType'].iloc[i]]] = 1.0

        images_811k.append(img)
        labels_811k.append(hot_encoding)

    images_811k = np.stack(images_811k)
    labels_811k = np.stack(labels_811k)
    wm_images = np.expand_dims(images_811k, axis=1)

    print(f"WM811K Images shape: {wm_images.shape}")
    print(f"WM811K Labels shape: {labels_811k.shape}")

    return wm_images, labels_811k


def getWM811KLoaders(wm_images, labels_811k, batch_size=64, test_split=0.2, random_state=42):
    wm_image_tensor = torch.tensor(wm_images)
    wm_label_tensor = torch.tensor(labels_811k).float()

    wm_X_train, wm_X_test, wm_y_train, wm_y_test = train_test_split(
        wm_image_tensor, wm_label_tensor,
        test_size=test_split,
        random_state=random_state
    )

    wm_train_dataset = TensorDataset(wm_X_train, wm_y_train)
    wm_test_dataset = TensorDataset(wm_X_test, wm_y_test)

    wm_train_loader = DataLoader(wm_train_dataset, batch_size=batch_size, shuffle=True)
    wm_test_loader = DataLoader(wm_test_dataset, batch_size=batch_size)

    print(f"WM811K Train size: {len(wm_train_dataset)} Test size: {len(wm_test_dataset)}")

    return wm_train_loader, wm_test_loader, wm_X_train, wm_X_test, wm_y_train, wm_y_test