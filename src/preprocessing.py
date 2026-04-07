import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

# Paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
TRAIN_DIR = os.path.join(BASE_DIR, 'data', 'raw', 'Training')
TEST_DIR = os.path.join(BASE_DIR, 'data', 'raw', 'Testing')

# Settings
IMG_SIZE = (224, 224)
CLASSES = ['glioma', 'meningioma', 'notumor', 'pituitary']

# ImageNet normalization values
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])


def load_image(path):
    """Load image, convert to RGB, resize to 224x224."""
    img = Image.open(path).convert('RGB')
    img = img.resize(IMG_SIZE)
    img = np.array(img) / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    return img.astype(np.float32)


def load_dataset(directory):
    """Load all images and labels from a directory."""
    images, labels = [], []
    for label, cls in enumerate(CLASSES):
        cls_path = os.path.join(directory, cls)
        for img_name in os.listdir(cls_path):
            img_path = os.path.join(cls_path, img_name)
            try:
                img = load_image(img_path)
                images.append(img)
                labels.append(label)
            except Exception as e:
                print(f"Skipping {img_path}: {e}")
    return np.array(images), np.array(labels)


def get_splits():
    """Return train, validation, and test splits."""
    print("Loading training data...")
    X_train_full, y_train_full = load_dataset(TRAIN_DIR)

    print("Loading test data...")
    X_test, y_test = load_dataset(TEST_DIR)

    # Split training into train and validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full,
        test_size=0.2,
        random_state=42,
        stratify=y_train_full
    )

    print(f"\nTrain: {X_train.shape}")
    print(f"Validation: {X_val.shape}")
    print(f"Test: {X_test.shape}")

    return X_train, X_val, X_test, y_train, y_val, y_test