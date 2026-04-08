import os
import numpy as np
from PIL import Image
import tensorflow as tf

CLASSES = ['glioma', 'meningioma', 'notumor', 'pituitary']
IMG_SIZE = (224, 224)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])


def load_image(path):
    """Load image, convert to RGB, resize to 224x224, normalize."""
    img = Image.open(path).convert('RGB')
    img = img.resize(IMG_SIZE)
    img = np.array(img) / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    return img.astype(np.float32)


def create_dataset(directory, batch_size=32, shuffle=False, augment=False):
    """
    Create a tf.data.Dataset that loads images from disk in batches.
    Much more memory efficient than loading everything at once.
    """
    image_paths = []
    labels = []

    for label, cls in enumerate(CLASSES):
        cls_path = os.path.join(directory, cls)
        for img_name in os.listdir(cls_path):
            if img_name.startswith('.'):
                continue
            image_paths.append(os.path.join(cls_path, img_name))
            labels.append(label)

    image_paths = np.array(image_paths)
    labels = np.array(labels)

    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))

    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(image_paths), seed=42)

    def load_and_preprocess(path, label):
        img = tf.py_function(
            lambda p: load_image(p.numpy().decode()),
            [path], tf.float32
        )
        img.set_shape([224, 224, 3])
        if augment:
            img = tf.image.random_flip_left_right(img)
            img = tf.image.random_brightness(img, 0.2)
            img = tf.image.random_contrast(img, 0.8, 1.2)
        return img, label

    dataset = dataset.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset, len(image_paths)


def get_datasets(train_dir, test_dir, batch_size=32, val_split=0.2):
    """
    Returns train, validation, and test datasets.
    Splits training directory into train and validation.
    """
    # Get all paths and labels
    image_paths = []
    labels = []

    for label, cls in enumerate(CLASSES):
        cls_path = os.path.join(train_dir, cls)
        for img_name in os.listdir(cls_path):
            if img_name.startswith('.'):
                continue
            image_paths.append(os.path.join(cls_path, img_name))
            labels.append(label)

    image_paths = np.array(image_paths)
    labels = np.array(labels)

    # Shuffle and split
    indices = np.random.RandomState(42).permutation(len(image_paths))
    val_size = int(len(indices) * val_split)
    val_idx = indices[:val_size]
    train_idx = indices[val_size:]

    train_paths, train_labels = image_paths[train_idx], labels[train_idx]
    val_paths, val_labels = image_paths[val_idx], labels[val_idx]

    print(f"Train: {len(train_paths)} images")
    print(f"Validation: {len(val_paths)} images")

    def make_dataset(paths, labs, shuffle=False, augment=False):
        ds = tf.data.Dataset.from_tensor_slices((paths, labs))
        if shuffle:
            ds = ds.shuffle(buffer_size=len(paths), seed=42)

        def load_and_preprocess(path, label):
            img = tf.py_function(
                lambda p: load_image(p.numpy().decode()),
                [path], tf.float32
            )
            img.set_shape([224, 224, 3])
            if augment:
                img = tf.image.random_flip_left_right(img)
                img = tf.image.random_brightness(img, 0.2)
                img = tf.image.random_contrast(img, 0.8, 1.2)
            return img, label

        ds = ds.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.batch(batch_size)
        ds = ds.prefetch(tf.data.AUTOTUNE)
        return ds

    train_ds = make_dataset(train_paths, train_labels, shuffle=True, augment=False)
    train_aug_ds = make_dataset(train_paths, train_labels, shuffle=True, augment=True)
    val_ds = make_dataset(val_paths, val_labels)

    # Test dataset
    test_paths = []
    test_labels = []
    for label, cls in enumerate(CLASSES):
        cls_path = os.path.join(test_dir, cls)
        for img_name in os.listdir(cls_path):
            if img_name.startswith('.'):
                continue
            test_paths.append(os.path.join(cls_path, img_name))
            test_labels.append(label)

    test_ds = make_dataset(np.array(test_paths), np.array(test_labels))
    print(f"Test: {len(test_paths)} images")

    return train_ds, train_aug_ds, val_ds, test_ds, len(train_paths), len(val_paths)