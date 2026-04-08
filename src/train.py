import tensorflow as tf
import numpy as np
import os
from src.models import build_baseline_cnn, build_mobilenetv2, build_resnet50, compile_model

# Settings
EPOCHS_HEAD = 10        # Epochs to train classification head only
EPOCHS_FINETUNE = 20    # Epochs to fine-tune unfrozen layers
BATCH_SIZE = 32
UNFREEZE_LAYERS = 30    # Number of top layers to unfreeze for fine-tuning

MODELS_DIR = 'models'


def train_baseline(X_train, y_train, X_val, y_val, augment=False):
    """
    Train baseline CNN from scratch.
    augment=False -> E1
    augment=True  -> E2
    """
    if augment:
        print("\n=== E2: Baseline CNN + Augmentation ===")
        X_train = apply_augmentation(X_train)
    else:
        print("\n=== E1: Baseline CNN, No Augmentation ===")

    model = build_baseline_cnn()
    model = compile_model(model, learning_rate=1e-3)
    model.summary()

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS_HEAD + EPOCHS_FINETUNE,
        batch_size=BATCH_SIZE,
        verbose=1
    )

    # Save model
    name = 'e2_baseline_augmented' if augment else 'e1_baseline'
    model.save(os.path.join(MODELS_DIR, f'{name}.h5'))
    print(f"Model saved to models/{name}.h5")

    return model, history


def train_mobilenetv2(X_train, y_train, X_val, y_val):
    """
    Train MobileNetV2 with transfer learning and fine-tuning.
    Experiment E3.
    """
    print("\n=== E3: MobileNetV2 Transfer Learning ===")

    model, base_model = build_mobilenetv2()
    model = compile_model(model, learning_rate=1e-3)

    # Phase 1 - train head only
    print("\nPhase 1: Training classification head (backbone frozen)...")
    history_head = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS_HEAD,
        batch_size=BATCH_SIZE,
        verbose=1
    )

    # Phase 2 - unfreeze top layers and fine-tune
    print(f"\nPhase 2: Fine-tuning top {UNFREEZE_LAYERS} layers...")
    for layer in base_model.layers[-UNFREEZE_LAYERS:]:
        layer.trainable = True

    # Recompile with lower learning rate
    model = compile_model(model, learning_rate=1e-5)

    history_finetune = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS_FINETUNE,
        batch_size=BATCH_SIZE,
        verbose=1
    )

    model.save(os.path.join(MODELS_DIR, 'e3_mobilenetv2.h5'))
    print("Model saved to models/e3_mobilenetv2.h5")

    return model, history_head, history_finetune


def train_resnet50(X_train, y_train, X_val, y_val):
    """
    Train ResNet50 with transfer learning and fine-tuning.
    Experiment E4.
    """
    print("\n=== E4: ResNet50 Transfer Learning ===")

    model, base_model = build_resnet50()
    model = compile_model(model, learning_rate=1e-3)

    # Phase 1 - train head only
    print("\nPhase 1: Training classification head (backbone frozen)...")
    history_head = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS_HEAD,
        batch_size=BATCH_SIZE,
        verbose=1
    )

    # Phase 2 - unfreeze top layers and fine-tune
    print(f"\nPhase 2: Fine-tuning top {UNFREEZE_LAYERS} layers...")
    for layer in base_model.layers[-UNFREEZE_LAYERS:]:
        layer.trainable = True

    model = compile_model(model, learning_rate=1e-5)

    history_finetune = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS_FINETUNE,
        batch_size=BATCH_SIZE,
        verbose=1
    )

    model.save(os.path.join(MODELS_DIR, 'e4_resnet50.h5'))
    print("Model saved to models/e4_resnet50.h5")

    return model, history_head, history_finetune


def apply_augmentation(X_train):
    """Apply data augmentation to training images."""
    print("Applying augmentation...")
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=15,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],
        zoom_range=0.1
    )
    augmented = []
    for img in X_train:
        img_aug = datagen.random_transform(img)
        augmented.append(img_aug)
    return np.array(augmented)