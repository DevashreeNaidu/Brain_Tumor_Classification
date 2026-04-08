import tensorflow as tf
import os

EPOCHS_HEAD = 10
EPOCHS_FINETUNE = 20
BATCH_SIZE = 32
UNFREEZE_LAYERS = 30

MODELS_DIR = 'models'


def train_baseline(train_ds, val_ds, augment=False):
    """
    Train baseline CNN from scratch.
    augment=False -> E1
    augment=True  -> E2
    """
    from src.models import build_baseline_cnn, compile_model

    if augment:
        print("\n=== E2: Baseline CNN + Augmentation ===")
    else:
        print("\n=== E1: Baseline CNN, No Augmentation ===")

    model = build_baseline_cnn()
    model = compile_model(model, learning_rate=1e-3)
    model.summary()

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS_HEAD + EPOCHS_FINETUNE,
        verbose=1
    )

    name = 'e2_baseline_augmented' if augment else 'e1_baseline'
    save_path = f'/content/drive/MyDrive/MS/Project Tumor/models/{name}.h5'
    model.save(save_path)
    print(f"Model saved to {save_path}")

    return model, history


def train_mobilenetv2(train_ds, val_ds):
    """
    Train MobileNetV2 with transfer learning.
    Experiment E3.
    """
    from src.models import build_mobilenetv2, compile_model

    print("\n=== E3: MobileNetV2 Transfer Learning ===")

    model, base_model = build_mobilenetv2()
    model = compile_model(model, learning_rate=1e-3)

    print("\nPhase 1: Training classification head (backbone frozen)...")
    history_head = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS_HEAD,
        verbose=1
    )

    print(f"\nPhase 2: Fine-tuning top {UNFREEZE_LAYERS} layers...")
    for layer in base_model.layers[-UNFREEZE_LAYERS:]:
        layer.trainable = True

    model = compile_model(model, learning_rate=1e-5)

    history_finetune = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS_FINETUNE,
        verbose=1
    )

    save_path = '/content/drive/MyDrive/MS/Project Tumor/models/e3_mobilenetv2.h5'
    model.save(save_path)
    print(f"Model saved to {save_path}")

    return model, history_head, history_finetune


def train_resnet50(train_ds, val_ds):
    """
    Train ResNet50 with transfer learning.
    Experiment E4.
    """
    from src.models import build_resnet50, compile_model

    print("\n=== E4: ResNet50 Transfer Learning ===")

    model, base_model = build_resnet50()
    model = compile_model(model, learning_rate=1e-3)

    print("\nPhase 1: Training classification head (backbone frozen)...")
    history_head = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS_HEAD,
        verbose=1
    )

    print(f"\nPhase 2: Fine-tuning top {UNFREEZE_LAYERS} layers...")
    for layer in base_model.layers[-UNFREEZE_LAYERS:]:
        layer.trainable = True

    model = compile_model(model, learning_rate=1e-5)

    history_finetune = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS_FINETUNE,
        verbose=1
    )

    save_path = '/content/drive/MyDrive/MS/Project Tumor/models/e4_resnet50.h5'
    model.save(save_path)
    print(f"Model saved to {save_path}")

    return model, history_head, history_finetune