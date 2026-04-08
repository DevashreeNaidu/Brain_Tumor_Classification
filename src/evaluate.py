import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report
)
import os

CLASSES = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
FIGURES_DIR = 'results/figures'


def evaluate_model(model, test_ds, experiment_name):
    """
    Evaluate a trained model on the test dataset.
    Returns accuracy, macro F1, and saves confusion matrix.
    """
    print(f"\n=== Evaluating {experiment_name} ===")

    # Get predictions
    y_pred_probs = model.predict(test_ds, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # Get true labels from dataset
    y_test = np.concatenate([y for x, y in test_ds], axis=0)

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average='macro')
    per_class_f1 = f1_score(y_test, y_pred, average=None)

    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Macro F1:  {macro_f1:.4f}")
    print("\nPer-class F1:")
    for cls, score in zip(CLASSES, per_class_f1):
        print(f"  {cls}: {score:.4f}")

    print("\nFull classification report:")
    print(classification_report(y_test, y_pred, target_names=CLASSES))

    # Confusion matrix
    plot_confusion_matrix(y_test, y_pred, experiment_name)

    return {
        'experiment': experiment_name,
        'accuracy': accuracy,
        'macro_f1': macro_f1,
        'per_class_f1': per_class_f1
    }


def plot_confusion_matrix(y_test, y_pred, experiment_name):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_test, y_pred)
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm_percent,
        annot=True,
        fmt='.1f',
        cmap='Blues',
        xticklabels=CLASSES,
        yticklabels=CLASSES,
        ax=ax
    )
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title(f'Confusion Matrix — {experiment_name} (%)')

    plt.tight_layout()
    filename = f"{experiment_name.lower().replace(' ', '_')}_confusion_matrix.png"
    plt.savefig(os.path.join(FIGURES_DIR, filename), dpi=150)
    plt.close()
    print(f"Confusion matrix saved to results/figures/{filename}")


def plot_training_history(history, experiment_name):
    """Plot training and validation accuracy/loss curves."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Accuracy
    axes[0].plot(history.history['accuracy'], label='Train')
    axes[0].plot(history.history['val_accuracy'], label='Validation')
    axes[0].set_title(f'Accuracy — {experiment_name}')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()

    # Loss
    axes[1].plot(history.history['loss'], label='Train')
    axes[1].plot(history.history['val_loss'], label='Validation')
    axes[1].set_title(f'Loss — {experiment_name}')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()

    plt.tight_layout()
    filename = f"{experiment_name.lower().replace(' ', '_')}_training_curves.png"
    plt.savefig(os.path.join(FIGURES_DIR, filename), dpi=150)
    plt.close()
    print(f"Training curves saved to results/figures/{filename}")


def summarize_all_experiments(results_list):
    """Print a summary table of all experiments."""
    print("\n" + "="*60)
    print("ABLATION STUDY SUMMARY")
    print("="*60)
    print(f"{'Experiment':<30} {'Accuracy':>10} {'Macro F1':>10}")
    print("-"*60)
    for r in results_list:
        print(f"{r['experiment']:<30} {r['accuracy']:>10.4f} {r['macro_f1']:>10.4f}")
    print("="*60)