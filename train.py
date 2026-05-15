"""
train.py — Pipeline de entrenamiento para el reconocedor de edificios.

Uso:
    python train.py --data_dir ./dataset --epochs 25 --batch_size 32 --lr 0.001

Estructura esperada del dataset:
    dataset/
    ├── edificios/          # Imágenes positivas (son edificios)
    │   ├── img001.jpg
    │   └── ...
    └── no_edificios/       # Imágenes negativas (NO son edificios)
        ├── img001.jpg
        └── ...
"""

import os
import argparse
import json
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix
)
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from model_generator import BuildingRecognizer


# ──────────────────────────────────────────────
#  Configuración de Transformaciones
# ──────────────────────────────────────────────

train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),         # Convierte a [C, H, W] y normaliza a [0, 1]
])

test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


# ──────────────────────────────────────────────
#  Funciones de Entrenamiento
# ──────────────────────────────────────────────

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_preds, all_labels = [], []

    for images, labels in tqdm(dataloader, desc="  Entrenando", leave=False):
        images = images.to(device)
        # Convertir de NCHW [B, 3, 224, 224] → NHWC [B, 224, 224, 3]
        images = images.permute(0, 2, 3, 1)
        labels = labels.float().unsqueeze(1).to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = (outputs > 0.5).float()
        all_preds.extend(preds.cpu().numpy().flatten())
        all_labels.extend(labels.cpu().numpy().flatten())

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    return epoch_loss, epoch_acc


def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="  Evaluando", leave=False):
            images = images.to(device)
            images = images.permute(0, 2, 3, 1)
            labels = labels.float().unsqueeze(1).to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            preds = (outputs > 0.5).float()
            all_preds.extend(preds.cpu().numpy().flatten())
            all_labels.extend(labels.cpu().numpy().flatten())

    epoch_loss = running_loss / len(dataloader.dataset)
    metrics = {
        "accuracy": accuracy_score(all_labels, all_preds),
        "precision": precision_score(all_labels, all_preds, zero_division=0),
        "recall": recall_score(all_labels, all_preds, zero_division=0),
        "f1": f1_score(all_labels, all_preds, zero_division=0),
        "confusion_matrix": confusion_matrix(all_labels, all_preds).tolist()
    }
    return epoch_loss, metrics


# ──────────────────────────────────────────────
#  Generación de Gráficos
# ──────────────────────────────────────────────

def plot_training_history(train_losses, val_losses, train_accs, val_accs, output_dir):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(train_losses, label='Train Loss', color='#6366f1', linewidth=2)
    ax1.plot(val_losses, label='Val Loss', color='#f43f5e', linewidth=2)
    ax1.set_title('Pérdida por Época', fontweight='bold')
    ax1.set_xlabel('Época')
    ax1.set_ylabel('BCE Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(train_accs, label='Train Acc', color='#6366f1', linewidth=2)
    ax2.plot(val_accs, label='Val Acc', color='#10b981', linewidth=2)
    ax2.set_title('Precisión por Época', fontweight='bold')
    ax2.set_xlabel('Época')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, 'training_history.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  📊 Gráfico guardado en: {path}")


def plot_confusion_matrix(cm, output_dir):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.set_title('Matriz de Confusión', fontweight='bold')
    plt.colorbar(im, ax=ax)
    classes = ['No Edificio', 'Edificio']
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(classes)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes)

    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, str(cm[i][j]),
                    ha="center", va="center",
                    color="white" if cm[i][j] > cm.max() / 2 else "black",
                    fontsize=16, fontweight='bold')

    ax.set_ylabel('Real')
    ax.set_xlabel('Predicción')
    plt.tight_layout()
    path = os.path.join(output_dir, 'confusion_matrix.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  📊 Matriz guardada en: {path}")


# ──────────────────────────────────────────────
#  Exportación ONNX
# ──────────────────────────────────────────────

def export_to_onnx(model, output_dir):
    model.eval()
    dummy_input = torch.randn(1, 224, 224, 3)
    onnx_path = os.path.join(output_dir, "building_model.onnx")

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=["cam_input"],
        output_names=["confidence_score"],
        opset_version=12,
        dynamic_axes={
            "cam_input": {0: "batch_size"},
            "confidence_score": {0: "batch_size"}
        }
    )
    print(f"  ✅ Modelo ONNX exportado: {onnx_path}")
    return onnx_path


# ──────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Entrenamiento del Reconocedor de Edificios")
    parser.add_argument('--data_dir', type=str, required=True, help='Directorio raíz del dataset')
    parser.add_argument('--epochs', type=int, default=25, help='Número de épocas')
    parser.add_argument('--batch_size', type=int, default=32, help='Tamaño del batch')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--output_dir', type=str, default='./output', help='Directorio de salida')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🖥️  Dispositivo: {device}")

    # ── Dataset ──
    full_dataset = datasets.ImageFolder(args.data_dir, transform=train_transforms)
    total = len(full_dataset)
    train_size = int(0.8 * total)
    test_size = total - train_size

    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])
    test_dataset.dataset.transform = test_transforms

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    print(f"📂 Dataset: {total} imágenes (Train: {train_size} | Test: {test_size})")
    print(f"   Clases: {full_dataset.classes}")

    # ── Modelo ──
    model = BuildingRecognizer().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)

    # ── Entrenamiento ──
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    best_f1 = 0.0

    print(f"\n🚀 Iniciando entrenamiento ({args.epochs} épocas)...\n")

    for epoch in range(1, args.epochs + 1):
        print(f"Época [{epoch}/{args.epochs}]")
        t_loss, t_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        v_loss, v_metrics = evaluate(model, test_loader, criterion, device)

        scheduler.step(v_loss)

        train_losses.append(t_loss)
        val_losses.append(v_loss)
        train_accs.append(t_acc)
        val_accs.append(v_metrics['accuracy'])

        print(f"  Train Loss: {t_loss:.4f} | Train Acc: {t_acc:.4f}")
        print(f"  Val Loss:   {v_loss:.4f} | Val Acc:   {v_metrics['accuracy']:.4f} | F1: {v_metrics['f1']:.4f}")

        # Guardar mejor modelo
        if v_metrics['f1'] > best_f1:
            best_f1 = v_metrics['f1']
            torch.save(model.state_dict(), os.path.join(args.output_dir, 'best_model.pth'))
            print(f"  💾 Mejor modelo guardado (F1: {best_f1:.4f})")

    # ── Evaluación Final ──
    print("\n📈 Evaluación final con el mejor modelo...")
    model.load_state_dict(torch.load(os.path.join(args.output_dir, 'best_model.pth')))
    _, final_metrics = evaluate(model, test_loader, criterion, device)

    print(f"\n{'='*50}")
    print(f"  RESULTADOS FINALES")
    print(f"{'='*50}")
    print(f"  Accuracy:  {final_metrics['accuracy']:.4f}")
    print(f"  Precision: {final_metrics['precision']:.4f}")
    print(f"  Recall:    {final_metrics['recall']:.4f}")
    print(f"  F1 Score:  {final_metrics['f1']:.4f}")
    print(f"{'='*50}\n")

    # ── Gráficos ──
    plot_training_history(train_losses, val_losses, train_accs, val_accs, args.output_dir)
    plot_confusion_matrix(np.array(final_metrics['confusion_matrix']), args.output_dir)

    # ── Exportar ONNX ──
    export_to_onnx(model, args.output_dir)

    # ── Guardar métricas como JSON ──
    report = {
        "fecha": datetime.now().isoformat(),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "device": str(device),
        "dataset_total": total,
        "train_size": train_size,
        "test_size": test_size,
        "final_metrics": final_metrics
    }
    report_path = os.path.join(args.output_dir, 'training_report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"  📄 Reporte guardado: {report_path}")
    print("\n✅ Entrenamiento completado exitosamente.\n")


if __name__ == "__main__":
    main()
