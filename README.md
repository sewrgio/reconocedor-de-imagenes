# 🏗️ NexusVision — Reconocedor de Edificios

Sistema inteligente de reconocimiento de imágenes para clasificación binaria de edificios, basado en una CNN personalizada con despliegue en formato ONNX.

## 📋 Características

- 🧠 **CNN personalizada** con ~3.3M parámetros, optimizada para detección arquitectónica
- 📦 **Exportación ONNX v12+** para despliegue multiplataforma
- 🌐 **Interfaz web** con cámara en vivo usando ONNX Runtime Web
- 📊 **Pipeline de entrenamiento** completo con métricas, gráficos y reportes
- 🎯 **Predicción individual** por línea de comandos

## 🚀 Inicio Rápido

### 1. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 2. Preparar el dataset

Organiza tus imágenes en la siguiente estructura:

```
dataset/
├── edificios/          # Imágenes de edificios
│   ├── img001.jpg
│   └── ...
└── no_edificios/       # Imágenes que NO son edificios
    ├── img001.jpg
    └── ...
```

> **Recomendación:** Mínimo 1,000 imágenes por clase para buenos resultados.

### 3. Entrenar el modelo

```bash
python train.py --data_dir ./dataset --epochs 25 --batch_size 32 --lr 0.001
```

El entrenamiento generará en `./output/`:
- `best_model.pth` — Pesos del mejor modelo
- `building_model.onnx` — Modelo exportado
- `training_history.png` — Gráficos de pérdida y precisión
- `confusion_matrix.png` — Matriz de confusión
- `training_report.json` — Métricas completas

### 4. Predecir con una imagen

```bash
python predict.py --model ./output/building_model.onnx --image ./mi_foto.jpg
```

### 5. Usar la interfaz web

```bash
# Copiar el modelo ONNX al directorio raíz
cp ./output/building_model.onnx ./building_model.onnx

# Servir con cualquier servidor HTTP
python -m http.server 8080
```

Abrir `http://localhost:8080` en el navegador.

## 🏛️ Especificaciones del Modelo

| Parámetro | Valor |
| :--- | :--- |
| Arquitectura | CNN (3 bloques conv + clasificador) |
| Input | `[1, 224, 224, 3]` — NHWC, float32 |
| Output | `[1, 1]` — Confianza (0.0–1.0) |
| Activación | Sigmoid (binaria) |
| Formato | ONNX v12+ |
| I/O Names | `cam_input` → `confidence_score` |

## 📁 Estructura del Proyecto

```
├── index.html           # Interfaz web
├── style.css            # Estilos de la UI
├── app.js               # Lógica de inferencia web
├── model_generator.py   # Arquitectura CNN + exportación rápida
├── train.py             # Pipeline de entrenamiento
├── predict.py           # Predicción por CLI
├── requirements.txt     # Dependencias Python
├── INFORME_ANALISIS.md  # Informe técnico completo
└── README.md            # Este archivo
```

## 📄 Documentación

Consulta el [Informe Técnico](INFORME_ANALISIS.md) para detalles completos sobre la arquitectura, métricas, y análisis del sistema.

---

*Desarrollado como parte del proyecto NexusVision*
