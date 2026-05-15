# Informe Técnico: Reconocedor de Imágenes de Edificios

## 1. Resumen Ejecutivo

Este documento presenta el análisis técnico completo del sistema **NexusVision**, un reconocedor de imágenes especializado en la identificación y clasificación binaria de edificios. El sistema emplea una Red Neuronal Convolucional (CNN) personalizada, entrenada con PyTorch y exportada al formato estándar **ONNX v12+**, lo que permite su despliegue tanto en servidores como en navegadores web mediante ONNX Runtime Web.

---

## 2. Especificaciones del Modelo

### 2.1 Tabla de Especificaciones

| Parámetro | Especificación |
| :--- | :--- |
| **Arquitectura** | CNN (Convolutional Neural Network) personalizada |
| **Formato del Modelo** | ONNX v12+ |
| **Tipo de Clasificación** | Binaria (Edificio / No Edificio) |
| **Función de Activación (Salida)** | Sigmoid |
| **Rango de Salida** | 0.0 – 1.0 (puntaje de confianza) |
| **Precisión de Datos** | float32 |
| **Función de Pérdida** | BCE (Binary Cross-Entropy) |
| **Optimizador** | Adam (lr=0.001) |
| **Scheduler** | ReduceLROnPlateau (patience=3, factor=0.5) |

### 2.2 Estructura de Tensores de Entrada/Salida (I/O)

```
┌────────────────────────────────────────────────────────────┐
│                    TENSOR DE ENTRADA                       │
├────────────┬───────────────────────────────────────────────┤
│ Nombre     │ "cam_input"                                  │
│ Shape      │ [1, 224, 224, 3]                             │
│ Orden      │ NHWC (Batch, Height, Width, Channels)        │
│ Tipo       │ float32                                      │
│ Valores    │ Normalizados entre 0.0 y 1.0                 │
└────────────┴───────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────┐
│                    TENSOR DE SALIDA                        │
├────────────┬───────────────────────────────────────────────┤
│ Nombre     │ "confidence_score"                           │
│ Shape      │ [1, 1]                                       │
│ Tipo       │ float32                                      │
│ Valores    │ 0.0 (No edificio) ↔ 1.0 (Edificio)          │
│ Umbral     │ > 0.5 = Edificio                             │
└────────────┴───────────────────────────────────────────────┘
```

### 2.3 Dimensiones de la Imagen

- **Resolución de entrada**: 224 × 224 píxeles
- **Canales de color**: 3 (RGB)
- **Preprocesamiento requerido**: Redimensionar a 224×224, normalizar píxeles a rango [0.0, 1.0]

---

## 3. Arquitectura de la CNN

### 3.1 Diagrama de Capas

```
Input [1, 224, 224, 3]
        │
        ▼ (permute → NCHW)
   ┌─────────────────────────────────┐
   │  Conv2d(3→32, 3×3, pad=1)      │  ← Detección de bordes y texturas básicas
   │  ReLU                           │
   │  MaxPool2d(2×2)                 │  → [32, 112, 112]
   ├─────────────────────────────────┤
   │  Conv2d(32→64, 3×3, pad=1)     │  ← Patrones geométricos (ventanas, líneas)
   │  ReLU                           │
   │  MaxPool2d(2×2)                 │  → [64, 56, 56]
   ├─────────────────────────────────┤
   │  Conv2d(64→128, 3×3, pad=1)    │  ← Estructuras complejas (fachadas, techos)
   │  ReLU                           │
   │  AdaptiveAvgPool2d(7×7)        │  → [128, 7, 7]
   └─────────────┬───────────────────┘
                  │ Flatten → [6272]
                  ▼
   ┌─────────────────────────────────┐
   │  Linear(6272 → 512)            │
   │  ReLU                           │
   │  Dropout(0.5)                   │  ← Regularización contra overfitting
   │  Linear(512 → 1)               │
   │  Sigmoid                        │  → Probabilidad [0.0, 1.0]
   └─────────────────────────────────┘
                  │
                  ▼
        Output [1, 1] (confidence_score)
```

### 3.2 Parámetros del Modelo

| Capa | Parámetros | Descripción |
| :--- | ---: | :--- |
| Conv2d (3→32) | 896 | Extracción de características de bajo nivel |
| Conv2d (32→64) | 18,496 | Detección de patrones intermedios |
| Conv2d (64→128) | 73,856 | Reconocimiento de estructuras complejas |
| Linear (6272→512) | 3,211,776 | Clasificador denso |
| Linear (512→1) | 513 | Capa de decisión |
| **Total** | **~3.3M** | Modelo compacto, ideal para despliegue web |

### 3.3 Justificación de la Arquitectura

- **3 bloques convolucionales**: Suficientes para capturar características arquitectónicas (bordes rectos, simetría, repetición de ventanas).
- **AdaptiveAvgPool2d**: Garantiza un tamaño de salida fijo independientemente de variaciones menores en la entrada.
- **Dropout(0.5)**: Previene el sobreajuste en datasets pequeños-medianos.
- **Sigmoid en salida**: Clasificación binaria directa con puntaje de confianza interpretable.

---

## 4. Estrategia de Entrenamiento

### 4.1 División del Dataset (Data Split)

```
     Dataset Total
    ┌───────────────────────────────────────┐
    │         80% Training                  │ 20% Test │
    └───────────────────────────────┬───────┴──────────┘
                                    │
                              Split aleatorio
                           (torch random_split)
```

| Conjunto | Porcentaje | Propósito |
| :--- | :---: | :--- |
| **Training** | 80% | Ajuste de pesos mediante backpropagation |
| **Test** | 20% | Evaluación final y medición de generalización |

### 4.2 Data Augmentation (Solo en Training)

| Transformación | Parámetro | Justificación |
| :--- | :--- | :--- |
| RandomHorizontalFlip | p=0.5 | Edificios simétricos: invariancia horizontal |
| RandomRotation | ±15° | Compensar ángulos de cámara |
| ColorJitter | brightness/contrast/saturation ±0.2 | Robustez ante condiciones de iluminación |
| Resize | 224×224 | Normalización de dimensiones |
| ToTensor + Normalize | [0.0, 1.0] | Compatibilidad con la red |

### 4.3 Hiperparámetros

| Hiperparámetro | Valor | Notas |
| :--- | :--- | :--- |
| Learning Rate | 0.001 | Estándar para Adam |
| Batch Size | 32 | Balance entre velocidad y estabilidad |
| Épocas | 25 | Con early-stopping implícito (best model) |
| Optimizador | Adam | Convergencia rápida, adapta learning rate |
| Scheduler | ReduceLROnPlateau | Reduce LR cuando la pérdida deja de mejorar |

---

## 5. Métricas de Evaluación

El sistema reporta las siguientes métricas sobre el conjunto de test:

| Métrica | Fórmula | Significado |
| :--- | :--- | :--- |
| **Accuracy** | (TP+TN) / Total | Porcentaje de predicciones correctas |
| **Precision** | TP / (TP+FP) | De los que predijo "Edificio", ¿cuántos lo eran realmente? |
| **Recall** | TP / (TP+FN) | De los edificios reales, ¿cuántos detectó? |
| **F1 Score** | 2·(P·R)/(P+R) | Media armónica entre Precision y Recall |

Adicionalmente se genera:
- **Matriz de Confusión** → visualización gráfica (PNG)
- **Curvas de Pérdida y Precisión** → evolución por época (PNG)
- **Reporte JSON** → métricas completas exportables

---

## 6. Pipeline de Despliegue

### 6.1 Flujo Completo

```
 ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
 │  Dataset  │ →  │  Train   │ →  │  ONNX    │ →  │  Deploy  │
 │  (imgs)   │    │  (PyTorch)│   │  Export  │    │  (Web)   │
 └──────────┘    └──────────┘    └──────────┘    └──────────┘
                      │               │               │
                 train.py        model_gen.py     index.html
                                                  + app.js
                                                  + ort.min.js
```

### 6.2 Opciones de Despliegue

| Plataforma | Tecnología | Rendimiento |
| :--- | :--- | :--- |
| **Navegador Web** | ONNX Runtime Web (WebAssembly) | ~50-100ms/inferencia |
| **Servidor Python** | ONNX Runtime (CPU/GPU) | ~10-30ms/inferencia |
| **Móvil (Android/iOS)** | ONNX Runtime Mobile | ~30-80ms/inferencia |
| **Edge Devices** | ONNX Runtime (ARM) | ~50-150ms/inferencia |

---

## 7. Estructura del Proyecto

```
reconocedor-de-imagenes/
├── index.html              # Interfaz web del reconocedor
├── style.css               # Estilos premium de la UI
├── app.js                  # Lógica de inferencia en navegador
├── model_generator.py      # Definición de la arquitectura CNN + exportación rápida
├── train.py                # Pipeline completo de entrenamiento
├── predict.py              # Predicción individual por línea de comandos
├── requirements.txt        # Dependencias de Python
├── INFORME_ANALISIS.md     # Este documento
├── README.md               # Guía de uso del proyecto
└── output/                 # (generado después del entrenamiento)
    ├── best_model.pth      # Pesos del mejor modelo PyTorch
    ├── building_model.onnx # Modelo exportado a ONNX
    ├── training_history.png# Gráficos de entrenamiento
    ├── confusion_matrix.png# Matriz de confusión
    └── training_report.json# Reporte de métricas
```

---

## 8. Requisitos del Sistema

### 8.1 Para Entrenamiento
- Python 3.8+
- PyTorch 2.0+
- GPU (opcional, recomendada para datasets grandes)
- 4GB+ RAM

### 8.2 Para Inferencia Web
- Navegador moderno con soporte WebAssembly (Chrome 80+, Firefox 78+, Safari 14+)
- Cámara web (opcional, para modo en vivo)

---

## 9. Conclusiones y Recomendaciones

1. **Arquitectura adecuada**: La CNN de 3 capas convolucionales con ~3.3M parámetros ofrece un balance óptimo entre precisión y velocidad para clasificación binaria.

2. **Formato ONNX**: Permite despliegue multiplataforma sin dependencia de framework específico.

3. **Escalabilidad**: El diseño modular permite fácilmente:
   - Añadir más clases (multi-label con softmax)
   - Migrar a arquitecturas más potentes (ResNet, EfficientNet)
   - Integrar transfer learning con pesos pre-entrenados

4. **Recomendación de Dataset**: Para resultados óptimos se recomienda un mínimo de **1,000 imágenes por clase** (2,000 total) con variedad en:
   - Tipos de edificios (residenciales, comerciales, industriales)
   - Condiciones de iluminación (día, noche, nublado)
   - Ángulos de captura (frontal, lateral, aéreo)

---

*Preparado por: Antigravity AI Assistant*
*Fecha: 15 de mayo de 2026*
*Versión: 2.0*
