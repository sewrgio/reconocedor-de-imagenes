"""
predict.py — Utilidad de inferencia para clasificar imágenes individuales.

Uso:
    python predict.py --model ./output/building_model.onnx --image ./test.jpg
"""

import argparse
import numpy as np
from PIL import Image

try:
    import onnxruntime as ort
except ImportError:
    print("Instala onnxruntime: pip install onnxruntime")
    exit(1)


def preprocess_image(image_path: str) -> np.ndarray:
    """
    Carga y preprocesa una imagen para el modelo.
    Retorna un tensor numpy de shape [1, 224, 224, 3] float32.
    """
    img = Image.open(image_path).convert('RGB')
    img = img.resize((224, 224), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    # Expandir para batch dimension → [1, 224, 224, 3]
    return np.expand_dims(arr, axis=0)


def predict(model_path: str, image_path: str) -> dict:
    """
    Ejecuta inferencia sobre una imagen y retorna el resultado.
    """
    session = ort.InferenceSession(model_path)
    input_tensor = preprocess_image(image_path)

    feeds = {"cam_input": input_tensor}
    result = session.run(["confidence_score"], feeds)
    confidence = float(result[0][0][0])

    is_building = confidence > 0.5
    label = "Edificio" if is_building else "No Edificio"

    return {
        "image": image_path,
        "label": label,
        "confidence": confidence,
        "confidence_pct": f"{confidence * 100:.2f}%"
    }


def main():
    parser = argparse.ArgumentParser(description="Predicción con modelo ONNX de edificios")
    parser.add_argument('--model', type=str, required=True, help='Ruta al modelo .onnx')
    parser.add_argument('--image', type=str, required=True, help='Ruta a la imagen a clasificar')
    args = parser.parse_args()

    result = predict(args.model, args.image)

    print(f"\n{'='*40}")
    print(f"  RESULTADO DE PREDICCIÓN")
    print(f"{'='*40}")
    print(f"  Imagen:     {result['image']}")
    print(f"  Predicción: {result['label']}")
    print(f"  Confianza:  {result['confidence_pct']}")
    print(f"{'='*40}\n")


if __name__ == "__main__":
    main()
