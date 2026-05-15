import torch
import torch.nn as nn
import torch.onnx

class BuildingRecognizer(nn.Module):
    def __init__(self):
        super(BuildingRecognizer, self).__init__()
        # Arquitectura CNN básica pero robusta
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((7, 7))
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 7 * 7, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # El input es NHWC [1, 224, 224, 3] pero PyTorch usa NCHW [1, 3, 224, 224]
        # Hacemos el permute para adaptarnos al requerimiento del usuario
        x = x.permute(0, 3, 1, 2) 
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

def export_model():
    model = BuildingRecognizer()
    model.eval()

    # Input dummy siguiendo la especificación [1, 224, 224, 3]
    dummy_input = torch.randn(1, 224, 224, 3)
    
    input_names = ["cam_input"]
    output_names = ["confidence_score"]

    torch.onnx.export(
        model,
        dummy_input,
        "building_model.onnx",
        verbose=True,
        input_names=input_names,
        output_names=output_names,
        opset_version=18
    )
    print("Modelo exportado exitosamente como 'building_model.onnx'")

if __name__ == "__main__":
    export_model()
