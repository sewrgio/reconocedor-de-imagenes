let session;
let video = document.getElementById('video');
let canvas = document.getElementById('canvas-preview');
let startBtn = document.getElementById('start-btn');
let captureBtn = document.getElementById('capture-btn');
let statusText = document.getElementById('status-text');
let modelLoader = document.getElementById('model-loader');
let modelStatus = document.getElementById('model-status');
let confidencePct = document.getElementById('confidence-pct');
let meterFill = document.getElementById('meter-fill');

const MODEL_PATH = './building_model.onnx';

// Cargar el modelo ONNX
async function loadModel() {
    try {
        // En un entorno real, el modelo debe existir en el servidor
        session = await ort.InferenceSession.create(MODEL_PATH);
        statusText.innerText = "Modelo Listo";
        modelLoader.classList.add('hidden');
        modelStatus.classList.add('active');
        console.log("ONNX Model loaded successfully");
    } catch (e) {
        statusText.innerText = "Error al cargar modelo (Asegúrate de generar el .onnx)";
        modelLoader.style.borderColor = "#ef4444";
        console.error("Failed to load model:", e);
    }
}

// Iniciar Cámara
startBtn.onclick = async () => {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment' } });
        video.srcObject = stream;
        captureBtn.disabled = false;
        startBtn.innerHTML = "<span>🔄</span> Cámara Activa";
    } catch (e) {
        alert("No se pudo acceder a la cámara.");
    }
};

// Capturar y Analizar
captureBtn.onclick = async () => {
    if (!session) {
        alert("El modelo aún no está cargado.");
        return;
    }

    // 1. Preprocesamiento: Capturar imagen del video y redimensionar a 224x224
    const ctx = canvas.getContext('2d');
    canvas.width = 224;
    canvas.height = 224;
    ctx.drawImage(video, 0, 0, 224, 224);

    // 2. Convertir imagen a Tensor [1, 224, 224, 3]
    const imageData = ctx.getImageData(0, 0, 224, 224).data;
    const float32Data = new Float32Array(224 * 224 * 3);

    // Ignoramos el canal Alpha y normalizamos (si el modelo lo requiere, aquí / 255.0)
    for (let i = 0, j = 0; i < imageData.length; i += 4) {
        float32Data[j++] = imageData[i] / 255.0;     // R
        float32Data[j++] = imageData[i + 1] / 255.0; // G
        float32Data[j++] = imageData[i + 2] / 255.0; // B
    }

    const inputTensor = new ort.Tensor('float32', float32Data, [1, 224, 224, 3]);

    // 3. Ejecutar Inferencia
    try {
        const feeds = { cam_input: inputTensor };
        const results = await session.run(feeds);
        const confidence = results.confidence_score.data[0];

        // 4. Mostrar Resultados
        updateUI(confidence);
    } catch (e) {
        console.error("Inference failed:", e);
    }
};

function updateUI(confidence) {
    const percentage = (confidence * 100).toFixed(2);
    confidencePct.innerText = `${percentage}%`;
    meterFill.style.width = `${percentage}%`;
    
    // Cambiar color basado en confianza
    if (confidence > 0.7) {
        meterFill.style.background = "linear-gradient(to right, #10b981, #34d399)";
    } else {
        meterFill.style.background = "linear-gradient(to right, #6366f1, #818cf8)";
    }
}

loadModel();
