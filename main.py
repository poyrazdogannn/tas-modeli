from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

import numpy as np
from PIL import Image
import pydicom
import tensorflow as tf
import io
import base64

app = FastAPI()

# Jinja2 template ve statik dosya ayarı
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# -------- MODEL AYARLARI --------
IMG_SIZE = (224, 224)
CLASS_LABELS = ["yok", "var"]   # sıralamayı kendi modeline göre gerekirse değiştir
MODEL_PATH = "smallcnn_224_best.tflite"

# TFLite modeli yükle
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


# -------- YARDIMCI FONKSİYONLAR --------

def load_dicom(uploaded: UploadFile) -> Image.Image:
    """
    DICOM dosyasını UploadFile'den okuyup PIL Image'e çevirir.
    """
    uploaded.file.seek(0)  # dosya imlecini başa al
    ds = pydicom.dcmread(uploaded.file)
    arr = ds.pixel_array.astype(np.float32)

    # normalize 0–255
    arr = arr - np.min(arr)
    if np.max(arr) != 0:
        arr = arr / np.max(arr)
    arr = (arr * 255).astype(np.uint8)

    img = Image.fromarray(arr).convert("RGB")
    return img


def load_image(uploaded: UploadFile) -> Image.Image:
    """
    JPG/PNG gibi normal görüntü dosyalarını PIL Image'e çevirir.
    """
    uploaded.file.seek(0)
    img = Image.open(uploaded.file).convert("RGB")
    return img


def preprocess(img: Image.Image) -> np.ndarray:
    """
    Görüntüyü modele uygun boyuta ve formata getirir.
    """
    img = img.resize(IMG_SIZE)
    arr = np.array(img).astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr


def predict(arr: np.ndarray):
    """
    TFLite interpreter ile tahmin yapar.
    """
    input_index = input_details[0]["index"]
    output_index = output_details[0]["index"]

    # giriş tipini input_details'e göre ayarla
    arr = arr.astype(input_details[0]["dtype"])
    interpreter.set_tensor(input_index, arr)
    interpreter.invoke()
    output = interpreter.get_tensor(output_index)[0]

    # Softmax ile olasılık
    e = np.exp(output - np.max(output))
    probs = e / e.sum()

    idx = int(np.argmax(probs))
    return CLASS_LABELS[idx], float(probs[idx]), probs.tolist()


# -------- ROUTES --------

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """
    Arayüzü gösteren ana sayfa.
    """
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict")
async def predict_view(request: Request, file: UploadFile = File(...)):
    """
    Yüklenen dosyadan (DICOM veya resim) tahmin yapan endpoint.
    """
    filename = (file.filename or "").lower()

    # Dosya tipine göre okuma
    if filename.endswith(".dcm"):
        img = load_dicom(file)
    else:
        img = load_image(file)

    # Preprocess + tahmin
    arr = preprocess(img)
    label, prob, probs = predict(arr)

    # Görüntüyü base64 string'e çevir (HTML'de göstermek için)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    img_bytes = buf.getvalue()
    img_b64 = base64.b64encode(img_bytes).decode("utf-8")

    context = {
        "request": request,
        "result": label,
        "probability": f"%{prob * 100:.1f}",
        "image_data": img_b64,
    }
    return templates.TemplateResponse("index.html", context)

