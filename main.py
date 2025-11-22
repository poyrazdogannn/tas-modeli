from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import numpy as np
from PIL import Image
import tensorflow as tf
import pydicom
import io

app = FastAPI()

templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

IMG_SIZE = (224, 224)
CLASS_LABELS = ["yok", "var"]
MODEL_PATH = "smallcnn_224_best.tflite"

interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


def load_dicom_bytes(data: bytes):
    ds = pydicom.dcmread(io.BytesIO(data))
    arr = ds.pixel_array.astype(np.float32)
    arr = arr - np.min(arr)
    if np.max(arr) != 0:
        arr = arr / np.max(arr)
    arr = (arr * 255).astype(np.uint8)
    img = Image.fromarray(arr).convert("RGB")
    return img


def load_image_bytes(data: bytes):
    img = Image.open(io.BytesIO(data)).convert("RGB")
    return img


def preprocess(img):
    img = img.resize(IMG_SIZE)
    arr = np.array(img).astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr


def predict(arr):
    input_index = input_details[0]["index"]
    output_index = output_details[0]["index"]
    interpreter.set_tensor(input_index, arr.astype(input_details[0]["dtype"]))
    interpreter.invoke()
    output = interpreter.get_tensor(output_index)[0]
    e = np.exp(output - np.max(output))
    probs = e / e.sum()
    idx = int(np.argmax(probs))
    return CLASS_LABELS[idx], float(probs[idx]), probs.tolist()


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict")
async def predict_view(request: Request, file: UploadFile = File(...)):
    filename = file.filename.lower()

    # Dosyayı bayt olarak oku (async!)
    file_bytes = await file.read()

    # Uzantıya göre DICOM mu, normal resim mi karar ver
    if filename.endswith(".dcm"):
        img = load_dicom_bytes(file_bytes)
    else:
        img = load_image_bytes(file_bytes)

    # Model için hazırlık
    arr = preprocess(img)
    label, prob, probs = predict(arr)

    # Görüntüyü tekrar PNG'ye çevirip base64 olarak sayfada göstermek için
    img_bytes = io.BytesIO()
    img.save(img_bytes, format="PNG")
    image_base = img_bytes.getvalue()

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "result": label,
            "probability": f"%{prob*100:.1f}",
            "image_data": image_base,
        },
    )
