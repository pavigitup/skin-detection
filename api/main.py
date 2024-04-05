from fastapi import FastAPI,File, UploadFile
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000",
]

# Set up CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


MODEL = tf.keras.models.load_model(r".\model\4.keras")


CLASS_NAMES = ['Actinic keratosis',
 'Atopic Dermatitis',
 'Benign keratosis',
 'Dermatofibroma',
 'Melanocytic nevus',
 'Melanoma',
 'Squamous cell carcinoma',
 'Tinea Ringworm Candidiasis',
 'Vascular lesion']

@app.get("/ping")
async def ping():
    return "Hello, I am alive"

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):
    try:
        image = read_file_as_image(await file.read()) 
        img_batch = np.expand_dims(image, 0)

        predictions = MODEL.predict(img_batch)

        predicted_class = CLASS_NAMES[np.argmax(predictions)]
        confidence = np.max(predictions)
        return {
            'class': predicted_class,
            'confidence': float(confidence)
        }
    except Exception as e:
        return {"error": str(e)}


    

if __name__ == "__main__":
    uvicorn.run(app, host='localhost' , port=8080)