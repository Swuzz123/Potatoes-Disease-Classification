from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn 
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf

app = FastAPI()

# Add CORS middleware to allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this in production to specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL =  tf.keras.models.load_model("D:\\Workspace\\project\\Potato Disease Classification\\potatoes.h5")
CLASS_NAMES = ['Early Blight', "Late Blight", "Healthy"]

@app.get("/ping")
async def ping():
    return "Hello World!"

def read_file_as_image(data):
    image = np.array(Image.open(BytesIO(data)))
    return image

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):
    
    image = read_file_as_image(await file.read()) # the shape is (W, H, channels)
    image_batch = np.expand_dims(image, 0) # convert the shape to (BATCH_SIZE, W, H, CHANNELS)
    
    prediction = MODEL.predict(image_batch)
    
    prediction_class = CLASS_NAMES[np.argmax(prediction[0])]
    confidence = np.max(prediction[0]) * 100
    
    return {
        'Prediction': prediction_class,
        'Confidence': float(confidence)
    }
    

if __name__ == '__main__':
    uvicorn.run(app, host= 'localhost', port= 8000)