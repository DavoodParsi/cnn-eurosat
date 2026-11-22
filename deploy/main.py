from io import BytesIO
import logging
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image, UnidentifiedImageError

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel


# --------------------------------------------
# Project Configuration
# --------------------------------------------

# Root directory of the deployment package
PROJECT_DIR = Path(__file__).resolve().parent

# Path to the final trained CNN model
MODEL_PATH = PROJECT_DIR / "model" / "high_final_model.keras"

# Maximum upload size: 5 MB
MAX_FILE_SIZE = 5 * 1024 * 1024

# Allowed image content types
ALLOWED_CONTENT_TYPES = {
    "image/jpeg",
    "image/png",
}


# --------------------------------------------
# Class Mapping
# --------------------------------------------

# Class order must match the training dataset order
CLASS_NAMES = [
    "AnnualCrop",
    "Forest",
    "HerbaceousVegetation",
    "Highway",
    "Industrial",
    "Pasture",
    "PermanentCrop",
    "Residential",
    "River",
    "SeaLake",
]


# --------------------------------------------
# Logging Configuration
# --------------------------------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# --------------------------------------------
# Model Loading
# --------------------------------------------

try:
    # Make sure the model file exists
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

    # Load the model once at startup
    model = tf.keras.models.load_model(MODEL_PATH)

    # Validate input shape: (batch, height, width, channels)
    if len(model.input_shape) != 4:
        raise ValueError("Invalid model input shape.")

    IMAGE_HEIGHT = int(model.input_shape[1])
    IMAGE_WIDTH = int(model.input_shape[2])

    # The model must accept RGB images
    if model.input_shape[-1] != 3:
        raise ValueError("Model must accept RGB images.")

    # Output size must match the class list
    if model.output_shape[-1] != len(CLASS_NAMES):
        raise ValueError("Model output and class mapping mismatch.")

    # The final layer must use Softmax
    output_activation = getattr(
        model.layers[-1].activation,
        "__name__",
        None,
    )
    if output_activation != "softmax":
        raise ValueError("Model output must use Softmax.")

except Exception as exc:
    logger.exception("CNN model loading failed.")

    # Do not start the app without a valid model
    raise RuntimeError(
        "CNN model could not be loaded. Check MODEL_PATH."
    ) from exc


# --------------------------------------------
# FastAPI Application
# --------------------------------------------

app = FastAPI(
    title="EuroSAT CNN Inference API",
    description="Land-cover image classification API using a custom residual CNN.",
    version="1.0.0",
)


# --------------------------------------------
# Response Models
# --------------------------------------------

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


class PredictionResponse(BaseModel):
    class_name: str
    confidence: float


# --------------------------------------------
# Image Decoding
# --------------------------------------------

def decode_image(contents: bytes):
    """Decode raw bytes into an RGB PIL image."""
    try:
        with Image.open(BytesIO(contents)) as source_image:
            # Force a full decode to detect corrupted files
            source_image.load()
            return source_image.convert("RGB")

    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise HTTPException(
            status_code=400,
            detail={
                "code": "INVALID_IMAGE",
                "message": "Uploaded file is not a valid or readable image.",
            },
        ) from exc


# --------------------------------------------
# Image Preprocessing
# --------------------------------------------

def preprocess_image(image):
    """Resize the image and add a batch dimension.

    No /255 scaling here: the model contains a Rescaling layer.
    """
    image_array = np.asarray(image, dtype=np.float32)

    resized_image = tf.image.resize(
        image_array,
        [IMAGE_HEIGHT, IMAGE_WIDTH],
        method="bilinear",
    )

    image_batch = tf.expand_dims(resized_image, axis=0)

    return image_batch.numpy()


# --------------------------------------------
# CNN Inference
# --------------------------------------------

def run_inference(image_batch):
    """Run the model and return the top prediction."""
    try:
        predictions = model.predict(image_batch, verbose=0)
        prediction = predictions[0]

        # Validate output shape
        if prediction.shape != (len(CLASS_NAMES),):
            raise ValueError("Unexpected prediction output shape.")

        # Validate output values
        if not np.all(np.isfinite(prediction)):
            raise ValueError("Prediction contains invalid values.")

        # Softmax probabilities must sum to 1
        if not np.isclose(np.sum(prediction), 1.0, atol=1e-5):
            raise ValueError("Invalid Softmax probabilities.")

        predicted_index = int(np.argmax(prediction))

        return PredictionResponse(
            class_name=CLASS_NAMES[predicted_index],
            confidence=float(prediction[predicted_index]),
        )

    except Exception as exc:
        logger.exception("CNN inference failed.")

        # Generic message: internal details stay in the logs
        raise HTTPException(
            status_code=500,
            detail={
                "code": "INFERENCE_FAILED",
                "message": "Prediction could not be completed.",
            },
        ) from exc


# --------------------------------------------
# GET /health
# --------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Report that the API is running and the model is loaded."""
    return HealthResponse(status="ok", model_loaded=True)


# --------------------------------------------
# POST /predict
# --------------------------------------------

@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(image: UploadFile = File(...)):
    """Classify an uploaded JPEG/PNG image."""
    try:
        # Check the declared content type
        if image.content_type not in ALLOWED_CONTENT_TYPES:
            raise HTTPException(
                status_code=415,
                detail={
                    "code": "UNSUPPORTED_MEDIA_TYPE",
                    "message": "Only JPEG and PNG images are supported.",
                },
            )

        # Read at most MAX_FILE_SIZE + 1 bytes to detect oversized files
        contents = await image.read(MAX_FILE_SIZE + 1)

        if len(contents) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail={
                    "code": "FILE_TOO_LARGE",
                    "message": "Uploaded file exceeds the maximum allowed size.",
                },
            )

        if not contents:
            raise HTTPException(
                status_code=400,
                detail={
                    "code": "EMPTY_FILE",
                    "message": "Uploaded file is empty.",
                },
            )

        decoded_image = decode_image(contents)

        try:
            image_batch = preprocess_image(decoded_image)
        finally:
            decoded_image.close()

        return run_inference(image_batch)

    finally:
        await image.close()