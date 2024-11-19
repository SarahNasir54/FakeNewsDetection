from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
from torchvision import models, transforms
from PIL import Image
import io

# Initialize FastAPI app
app = FastAPI()

# Mount the frontend folder to serve index.html
app.mount("/frontend", StaticFiles(directory="frontend"), name="frontend")

# Root endpoint to serve the index.html from frontend folder
@app.get("/")
async def read_root():
    return FileResponse("frontend/index.html")

# Mock user profile data (Replace with real database if available)
USER_PROFILES = {
    "1": {"name": "Alice", "email": "alice@example.com", "joined": "2023-01-15"},
    "2": {"name": "Bob", "email": "bob@example.com", "joined": "2022-11-20"},
    "3": {"name": "Charlie", "email": "charlie@example.com", "joined": "2021-08-05"},
}

# Load DistilBERT-based fake news detection model from Hugging Face
tokenizer = AutoTokenizer.from_pretrained("therealcyberlord/fake-news-classification-distilbert")
model = AutoModelForSequenceClassification.from_pretrained("therealcyberlord/fake-news-classification-distilbert")

# Use the Hugging Face pipeline for text classification
text_classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)

# Load image-based model (ResNet for image classification)
image_model = models.resnet50(pretrained=True)
image_model.eval()  # Set model to evaluation mode

# Image transforms to match ResNet input requirements
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Endpoint for multimodal fake news detection (text, image, and user profile)
@app.post("/predict")
async def predict(
    profile_id: str = Form(None),  # User profile ID
    text: str = Form(None),  # Accept text from form
    file: UploadFile = File(None)  # Accept file upload
):
    # Fetch user profile information
    user_profile = USER_PROFILES.get(profile_id)
    if not user_profile:
        return {"error": "User profile not found. Please provide a valid profile ID."}

    # Text prediction (if provided)
    text_label, text_score = None, None
    if text:
        text_result = text_classifier(text)
        text_label = text_result[0]['label']
        text_score = text_result[0]['score']

    # Image prediction (if provided)
    image_label, image_score = None, None
    if file:
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        image_tensor = transform(image).unsqueeze(0)
        with torch.no_grad():
            image_output = image_model(image_tensor)
            image_probs = torch.nn.functional.softmax(image_output[0], dim=0)
            image_label = image_probs.argmax().item()
            image_score = image_probs.max().item()

    # Decision Logic
    if text_label == "LABEL_0" and image_label == 1:  # Both indicate fake
        final_label = "Fake"
        final_score = min(text_score or 1, image_score or 1)
    else:
        final_label = "Real"
        final_score = max(text_score or 0, image_score or 0)

    return {
        "user_profile": user_profile,
        "final_label": final_label,
        "final_score": final_score,
        "text_prediction": {"label": text_label, "score": text_score},
        "image_prediction": {"label": image_label, "score": image_score},
    }

# Run the app with uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
