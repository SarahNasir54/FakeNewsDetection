from fastapi import FastAPI, HTTPException, File, UploadFile
from pydantic import BaseModel
from transformers import BertForSequenceClassification, BertTokenizer, pipeline
from typing import List

app = FastAPI()

# Load the pre-trained 'bert-base-cased' model and tokenizer
model_path = "bert-base-cased"  # Using the 'bert-base-cased' pre-trained model from HuggingFace
text_model = BertForSequenceClassification.from_pretrained(model_path)
tokenizer = BertTokenizer.from_pretrained(model_path)
text_classifier = pipeline("text-classification", model=text_model, tokenizer=tokenizer)

# Sample user profiles
user_profiles = {
    1: {"id": 1, "name": "Alice", "email": "alice@example.com", "age": 25, "bio": "AI enthusiast.", "followers": 100, "following": 150},
    2: {"id": 2, "name": "Bob", "email": "bob@example.com", "age": 30, "bio": "Photographer.", "followers": 200, "following": 180},
    3: {"id": 3, "name": "Charlie", "email": "charlie@example.com", "age": 35, "bio": "Gamer.", "followers": 500, "following": 200},
}

# Models for API
class TextRequest(BaseModel):
    text: str

# Endpoint to fetch user profile
@app.get("/profiles/{profile_id}")
def get_profile(profile_id: int):
    if profile_id not in user_profiles:
        raise HTTPException(status_code=404, detail="Profile not found")
    return user_profiles[profile_id]

# Endpoint for fake news detection (using the 'bert-base-cased' model)
@app.post("/predict")
async def predict(request: TextRequest):
    text_result = text_classifier(request.text)
    label = text_result[0]["label"]
    score = text_result[0]["score"]

    return {
        "final_label": "Fake" if label == "LABEL_0" else "Real",
        "final_score": score,
    }

# Endpoint to handle file uploads
@app.post("/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    file_names = []
    for file in files:
        # Save the file or perform any other necessary actions
        file_names.append(file.filename)  # Just appending filename as example
    return {"files": file_names}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
from fastapi import FastAPI, UploadFile, File, Form  # Added Form
from pydantic import BaseModel
from transformers import pipeline, BertTokenizer, BertForSequenceClassification
import torch
from torchvision import models, transforms
from PIL import Image
import io

# Initialize FastAPI app
app = FastAPI()

# Load text-based model (BERT for fake news detection)
text_model = BertForSequenceClassification.from_pretrained("bert-base-uncased")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
text_classifier = pipeline("text-classification", model=text_model, tokenizer=tokenizer)

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

# Define request body structure
class TextRequest(BaseModel):
    text: str

# Endpoint for multimodal fake news detection (text and image)
@app.post("/predict")
async def predict(
    text: str = Form(None),  # Accept text from form
    file: UploadFile = File(None)  # Accept file upload
):
    if not text and not file:
        return {"error": "Please provide either text or an image."}

    # Text prediction (if provided)
    text_result = None
    if text:
        text_result = text_classifier(text)
        text_label = text_result[0]['label']
        text_score = text_result[0]['score']
    else:
        text_label = None
        text_score = None

    # Image prediction (if provided)
    image_result = None
    if file:
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        image_tensor = transform(image).unsqueeze(0)
        with torch.no_grad():
            image_output = image_model(image_tensor)
            image_probs = torch.nn.functional.softmax(image_output[0], dim=0)
            image_label = image_probs.argmax().item()
            image_score = image_probs.max().item()
    else:
        image_label = None
        image_score = None

    # Decision Logic
    if text_label == "LABEL_0" and image_label == 1:  # Both indicate fake
        final_label = "Fake"
        final_score = min(text_score or 1, image_score or 1)
    else:
        final_label = "Real"
        final_score = max(text_score or 0, image_score or 0)

    return {
        "final_label": final_label,
        "final_score": final_score,
        "text_prediction": {"label": text_label, "score": text_score},
        "image_prediction": {"label": image_label, "score": image_score},
    }

# Run the app with uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)