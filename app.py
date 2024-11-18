from fastapi import FastAPI, UploadFile, File
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
async def predict(request: TextRequest, file: UploadFile = File(...)):
    # Text prediction
    text_result = text_classifier(request.text)
    text_label = text_result[0]['label']
    text_score = text_result[0]['score']
    
    # Image prediction
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data))  # Open the image
    image_tensor = transform(image).unsqueeze(0)  # Apply transforms and add batch dimension
    
    with torch.no_grad():
        image_output = image_model(image_tensor)
        image_probs = torch.nn.functional.softmax(image_output[0], dim=0)
        image_label = image_probs.argmax().item()
        image_score = image_probs.max().item()

    # Combine text and image results for final decision
    # Simple logic: if both predict fake, classify as fake
    if text_label == "LABEL_0" and image_label == 1:  # Fake and Image shows 'fake'
        final_label = "Fake"
        final_score = min(text_score, image_score)
    else:
        final_label = "Real"
        final_score = max(text_score, image_score)

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
