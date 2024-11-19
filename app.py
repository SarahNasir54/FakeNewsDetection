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
