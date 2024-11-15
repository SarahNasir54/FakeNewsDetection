from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline
from fastapi.middleware.cors import CORSMiddleware

# Initialize FastAPI app
app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the model
MODEL = "jy46604790/Fake-News-Bert-Detect"
clf = pipeline("text-classification", model=MODEL, tokenizer=MODEL)

# Label mapping
label_map = {
    "LABEL_0": "Fake",
    "LABEL_1": "Real"
}

# Define request body structure
class TextRequest(BaseModel):
    text: str

# Define endpoint for fake news detection
@app.post("/predict")
async def predict(request: TextRequest):
    result = clf(request.text)
    # Map model label to readable label
    label = label_map.get(result[0]["label"], "Unknown")
    score = result[0]["score"]
    return {"label": label, "score": score}

# Run the app with uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
