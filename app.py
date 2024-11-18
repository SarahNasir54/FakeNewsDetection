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
MODEL = "roberta-base"  # Use RoBERTa base model
clf = pipeline(
    "text-classification",
    model=MODEL,
    tokenizer=MODEL,
    return_all_scores=True,  # Include this to see probabilities for all labels
)

# Label mapping
label_map = {
    0: "Fake",
    1: "Real",
}

# Define request body structure
class TextRequest(BaseModel):
    text: str

# Define endpoint for fake news detection
@app.post("/predict")
async def predict(request: TextRequest):
    result = clf(request.text)[0]  # Only process the first result
    print("Debugging Output:", result)  # Inspect raw model output

    # Select label with the highest score
    best_prediction = max(result, key=lambda x: x["score"])
    label = label_map.get(best_prediction["label"], "Unknown")
    score = best_prediction["score"]

    return {"label": label, "score": score}

# Run the app with uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
