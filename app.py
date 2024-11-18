from fastapi import FastAPI, UploadFile, File, Form
from pydantic import BaseModel
from transformers import pipeline
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io

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

# Load the text model
MODEL = "jy46604790/Fake-News-Bert-Detect"
clf = pipeline("text-classification", model=MODEL, tokenizer=MODEL)

# Label mapping for text detection
label_map = {
    "LABEL_0": "Fake",
    "LABEL_1": "Real"
}

# Dummy function for image fake detection
def detect_fake_image(image: Image.Image):
    """
    Mock function to detect fake images.
    Replace this with an actual model or detection algorithm.
    """
    # For now, we return fake for demonstration
    return "Fake", 0.99

# Define endpoint for fake news detection
@app.post("/predict")
async def predict(text: str = Form(None), image: UploadFile = File(None)):
    if text:
        # Process text input
        result = clf(text)
        print("Text Debugging Output:", result)  # Inspect raw model output
        label = label_map.get(result[0]["label"], "Unknown")
        score = result[0]["score"]
        return {"label": label, "score": score}

    elif image:
        # Process image input
        try:
            # Open the image using PIL
            image_data = await image.read()
            pil_image = Image.open(io.BytesIO(image_data))
            
            # Analyze the image (replace with actual logic/model)
            label, score = detect_fake_image(pil_image)
            return {"label": label, "score": score}
        except Exception as e:
            return {"error": f"Failed to process image: {str(e)}"}

    # If neither text nor image is provided
    return {"error": "Please provide either text or an image for prediction."}

# Run the app with uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
