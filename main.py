from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()

# Define the input data model
class NewsInput(BaseModel):
    text: str

# Enable CORS to allow the front-end to communicate with the back-end
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows requests from any origin; modify this list for specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/fake-news-check")
async def fake_news_check(news: NewsInput):
    # Placeholder logic for fake news detection
    if "fake" in news.text.lower():
        return {"isFake": True, "message": "This is likely fake news."}
    else:
        return {"isFake": False, "message": "This appears to be real news."}
