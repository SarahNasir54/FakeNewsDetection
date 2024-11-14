from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

class NewsInput(BaseModel):
    text: str

@app.post("/fake-news-check")
async def fake_news_check(news: NewsInput):
    # Here you would add your fake news detection logic
    # For now, let's use a placeholder that says any text containing "fake" is fake news.

    if "fake" in news.text.lower():
        return {"isFake": True, "message": "This is likely fake news."}
    else:
        return {"isFake": False, "message": "This appears to be real news."}
