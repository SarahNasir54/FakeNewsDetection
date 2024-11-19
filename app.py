import requests
from bs4 import BeautifulSoup
from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
from torchvision import models, transforms
from PIL import Image
import io
from datetime import datetime, timedelta
import asyncio
import logging

# Initialize FastAPI app
app = FastAPI()

# Set up logging
logging.basicConfig(level=logging.DEBUG)

# Mount the frontend folder to serve index.html
app.mount("/frontend", StaticFiles(directory="frontend"), name="frontend")

# Root endpoint to serve the index.html from frontend folder
@app.get("/")
async def read_root():
    return FileResponse("frontend/index.html")

# Simple cache for user profiles (can be replaced with a database)
PROFILE_CACHE = {}
CACHE_EXPIRY = timedelta(days=1)  # Cache expiry time, e.g., 1 day

# Function to scrape and cache the user profile
async def scrape_user_profile(url: str):
    current_time = datetime.now()
    if url in PROFILE_CACHE:
        cached_data = PROFILE_CACHE[url]
        # Check if the cached data has expired
        if current_time - cached_data['timestamp'] < CACHE_EXPIRY:
            return cached_data['profile']

    try:
        # Fetch the page content
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')

        # Example of extracting basic user profile data
        name = soup.find('h1', class_='profile-name').text.strip() if soup.find('h1', class_='profile-name') else 'Not Available'
        email = soup.find('a', class_='profile-email').text.strip() if soup.find('a', class_='profile-email') else 'Not Available'
        joined = soup.find('span', class_='profile-joined').text.strip() if soup.find('span', class_='profile-joined') else 'Not Available'

        # Cache the scraped data
        PROFILE_CACHE[url] = {
            'profile': {"name": name, "email": email, "joined": joined},
            'timestamp': current_time
        }

        return {"name": name, "email": email, "joined": joined}
    except Exception as e:
        logging.error(f"Error scraping profile: {e}")
        return {"error": f"Failed to scrape user profile: {str(e)}"}

# Use transformers to detect fake news from text
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
fake_news_classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)

# Pretrained model for image classification
transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

# Predict endpoint for checking fake news and processing user profile
@app.post("/predict")
async def predict(
    profile_url: str = Form(...),
    text: str = Form(None),
    file: UploadFile = File(None),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    # Add scraping to background task
    background_tasks.add_task(scrape_user_profile, profile_url)

    # Just a placeholder while scraping is done in the background
    user_profile = {"name": "Loading...", "email": "Loading...", "joined": "Loading..."}

    # Predict fake news
    fake_news_result = fake_news_classifier(text)
    final_label = fake_news_result[0]['label']
    final_score = fake_news_result[0]['score']

    return {
        "user_profile": user_profile,
        "final_label": final_label,
        "final_score": final_score
    }
