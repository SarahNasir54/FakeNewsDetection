import uvicorn
from fastapi import FastAPI  # Import the FastAPI instance from app.py
app = FastAPI()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
