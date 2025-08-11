from fastapi import FastAPI
import logging
import sys

# Add local modules to path
sys.path.append('.')

from api.endpoints import router, ensemble_model
from training.trainer import ModelTrainer
from config import Config

# Configure logging
logging.basicConfig(level=getattr(logging, Config.LOG_LEVEL))
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title=Config.API_TITLE,
    description=Config.API_DESCRIPTION,
    version=Config.API_VERSION
)

# Include API routes
app.include_router(router)

@app.on_event("startup")
async def startup_event():
    """Initialize the application on startup"""
    logger.info("Starting fraud detection service...")
    
    # Try to load existing model, train new one if not found
    if not ensemble_model.load_model():
        logger.info("No existing model found. Training new model...")
        trainer = ModelTrainer()
        trainer.train_model()
    else:
        logger.info("Existing model loaded successfully")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=Config.API_HOST, port=Config.API_PORT)