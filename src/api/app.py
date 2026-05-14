from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.core.pipeline import Pipeline
from src.core.logger import get_logger
from dotenv import load_dotenv
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
load_dotenv(ROOT_DIR / ".env")

class GenerateRequest(BaseModel):
    user_id: str
    prompt: str

    class Config:
        arbitrary_types_allowed = True 


class GenerateResponse(BaseModel):
    output: str

    class Config:
        arbitrary_types_allowed = True 

app = FastAPI()
logger = get_logger()
pipeline = Pipeline()

@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    try:
        out = pipeline.run(req.user_id, req.prompt)
        return GenerateResponse(output=out)
    except Exception as e:
        logger.error(f"Error in pipeline: {e}")
        raise HTTPException(status_code=400, detail=str(e))
