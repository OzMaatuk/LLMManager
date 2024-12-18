# server.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from llm_utils.manager import LLMUtils

app = FastAPI()

class GenerateRequest(BaseModel):
    model_name: str
    prompt: str

@app.post("/generate")
async def generate_text(request: GenerateRequest):
    model_name = request.model_name
    prompt = request.prompt

    if not model_name:
        raise HTTPException(status_code=400, detail="Model name is required")

    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt is required")

    try:
        # Initialize the LLMUtils with the specified model
        llm_utils = LLMUtils(model_name)
        generated_text = llm_utils.generate_text(prompt)
        
        if generated_text:
            return {"generated_text": generated_text}
        else:
            raise HTTPException(status_code=500, detail="Failed to generate text")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == '__main__':
    load_dotenv()
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=5000)