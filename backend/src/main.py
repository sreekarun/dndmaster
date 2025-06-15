from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import Optional

app = FastAPI(
    title="AI Dungeon & Dragons Master",
    description="A FastAPI backend for an AI-powered D&D master using OpenAI APIs.",
    version="0.1.0"
)

class PlayerAction(BaseModel):
    player_name: str
    action: str
    context: Optional[str] = None

@app.get("/")
async def root():
    return {"message": "Welcome to the AI Dungeon & Dragons Master API!"}

@app.post("/dnd/action")
async def process_action(action: PlayerAction):
    # Placeholder for OpenAI integration
    # In production, call your OpenAI API logic here
    response = {
        "narration": f"{action.player_name} attempts: {action.action}. (AI response would go here.)",
        "context": action.context
    }
    return response