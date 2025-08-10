# Your First AI Feature

Time to make your first real AI integration! We'll modify the D&D Master application to generate actual AI responses instead of placeholder text.

## Learning Objectives

By the end of this tutorial, you'll have:
- Integrated OpenAI API into the backend
- Created dynamic AI responses for player actions
- Tested your AI integration
- Understood error handling basics

## Step 1: Set Up OpenAI Integration

### Install Required Dependencies

The `openai` package should already be in your `requirements.txt`, but let's make sure it's installed:

```bash
cd backend
pip install openai python-dotenv
```

### Create Your Environment File

Create a `.env` file in the `backend` directory:

```bash
# backend/.env
OPENAI_API_KEY=your_actual_api_key_here
OPENAI_MODEL=gpt-3.5-turbo
MAX_TOKENS=200
TEMPERATURE=0.8
```

**Important**: Replace `your_actual_api_key_here` with your real OpenAI API key!

## Step 2: Modify the Backend Code

Let's update `backend/src/main.py` to include real AI integration:

```python
from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
from typing import Optional
import openai
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = FastAPI(
    title="AI Dungeon & Dragons Master",
    description="A FastAPI backend for an AI-powered D&D master using OpenAI APIs.",
    version="0.1.0"
)

# Initialize OpenAI client
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class PlayerAction(BaseModel):
    player_name: str
    action: str
    context: Optional[str] = None

@app.get("/")
async def root():
    return {"message": "Welcome to the AI Dungeon & Dragons Master API!"}

@app.post("/dnd/action")
async def process_action(action: PlayerAction):
    try:
        # Create the prompt for the AI
        system_prompt = """You are an experienced Dungeon Master for a D&D game. 
        Respond to player actions with engaging, descriptive narratives. 
        Keep responses exciting but appropriate for teenagers. 
        Include dice roll results when relevant."""
        
        user_prompt = f"""
        Player: {action.player_name}
        Action: {action.action}
        Context: {action.context or "No additional context"}
        
        Describe what happens as a result of this action in 2-3 sentences.
        """
        
        # Call OpenAI API
        response = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=int(os.getenv("MAX_TOKENS", 200)),
            temperature=float(os.getenv("TEMPERATURE", 0.8))
        )
        
        ai_narration = response.choices[0].message.content.strip()
        
        return {
            "narration": ai_narration,
            "context": action.context,
            "player_name": action.player_name,
            "success": True
        }
        
    except openai.APIError as e:
        raise HTTPException(status_code=500, detail=f"OpenAI API error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "ai_enabled": bool(os.getenv("OPENAI_API_KEY"))}
```

## Step 3: Test Your Integration

### Start the Backend

```bash
cd backend/src
uvicorn main:app --reload
```

### Test with the API Documentation

1. Open `http://localhost:8000/docs` in your browser
2. Click on the `/dnd/action` endpoint
3. Click "Try it out"
4. Enter test data:

```json
{
  "player_name": "Gandalf",
  "action": "I cast a fireball spell at the group of goblins",
  "context": "We are in a dark cave with 3 goblins blocking our path"
}
```

5. Click "Execute"

You should see an AI-generated response!

### Test with curl (Command Line)

```bash
curl -X POST "http://localhost:8000/dnd/action" \
     -H "Content-Type: application/json" \
     -d '{
       "player_name": "Aragorn",
       "action": "I draw my sword and charge at the orc",
       "context": "Battle in the throne room"
     }'
```

## Step 4: Understanding What Just Happened

### The AI Pipeline

1. **Input Processing**: Your player action gets formatted into a prompt
2. **System Prompt**: Tells the AI how to behave (as a D&D master)
3. **User Prompt**: Contains the specific action and context
4. **AI Generation**: OpenAI generates a response
5. **Output Processing**: We clean up and return the response

### Key Code Sections

**Environment Loading**:
```python
from dotenv import load_dotenv
load_dotenv()
```
This loads your API key securely from the `.env` file.

**Prompt Engineering**:
```python
system_prompt = """You are an experienced Dungeon Master..."""
```
This tells the AI exactly how to behave and respond.

**Error Handling**:
```python
try:
    # AI call here
except openai.APIError as e:
    # Handle API-specific errors
```
This catches and handles errors gracefully.

## Step 5: Experiment and Improve

### Try Different Prompts

Experiment with different system prompts:

```python
# More dramatic responses
system_prompt = """You are a dramatic Dungeon Master who loves epic storytelling. 
Make every action sound heroic and exciting with vivid descriptions."""

# More tactical responses  
system_prompt = """You are a tactical Dungeon Master focused on game mechanics. 
Include dice rolls, damage numbers, and strategic implications in your responses."""
```

### Adjust AI Parameters

Try different settings in your `.env` file:

```bash
# More creative responses
TEMPERATURE=1.0

# More consistent responses
TEMPERATURE=0.3

# Longer responses
MAX_TOKENS=300

# Shorter responses
MAX_TOKENS=100
```

### Test Edge Cases

Try unusual inputs to see how your AI handles them:
- Very long actions
- Nonsensical actions
- Empty context
- Special characters

## Step 6: Add More Features

### Health Check Endpoint

Test if your AI integration is working:

```bash
curl http://localhost:8000/health
```

This returns whether the AI is properly configured.

### Context Memory (Advanced)

For a more advanced feature, you could store conversation history:

```python
# This is a simplified example - you'd want to use a database in production
conversation_history = {}

@app.post("/dnd/action")
async def process_action(action: PlayerAction):
    # Get previous context for this player
    history = conversation_history.get(action.player_name, [])
    
    # Add to messages
    messages = [{"role": "system", "content": system_prompt}]
    
    # Add conversation history
    for h in history[-5:]:  # Last 5 interactions
        messages.append({"role": "user", "content": h["action"]})
        messages.append({"role": "assistant", "content": h["response"]})
    
    # Add current action
    messages.append({"role": "user", "content": user_prompt})
    
    # ... rest of the AI call
    
    # Store this interaction
    if action.player_name not in conversation_history:
        conversation_history[action.player_name] = []
    conversation_history[action.player_name].append({
        "action": action.action,
        "response": ai_narration
    })
```

## Troubleshooting

### Common Issues

**"Invalid API Key" Error**:
- Check your `.env` file exists in the right location
- Verify your API key is correct (starts with `sk-`)
- Make sure you have credits in your OpenAI account

**"Module not found" Error**:
- Make sure you're in the right directory
- Activate your virtual environment: `source venv/bin/activate`
- Install dependencies: `pip install -r requirements.txt`

**Slow Responses**:
- Normal for AI APIs (2-5 seconds)
- You can reduce `MAX_TOKENS` for faster responses
- Consider caching common responses

## Testing Your Knowledge

Try implementing these challenges:

1. **Character Creator**: Add an endpoint that generates character backgrounds
2. **Dice Roller**: Create an endpoint that rolls dice and includes AI commentary
3. **Location Generator**: Build a feature that creates random locations with descriptions

## What's Next?

Congratulations! You've successfully integrated AI into a real application. You now understand:
- How to call AI APIs from a backend
- Prompt engineering basics
- Error handling for AI services
- Testing AI integrations

Ready for more advanced features? Continue to [Building Advanced Integrations](./04-advanced-integrations.md) to learn about character generation, memory systems, and more complex AI interactions.

## Code Summary

Here's what we built:
- ✅ Real OpenAI integration
- ✅ Secure API key handling
- ✅ Error handling
- ✅ Configurable AI parameters
- ✅ Testing endpoints

Your D&D Master now has real AI powers! 🧙‍♂️