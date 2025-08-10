# Building Advanced Integrations

Now that you have basic AI integration working, let's build more sophisticated features that showcase the real power of AI in applications.

## Learning Objectives

In this section, you'll learn to:
- Build character generation systems
- Implement conversation memory
- Create dynamic storytelling features
- Handle complex AI workflows
- Optimize performance and costs

## Advanced Feature 1: AI Character Generator

Let's create an endpoint that generates complete D&D characters with backstories.

### Step 1: Add Character Generation Endpoint

Add this to your `backend/src/main.py`:

```python
from pydantic import BaseModel
from typing import Optional, List
import json

class CharacterRequest(BaseModel):
    character_class: Optional[str] = None
    race: Optional[str] = None
    background: Optional[str] = None
    personality_traits: Optional[List[str]] = None

@app.post("/dnd/generate-character")
async def generate_character(request: CharacterRequest):
    try:
        system_prompt = """You are an expert D&D character creator. Generate complete, 
        interesting characters with stats, backstories, and personality traits. 
        Return responses in JSON format with these fields:
        - name
        - race  
        - class
        - level
        - stats (strength, dexterity, constitution, intelligence, wisdom, charisma)
        - backstory (2-3 sentences)
        - personality_traits (3 traits)
        - equipment (list of starting equipment)
        """
        
        user_prompt = f"""
        Create a D&D character with these preferences:
        Class: {request.character_class or "any"}
        Race: {request.race or "any"}  
        Background: {request.background or "any"}
        Personality hints: {request.personality_traits or "surprise me"}
        
        Make them interesting and unique!
        """
        
        response = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=500,
            temperature=0.9  # Higher creativity for character generation
        )
        
        ai_response = response.choices[0].message.content.strip()
        
        # Try to parse as JSON, fallback to text if it fails
        try:
            character_data = json.loads(ai_response)
        except json.JSONDecodeError:
            character_data = {"description": ai_response}
        
        return {
            "character": character_data,
            "success": True
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Character generation error: {str(e)}")
```

### Step 2: Test Character Generation

Test with different inputs:

```bash
curl -X POST "http://localhost:8000/dnd/generate-character" \
     -H "Content-Type: application/json" \
     -d '{
       "character_class": "wizard",
       "race": "elf",
       "personality_traits": ["curious", "bookish"]
     }'
```

## Advanced Feature 2: Conversation Memory System

Create a memory system that remembers past interactions for better continuity.

### Step 1: Set Up Memory Storage

```python
from datetime import datetime
import uuid

# In-memory storage (use a database in production)
conversation_memories = {}
player_sessions = {}

class ConversationMemory:
    def __init__(self):
        self.interactions = []
        self.context = {}
        self.last_updated = datetime.now()
    
    def add_interaction(self, player_action, ai_response, context=None):
        self.interactions.append({
            "timestamp": datetime.now(),
            "player_action": player_action,
            "ai_response": ai_response,
            "context": context
        })
        
        # Keep only last 10 interactions to manage token usage
        if len(self.interactions) > 10:
            self.interactions = self.interactions[-10:]
        
        self.last_updated = datetime.now()
    
    def get_context_summary(self):
        if not self.interactions:
            return "No previous interactions."
        
        recent_actions = [
            f"Player: {interaction['player_action']}" 
            for interaction in self.interactions[-3:]
        ]
        return "Recent actions: " + " | ".join(recent_actions)

def get_or_create_memory(player_name: str) -> ConversationMemory:
    if player_name not in conversation_memories:
        conversation_memories[player_name] = ConversationMemory()
    return conversation_memories[player_name]
```

### Step 2: Enhanced Action Processing with Memory

Update your action endpoint:

```python
@app.post("/dnd/action")
async def process_action(action: PlayerAction):
    try:
        # Get player's conversation memory
        memory = get_or_create_memory(action.player_name)
        context_summary = memory.get_context_summary()
        
        system_prompt = """You are an experienced Dungeon Master for a D&D game. 
        Use the conversation history to maintain continuity and remember what has happened.
        Respond to player actions with engaging, descriptive narratives that build on previous events.
        Keep responses exciting but appropriate for teenagers."""
        
        user_prompt = f"""
        Player: {action.player_name}
        Current Action: {action.action}
        Current Context: {action.context or "No additional context"}
        Previous Context: {context_summary}
        
        Describe what happens as a result of this action, taking into account the previous interactions.
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
        
        # Store this interaction in memory
        memory.add_interaction(action.action, ai_narration, action.context)
        
        return {
            "narration": ai_narration,
            "context": action.context,
            "player_name": action.player_name,
            "session_context": context_summary,
            "success": True
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Action processing error: {str(e)}")
```

## Advanced Feature 3: Dynamic Story Generation

Create a system that generates entire story scenarios.

### Step 1: Story Generator Endpoint

```python
class StoryRequest(BaseModel):
    setting: Optional[str] = None
    theme: Optional[str] = None
    difficulty: Optional[str] = "medium"
    party_size: Optional[int] = 4

@app.post("/dnd/generate-story")
async def generate_story(request: StoryRequest):
    try:
        system_prompt = """You are a creative D&D campaign writer. Generate engaging story 
        scenarios with clear objectives, interesting NPCs, and potential plot twists.
        Structure your response with:
        - Setting description
        - Main objective
        - Key NPCs
        - Potential challenges
        - Plot hooks
        """
        
        user_prompt = f"""
        Create a D&D story scenario with:
        Setting: {request.setting or "fantasy medieval"}
        Theme: {request.theme or "adventure"}
        Difficulty: {request.difficulty}
        Party size: {request.party_size} players
        
        Make it engaging and suitable for a {request.difficulty} difficulty level.
        """
        
        response = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=600,
            temperature=0.9
        )
        
        story_content = response.choices[0].message.content.strip()
        
        return {
            "story": story_content,
            "setting": request.setting,
            "theme": request.theme,
            "difficulty": request.difficulty,
            "success": True
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Story generation error: {str(e)}")
```

## Advanced Feature 4: Smart Dice Rolling with AI Commentary

Add intelligent dice rolling that adapts to the situation.

```python
import random

class DiceRoll(BaseModel):
    dice_type: str  # e.g., "d20", "2d6", "1d4+2"
    action_context: Optional[str] = None
    difficulty: Optional[int] = None

@app.post("/dnd/roll-dice")
async def roll_dice(roll_request: DiceRoll):
    try:
        # Parse dice notation (simplified)
        def parse_and_roll(dice_str):
            # Handle formats like "d20", "2d6", "1d4+2"
            dice_str = dice_str.lower().replace(" ", "")
            
            if "d" not in dice_str:
                return None, "Invalid dice format"
            
            # Simple parsing - you could make this more robust
            parts = dice_str.split("d")
            num_dice = int(parts[0]) if parts[0] else 1
            
            if "+" in parts[1]:
                die_size, modifier = parts[1].split("+")
                modifier = int(modifier)
            elif "-" in parts[1]:
                die_size, modifier = parts[1].split("-")
                modifier = -int(modifier)
            else:
                die_size = parts[1]
                modifier = 0
            
            die_size = int(die_size)
            
            # Roll the dice
            rolls = [random.randint(1, die_size) for _ in range(num_dice)]
            total = sum(rolls) + modifier
            
            return {
                "individual_rolls": rolls,
                "modifier": modifier,
                "total": total,
                "dice_notation": dice_str
            }, None
        
        roll_result, error = parse_and_roll(roll_request.dice_type)
        
        if error:
            raise HTTPException(status_code=400, detail=error)
        
        # Generate AI commentary on the roll
        system_prompt = """You are a D&D Dungeon Master providing colorful commentary on dice rolls.
        React appropriately to high rolls (celebrate), low rolls (commiserate), and average rolls.
        Keep it fun and engaging."""
        
        user_prompt = f"""
        A player rolled {roll_request.dice_type} for: {roll_request.action_context or "an action"}
        Result: {roll_result['total']} (rolled {roll_result['individual_rolls']}, modifier: {roll_result['modifier']})
        Difficulty: {roll_request.difficulty or "unknown"}
        
        Provide entertaining commentary on this roll result!
        """
        
        response = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=150,
            temperature=1.0
        )
        
        ai_commentary = response.choices[0].message.content.strip()
        
        return {
            "roll_result": roll_result,
            "commentary": ai_commentary,
            "success": roll_result['total'] >= (roll_request.difficulty or 10),
            "action_context": roll_request.action_context
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Dice rolling error: {str(e)}")
```

## Performance Optimization

### Caching Strategies

```python
import time
from functools import lru_cache

# Simple cache for common requests
response_cache = {}

def get_cached_response(prompt_hash, max_age_seconds=300):
    """Get cached response if it exists and isn't too old"""
    if prompt_hash in response_cache:
        cached_data = response_cache[prompt_hash]
        if time.time() - cached_data['timestamp'] < max_age_seconds:
            return cached_data['response']
    return None

def cache_response(prompt_hash, response):
    """Cache a response"""
    response_cache[prompt_hash] = {
        'response': response,
        'timestamp': time.time()
    }
    
    # Clean old entries (simple cleanup)
    if len(response_cache) > 100:
        oldest_key = min(response_cache.keys(), 
                        key=lambda k: response_cache[k]['timestamp'])
        del response_cache[oldest_key]
```

### Token Usage Monitoring

```python
class TokenUsageTracker:
    def __init__(self):
        self.total_tokens = 0
        self.requests_count = 0
        self.start_time = time.time()
    
    def add_usage(self, tokens_used):
        self.total_tokens += tokens_used
        self.requests_count += 1
    
    def get_stats(self):
        runtime = time.time() - self.start_time
        return {
            "total_tokens": self.total_tokens,
            "requests": self.requests_count,
            "runtime_hours": runtime / 3600,
            "tokens_per_hour": self.total_tokens / (runtime / 3600) if runtime > 0 else 0
        }

# Global tracker
usage_tracker = TokenUsageTracker()

@app.get("/dnd/usage-stats")
async def get_usage_stats():
    return usage_tracker.get_stats()
```

## Testing Your Advanced Features

### Character Generation Test

```bash
curl -X POST "http://localhost:8000/dnd/generate-character" \
     -H "Content-Type: application/json" \
     -d '{
       "character_class": "rogue",
       "race": "halfling",
       "background": "criminal"
     }'
```

### Story Generation Test

```bash
curl -X POST "http://localhost:8000/dnd/generate-story" \
     -H "Content-Type: application/json" \
     -d '{
       "setting": "haunted mansion",
       "theme": "mystery",
       "difficulty": "hard"
     }'
```

### Dice Rolling Test

```bash
curl -X POST "http://localhost:8000/dnd/roll-dice" \
     -H "Content-Type: application/json" \
     -d '{
       "dice_type": "d20",
       "action_context": "attacking a dragon",
       "difficulty": 15
     }'
```

## Building Your Own Features

Now try creating these advanced features:

### Challenge 1: NPC Dialogue System
Create an endpoint that generates realistic NPC conversations based on character personality and situation.

### Challenge 2: Quest Generator
Build a system that creates multi-step quests with objectives, rewards, and branching paths.

### Challenge 3: Combat AI
Develop an AI that can run combat encounters, managing multiple enemies and tactical decisions.

### Challenge 4: World Builder
Create a tool that generates consistent fantasy worlds with geography, politics, and history.

## Best Practices for Advanced AI

### Prompt Engineering Tips
1. **Be specific**: Clear instructions get better results
2. **Use examples**: Show the AI what you want
3. **Set constraints**: Define what you don't want
4. **Test iteratively**: Refine prompts based on results

### Error Handling
```python
async def safe_ai_call(messages, max_retries=3):
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
                messages=messages,
                max_tokens=200,
                temperature=0.8
            )
            return response.choices[0].message.content.strip()
        except openai.RateLimitError:
            if attempt < max_retries - 1:
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
            else:
                raise
        except Exception as e:
            if attempt == max_retries - 1:
                raise
```

## What's Next?

You've now built sophisticated AI integrations! You understand:
- Advanced prompt engineering
- Memory and context management
- Performance optimization
- Complex AI workflows

Ready to learn about deployment and best practices? Continue to [Best Practices & Troubleshooting](./05-best-practices.md) to learn about production deployment, security, and handling edge cases.

## Feature Summary

You've built:
- ✅ AI Character Generator
- ✅ Conversation Memory System  
- ✅ Dynamic Story Generation
- ✅ Smart Dice Rolling with Commentary
- ✅ Performance Monitoring
- ✅ Caching Systems

Your D&D Master is now a powerful AI-driven application! 🚀