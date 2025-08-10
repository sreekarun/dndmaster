# Best Practices & Troubleshooting

This guide covers production-ready practices, security considerations, and solutions to common problems you'll encounter when building AI-powered applications.

## Production Deployment

### Environment Configuration

#### Production Environment Variables

Create different `.env` files for different environments:

```bash
# .env.development
OPENAI_API_KEY=sk-test-key-here
OPENAI_MODEL=gpt-3.5-turbo
MAX_TOKENS=200
TEMPERATURE=0.8
ENVIRONMENT=development
DEBUG=true

# .env.production  
OPENAI_API_KEY=sk-production-key-here
OPENAI_MODEL=gpt-3.5-turbo
MAX_TOKENS=150
TEMPERATURE=0.7
ENVIRONMENT=production
DEBUG=false
RATE_LIMIT_PER_MINUTE=60
```

#### Configuration Management

```python
from pydantic import BaseSettings

class Settings(BaseSettings):
    openai_api_key: str
    openai_model: str = "gpt-3.5-turbo"
    max_tokens: int = 200
    temperature: float = 0.8
    environment: str = "development"
    debug: bool = False
    rate_limit_per_minute: int = 60
    
    class Config:
        env_file = ".env"

settings = Settings()
```

### Security Best Practices

#### API Key Security

**Never commit API keys to version control:**

```bash
# Add to .gitignore
.env
.env.*
config/secrets.json
*.key
```

**Use environment variables in production:**

```python
import os

# Good: Use environment variables
api_key = os.getenv("OPENAI_API_KEY")

# Bad: Hard-coded keys
api_key = "sk-1234567890abcdef"  # Never do this!
```

#### Input Validation and Sanitization

```python
from pydantic import BaseModel, validator
import re

class PlayerAction(BaseModel):
    player_name: str
    action: str
    context: Optional[str] = None
    
    @validator('player_name')
    def validate_player_name(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError('Player name cannot be empty')
        if len(v) > 50:
            raise ValueError('Player name too long')
        # Remove potentially harmful characters
        if re.search(r'[<>"\'\&]', v):
            raise ValueError('Player name contains invalid characters')
        return v.strip()
    
    @validator('action')
    def validate_action(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError('Action cannot be empty')
        if len(v) > 500:
            raise ValueError('Action too long')
        return v.strip()
```

#### Rate Limiting

```python
from fastapi import HTTPException
import time
from collections import defaultdict

class RateLimiter:
    def __init__(self, max_requests_per_minute=60):
        self.max_requests = max_requests_per_minute
        self.requests = defaultdict(list)
    
    def is_allowed(self, client_id: str) -> bool:
        now = time.time()
        minute_ago = now - 60
        
        # Clean old requests
        self.requests[client_id] = [
            req_time for req_time in self.requests[client_id] 
            if req_time > minute_ago
        ]
        
        # Check if under limit
        if len(self.requests[client_id]) >= self.max_requests:
            return False
        
        # Add current request
        self.requests[client_id].append(now)
        return True

rate_limiter = RateLimiter(max_requests_per_minute=60)

@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    client_ip = request.client.host
    
    if not rate_limiter.is_allowed(client_ip):
        raise HTTPException(
            status_code=429, 
            detail="Rate limit exceeded. Try again later."
        )
    
    response = await call_next(request)
    return response
```

### Content Filtering

#### Implement Content Safety

```python
import openai

async def check_content_safety(text: str) -> tuple[bool, str]:
    """Check if content is safe using OpenAI's moderation API"""
    try:
        response = client.moderations.create(input=text)
        result = response.results[0]
        
        if result.flagged:
            categories = [cat for cat, flagged in result.categories.__dict__.items() if flagged]
            return False, f"Content flagged for: {', '.join(categories)}"
        
        return True, "Content is safe"
    
    except Exception as e:
        # Default to safe if moderation fails
        return True, f"Moderation check failed: {str(e)}"

@app.post("/dnd/action")
async def process_action(action: PlayerAction):
    # Check input safety
    is_safe, safety_message = await check_content_safety(action.action)
    if not is_safe:
        raise HTTPException(status_code=400, detail=f"Inappropriate content: {safety_message}")
    
    # ... rest of your AI processing
```

#### Custom Content Filters

```python
import re

class ContentFilter:
    def __init__(self):
        # Add words/patterns you want to filter
        self.blocked_patterns = [
            r'\b(inappropriate_word1|inappropriate_word2)\b',
            r'harmful_pattern',
        ]
        self.blocked_themes = [
            'violence against real people',
            'self-harm',
            'illegal activities'
        ]
    
    def is_safe(self, text: str) -> tuple[bool, str]:
        text_lower = text.lower()
        
        # Check blocked patterns
        for pattern in self.blocked_patterns:
            if re.search(pattern, text_lower):
                return False, "Content contains blocked words"
        
        # Check themes (you could use AI for this too)
        for theme in self.blocked_themes:
            if theme in text_lower:
                return False, f"Content contains inappropriate theme: {theme}"
        
        return True, "Content is appropriate"

content_filter = ContentFilter()
```

## Error Handling and Resilience

### Comprehensive Error Handling

```python
import asyncio
from enum import Enum

class ErrorType(Enum):
    API_ERROR = "api_error"
    RATE_LIMIT = "rate_limit"
    TIMEOUT = "timeout"
    INVALID_INPUT = "invalid_input"
    UNKNOWN = "unknown"

class AIServiceError(Exception):
    def __init__(self, error_type: ErrorType, message: str, details: dict = None):
        self.error_type = error_type
        self.message = message
        self.details = details or {}
        super().__init__(self.message)

async def robust_ai_call(messages, max_retries=3, timeout=30):
    """Make an AI call with proper error handling and retries"""
    
    for attempt in range(max_retries):
        try:
            response = await asyncio.wait_for(
                client.chat.completions.create(
                    model=settings.openai_model,
                    messages=messages,
                    max_tokens=settings.max_tokens,
                    temperature=settings.temperature
                ),
                timeout=timeout
            )
            
            return response.choices[0].message.content.strip()
            
        except openai.RateLimitError as e:
            if attempt < max_retries - 1:
                wait_time = min(2 ** attempt, 60)  # Exponential backoff, max 60s
                await asyncio.sleep(wait_time)
                continue
            raise AIServiceError(ErrorType.RATE_LIMIT, "Rate limit exceeded", {"attempt": attempt})
            
        except openai.APITimeoutError:
            if attempt < max_retries - 1:
                await asyncio.sleep(2)
                continue
            raise AIServiceError(ErrorType.TIMEOUT, "API request timed out")
            
        except openai.APIError as e:
            raise AIServiceError(ErrorType.API_ERROR, f"OpenAI API error: {str(e)}")
            
        except asyncio.TimeoutError:
            raise AIServiceError(ErrorType.TIMEOUT, "Request timed out")
            
        except Exception as e:
            raise AIServiceError(ErrorType.UNKNOWN, f"Unexpected error: {str(e)}")

@app.exception_handler(AIServiceError)
async def ai_service_error_handler(request: Request, exc: AIServiceError):
    error_responses = {
        ErrorType.RATE_LIMIT: {
            "status_code": 429,
            "message": "Service is busy. Please try again in a few moments.",
            "retry_after": 60
        },
        ErrorType.TIMEOUT: {
            "status_code": 504,
            "message": "Request took too long. Please try again.",
        },
        ErrorType.API_ERROR: {
            "status_code": 502,
            "message": "AI service is temporarily unavailable.",
        },
        ErrorType.INVALID_INPUT: {
            "status_code": 400,
            "message": "Invalid input provided.",
        },
        ErrorType.UNKNOWN: {
            "status_code": 500,
            "message": "An unexpected error occurred.",
        }
    }
    
    error_info = error_responses.get(exc.error_type, error_responses[ErrorType.UNKNOWN])
    
    return JSONResponse(
        status_code=error_info["status_code"],
        content={
            "error": exc.error_type.value,
            "message": error_info["message"],
            "details": exc.details if settings.debug else {}
        }
    )
```

### Fallback Strategies

```python
async def get_ai_response_with_fallback(prompt: str) -> str:
    """Get AI response with fallback to simpler models or pre-written responses"""
    
    # Try primary model
    try:
        return await robust_ai_call([{"role": "user", "content": prompt}])
    except AIServiceError as e:
        if e.error_type == ErrorType.RATE_LIMIT:
            # Try a simpler, cheaper model
            try:
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",  # Fallback to cheaper model
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=100,  # Reduced tokens
                    temperature=0.5
                )
                return response.choices[0].message.content.strip()
            except:
                pass
        
        # Final fallback to pre-written responses
        return get_fallback_response(prompt)

def get_fallback_response(prompt: str) -> str:
    """Return appropriate fallback responses when AI is unavailable"""
    fallback_responses = {
        "character": "A mysterious adventurer appears before you, ready for whatever challenges lie ahead.",
        "action": "Your action creates an interesting situation. The outcome depends on your next move.",
        "story": "You find yourself in a land of endless possibilities, where adventure awaits around every corner.",
        "default": "The magical energies swirl around you, creating new possibilities for adventure."
    }
    
    # Simple keyword matching for fallbacks
    prompt_lower = prompt.lower()
    for key, response in fallback_responses.items():
        if key in prompt_lower:
            return response
    
    return fallback_responses["default"]
```

## Performance Optimization

### Caching Strategies

```python
import hashlib
import json
from datetime import datetime, timedelta

class ResponseCache:
    def __init__(self, max_size=1000, default_ttl=300):
        self.cache = {}
        self.max_size = max_size
        self.default_ttl = default_ttl
    
    def _get_cache_key(self, prompt: str, model: str, temperature: float) -> str:
        """Generate cache key from prompt parameters"""
        key_data = f"{prompt}:{model}:{temperature}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def get(self, prompt: str, model: str, temperature: float) -> Optional[str]:
        """Get cached response if valid"""
        key = self._get_cache_key(prompt, model, temperature)
        
        if key in self.cache:
            cached_item = self.cache[key]
            if datetime.now() < cached_item['expires']:
                return cached_item['response']
            else:
                del self.cache[key]
        
        return None
    
    def set(self, prompt: str, model: str, temperature: float, response: str, ttl: int = None):
        """Cache a response"""
        key = self._get_cache_key(prompt, model, temperature)
        ttl = ttl or self.default_ttl
        
        # Clean up if cache is full
        if len(self.cache) >= self.max_size:
            self._cleanup_expired()
            if len(self.cache) >= self.max_size:
                # Remove oldest entry
                oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k]['created'])
                del self.cache[oldest_key]
        
        self.cache[key] = {
            'response': response,
            'created': datetime.now(),
            'expires': datetime.now() + timedelta(seconds=ttl)
        }
    
    def _cleanup_expired(self):
        """Remove expired entries"""
        now = datetime.now()
        expired_keys = [
            key for key, item in self.cache.items() 
            if now >= item['expires']
        ]
        for key in expired_keys:
            del self.cache[key]

# Global cache instance
response_cache = ResponseCache()

async def cached_ai_call(prompt: str, model: str = None, temperature: float = None) -> str:
    """Make AI call with caching"""
    model = model or settings.openai_model
    temperature = temperature or settings.temperature
    
    # Check cache first
    cached_response = response_cache.get(prompt, model, temperature)
    if cached_response:
        return cached_response
    
    # Make AI call
    response = await robust_ai_call([{"role": "user", "content": prompt}])
    
    # Cache the response
    response_cache.set(prompt, model, temperature, response)
    
    return response
```

### Monitoring and Analytics

```python
import time
from dataclasses import dataclass
from typing import Dict, List
import statistics

@dataclass
class RequestMetrics:
    endpoint: str
    response_time: float
    tokens_used: int
    success: bool
    error_type: Optional[str] = None
    timestamp: float = time.time()

class PerformanceMonitor:
    def __init__(self):
        self.metrics: List[RequestMetrics] = []
        self.max_metrics = 10000  # Keep last 10k requests
    
    def record_request(self, metrics: RequestMetrics):
        self.metrics.append(metrics)
        if len(self.metrics) > self.max_metrics:
            self.metrics = self.metrics[-self.max_metrics:]
    
    def get_stats(self, hours: int = 24) -> Dict:
        """Get performance statistics for the last N hours"""
        cutoff_time = time.time() - (hours * 3600)
        recent_metrics = [m for m in self.metrics if m.timestamp > cutoff_time]
        
        if not recent_metrics:
            return {"message": "No recent data"}
        
        response_times = [m.response_time for m in recent_metrics]
        successful_requests = [m for m in recent_metrics if m.success]
        failed_requests = [m for m in recent_metrics if not m.success]
        
        return {
            "total_requests": len(recent_metrics),
            "successful_requests": len(successful_requests),
            "failed_requests": len(failed_requests),
            "success_rate": len(successful_requests) / len(recent_metrics) * 100,
            "avg_response_time": statistics.mean(response_times),
            "median_response_time": statistics.median(response_times),
            "total_tokens": sum(m.tokens_used for m in recent_metrics),
            "errors_by_type": {
                error_type: len([m for m in failed_requests if m.error_type == error_type])
                for error_type in set(m.error_type for m in failed_requests if m.error_type)
            }
        }

# Global monitor
performance_monitor = PerformanceMonitor()

@app.get("/admin/performance")
async def get_performance_stats(hours: int = 24):
    """Get performance statistics (protect this endpoint in production!)"""
    return performance_monitor.get_stats(hours)
```

## Common Issues and Solutions

### Issue: High API Costs

**Symptoms:**
- Unexpected high bills from OpenAI
- Many long responses
- Repeated similar requests

**Solutions:**

```python
# 1. Implement strict token limits
MAX_RESPONSE_TOKENS = 150
MAX_PROMPT_TOKENS = 500

def limit_prompt_length(prompt: str, max_tokens: int = MAX_PROMPT_TOKENS) -> str:
    """Truncate prompt if too long"""
    # Rough estimate: 4 characters per token
    max_chars = max_tokens * 4
    if len(prompt) > max_chars:
        return prompt[:max_chars] + "..."
    return prompt

# 2. Use cheaper models when possible
def choose_model(complexity: str) -> str:
    """Choose appropriate model based on complexity"""
    if complexity == "simple":
        return "gpt-3.5-turbo"
    elif complexity == "complex":
        return "gpt-4"
    else:
        return "gpt-3.5-turbo"  # Default to cheaper option

# 3. Implement usage quotas per user
class UsageTracker:
    def __init__(self):
        self.user_usage = defaultdict(int)
        self.daily_limits = {"free": 100, "premium": 1000}
    
    def can_make_request(self, user_id: str, user_tier: str = "free") -> bool:
        daily_limit = self.daily_limits.get(user_tier, 100)
        return self.user_usage[user_id] < daily_limit
    
    def record_usage(self, user_id: str, tokens_used: int):
        self.user_usage[user_id] += tokens_used
```

### Issue: Slow Response Times

**Symptoms:**
- Users wait too long for responses
- Timeouts occur frequently
- Poor user experience

**Solutions:**

```python
# 1. Implement streaming responses (for supported models)
@app.post("/dnd/action-stream")
async def process_action_stream(action: PlayerAction):
    async def generate_stream():
        try:
            stream = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": action.action}],
                stream=True,
                max_tokens=200
            )
            
            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    yield f"data: {json.dumps({'content': chunk.choices[0].delta.content})}\n\n"
            
            yield f"data: {json.dumps({'done': True})}\n\n"
            
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
    
    return StreamingResponse(generate_stream(), media_type="text/plain")

# 2. Implement response caching aggressively
@lru_cache(maxsize=1000)
def get_common_response(action_type: str, context: str) -> str:
    """Cache responses for common actions"""
    # This would be populated with pre-generated responses
    common_responses = {
        "attack": "Your attack strikes true, dealing significant damage!",
        "defend": "You raise your shield, preparing for the incoming attack.",
        # ... more common responses
    }
    return common_responses.get(action_type, "")

# 3. Use background tasks for non-critical operations
from fastapi import BackgroundTasks

@app.post("/dnd/action")
async def process_action(action: PlayerAction, background_tasks: BackgroundTasks):
    # Get immediate response
    response = await get_quick_response(action)
    
    # Process additional features in background
    background_tasks.add_task(update_player_stats, action.player_name)
    background_tasks.add_task(log_action_analytics, action)
    
    return response
```

### Issue: AI Responses Are Inconsistent

**Symptoms:**
- AI forgets previous context
- Responses don't match the game's tone
- Characters act out of character

**Solutions:**

```python
# 1. Improve prompt engineering
def create_consistent_prompt(action: PlayerAction, game_context: dict) -> str:
    """Create a prompt that maintains consistency"""
    
    base_prompt = f"""
    You are the Dungeon Master for an ongoing D&D campaign.
    
    CAMPAIGN SETTING: {game_context.get('setting', 'Fantasy medieval')}
    CURRENT LOCATION: {game_context.get('location', 'Unknown')}
    TONE: {game_context.get('tone', 'Adventurous but appropriate for teens')}
    
    IMPORTANT RULES:
    - Stay consistent with the established setting
    - Remember character relationships and past events
    - Keep responses between 2-4 sentences
    - Always end with a clear situation for the player to respond to
    
    RECENT EVENTS:
    {game_context.get('recent_events', 'The adventure begins...')}
    
    CURRENT ACTION: {action.action}
    PLAYER: {action.player_name}
    
    Describe what happens and present the next situation:
    """
    
    return base_prompt

# 2. Implement better context management
class GameSession:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.setting = "Fantasy medieval"
        self.location = "Starting tavern"
        self.tone = "Adventurous"
        self.events = []
        self.characters = {}
    
    def add_event(self, event: str):
        self.events.append(event)
        # Keep only last 10 events for context
        if len(self.events) > 10:
            self.events = self.events[-10:]
    
    def get_context(self) -> dict:
        return {
            "setting": self.setting,
            "location": self.location,
            "tone": self.tone,
            "recent_events": " | ".join(self.events[-5:])
        }

# 3. Use few-shot examples in prompts
def create_few_shot_prompt(action: PlayerAction) -> str:
    """Include examples of good responses"""
    
    examples = """
    Example 1:
    Player Action: "I swing my sword at the goblin"
    Good Response: "Your blade catches the morning light as it arcs toward the goblin. The creature tries to dodge but your training pays off - you score a solid hit across its shoulder! The goblin staggers back, wounded but still fighting. What's your next move?"
    
    Example 2:
    Player Action: "I try to convince the guard to let us pass"
    Good Response: "You step forward with confidence and speak persuasively to the guard. He listens intently, then slowly nods. 'Your words ring true, traveler. You may pass, but be warned - strange things have been happening in the woods lately.' He steps aside and gestures toward the path ahead."
    """
    
    return f"{examples}\n\nNow respond to this action: {action.action}"
```

## Deployment Checklist

### Pre-Deployment

- [ ] **Security**
  - [ ] API keys stored in environment variables
  - [ ] Input validation implemented
  - [ ] Rate limiting configured
  - [ ] Content filtering active

- [ ] **Performance**  
  - [ ] Caching implemented
  - [ ] Error handling robust
  - [ ] Monitoring in place
  - [ ] Resource limits set

- [ ] **Testing**
  - [ ] All endpoints tested
  - [ ] Error scenarios covered
  - [ ] Load testing completed
  - [ ] Content safety verified

### Production Monitoring

```python
# Set up logging
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    
    response = await call_next(request)
    
    process_time = time.time() - start_time
    logger.info(
        f"{request.method} {request.url.path} - "
        f"Status: {response.status_code} - "
        f"Time: {process_time:.3f}s"
    )
    
    return response
```

## Getting Help

### Debugging Tips

1. **Check logs first**: Most issues leave traces in logs
2. **Test with simple inputs**: Start with basic cases
3. **Verify API keys**: Ensure they're valid and have credits
4. **Check rate limits**: See if you're hitting service limits
5. **Monitor token usage**: Understand your consumption patterns

### Resources

- **OpenAI Documentation**: [platform.openai.com/docs](https://platform.openai.com/docs)
- **FastAPI Documentation**: [fastapi.tiangolo.com](https://fastapi.tiangolo.com)
- **AI Safety Guidelines**: [platform.openai.com/docs/guides/safety-best-practices](https://platform.openai.com/docs/guides/safety-best-practices)

### Community

- Stack Overflow: Tag questions with `openai`, `fastapi`, `python`
- Reddit: r/MachineLearning, r/OpenAI
- Discord: FastAPI and OpenAI communities

## Congratulations! 🎉

You've completed the AI Integration Training Guide! You now have the knowledge and tools to:

- Build production-ready AI applications
- Handle errors gracefully and securely
- Optimize performance and manage costs
- Deploy and monitor AI services

Keep experimenting, keep learning, and most importantly - keep building amazing AI-powered applications!

---

**Final Project Ideas:**
- Build your own AI assistant for a different domain
- Create an AI-powered game master for other RPG systems
- Develop educational tools using AI
- Build creative writing assistants

The future of AI development is in your hands! 🚀