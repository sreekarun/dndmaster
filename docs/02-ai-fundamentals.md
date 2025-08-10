# AI Integration Fundamentals

## What Are AI APIs?

An API (Application Programming Interface) is a way for different software programs to communicate. An AI API specifically allows your application to communicate with artificial intelligence services.

Think of it like ordering food at a restaurant:
- You (your app) place an order (send a request)
- The kitchen (AI service) prepares your food (processes the request)
- The waiter (API) brings you the finished meal (returns the response)

## Popular AI Services

### OpenAI
- **GPT Models**: Great for text generation, conversation, and creative writing
- **DALL-E**: Creates images from text descriptions
- **Whisper**: Converts speech to text

### Other Services
- **Google Cloud AI**: Various AI tools and models
- **Amazon AWS AI**: Machine learning services
- **Anthropic Claude**: Advanced conversational AI

## How AI APIs Work

### The Request-Response Cycle

1. **Your App**: Sends a request with:
   - Text prompt or input
   - Configuration parameters (temperature, max tokens, etc.)
   - Authentication (API key)

2. **AI Service**: 
   - Processes your request
   - Generates a response using AI models
   - Returns structured data

3. **Your App**: 
   - Receives the response
   - Processes and displays it to users

### Example API Call

Here's what a simple OpenAI API call looks like:

```python
import openai

# Set up the client
client = openai.OpenAI(api_key="your-api-key")

# Make a request
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "user", "content": "Tell me about dragons in D&D"}
    ],
    max_tokens=150,
    temperature=0.7
)

# Get the response
ai_response = response.choices[0].message.content
print(ai_response)
```

## Key Concepts

### Prompts
The text you send to the AI. Good prompts:
- Are clear and specific
- Provide context
- Include examples when helpful

**Example Prompts for D&D:**
```
Bad: "What happens?"
Good: "The player John swings his sword at the goblin. Describe the outcome in an exciting way, keeping it appropriate for teens."
```

### Temperature
Controls randomness in responses:
- **0.0**: Very predictable, consistent responses
- **0.7**: Balanced creativity and consistency  
- **1.0**: Very creative, unpredictable responses

### Tokens
Units of text the AI processes:
- ~4 characters = 1 token
- Costs are based on tokens used
- You can limit tokens to control costs

### Context Window
The amount of previous conversation the AI remembers:
- GPT-3.5: ~4,000 tokens
- GPT-4: ~8,000+ tokens
- Affects how much history you can include

## AI in Web Applications

### Where AI Fits

In a typical web app with AI:

```
Frontend (React) → Backend (FastAPI) → AI Service (OpenAI)
     ↑                    ↓
User sees response ← Processes AI response ← Gets AI response
```

### Why Use a Backend?

**Security**: API keys stay on your server, not exposed to users
**Cost Control**: You can limit and monitor usage
**Processing**: You can modify AI responses before sending to users
**Caching**: Store common responses to save money

## Practical Example: D&D Context

In our D&D application, AI helps with:

### 1. Action Resolution
**Input**: "I cast fireball at the goblins"
**AI Processing**: Determines outcome, describes effects
**Output**: "Your fireball explodes among the goblins, dealing 8 damage to each..."

### 2. Story Generation
**Input**: Player enters a new room
**AI Processing**: Creates description based on context
**Output**: "You enter a dusty library filled with ancient tomes..."

### 3. Character Interactions
**Input**: Player talks to an NPC
**AI Processing**: Generates appropriate dialogue
**Output**: "The wise wizard strokes his beard and says..."

## Cost Considerations

AI APIs charge based on usage:

### OpenAI Pricing (approximate)
- GPT-3.5-turbo: ~$0.002 per 1,000 tokens
- GPT-4: ~$0.03 per 1,000 tokens

### Cost Management Tips
1. **Set usage limits** in your OpenAI account
2. **Cache common responses** to avoid repeated calls
3. **Use cheaper models** when possible (GPT-3.5 vs GPT-4)
4. **Limit token counts** for responses
5. **Implement user limits** in your app

## Ethical Considerations

### Content Filtering
- AI can generate inappropriate content
- Implement content filters
- Set clear guidelines in prompts

### Privacy
- Don't send sensitive user data to AI services
- Be transparent about AI usage
- Follow data protection laws

### Bias and Fairness
- AI models can reflect training data biases
- Test with diverse scenarios
- Have human oversight for important decisions

## Getting Your API Key

### OpenAI Setup
1. Go to [platform.openai.com](https://platform.openai.com)
2. Create an account
3. Navigate to API Keys section
4. Create a new secret key
5. **Important**: Keep this key secure and never commit it to version control

### Setting Up Environment Variables

Create a `.env` file in your backend directory:

```bash
# backend/.env
OPENAI_API_KEY=sk-your-actual-api-key-here
OPENAI_MODEL=gpt-3.5-turbo
MAX_TOKENS=150
TEMPERATURE=0.7
```

**Security Note**: Add `.env` to your `.gitignore` file so it's never committed to Git.

## What's Next?

Now that you understand AI API fundamentals, you're ready to implement your first AI feature! Continue to [Your First AI Feature](./03-first-ai-feature.md) to add real AI responses to the D&D Master application.

## Quick Review

Before moving on, make sure you understand:
- [ ] What an AI API is and how it works
- [ ] The request-response cycle
- [ ] Key concepts: prompts, temperature, tokens
- [ ] Why we use a backend for AI integration
- [ ] Cost and ethical considerations
- [ ] How to get and secure an API key

Questions? Check the [Best Practices & Troubleshooting](./05-best-practices.md) guide!