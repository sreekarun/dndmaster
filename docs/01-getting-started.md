# Getting Started

## Project Overview

The DnD Master project is an AI-powered Dungeons & Dragons assistant that demonstrates how to integrate artificial intelligence into a web application. It consists of:

- **Backend**: A FastAPI server that handles game logic and AI integration
- **Frontend**: A React application for the user interface
- **AI Integration**: OpenAI API integration for generating dynamic content

## Understanding the Architecture

```
dndmaster/
├── backend/              # FastAPI server
│   ├── src/
│   │   └── main.py      # Main API endpoints
│   └── requirements.txt  # Python dependencies
├── clients/
│   └── frontend/        # React application
│       ├── src/         # React components
│       └── package.json # Node.js dependencies
└── docs/               # This documentation
```

## Development Environment Setup

### Prerequisites

Before you start, make sure you have:
- Python 3.8+ installed
- Node.js 16+ installed
- A text editor (VS Code recommended)
- Git installed

### Step 1: Clone the Repository

```bash
git clone https://github.com/sreekarun/dndmaster.git
cd dndmaster
```

### Step 2: Set Up the Backend

1. Navigate to the backend directory:
```bash
cd backend
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file for your API keys:
```bash
# backend/.env
OPENAI_API_KEY=your_openai_api_key_here
```

5. Start the backend server:
```bash
cd src
uvicorn main:app --reload
```

The backend will be available at `http://localhost:8000`

### Step 3: Set Up the Frontend

1. Open a new terminal and navigate to the frontend:
```bash
cd clients/frontend
```

2. Install dependencies:
```bash
npm install
```

3. Start the development server:
```bash
npm run dev
```

The frontend will be available at `http://localhost:5173`

## Testing Your Setup

1. Visit `http://localhost:8000` in your browser - you should see a welcome message
2. Visit `http://localhost:8000/docs` to see the API documentation
3. Visit `http://localhost:5173` to see the React application

## Understanding the Current Code

### Backend Analysis

The main API file (`backend/src/main.py`) contains:

```python
@app.post("/dnd/action")
async def process_action(action: PlayerAction):
    # This is where AI integration will happen
    response = {
        "narration": f"{action.player_name} attempts: {action.action}. (AI response would go here.)",
        "context": action.context
    }
    return response
```

**Key Points:**
- The endpoint receives player actions
- Currently returns placeholder text
- This is where we'll add AI integration

### Frontend Analysis

The React application provides the user interface for:
- Submitting player actions
- Displaying AI-generated responses
- Managing game state

## What's Next?

Now that you have the project running, you're ready to learn about AI integration fundamentals. Continue to [AI Integration Fundamentals](./02-ai-fundamentals.md) to understand how AI APIs work and how they fit into this application.

## Troubleshooting

### Common Issues

**Backend won't start:**
- Check that Python 3.8+ is installed: `python --version`
- Ensure virtual environment is activated
- Verify all dependencies are installed: `pip list`

**Frontend won't start:**
- Check that Node.js 16+ is installed: `node --version`
- Clear node_modules and reinstall: `rm -rf node_modules && npm install`

**Port conflicts:**
- Backend: Change port with `uvicorn main:app --port 8001`
- Frontend: Usually auto-assigns an available port

Need more help? Check the [Best Practices & Troubleshooting](./05-best-practices.md) guide.