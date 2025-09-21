#!/usr/bin/env python
"""
LangServe Server for the Milvus RAG Application

This script creates a FastAPI web server to expose the RAG (Retrieval-Augmented Generation)
chain, which is defined in `rag_app.py`. It provides several endpoints for interaction:

1.  **Standard LangServe Endpoints**: Includes a web-based playground for testing the RAG
    chain directly, along with `invoke` and `stream` endpoints.
2.  **OpenAI-Compatible Endpoints**: Mimics the OpenAI API structure (`/v1/models` and
    `/v1/chat/completions`), allowing it to be used as a drop-in replacement with
    clients like Open WebUI.
3.  **Health Check**: A simple `/health` endpoint to verify the server is running.
"""

# --- 1. IMPORTS AND ENVIRONMENT SETUP ---
import os
import sys
import json
from importlib import import_module
from typing import List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from langserve import add_routes
from langchain_core.runnables import RunnableLambda

# Add the current directory to the Python path to ensure `rag_app` can be imported
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Load environment variables from a .env file (e.g., OPENAI_API_KEY)
load_dotenv()

# Ensure the OpenAI API key is set for the LangChain components
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not found in environment variables.")
os.environ["OPENAI_API_KEY"] = api_key

print("🚀 Starting LangServe RAG Server...")

# --- 2. LAZY LOADING OF THE RAG CHAIN ---

# Attempt to import the RAG chain from rag_app.py at startup.
# This provides a "best-effort" initial load.
try:
    # The `chain` object is built and configured in `rag_app.py`
    from rag_app import chain as rag_chain_instance
    rag_available = True
    print("✅ RAG chain loaded successfully at startup.")
except Exception as e:
    rag_chain_instance = None
    rag_available = False
    print(f"⚠️ RAG chain not available at startup. It will be loaded on first request. Error: {e}")

# To ensure the RAG chain can be re-imported or reflects changes without a server restart
# during development, we use a lazy-loading mechanism. This runnable re-imports and
# invokes the chain on each request.
def lazy_rag_chain_invoke(input_data):
    """Dynamically re-imports and invokes the RAG chain."""
    try:
        # Re-import the rag_app module to get the latest version of the chain
        rag_app_module = import_module("rag_app")
        # Get the 'chain' attribute from the reloaded module
        chain = getattr(rag_app_module, "chain")
        # Invoke the chain with the provided input
        return chain.invoke(input_data)
    except Exception as e:
        import traceback
        print(f"❌ Error loading/invoking RAG chain: {e}")
        print("Full traceback:")
        print(traceback.format_exc())
        return f"RAG system error: {e}"

# Wrap the lazy loading function in a LangChain RunnableLambda
# This makes our custom function compatible with the LangChain ecosystem.
chain = RunnableLambda(lazy_rag_chain_invoke)

print("📦 LangServe server ready.")

# --- 3. FASTAPI APPLICATION SETUP ---

# Initialize the FastAPI application
app = FastAPI(
    title="Wimpy Kid RAG API",
    version="1.0.0",
    description="A LangServe-powered RAG API for answering questions about the 'Diary of a Wimpy Kid' series.",
)

# Configure CORS (Cross-Origin Resource Sharing) middleware
# This allows web applications from any origin to access the API, which is useful for development.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

# --- 4. API ENDPOINTS ---

@app.get("/")
async def root():
    """Root endpoint providing basic server information and available endpoints."""
    return {
        "message": "Wimpy Kid RAG API (LangServe)",
        "status": "healthy",
        "rag_available_on_startup": rag_available,
        "endpoints": {
            "playground": "/wimpy-kid-rag/playground",
            "invoke": "/wimpy-kid-rag/invoke",
            "stream": "/wimpy-kid-rag/stream",
            "openai_compatible": "/v1/chat/completions"
        }
    }

@app.get("/health")
async def health():
    """A simple health check endpoint."""
    return {"status": "healthy"}

# Add the standard LangServe routes to the FastAPI app.
# This automatically creates endpoints for the playground, invoke, stream, etc.
add_routes(
    app,
    chain,  # The runnable chain to be served
    path="/wimpy-kid-rag",
    enable_feedback_endpoint=True,
    enable_public_trace_link_endpoint=True,
    playground_type="default",
)

# --- 5. OPENAI-COMPATIBLE ENDPOINTS ---

# Pydantic models for request and response validation, matching OpenAI's structure.
class Message(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str = "wimpy-kid-rag"
    messages: List[Message]
    stream: Optional[bool] = False
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 1000

@app.get("/v1/models")
async def list_models():
    """Provides a list of available models, compatible with the OpenAI API."""
    return JSONResponse({
        "object": "list",
        "data": [{
            "id": "wimpy-kid-rag",
            "object": "model",
            "created": 1677610602, # Placeholder timestamp
            "owned_by": "custom"
        }]
    })

async def handle_chat_request(request: ChatCompletionRequest):
    """
    Core logic for handling OpenAI-compatible chat completion requests.
    This function is shared by both `/api/chat/completions` and `/v1/chat/completions`.
    """
    print(f"🔍 API Request received: model={request.model}, stream={request.stream}, messages={len(request.messages)}")

    # Extract the last user message from the conversation history
    user_message = ""
    for msg in reversed(request.messages):
        if msg.role == "user":
            user_message = msg.content
            break
    
    if not user_message:
        return JSONResponse({"error": "No user message found"}, status_code=400)
    
    print(f"💬 Extracted user message: {user_message[:100]}...")

    # Handle streaming responses
    if request.stream:
        print("📡 Streaming response requested.")
        
        async def generate_stream():
            """Generator function to stream the RAG chain's response chunk by chunk."""
            try:
                # Use `astream` for asynchronous streaming from the RAG chain
                async for chunk in chain.astream(user_message):
                    # Format the chunk to match the OpenAI streaming API
                    delta_chunk = {
                        "id": f"chatcmpl-stream-{hash(chunk)}",
                        "object": "chat.completion.chunk",
                        "created": 1677652288, # Placeholder
                        "model": request.model,
                        "choices": [{"index": 0, "delta": {"content": chunk}, "finish_reason": None}]
                    }
                    yield f"data: {json.dumps(delta_chunk)}\n\n"
                
                # Send the final "stop" message to terminate the stream
                final_chunk = {
                    "id": "chatcmpl-final",
                    "object": "chat.completion.chunk",
                    "created": 1677652288,
                    "model": request.model,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]
                }
                yield f"data: {json.dumps(final_chunk)}\n\n"
                yield "data: [DONE]\n\n"
                print("✅ Stream completed successfully.")
            except Exception as e:
                print(f"❌ Streaming error: {e}")
                # Send an error message within the stream if something goes wrong
                error_chunk = {
                    "choices": [{"delta": {"content": f"Error: {e}"}, "finish_reason": "stop"}]
                }
                yield f"data: {json.dumps(error_chunk)}\n\n"
                yield "data: [DONE]\n\n"
        
        return StreamingResponse(generate_stream(), media_type="text/event-stream")

    # Handle non-streaming (blocking) responses
    else:
        print("📄 Non-streaming response requested.")
        try:
            response_text = chain.invoke(user_message)
            print(f"🤖 RAG Response: {response_text[:200]}...")
            
            # Format the response to match the OpenAI API
            result = {
                "id": "chatcmpl-123",
                "object": "chat.completion",
                "created": 1677652288,
                "model": request.model,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": response_text}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0} # Usage data is not tracked here
            }
            return JSONResponse(result)
        except Exception as e:
            print(f"❌ Non-streaming error: {e}")
            return JSONResponse({"error": str(e)}, status_code=500)

@app.post("/api/chat/completions")
async def api_chat_completions(request: ChatCompletionRequest):
    """Endpoint for older clients that might use `/api/chat/completions`."""
    return await handle_chat_request(request)

@app.post("/v1/chat/completions")
async def v1_chat_completions(request: ChatCompletionRequest):
    """Main OpenAI-compatible endpoint for chat completions."""
    return await handle_chat_request(request)

# --- 6. SERVER EXECUTION ---

if __name__ == "__main__":
    import uvicorn
    
    print("\n🌟 LangServe endpoints available:")
    print(f"   - 📊 Playground: http://localhost:8001/wimpy-kid-rag/playground")
    print(f"   - 🤖 OpenAI API: http://localhost:8001/v1/chat/completions")
    
    # Run the FastAPI server using uvicorn
    uvicorn.run(
        app, 
        host="0.0.0.0",  # Listen on all available network interfaces
        port=8001,
        reload=False     # `reload=True` is useful for development but not for production
    )