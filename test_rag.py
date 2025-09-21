#!/usr/bin/env python3
"""
RAG System Integration Test Script

This script runs a series of integration tests against the LangServe RAG API 
to ensure all its components are functioning correctly. It does not test the 
UI, but rather the API endpoints that the UI consumes.

Tests include:
1.  **Health Checks**: Verifies that the server is running and responsive.
2.  **Model Availability**: Checks the OpenAI-compatible `/v1/models` endpoint to
    ensure the custom RAG model is listed.
3.  **RAG Queries (Non-Streaming)**: Sends questions to the `/v1/chat/completions`
    endpoint and validates that a meaningful response is returned.
4.  **RAG Queries (Streaming)**: Tests the streaming functionality to ensure that
    response chunks are sent correctly.
"""

# --- 1. IMPORTS ---
import requests
import json
import time
import sys

# --- 2. CONFIGURATION ---
# Base URL for the LangServe API server.
LANGSERVE_URL = "http://localhost:8001"
# A list of questions to test the RAG chain's knowledge.
TEST_QUESTIONS = [
    "Who is the main character of 'Diary of a Wimpy Kid'?",
    "What is the Cheese Touch?",
    "Describe Greg's relationship with his brother Rodrick.",
]

def run_test(test_function):
    """
    A simple test runner that executes a given test function and prints the result.
    
    Args:
        test_function (function): The test function to execute.
        
    Returns:
        bool: True if the test passes, False otherwise.
    """
    try:
        if test_function():
            print(f"✅ PASS: {test_function.__name__}")
            return True
        else:
            print(f"❌ FAIL: {test_function.__name__}")
            return False
    except Exception as e:
        print(f"💥 ERROR in {test_function.__name__}: {e}")
        return False

def _make_request(method, url, **kwargs):
    """
    A helper function to make HTTP requests with error handling and consistent logging.
    """
    try:
        # Set a default timeout for all requests.
        kwargs.setdefault("timeout", 30)
        response = requests.request(method, url, **kwargs)
        # Raise an exception for bad status codes (4xx or 5xx).
        response.raise_for_status()
        print(f"   -> {method} {url} | Status: {response.status_code}")
        return response
    except requests.exceptions.RequestException as e:
        print(f"   -> ❌ {method} {url} | Error: {e}")
        return None

# --- 3. TEST FUNCTIONS ---

def test_health_check():
    """
    Tests the server's root (`/`) and health (`/health`) endpoints.
    """
    print("\n[Test 1: Health Check]")
    root_response = _make_request("GET", f"{LANGSERVE_URL}/")
    health_response = _make_request("GET", f"{LANGSERVE_URL}/health")
    
    if root_response and health_response:
        data = root_response.json()
        print(f"   Server Message: {data.get('message')}")
        return data.get("status") == "healthy"
    return False

def test_model_availability():
    """
    Tests the OpenAI-compatible `/v1/models` endpoint to ensure our custom
    RAG model is correctly advertised.
    """
    print("\n[Test 2: Model Availability]")
    response = _make_request("GET", f"{LANGSERVE_URL}/v1/models")
    
    if response:
        data = response.json()
        models = [model.get("id") for model in data.get("data", [])]
        print(f"   Available Models: {models}")
        # Check if our specific RAG model is in the list.
        return "wimpy-kid-rag" in models
    return False

def test_rag_non_streaming_queries():
    """
    Sends several questions to the RAG chain via the non-streaming chat endpoint
    and checks for valid responses.
    """
    print("\n[Test 3: RAG Non-Streaming Queries]")
    headers = {"Content-Type": "application/json"}
    
    for i, question in enumerate(TEST_QUESTIONS, 1):
        print(f"\n   Query {i}: '{question}'")
        
        payload = {
            "model": "wimpy-kid-rag",
            "messages": [{"role": "user", "content": question}],
            "stream": False  # Explicitly request a non-streaming response.
        }
        
        response = _make_request("POST", f"{LANGSERVE_URL}/v1/chat/completions", json=payload, headers=headers)
        
        if not response:
            return False
            
        result = response.json()
        content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
        
        if not content or "error" in content.lower():
            print(f"   ❌ Invalid or empty response: {content}")
            return False
        
        print(f"   ✅ Response received ({len(content)} chars): '{content[:80]}...'")
        time.sleep(1) # Small delay to avoid overwhelming the server.
        
    return True

def test_rag_streaming_query():
    """
    Sends a single question to the RAG chain via the streaming chat endpoint
    and verifies that multiple chunks are received.
    """
    print("\n[Test 4: RAG Streaming Query]")
    headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}
    question = "What is the Cheese Touch?"
    print(f"\n   Query: '{question}'")
    
    payload = {
        "model": "wimpy-kid-rag",
        "messages": [{"role": "user", "content": question}],
        "stream": True  # Request a streaming response.
    }
    
    try:
        with requests.post(f"{LANGSERVE_URL}/v1/chat/completions", json=payload, headers=headers, stream=True, timeout=30) as response:
            response.raise_for_status()
            
            chunk_count = 0
            full_response = ""
            
            for line in response.iter_lines():
                if line:
                    line_str = line.decode('utf-8')
                    if line_str.startswith("data:"):
                        if "[DONE]" in line_str:
                            print("   ✅ End-of-stream marker [DONE] received.")
                            break
                        
                        try:
                            data_str = line_str[len("data: "):]
                            chunk = json.loads(data_str)
                            content = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
                            if content:
                                full_response += content
                                chunk_count += 1
                                sys.stdout.write("▪") # Print a character for each chunk received.
                                sys.stdout.flush()
                        except json.JSONDecodeError:
                            print(f"\n   ❌ Failed to decode JSON from line: {line_str}")
                            continue
            
            print(f"\n   ✅ Received {chunk_count} stream chunks.")
            print(f"   Full response ({len(full_response)} chars): '{full_response[:100]}...'")
            
            # A successful stream should have multiple chunks and a non-empty response.
            return chunk_count > 1 and len(full_response) > 10
            
    except requests.exceptions.RequestException as e:
        print(f"   ❌ Streaming request failed: {e}")
        return False

# --- 4. MAIN EXECUTION ---

def main():
    """
    Main function to run all integration tests in sequence.
    """
    print("--- Starting RAG API Integration Tests ---")
    
    # List of all test functions to be executed.
    tests_to_run = [
        test_health_check,
        test_model_availability,
        test_rag_non_streaming_queries,
        test_rag_streaming_query,
    ]
    
    results = [run_test(test) for test in tests_to_run]
    
    print("\n--- Test Summary ---")
    passed_count = sum(results)
    total_count = len(results)
    
    if passed_count == total_count:
        print(f"🎉 All {total_count} tests passed successfully!")
        sys.exit(0)
    else:
        print(f"🔥 {total_count - passed_count} out of {total_count} tests failed.")
        sys.exit(1)

if __name__ == "__main__":
    main()