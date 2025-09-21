#!/usr/bin/env python3
"""
End-to-End Playwright Test for Open WebUI RAG Functionality

This script uses Playwright to automate browser interaction and verify the complete
Retrieval-Augmented Generation (RAG) pipeline. It ensures that the frontend (Open WebUI)
can correctly communicate with the backend (LangServe API) and display the expected
response from the RAG chain.
"""

# --- 1. IMPORTS ---
import asyncio
import pytest
from playwright.async_api import async_playwright, Page, expect

# --- 2. PYTEST CONFIGURATION ---

# Mark the test as an asyncio test, allowing the use of `await`.
@pytest.mark.asyncio
async def test_openwebui_rag_functionality():
    """
    Tests the end-to-end RAG functionality with Open WebUI.
    
    Steps:
    1.  Launches a headless browser using Playwright.
    2.  Navigates to the Open WebUI frontend.
    3.  Locates the chat input, types a specific question, and submits it.
    4.  Waits for the assistant's response container to appear and verify it contains
        the expected answer ("Greg Heffley").
    5.  Captures a screenshot on failure for easier debugging.
    """
    async with async_playwright() as p:
        # --- 3. BROWSER SETUP ---
        print("\n🚀 Launching browser...")
        # Launch a headless Chromium browser. `headless=True` is essential for CI environments.
        browser = await p.chromium.launch(headless=True)
        # Create a new browser context with a specific viewport size.
        context = await browser.new_context(viewport={'width': 1920, 'height': 1080})
        page: Page = await context.new_page()

        # --- 4. NETWORK LOGGING (FOR DEBUGGING) ---
        # These lists will store network requests and responses related to the chat API.
        api_requests = []
        api_responses = []

        def log_request(request):
            """Callback to log outgoing requests to the chat completions API."""
            if 'chat/completions' in request.url:
                print(f"➡️  API Request Sent: {request.method} {request.url}")
                api_requests.append(f"{request.method} {request.url}")

        def log_response(response):
            """Callback to log incoming responses from the chat completions API."""
            if 'chat/completions' in response.url:
                print(f"⬅️  API Response Received: {response.status} {response.url}")
                api_responses.append(f"{response.status} {response.url}")

        # Register the logging callbacks with the page.
        page.on("request", log_request)
        page.on("response", log_response)

        try:
            # --- 5. TEST EXECUTION ---
            print("🌐 Navigating to http://localhost:3000...")
            # Navigate to the web UI. `wait_until="networkidle"` waits for network activity to cease.
            await page.goto("http://localhost:3000", wait_until="networkidle", timeout=20000)
            print("✅ Page loaded successfully.")

            # Define the question and the expected text in the answer.
            question = "What is the main character name in Diary of a Wimpy Kid?"
            expected_text = "Greg Heffley"

            # Locate the chat input field (a contenteditable div) and type the question.
            print(f"💬 Typing question: '{question}'")
            chat_input_locator = page.locator('[contenteditable="true"]')
            await chat_input_locator.fill(question)
            
            # Click the submit button to send the message.
            print("📤 Clicking send button...")
            await page.locator('button[type="submit"]').click()

            # This is the core assertion of the test.
            # We locate the container for the assistant's messages and assert that it
            # contains the expected text. `expect` handles waiting automatically.
            print(f"⏳ Waiting for response containing '{expected_text}'...")
            assistant_response_locator = page.locator(".chat-assistant").last()
            
            await expect(assistant_response_locator).to_contain_text(
                expected_text, 
                timeout=30000  # Generous 30-second timeout for the RAG chain to respond.
            )

            print(f"🎉 TEST PASSED: Found response containing '{expected_text}'.")
            
            # Optional: Log the final response for verification.
            final_response_text = await assistant_response_locator.inner_text()
            print(f"📝 Assistant's final response preview: {final_response_text[:300]}...")

        except Exception as e:
            # --- 6. FAILURE HANDLING ---
            print(f"❌ TEST FAILED: An error occurred.")
            print(f"Error details: {e}")
            # Save a screenshot to diagnose the UI state at the time of failure.
            await page.screenshot(path="test_failure_screenshot.png")
            print("📸 Screenshot saved to 'test_failure_screenshot.png'")
            # Re-raise the exception to ensure the test is marked as failed by pytest.
            raise

        finally:
            # --- 7. CLEANUP AND DEBUGGING OUTPUT ---
            print("\n--- 🕵️‍♂️ Debugging Info ---")
            print(f"Total API Requests to 'chat/completions': {len(api_requests)}")
            print(f"Total API Responses from 'chat/completions': {len(api_responses)}")
            print("--------------------------\n")
            # Close the browser to clean up resources.
            await browser.close()
            print("🚪 Browser closed.")

# --- 8. SCRIPT EXECUTION ---
if __name__ == "__main__":
    """
    Allows the script to be run directly for debugging purposes without needing pytest.
    Example: `python test_openwebui.py`
    """
    print("Running test directly without pytest...")
    asyncio.run(test_openwebui_rag_functionality())
