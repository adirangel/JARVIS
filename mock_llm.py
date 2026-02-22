"""Mock LLM for testing."""
import asyncio

def mock_stream_response(messages):
    response = """Sir, the current system time is 9:15 PM, Saturday, February 22, 2026. 
All systems are operational. I hope you are having a pleasant evening."""
    for word in response.split():
        yield word + " "
        asyncio.sleep(0.05)
