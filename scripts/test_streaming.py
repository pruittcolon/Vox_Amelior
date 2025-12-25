import asyncio
import websockets
import json

async def test_streaming_connection():
    uri = "ws://localhost:8000/api/v1/salesforce/stream"
    try:
        print(f"Connecting to {uri}...")
        async with websockets.connect(uri) as websocket:
            print("Successfully connected to Salesforce Streaming WebSocket!")
            
            # Send a ping
            await websocket.send("ping")
            print("Sent: ping")
            
            # Wait for response
            response = await websocket.recv()
            print(f"Received: {response}")
            
            assert response == "pong"
            print("TEST PASSED: WebSocket connection established and responsive.")
            
    except Exception as e:
        print(f"TEST FAILED: {e}")

if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(test_streaming_connection())
