#!/usr/bin/env python

import asyncio
import websockets

def send_text(msg, event_loop):
    async def send_(text):
        uri = "ws://localhost:8000"
        async with websockets.connect(uri) as websocket:
            await websocket.send('{"text":"%s", "label":"msg"}' % text)
    
    try:
        event_loop.run_until_complete(send_(msg))
    except Exception as e:
        print(e)
    
if __name__ == '__main__':
    event_loop = asyncio.get_event_loop()
    send_text("Hello", event_loop)