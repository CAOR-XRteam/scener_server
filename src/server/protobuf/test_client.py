import asyncio
import websockets
import message_pb2  # your generated protobuf module

async def main():
    uri = "ws://localhost:8765"
    async with websockets.connect(uri) as ws:
        loop = asyncio.get_event_loop()
        while True:
            line = await loop.run_in_executor(None, input, "> ")
            if line.lower() in {"exit", "quit"}:
                break

            # Wrap the typed line in protobuf message
            msg = message_pb2.Content()
            msg.type = "text"
            msg.json = f'{{"user":"{line}"}}'

            await ws.send(msg.SerializeToString())
            print("Protobuf message sent.")

asyncio.run(main())
