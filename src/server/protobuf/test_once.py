import asyncio
import websockets
import message_pb2  # your generated protobuf module

async def main():
    uri = "ws://localhost:8765"
    async with websockets.connect(uri) as ws:
        # Send initial protobuf message
        initial_msg = message_pb2.Content()
        initial_msg.type = "json"
        initial_msg.json = '{"hello":"world"}'
        await ws.send(initial_msg.SerializeToString())

        print("Protobuf message sent.")

asyncio.run(main())
