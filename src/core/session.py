import json
import asyncio
#import model.llm.chat


class Session:
    def __init__(self, client):
        self.client = client

    async def run(self):
        while self.client.is_active:
            try:
                message = await self.client.queue_input.get()
                await self.handle_message(message)
            except Exception as e:
                # Optional: log or handle processing errors
                print(f"Session error: {e}")
                break

    async def handle_message(self, message):
        j = json.loads(message)
        if j.get("command") == "chat":
            input = j.get("message")
            #output = model.llm.chat.prompt(input)
            await self.client.send_message("success", 200, output)
        else:
            await self.client.send_message("error", 400, "Command not recognized")
