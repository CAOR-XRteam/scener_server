import asyncio
import uuid
import json

from beartype import beartype
from server.client import Client
from lib import logger
from server.valider import (
    InputMessage,
    OutputMessage,
    OutputMessageWrapper,
    parse_agent_response,
    convert_image_to_bytes,
)


@beartype
class Session:
    def __init__(self, client: Client):
        """init session by client and assign an ID"""
        self.client = client
        self.thread_id = uuid.uuid1()

    async def run(self):
        """While client keep being actif, handle input messages"""
        while self.client.is_active:
            try:
                message = await self.client.queue_input.get()
                await self.handle_message(message)
                self.client.queue_input.task_done()
            except asyncio.CancelledError:
                logger.info(
                    f"Session {self.thread_id} cancelled for websocket {self.client.websocket.remote_address}"
                )
                break
            except Exception as e:
                # Optional: log or handle processing errors
                logger.error(f"Session error: {e}")
                await self.client.send_message(
                    OutputMessage(
                        status="error",
                        code=500,
                        message=f"Internal server error in thread {self.thread_id}",
                    )
                )
                break

    async def handle_message(self, input_message: InputMessage):
        """handle one client input message - send it to async chat"""
        message = input_message.message
        logger.info(f"Received message in thread {self.thread_id}: {message}")

        try:
            output_generator = self.client.agent.achat(message, str(self.thread_id))
            async for token in output_generator:
                logger.info(f"Token received in thread {self.thread_id}: {token}")
                thinking, final_answer = parse_agent_response(
                    token
                )  # When using stream with qwen, it returns the thinking part and the final answer as one token

                await self.client.send_message(
                    OutputMessageWrapper(
                        output_message=OutputMessage(
                            status="stream",
                            code=200,
                            action="thinking_process",
                            message=thinking,
                        ),
                        additional_data=None,
                    )
                )

                json_response = json.loads(final_answer)
                if json_response.get("action") == "image_generation":
                    try:
                        generated_images_data = json_response.get(
                            "generated_images_data"
                        )
                        data = []
                        for image_data in generated_images_data:
                            image_path = image_data.get("path")
                            if image_path:
                                data.append(convert_image_to_bytes(image_path))

                        await self.client.send_message(
                            OutputMessageWrapper(
                                output_message=OutputMessage(
                                    status="stream",
                                    code=200,
                                    action="image_generation",
                                    message=json_response.get("message"),
                                ),
                                additional_data=data,
                            )
                        )
                    except Exception as e:
                        logger.error(f"Error converting image to bytes: {e}")
                        await self.client.send_message(
                            OutputMessageWrapper(
                                output_message=OutputMessage(
                                    status="error",
                                    code=500,
                                    action="image_generation",
                                    message=f"Error converting image to bytes: {e}",
                                ),
                                additional_data=None,
                            )
                        )
                elif json_response.get("action") == "agent_response":
                    await self.client.send_message(
                        OutputMessageWrapper(
                            output_message=OutputMessage(
                                status="stream",
                                code=200,
                                action="agent_response",
                                message=json_response.get("message"),
                            ),
                            additional_data=None,
                        )
                    )
                else:
                    await self.client.send_message(
                        OutputMessageWrapper(
                            output_message=OutputMessage(
                                status="error",
                                code=400,
                                action="agent_response",
                                message=f"Unknown action in response: {json_response.get('action')}",
                            ),
                            additional_data=None,
                        )
                    )

            logger.info(f"Stream completed for thread {self.thread_id}")
        except asyncio.CancelledError:
            logger.info(
                f"Stream cancelled for thread {self.thread_id} for websocket {self.client.websocket.remote_address}"
            )
            raise
        except Exception as e:
            logger.error(f"Error during chat stream: {e}")
            await self.client.send_message(
                OutputMessageWrapper(
                    OutputMessage(
                        status="error",
                        code=500,
                        action="agent_response",
                        message=f"Error during chat stream in thread {self.thread_id}",
                    ),
                    additional_data=None,
                )
            )
