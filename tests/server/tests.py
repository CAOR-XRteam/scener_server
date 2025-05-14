import asyncio
from unittest.mock import AsyncMock, MagicMock, patch, call, Mock
import pytest
from server.valider import InputMessage, OutputMessage

from server.client import Client
from server.session import Session
from server.server import Server

import websockets.exceptions
import uuid
import signal
from colorama import Fore


@pytest.fixture
def mock_ws():
    ws = AsyncMock()
    ws.remote_address = ("127.0.0.1", 12345)
    return ws


@pytest.fixture
def mock_agent():
    agent_instance = Mock()
    agent_instance.achat = AsyncMock()

    return agent_instance


@pytest.fixture
def mock_client(mock_ws, mock_agent):
    return Client(mock_ws, mock_agent)


class MockWsServer:
    def __init__(self):
        self.close = Mock()
        self.wait_closed = AsyncMock()
        self._is_serving = True

    def is_serving(self):
        return self._is_serving


class TestServer:
    def run_coroutine(self, coroutine):
        return asyncio.run(coroutine)

    @pytest.fixture
    def mock_server(self, mock_agent):
        with patch(
            "server.server.AgentAPI", new_callable=Mock, return_value=mock_agent
        ):
            with patch("server.server.logger"):
                server = Server(host="0.0.0.0", port=8765)
                server.handler_shutdown = Mock()

                return server

    @patch("server.server.AgentAPI")
    @patch("server.server.logger")
    def test_init_success(self, mock_logger, mock_agent_api, mock_agent):
        mock_agent_api.return_value = mock_agent

        server = Server(host="0.0.0.0", port=8765)

        assert server.host == "0.0.0.0"
        assert server.port == 8765
        assert server.list_client is not None
        assert server.server is None
        assert server.agent is mock_agent

        mock_agent_api.assert_called_once()

        mock_logger.info.assert_called_once_with(
            "AgentAPI initialized successfully at server startup."
        )

    @patch("server.server.AgentAPI")
    @patch("server.server.logger")
    def test_init_agent_error(self, mock_logger, mock_agent_api):
        err = ValueError("test")
        mock_agent_api.side_effect = err

        server = Server(host="0.0.0.0", port=8765)

        assert server.host == "0.0.0.0"
        assert server.port == 8765
        assert server.list_client is not None
        assert server.server is None
        assert server.agent is None

        mock_agent_api.assert_called_once()

        mock_logger.critical.assert_called_once_with(
            f"Failed to initialize AgentAPI at server startup: {err}"
        )

    @patch("server.server.asyncio.get_event_loop")
    @patch("server.server.logger")
    def test_start_success(self, mock_logger, mock_get_event_loop, mock_server):
        async def mock_run_server():
            mock_server.server = mock_ws_server
            await asyncio.sleep(0.01)
            mock_ws_server._is_serving = False

        mock_loop = MagicMock()
        mock_get_event_loop.return_value = mock_loop
        mock_ws_server = MockWsServer()
        mock_server.run = AsyncMock(side_effect=mock_run_server)
        mock_loop.run_until_complete.side_effect = self.run_coroutine

        mock_server.start()

        mock_server.run.assert_awaited_once()

        mock_get_event_loop.assert_called_once()
        assert mock_loop.run_until_complete.call_count == 2
        mock_loop.add_signal_handler.assert_called_once_with(
            signal.SIGINT, mock_server.handler_shutdown
        )

        mock_server.server.wait_closed.assert_awaited_once()

        mock_logger.info.assert_called_once_with("Server finished working.")

    @patch("server.server.asyncio.get_event_loop")
    @patch("server.server.logger")
    def test_start_exception(self, mock_logger, mock_get_event_loop, mock_server):
        mock_loop = MagicMock()
        mock_get_event_loop.return_value = mock_loop
        err = ValueError("test")
        mock_server.run = AsyncMock(side_effect=err)
        mock_server.server = MockWsServer()
        mock_loop.run_until_complete.side_effect = self.run_coroutine

        mock_server.start()

        mock_server.run.assert_awaited_once()

        mock_get_event_loop.assert_called_once()
        mock_loop.add_signal_handler.assert_called_once_with(
            signal.SIGINT, mock_server.handler_shutdown
        )
        assert mock_loop.run_until_complete.call_count == 1

        mock_logger.error.assert_called_once_with(
            f"Error in server's main execution: {err}"
        )

        mock_logger.info.assert_called_once_with("Server finished working.")

    @pytest.mark.asyncio
    @patch("server.server.websockets.serve", new_callable=AsyncMock)
    @patch("server.server.logger")
    async def test_run_success(self, mock_logger, mock_ws_serve, mock_server):
        mock_ws_server = MockWsServer()
        mock_ws_serve.return_value = mock_ws_server

        await mock_server.run()

        mock_ws_serve.assert_awaited_once_with(
            mock_server.handler_client, mock_server.host, mock_server.port
        )
        mock_logger.info.assert_called_once_with(
            f"Server running on {Fore.GREEN}ws://{mock_server.host}:{mock_server.port}{Fore.GREEN}"
        )
        mock_ws_server.wait_closed.assert_awaited_once()

    @pytest.mark.asyncio
    @patch("server.server.websockets.serve", new_callable=AsyncMock)
    @patch("server.server.logger")
    async def test_run_os_error(self, mock_logger, mock_ws_serve, mock_server):
        err = OSError("test")
        mock_ws_serve.side_effect = err

        await mock_server.run()

        assert mock_server.shutdown_event.is_set()

        mock_ws_serve.assert_awaited_once_with(
            mock_server.handler_client, mock_server.host, mock_server.port
        )

        mock_logger.error.assert_called_once_with(
            f"Could not start server on {Fore.GREEN}ws://{mock_server.host}:{mock_server.port}{Fore.GREEN}: {err}."
        )

    @pytest.mark.asyncio
    @patch("server.server.websockets.serve", new_callable=AsyncMock)
    @patch("server.server.logger")
    async def test_run_other_exception(self, mock_logger, mock_ws_serve, mock_server):
        err = ValueError("test")
        mock_ws_serve.side_effect = err

        await mock_server.run()

        assert mock_server.shutdown_event.is_set()

        mock_ws_serve.assert_awaited_once_with(
            mock_server.handler_client, mock_server.host, mock_server.port
        )

        mock_logger.error.assert_called_once_with(
            f"Internal error during server run: {err}"
        )

    @pytest.mark.asyncio
    @patch("server.server.logger")
    async def test_handler_client_success(self, mock_logger, mock_server, mock_client):
        pass


class TestSession:
    async def _mock_achat_gen(self, tokens):
        for token in tokens:
            await asyncio.sleep(0)
            yield token

    @pytest.fixture
    def mock_session(self, mock_client):
        with patch("server.session.uuid.uuid1") as mock_uuid1:
            with patch("server.session.logger") as mock_logger:
                mock_uuid1.return_value = uuid.UUID(
                    "11111111-1111-1111-1111-111111111111"
                )
                return Session(mock_client)

    @pytest.mark.asyncio
    @patch("server.session.logger")
    async def test_run_success(self, mock_logger, mock_session):

        message1 = InputMessage(command="chat", message="test1")
        message2 = InputMessage(command="chat", message="test2")

        mock_session.client.queue_input.get = AsyncMock(
            side_effect=[
                message1,
                message2,
                asyncio.CancelledError,
            ]
        )
        mock_session.client.queue_input.task_done = MagicMock()

        handle_message_mock = AsyncMock()
        mock_session.handle_message = handle_message_mock
        mock_session.client.is_active = True
        await mock_session.run()

        assert mock_session.client.queue_input.get.await_count == 3
        assert mock_session.client.queue_input.task_done.call_count == 2

        handle_message_mock.assert_has_awaits(
            [
                call(message1),
                call(message2),
            ]
        )

        mock_session.client.queue_input.task_done.assert_called_with()
        assert mock_session.client.queue_input.task_done.call_count == 2
        assert mock_session.handle_message.call_count == 2

        mock_logger.info.assert_called_once_with(
            f"Session {mock_session.thread_id} cancelled for websocket {mock_session.client.websocket.remote_address}"
        )

    @pytest.mark.asyncio
    @patch("server.session.logger")
    async def test_run_cancelled_error(self, mock_logger, mock_session):
        mock_session.client.queue_input.get = AsyncMock(
            side_effect=asyncio.CancelledError,
        )
        mock_session.client.queue_input.task_done = AsyncMock()
        handle_message_mock = AsyncMock()
        mock_session.handle_message = handle_message_mock

        await mock_session.run()

        mock_session.handle_message.assert_not_called()
        mock_session.client.queue_input.task_done.assert_not_called()

        mock_logger.info.assert_called_once_with(
            f"Session {mock_session.thread_id} cancelled for websocket {mock_session.client.websocket.remote_address}"
        )

    @pytest.mark.asyncio
    @patch("server.session.logger")
    async def test_run_other_exception(self, mock_logger, mock_session):
        err = ValueError("test")
        message = InputMessage(command="chat", message="test")
        mock_session.client.queue_input.get = AsyncMock(
            side_effect=[
                message,
                asyncio.CancelledError(),
            ]
        )
        mock_session.client.queue_input.task_done = AsyncMock()
        mock_session.client.send_message = AsyncMock()

        handle_message_mock = AsyncMock(side_effect=err)
        mock_session.handle_message = handle_message_mock

        await mock_session.run()

        mock_session.client.queue_input.get.assert_awaited_once()
        mock_session.client.queue_input.task_done.assert_not_called()

        handle_message_mock.assert_awaited_once_with(message)

        mock_session.client.send_message.assert_awaited_once_with(
            OutputMessage(
                status="error",
                code=500,
                message=f"Internal server error in thread {mock_session.thread_id}",
            )
        )

        mock_logger.error.assert_called_once_with(f"Session error: {err}")

    @pytest.mark.asyncio
    @patch("server.session.logger")
    async def test_handle_message_success(
        self,
        mock_logger,
        mock_session,
    ):
        message = InputMessage(command="chat", message="test test test")
        tokens = ["test ", "has...", "passed"]
        mock_session.client.send_message = AsyncMock()
        mock_session.client.agent.achat.return_value = self._mock_achat_gen(tokens)

        await mock_session.handle_message(message)

        mock_session.client.agent.achat.assert_called_once_with(
            message.message, str(mock_session.thread_id)
        )

        assert mock_session.client.send_message.await_count == len(tokens)
        mock_session.client.send_message.assert_has_awaits(
            [
                call(OutputMessage(status="stream", code=200, message=token))
                for token in tokens
            ],
            any_order=False,
        )

        assert mock_logger.info.call_count == 2
        mock_logger.info.assert_any_call(
            f"Received message in thread {mock_session.thread_id}: {message.message}"
        )
        mock_logger.info.assert_any_call(
            f"Stream completed for thread {mock_session.thread_id}"
        )

    @pytest.mark.asyncio
    @patch("server.session.logger")
    async def test_handle_message_achat_cancelled_error(
        self, mock_logger, mock_session
    ):
        message = InputMessage(command="chat", message="test")
        mock_session.client.agent.achat.side_effect = asyncio.CancelledError("oups")
        mock_session.client.send_message = AsyncMock()

        with pytest.raises(asyncio.CancelledError, match="oups"):
            await mock_session.handle_message(message)

        mock_session.client.send_message.assert_not_called()

        assert mock_logger.info.call_count == 2
        mock_logger.info.assert_any_call(
            f"Received message in thread {mock_session.thread_id}: {message.message}"
        )
        mock_logger.info.assert_any_call(
            f"Stream cancelled for thread {mock_session.thread_id} for websocket {mock_session.client.websocket.remote_address}"
        )

    @pytest.mark.asyncio
    @patch("server.session.logger")
    async def test_handle_message_achat_other_exception(
        self, mock_logger, mock_session
    ):
        err = ValueError("test")
        message = InputMessage(command="chat", message="test")
        mock_session.client.agent.achat.side_effect = err
        mock_session.client.send_message = AsyncMock()

        await mock_session.handle_message(message)

        mock_session.client.send_message.assert_awaited_once_with(
            OutputMessage(
                status="error",
                code=500,
                message=f"Error during chat stream in thread {mock_session.thread_id}",
            )
        )

        mock_logger.info.assert_called_once_with(
            f"Received message in thread {mock_session.thread_id}: {message.message}"
        )
        mock_logger.error.assert_called_once_with(f"Error during chat stream: {err}")

    @pytest.mark.asyncio
    @patch("server.session.logger")
    async def test_handle_message_stream_cancelled_error(
        self, mock_logger, mock_session
    ):
        message = InputMessage(command="chat", message="test test test")
        tokens = ["test ", "has...", "passed"]
        mock_session.client.send_message = AsyncMock(
            side_effect=asyncio.CancelledError("oups")
        )
        mock_session.client.agent.achat.return_value = self._mock_achat_gen(tokens)

        with pytest.raises(asyncio.CancelledError, match="oups"):
            await mock_session.handle_message(message)

        mock_session.client.agent.achat.assert_called_once_with(
            message.message, str(mock_session.thread_id)
        )

        mock_session.client.send_message.assert_awaited_once()

        assert mock_logger.info.call_count == 2
        mock_logger.info.assert_any_call(
            f"Received message in thread {mock_session.thread_id}: {message.message}"
        )
        mock_logger.info.assert_any_call(
            f"Stream cancelled for thread {mock_session.thread_id} for websocket {mock_session.client.websocket.remote_address}"
        )

    @pytest.mark.asyncio
    @patch("server.session.logger")
    async def test_handle_message_stream_other_exception(
        self, mock_logger, mock_session
    ):
        message = InputMessage(command="chat", message="test test test")
        tokens = ["test ", "has...", "passed"]
        err = ValueError("oups")
        mock_session.client.send_message = AsyncMock(side_effect=err)
        mock_session.client.agent.achat.return_value = self._mock_achat_gen(tokens)

        with pytest.raises(ValueError, match="oups"):
            await mock_session.handle_message(message)

        mock_session.client.agent.achat.assert_called_once_with(
            message.message, str(mock_session.thread_id)
        )
        assert mock_session.client.send_message.call_count == 2
        mock_session.client.send_message.assert_awaited_with(
            OutputMessage(
                status="error",
                code=500,
                message=f"Error during chat stream in thread {mock_session.thread_id}",
            )
        )

        mock_logger.info.assert_called_once_with(
            f"Received message in thread {mock_session.thread_id}: {message.message}"
        )
        mock_logger.error.assert_called_once_with(f"Error during chat stream: {err}")


class TestClient:
    @patch("server.session.Session")
    @pytest.mark.asyncio
    async def test_start_client(self, mock_session, mock_client):
        session_mock = MagicMock()
        mock_session.return_value = session_mock
        session_mock.run = AsyncMock()

        mock_client.start()

        assert mock_client.session is session_mock
        assert mock_client.is_active is True
        assert isinstance(mock_client.queue_input, asyncio.Queue)
        assert isinstance(mock_client.queue_output, asyncio.Queue)
        assert isinstance(mock_client.disconnection, asyncio.Event)
        assert mock_client.task_input is not None
        assert mock_client.task_output is not None
        assert mock_client.task_session is not None

        mock_session.assert_called_once_with(mock_client)
        session_mock.run.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_message_success(self, mock_client):
        message = OutputMessage(status="stream", code=200, message="test")
        await mock_client.send_message(message)

        assert not mock_client.queue_output.empty()

        queued_message = await mock_client.queue_output.get()

        assert queued_message == OutputMessage(
            status="stream", code=200, message="test"
        )

    @pytest.mark.asyncio
    @patch("server.client.logger")
    async def test_send_message_cancelled_error(self, mock_logger, mock_client):
        mock_client.queue_output.put = AsyncMock(
            side_effect=asyncio.CancelledError("test")
        )
        message = OutputMessage(status="stream", code=200, message="test")

        with pytest.raises(asyncio.CancelledError, match="test"):
            await mock_client.send_message(message)

        mock_client.queue_output.put.assert_awaited_once_with(message)
        mock_logger.error.assert_called_once_with(
            f"Task was cancelled while sending message to {Fore.GREEN}{mock_client.websocket.remote_address}{Fore.RESET}, initial message: {message}"
        )

    @pytest.mark.asyncio
    @patch("server.client.logger")
    async def test_send_message_other_exception(
        self, mock_logger, mock_ws, mock_client
    ):
        err = ValueError("error")
        mock_client.queue_output.put = AsyncMock(side_effect=err)
        message = OutputMessage(status="stream", code=200, message="test")

        await mock_client.send_message(message)

        mock_client.queue_output.put.assert_awaited_once_with(message)

        mock_logger.error.assert_called_once_with(
            f"Error queuing message for {Fore.GREEN}{mock_ws.remote_address}{Fore.RESET}: {err}, initial message: {message}"
        )

    @pytest.mark.asyncio
    @patch("server.client.logger")
    @patch("server.client.Client.close", new_callable=AsyncMock)
    async def test_loop_input_success(
        self,
        mock_close: AsyncMock,
        mock_logger: MagicMock,
        mock_client: Client,
        mock_ws: AsyncMock,
    ):
        mock_ws.__aiter__.return_value = ["test1", "test2"]

        async def close_side_effect():
            mock_client.is_active = False

        mock_close.side_effect = close_side_effect
        mock_client.is_active = True

        await mock_client.loop_input()

        mock_close.assert_awaited_once()
        mock_logger.error.assert_not_called()

        queued_messages = []
        while not mock_client.queue_input.empty():
            item = await asyncio.wait_for(mock_client.queue_input.get(), timeout=0.1)
            queued_messages.append(item)
            mock_client.queue_input.task_done()

        assert queued_messages == [
            InputMessage(command="chat", message="test1"),
            InputMessage(command="chat", message="test2"),
        ]

    @pytest.mark.asyncio
    @patch("server.client.logger")
    async def test_loop_input_empty_message(self, mock_logger, mock_client, mock_ws):
        mock_ws.__aiter__.return_value = [""]
        mock_client.queue_input.put = AsyncMock()
        mock_client.send_message = AsyncMock()

        await mock_client.loop_input()

        mock_client.queue_input.put.assert_not_awaited()
        mock_client.send_message.assert_awaited_once()

        error_message = mock_client.send_message.await_args.args[0]
        assert isinstance(error_message, OutputMessage)
        assert error_message.status == "error"
        assert error_message.code == 400
        assert "Invalid input" in error_message.message

        assert mock_client.is_active is False

        mock_logger.error.assert_called_once()
        log_message = mock_logger.error.call_args[0][0]
        assert (
            f"Validation error for client {mock_client.websocket.remote_address}:"
            in log_message
        )

    @pytest.mark.asyncio
    @patch("server.client.logger")
    async def test_loop_input_cancelled_error(self, mock_logger, mock_client, mock_ws):
        mock_ws.__aiter__.return_value = ["test"]
        mock_client.queue_input.put = AsyncMock(side_effect=asyncio.CancelledError)

        await mock_client.loop_input()

        mock_client.queue_input.put.assert_awaited_once()
        assert mock_client.queue_input.empty()
        assert mock_client.is_active is False

        mock_logger.error.assert_called_once_with(
            f"Task cancelled for {Fore.GREEN}{mock_ws.remote_address}{Fore.RESET}"
        )

    @pytest.mark.asyncio
    @patch("server.client.logger")
    async def test_loop_input_connection_closed(
        self, mock_logger, mock_client, mock_ws
    ):
        mock_ws.__aiter__ = MagicMock(return_value=mock_ws)
        err = websockets.exceptions.ConnectionClosed(rcvd=None, sent=None)
        mock_ws.__anext__.side_effect = err

        await mock_client.loop_input()

        assert mock_client.queue_input.empty()
        assert mock_client.is_active is False

        mock_logger.error.assert_called_once_with(
            f"Client {Fore.GREEN}{mock_ws.remote_address}{Fore.RESET} disconnected. Reason: {err}"
        )

    @pytest.mark.asyncio
    @patch("server.client.logger")
    async def test_loop_input_other_exception(self, mock_logger, mock_client, mock_ws):
        err = ValueError("test")
        mock_ws.__aiter__ = MagicMock(return_value=mock_ws)
        mock_ws.__anext__.side_effect = err

        await mock_client.loop_input()

        assert mock_client.queue_input.empty()
        assert mock_client.is_active is False

        mock_logger.error.assert_called_once_with(
            f"Error with client {Fore.GREEN}{mock_ws.remote_address}{Fore.RESET}: {err}"
        )

    @pytest.mark.asyncio
    @patch("server.client.logger")
    async def test_loop_output_success(self, mock_logger, mock_client, mock_ws):
        message = OutputMessage(status="stream", code=200, message="test")
        mock_client.queue_output.get = AsyncMock(
            side_effect=[message, ValueError("test")]
        )

        await mock_client.loop_output()

        mock_ws.send.assert_awaited_once_with(message.message)

        mock_logger.info.assert_called_once_with(
            f"Sent message to {Fore.GREEN}{mock_ws.remote_address}{Fore.RESET}:\n {message}"
        )

    @pytest.mark.asyncio
    @patch("server.client.logger")
    async def test_loop_output_cancelled_error(self, mock_logger, mock_client, mock_ws):
        mock_client.queue_output.get = AsyncMock(side_effect=asyncio.CancelledError)

        await mock_client.loop_output()

        mock_client.queue_output.get.assert_awaited_once()

        mock_ws.send.assert_not_awaited()

        mock_logger.info.assert_called_once_with(
            f"Task cancelled for {Fore.GREEN}{mock_ws.remote_address}{Fore.RESET}"
        )

    @pytest.mark.asyncio
    @patch("server.client.logger")
    async def test_loop_output_other_exception_on_get(
        self, mock_logger, mock_client, mock_ws
    ):
        err = ValueError("get_error")
        mock_client.queue_output.get = AsyncMock(side_effect=err)

        await mock_client.loop_output()

        mock_client.queue_output.get.assert_awaited_once()

        mock_ws.send.assert_not_awaited()

        mock_logger.error.assert_called_once_with(
            f"Error sending message to {Fore.GREEN}{mock_ws.remote_address}{Fore.RESET}: {err}"
        )

    @pytest.mark.asyncio
    @patch("server.client.logger")
    async def test_loop_output_other_exception_on_send(
        self, mock_logger, mock_client, mock_ws
    ):
        message = OutputMessage(status="stream", code=200, message="test")
        mock_client.queue_output.get = AsyncMock(
            side_effect=[message, asyncio.CancelledError]
        )
        err = ValueError("send_error")
        mock_ws.send.side_effect = err

        await mock_client.loop_output()

        mock_client.queue_output.get.assert_awaited_once()
        assert mock_client.is_active is False

        mock_ws.send.assert_awaited_once_with(message.message)

        mock_logger.error.assert_called_once_with(
            f"Error sending message to {Fore.GREEN}{mock_ws.remote_address}{Fore.RESET}: {err}"
        )

    @pytest.mark.asyncio
    @patch("server.client.logger")
    async def test_close_success(self, mock_logger, mock_client, mock_ws):
        mock_client.task_input = asyncio.create_task(asyncio.sleep(1))
        mock_client.task_output = asyncio.create_task(asyncio.sleep(1))
        mock_client.task_session = asyncio.create_task(asyncio.sleep(1))
        mock_client.disconnection = MagicMock(spec=asyncio.Event)

        await mock_client.queue_input.put(InputMessage(command="chat", message="test"))
        await mock_client.queue_output.put(
            OutputMessage(status="stream", code=123, message="test")
        )

        await mock_client.close()

        assert not mock_client.is_active
        mock_ws.close.assert_awaited_once()
        assert mock_client.queue_input.empty()
        assert mock_client.queue_output.empty()

        mock_client.disconnection.set.assert_called_once()

        mock_logger.info.assert_called_once_with(
            f"Closing connection for {mock_ws.remote_address}"
        )

    @pytest.mark.asyncio
    @patch("server.client.logger")
    async def test_close_ws_error(self, mock_logger, mock_client, mock_ws):
        mock_client.task_input = asyncio.create_task(asyncio.sleep(1))
        mock_client.task_output = asyncio.create_task(asyncio.sleep(1))
        mock_client.task_session = asyncio.create_task(asyncio.sleep(1))
        mock_client.disconnection = MagicMock(spec=asyncio.Event)

        err = websockets.exceptions.ConnectionClosed(rcvd=None, sent=None)
        mock_ws.close.side_effect = err

        await mock_client.queue_input.put(InputMessage(command="chat", message="test"))
        await mock_client.queue_output.put(
            OutputMessage(status="stream", code=123, message="test")
        )

        await mock_client.close()

        assert not mock_client.is_active
        mock_ws.close.assert_awaited_once()
        assert mock_client.queue_input.empty()
        assert mock_client.queue_output.empty()

        mock_client.disconnection.set.assert_called_once()

        mock_logger.info.assert_called_once_with(
            f"Closing connection for {mock_ws.remote_address}"
        )
        mock_logger.error.assert_called_once_with(
            f"Error closing websocket connection for {mock_ws.remote_address}: {err}"
        )
