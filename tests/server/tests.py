import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
from server.valider import InputMessage, OutputMessage

from server.client import Client

import websockets.exceptions
from colorama import Fore


class TestServer:
    # TODO
    pass


class TestSession:
    # TODO
    pass


class TestClient:
    @pytest.fixture
    def mock_ws(self):
        ws = AsyncMock()
        ws.remote_address = ("127.0.0.1", 12345)
        return ws

    @pytest.fixture
    def mock_client(self, mock_ws):
        return Client(mock_ws)

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
        msg = OutputMessage(status="stream", code=200, message="test")
        await mock_client.send_message(msg)

        assert not mock_client.queue_output.empty()
        queued_msg = await mock_client.queue_output.get()
        assert queued_msg == OutputMessage(status="stream", code=200, message="test")

    @pytest.mark.asyncio
    @patch("server.client.logger")
    async def test_send_message_cancelled_error(self, mock_logger, mock_client):
        mock_client.queue_output.put = AsyncMock()
        msg = OutputMessage(status="stream", code=200, message="test")
        mock_client.queue_output.put.side_effect = asyncio.CancelledError

        with pytest.raises(asyncio.CancelledError):
            await mock_client.send_message(msg)

        mock_client.queue_output.put.assert_awaited_once_with(msg)
        mock_logger.assert_not_called()

    @pytest.mark.asyncio
    @patch("server.client.logger")
    async def test_send_message_other_exception(self, mock_logger, mock_client):
        mock_client.queue_output.put = AsyncMock()
        msg = OutputMessage(status="stream", code=200, message="test")
        mock_client.queue_output.put.side_effect = ValueError("error")

        await mock_client.send_message(msg)

        mock_client.queue_output.put.assert_awaited_once_with(msg)

        mock_logger.error.assert_called_once()
        log_msg = mock_logger.error.call_args[0][0]

        assert "Error queuing message for" in log_msg

    @pytest.mark.asyncio
    @patch("server.client.logger")
    async def test_loop_input_success(self, mock_logger, mock_client, mock_ws):
        mock_ws.__aiter__.return_value = ["test1", "test2"]
        await mock_client.loop_input()

        queued_messages = []
        while not mock_client.queue_input.empty():
            queued_messages.append(await mock_client.queue_input.get())

        assert queued_messages == [
            InputMessage(command="chat", message="test1"),
            InputMessage(command="chat", message="test2"),
        ]
        assert mock_client.is_active is False
        mock_logger.assert_not_called()

    @pytest.mark.asyncio
    @patch("server.client.logger")
    async def test_loop_input_cancelled_error(self, mock_logger, mock_client, mock_ws):
        mock_ws.__aiter__.return_value = ["test"]
        mock_client.queue_input.put = AsyncMock(side_effect=asyncio.CancelledError)
        await mock_client.loop_input()

        mock_client.queue_input.put.assert_awaited_once()
        assert mock_client.queue_input.empty()
        mock_logger.error.assert_called_once()
        log_msg = mock_logger.error.call_args[0][0]
        assert (
            f"Task cancelled for {Fore.GREEN}{mock_ws.remote_address}{Fore.RESET}"
            in log_msg
        )
        assert mock_client.is_active is False

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

        mock_logger.error.assert_called_once()
        log_msg = mock_logger.error.call_args[0][0]
        assert (
            f"Client {Fore.GREEN}{mock_ws.remote_address}{Fore.RESET} disconnected. Reason: {err}"
            in log_msg
        )

        assert mock_client.is_active is False

    @pytest.mark.asyncio
    @patch("server.client.logger")
    async def test_loop_input_other_exception(self, mock_logger, mock_client, mock_ws):
        err = ValueError("test")
        mock_ws.__aiter__ = MagicMock(return_value=mock_ws)
        mock_ws.__anext__.side_effect = err

        await mock_client.loop_input()

        assert mock_client.queue_input.empty()

        mock_logger.error.assert_called_once()
        log_msg = mock_logger.error.call_args[0][0]
        assert (
            f"Error with client {Fore.GREEN}{mock_ws.remote_address}{Fore.RESET}: {err}"
            in log_msg
        )

        assert mock_client.is_active is False

    @pytest.mark.asyncio
    @patch("server.client.logger")
    async def test_loop_output_success(self, mock_logger, mock_client, mock_ws):
        msg = OutputMessage(status="stream", code=200, message="test")
        mock_client.queue_output.get = AsyncMock(
            side_effect=[msg, asyncio.CancelledError]
        )
        await mock_client.loop_output()

        mock_ws.send.assert_awaited_once_with(msg.message)

        mock_logger.info.assert_called_once()
        log_msg = mock_logger.info.call_args[0][0]
        assert (
            f"Sent message to {Fore.GREEN}{mock_ws.remote_address}{Fore.RESET}:\n {msg}"
            in log_msg
        )

    @pytest.mark.asyncio
    @patch("server.client.logger")
    async def test_loop_output_cancelled_error(self, mock_logger, mock_client, mock_ws):
        mock_client.queue_output.get = AsyncMock(side_effect=asyncio.CancelledError)
        await mock_client.loop_output()

        mock_client.queue_output.get.assert_awaited_once()

        mock_ws.send.assert_not_awaited()

        mock_logger.error.assert_called_once()
        log_msg = mock_logger.error.call_args[0][0]
        assert (
            f"Task cancelled for {Fore.GREEN}{mock_ws.remote_address}{Fore.RESET}"
            in log_msg
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

        mock_logger.error.assert_called_once()
        log_msg = mock_logger.error.call_args[0][0]
        assert (
            f"Error sending message to {Fore.GREEN}{mock_ws.remote_address}{Fore.RESET}: {err}"
            in log_msg
        )

    @pytest.mark.asyncio
    @patch("server.client.logger")
    async def test_loop_output_other_exception_on_send(
        self, mock_logger, mock_client, mock_ws
    ):
        msg = OutputMessage(status="stream", code=200, message="test")
        mock_client.queue_output.get = AsyncMock(
            side_effect=[msg, asyncio.CancelledError]
        )
        err = ValueError("send_error")

        mock_ws.send.side_effect = err
        await mock_client.loop_output()

        assert mock_client.is_active is False
        mock_client.queue_output.get.assert_awaited_once()

        mock_ws.send.assert_awaited_once_with(msg.message)

        mock_logger.error.assert_called_once()
        log_msg = mock_logger.error.call_args[0][0]
        assert (
            f"Error sending message to {Fore.GREEN}{mock_ws.remote_address}{Fore.RESET}: {err}"
            in log_msg
        )

    @pytest.mark.asyncio
    @patch("asyncio.gather", new_callable=AsyncMock)
    @patch("server.client.logger")
    async def test_close_success(self, mock_logger, mock_gather, mock_client, mock_ws):
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

        mock_gather.assert_awaited_once()
        gathered_tasks = mock_gather.call_args[0]
        assert set(gathered_tasks) == {
            mock_client.task_input,
            mock_client.task_output,
            mock_client.task_session,
        }

        mock_client.disconnection.set.assert_called_once()

        mock_logger.info.assert_called_once()
        assert (
            f"Closing connection for {mock_ws.remote_address}"
            in mock_logger.info.call_args[0][0]
        )

    @pytest.mark.asyncio
    @patch("asyncio.gather", new_callable=AsyncMock)
    @patch("server.client.logger")
    async def test_close_ws_error(self, mock_logger, mock_gather, mock_client, mock_ws):
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

        mock_gather.assert_awaited_once()
        gathered_tasks = mock_gather.call_args[0]
        assert set(gathered_tasks) == {
            mock_client.task_input,
            mock_client.task_output,
            mock_client.task_session,
        }
        assert mock_gather.call_args[1].get("return_exceptions") is True

        mock_client.disconnection.set.assert_not_called()

        mock_logger.info.assert_called_once()
        assert (
            f"Closing connection for {mock_ws.remote_address}"
            in mock_logger.info.call_args[0][0]
        )
        mock_logger.error.assert_called_once()
        assert (
            f"Error closing websocket connection for {mock_ws.remote_address}: {err}"
            in mock_logger.error.call_args[0][0]
        )
