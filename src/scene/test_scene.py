import pytest
from unittest.mock import patch
from .scene import SceneAnalyzer


class TestSceneAnalyzer:
    @pytest.fixture
    def analyzer(self):
        return SceneAnalyzer()

    @pytest.fixture
    def sample_scene(self):
        return {
            "objects": [
                {"name": "table", "position": [0, 0, 0]},
                {"name": "chair", "position": [1, 0, 1]},
                {"name": "lamp", "position": [0, 1, 0], "state": "off"},
            ],
            "lights": [{"id": "main", "intensity": 0.8}],
        }

    @patch("scene.scene.chat")
    def test_one_element_selection(self, mock_chat, analyzer, sample_scene):
        mock_response = '{"objects": [{"name": "table", "position": [0, 0, 0]}]}'
        mock_chat.return_value.message.content = mock_response

        user_input = "Add a vase on the table"
        result = analyzer.analyze(sample_scene, user_input)

        expected = {"objects": [{"name": "table", "position": [0, 0, 0]}]}
        assert result == expected

    @patch("scene.scene.chat")
    def test_multiple_elements_selection(self, mock_chat, analyzer, sample_scene):
        mock_response = '{"objects": [{"name": "lamp", "position": [0, 1, 0], "state": "off"}], "lights": [{"id": "main", "intensity": 0.8}]}'
        mock_chat.return_value.message.content = mock_response

        user_input = "Turn on the lamp and adjust main light"
        result = analyzer.analyze(sample_scene, user_input)

        expected = {
            "objects": [{"name": "lamp", "position": [0, 1, 0], "state": "off"}],
            "lights": [{"id": "main", "intensity": 0.8}],
        }
        assert result == expected

    @patch("scene.scene.chat")
    def test_invalid_json_response(self, mock_chat, analyzer, sample_scene):
        mock_chat.return_value.message.content = '{"objects": [invalidjson]}'

        user_input = "Blablabla"
        result = analyzer.analyze(sample_scene, user_input)

        assert "error" in result
        assert "raw_response" in result
