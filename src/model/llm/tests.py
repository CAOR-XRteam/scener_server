import pytest
from unittest.mock import patch
from .scene import SceneAnalyzer
from .decomposer import Decomposer


class TestDecomposer:
    @pytest.fixture
    def decomposer(self):
        return Decomposer()

    @pytest.fixture
    def sample_prompt(self):
        return "Generate a traditional Japanese theatre room with intricate wooden flooring, high wooden ceiling beams, elegant red and gold accents, and large silk curtains."

    @patch("src.model.llm.decomposer.chat")
    def test_decompose(self, mock_chat, decomposer, sample_prompt):
        mock_response = '{"scene": {"objects": [{"name": "theatre_room", "type": "room", "position": {"x": 0, "y": 0, "z": 0}, "rotation": {"x": 0, "y": 0, "z": 0}, "scale": {"x": 20, "y": 10, "z": 20}, "material": "traditional_wood_material", "prompt": "Generate an image of a squared traditional Japanese theatre room viewed from the outside at a 3/4 top-down perspective."}]}}'
        mock_chat.return_value.message.content = mock_response

        result = decomposer.decompose(sample_prompt)

        expected = {
            "scene": {
                "objects": [
                    {
                        "name": "theatre_room",
                        "type": "room",
                        "position": {"x": 0, "y": 0, "z": 0},
                        "rotation": {"x": 0, "y": 0, "z": 0},
                        "scale": {"x": 20, "y": 10, "z": 20},
                        "material": "traditional_wood_material",
                        "prompt": (
                            "Generate an image of a squared traditional Japanese theatre room "
                            "viewed from the outside at a 3/4 top-down perspective."
                        ),
                    }
                ]
            }
        }
        assert result == expected

    @patch("src.model.llm.decomposer.chat")
    def test_invalid_json_response(self, mock_chat, decomposer):
        mock_chat.return_value.message.content = '{"scene": [invalidjson]}'

        user_input = "Blablabla"
        result = decomposer.decompose(user_input)

        assert "error" in result
        assert "raw_response" in result


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

    @patch("src.model.llm.scene.chat")
    def test_one_element_selection(self, mock_chat, analyzer, sample_scene):
        mock_response = '{"objects": [{"name": "table", "position": [0, 0, 0]}]}'
        mock_chat.return_value.message.content = mock_response

        user_input = "Add a vase on the table"
        result = analyzer.analyze(sample_scene, user_input)

        expected = {"objects": [{"name": "table", "position": [0, 0, 0]}]}
        assert result == expected

    @patch("src.model.llm.scene.chat")
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

    @patch("src.model.llm.scene.chat")
    def test_invalid_json_response(self, mock_chat, analyzer, sample_scene):
        mock_chat.return_value.message.content = '{"objects": [invalidjson]}'

        user_input = "Blablabla"
        result = analyzer.analyze(sample_scene, user_input)

        assert "error" in result
        assert "raw_response" in result
