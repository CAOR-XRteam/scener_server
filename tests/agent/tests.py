import pytest

from agent.tools.scene import SceneAnalyzer
from agent.tools.decomposer import Decomposer
from agent.tools.improver import Improver
from langchain_core.messages import AIMessage
from langchain_core.exceptions import OutputParserException
from unittest.mock import patch, MagicMock


class TestImprover:
    @pytest.fixture
    def sample_prompt(self):
        return "Generate a Japanese theatre scene with samurai armor in the center"

    @pytest.fixture
    def mock_improver(self):
        with patch("agent.tools.improver.initialize_model") as mock_init:
            mock_llm_instance = MagicMock()
            mock_llm_invoke_method = MagicMock()

            mock_llm_instance.invoke = mock_llm_invoke_method
            mock_init.return_value = mock_llm_instance

            mock_improver = Improver(model_name="test")

            mock_init.assert_called_once_with("test", temperature=0.0)

            return mock_improver, mock_llm_instance

    def test_improve(self, mock_improver, sample_prompt):
        mock_improver, mock_llm_invoke = mock_improver
        mock_response = "Generate a traditional Japanese theatre scene with Samurai armor placed in the center of the stage. The room should have wooden flooring, simple red and gold accents, and folding screens in the background. The Samurai armor should be detailed, with elements like the kabuto (helmet) and yoroi (body armor), capturing the essence of a classical Japanese theatre setting."
        mock_llm_invoke.return_value = AIMessage(content=mock_response)

        result = mock_improver.improve_single_prompt(sample_prompt)

        assert result == mock_response
        mock_llm_invoke.assert_called_once()

    def test_improve_llm_api_error(self, mock_improver, sample_prompt):
        mock_improver, mock_llm_invoke = mock_improver
        error_message = "Ollama service unreachable"
        mock_llm_invoke.side_effect = ConnectionError(error_message)

        with pytest.raises(ConnectionError, match=error_message):
            mock_improver.improve_single_prompt(sample_prompt)

        mock_llm_invoke.assert_called_once()


class TestDecomposer:
    @pytest.fixture
    def sample_prompt(self):
        return "Generate a traditional Japanese theatre room with intricate wooden flooring, high wooden ceiling beams, elegant red and gold accents, and large silk curtains."

    @pytest.fixture
    def mock_decomposer(self):
        with patch("agent.tools.decomposer.initialize_model") as mock_init:
            mock_llm_instance = MagicMock()
            mock_llm_invoke_method = MagicMock()

            mock_llm_instance.invoke = mock_llm_invoke_method
            mock_init.return_value = mock_llm_instance

            mock_decomposer = Decomposer(model_name="test")

            mock_init.assert_called_once_with("test", temperature=0.0)

            return mock_decomposer, mock_llm_instance

    def test_decompose(self, mock_decomposer, sample_prompt):
        mock_decomposer, mock_llm_invoke = mock_decomposer
        mock_response = '{"scene": {"objects": [{"name": "theatre_room", "type": "room", "position": {"x": 0, "y": 0, "z": 0}, "rotation": {"x": 0, "y": 0, "z": 0}, "scale": {"x": 20, "y": 10, "z": 20}, "material": "traditional_wood_material", "prompt": "Generate an image of a squared traditional Japanese theatre room viewed from the outside at a 3/4 top-down perspective."}]}}'
        mock_llm_invoke.return_value = AIMessage(content=mock_response)

        result = mock_decomposer.decompose(sample_prompt)

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
        mock_llm_invoke.assert_called_once()

    def test_invalid_json_response(self, decomposer):
        mock_decomposer, mock_llm_invoke = mock_decomposer
        mock_llm_invoke.return_value = AIMessage(content='{"scene": [invalidjson]}')

        user_input = "Blablabla"
        with pytest.raises(OutputParserException) as e:
            mock_decomposer.decompose(user_input)

    def test_decompose_llm_api_error(self, mock_decomposer, sample_prompt):
        mock_decomposer, mock_llm_invoke = mock_decomposer
        error_message = "Ollama service unreachable"
        mock_llm_invoke.side_effect = ConnectionError(error_message)

        with pytest.raises(ConnectionError, match=error_message):
            mock_decomposer.decompose(sample_prompt)

        mock_llm_invoke.assert_called_once()


@pytest.mark.skip(reason="not implemented yet")
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

    @patch("langchain_ollama.llms.OllamaLLM.invoke")
    def test_one_element_selection(self, mock_llm_invoke, analyzer, sample_scene):
        mock_response = '{"objects": [{"name": "table", "position": [0, 0, 0]}]}'
        mock_llm_invoke.return_value = AIMessage(content=mock_response)

        user_input = "Add a vase on the table"
        result = analyzer.analyze(sample_scene, user_input)

        expected = {"objects": [{"name": "table", "position": [0, 0, 0]}]}
        assert result == expected

    @patch("langchain_ollama.llms.OllamaLLM.invoke")
    def test_multiple_elements_selection(self, mock_llm_invoke, analyzer, sample_scene):
        mock_response = '{"objects": [{"name": "lamp", "position": [0, 1, 0], "state": "off"}], "lights": [{"id": "main", "intensity": 0.8}]}'
        mock_llm_invoke.return_value = AIMessage(content=mock_response)

        user_input = "Turn on the lamp and adjust main light"
        result = analyzer.analyze(sample_scene, user_input)

        expected = {
            "objects": [{"name": "lamp", "position": [0, 1, 0], "state": "off"}],
            "lights": [{"id": "main", "intensity": 0.8}],
        }
        assert result == expected
        mock_llm_invoke.assert_called_once()

    @patch("langchain_ollama.llms.OllamaLLM.invoke")
    def test_invalid_json_response(self, mock_llm_invoke, analyzer, sample_scene):
        mock_llm_invoke.return_value = AIMessage(content='{"objects": [invalidjson]}')

        user_input = "Blablabla"
        with pytest.raises(OutputParserException) as e:
            analyzer.analyze(sample_scene, user_input)

        mock_llm_invoke.assert_called_once()

    @patch("langchain_ollama.llms.OllamaLLM.invoke")
    def test_analyze_llm_api_error(self, mock_llm_invoke, analyzer, sample_scene):
        error_message = "Ollama service unreachable"
        mock_llm_invoke.side_effect = ConnectionError(error_message)

        user_input = "Whatever"
        with pytest.raises(ConnectionError, match=error_message):
            analyzer.analyze(sample_scene, user_input)

        mock_llm_invoke.assert_called_once()
