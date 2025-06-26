from agent.tools.scene.improver import Improver
from agent.tools.scene.decomposer import InitialDecomposer, DecompositionOutput
from lib import load_config


def decompose_and_improve(user_input: str) -> DecompositionOutput:
    config = load_config()

    initial_decomposer_model_name = config.get("initial_decomposer_model")
    improver_model_name = config.get("improver_model")

    decomposer = InitialDecomposer(model_name=initial_decomposer_model_name)
    improver = Improver(model_name=improver_model_name)
    try:
        decomposed_scene = decomposer.decompose(user_input)
    except Exception as e:
        raise ValueError(f"Failed to decompose input: {e}")
    try:
        improved_scene = improver.improve(decomposed_scene)
    except Exception as e:
        raise ValueError(f"Failed to improve decomposed scene: {e}")

    return improved_scene
