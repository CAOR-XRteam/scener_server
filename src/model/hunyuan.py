import torch

from pathlib import Path
from beartype import beartype
from PIL import Image
from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
from hy3dgen.texgen import Hunyuan3DPaintPipeline
from hy3dgen.rembg import BackgroundRemover


@beartype
def read_glb(object_path: str):
    with open(object_path, "rb") as f:
        return f.read()


@beartype
def generate(image_path: Path, image_id: str):
    img = Image.open(image_path).convert("RGB").resize((512, 512))
    rembg = BackgroundRemover()
    img = rembg(img)

    try:
        shape_pipe = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
            "tencent/Hunyuan3D-2"
        )
        shape_pipe
        paint_pipe = Hunyuan3DPaintPipeline.from_pretrained("tencent/Hunyuan3D-2")

        mesh = shape_pipe(image=img, num_inference_steps=15)[0]
        import mesh_processor

        print("mesh_processor is:", mesh_processor)
        mesh_text = paint_pipe(mesh, image=img)
        mesh_text.export(image_path.parent / f"{image_id}.glb")
    except:
        raise


if __name__ == "__main__":
    generate(Path("steampunk_boat.png"), "1")
