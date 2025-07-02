import torch

from pathlib import Path
from beartype import beartype
from PIL import Image
from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
from hy3dgen.texgen import Hunyuan3DPaintPipeline


@beartype
def read_glb(object_path: str):
    with open(object_path, "rb") as f:
        return f.read()


@beartype
def generate(image_path: Path, image_id: str):
    img = Image.open(image_path).convert("RGB")

    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
    else:
        dtype = torch.float16

    shape_pipe = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        "tencent/Hunyuan3D-2", torch_dtype=dtype
    ).to("cuda")
    paint_pipe = Hunyuan3DPaintPipeline.from_pretrained(
        "tencent/Hunyuan3D-2", torch_dtype=dtype
    ).to("cuda")

    with torch.autocast("cuda", dtype=dtype):
        mesh = shape_pipe(image=img, num_inference_steps=15)[0]
        mesh_text = paint_pipe(mesh, image=img)

    mesh_text.export(image_path.parent / f"{image_id}.glb")


if __name__ == "__main__":
    generate(Path("steampunk_boat.png"), "1")
