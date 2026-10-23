
from comfy_api.latest import ComfyExtension, io
from typing_extensions import override
from .diffueraser_node import Propainter_Loader,DiffuEraser_Loader,DiffuEraser_PreData,Propainter_Sampler,DiffuEraser_Sampler

class DiffuEraser_SM_Extension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            Propainter_Loader,
            DiffuEraser_Loader,
            DiffuEraser_PreData,
            Propainter_Sampler,
            DiffuEraser_Sampler,
        ]


async def comfy_entrypoint() -> DiffuEraser_SM_Extension:  # ComfyUI calls this to load your extension and its nodes.
    return DiffuEraser_SM_Extension()
