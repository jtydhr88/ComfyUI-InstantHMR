from typing_extensions import override

from comfy_api.latest import ComfyExtension, io
from server import PromptServer

from .live import register_routes
from .nodes import InstantHMRLiveCapture, InstantHMRLoader, InstantHMRPredict

WEB_DIRECTORY = "./web"

register_routes(PromptServer.instance.routes)


class InstantHMRExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [InstantHMRLoader, InstantHMRPredict, InstantHMRLiveCapture]


async def comfy_entrypoint() -> InstantHMRExtension:
    return InstantHMRExtension()
