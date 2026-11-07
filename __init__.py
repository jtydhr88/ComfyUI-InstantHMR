from typing_extensions import override

from comfy_api.latest import ComfyExtension, io

from .nodes import InstantHMRLoader, InstantHMRPredict


class InstantHMRExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [InstantHMRLoader, InstantHMRPredict]


async def comfy_entrypoint() -> InstantHMRExtension:
    return InstantHMRExtension()
