from collections import defaultdict
from typing import Final, List
from PIL import Image

from gym_md.envs.md_env import MdEnvBase
from gym_md.envs.agent.companion_agent import CompanionAgent
from gym_md.envs.renderer.collab_renderer import CollabRenderer

class MdCollabEnv(MdEnvBase):
    def __init__(self, stage_name: str):
        super().__init__(stage_name)
        self.c_agent: CompanionAgent = CompanionAgent(self.grid, self.setting, self.random)
        self.c_renderer: Final[CollabRenderer] = CollabRenderer(self.grid, self.agent, self.setting, self.c_agent)

    def reset(self) -> List[int]:
        """環境をリセットする."""
        super().reset()
        self.c_agent.reset()
        return self._get_observation()

    def render(self, mode="human") -> Image:
        """画像の描画を行う.

        Notes
        -----
        画像自体も取得できるため，保存も可能.

        Returns
        -------
        Image
        """
        return self.c_renderer.render(mode=mode)

    def generate(self, mode="human") -> Image:
        """画像を生成する.

        Notes
        -----
        画像の保存などの処理はgym外で行う.

        Returns
        -------
        Image
        """
        return self.c_renderer.generate(mode=mode)