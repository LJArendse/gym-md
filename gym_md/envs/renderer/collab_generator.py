"""Image Generate module."""
from os import path
from typing import Final, List

from PIL import Image

from gym_md.envs.agent.agent import Agent
from gym_md.envs.agent.companion_agent import CompanionAgent
from gym_md.envs.grid import Grid
from gym_md.envs.renderer.generator import Generator

tiles_dir = path.join(path.dirname(__file__), path.pardir, "tiles")
tiles_names: Final[List[str]] = [
    "empty.png",
    "wall.png",
    "chest.png",
    "potion.png",
    "monster.png",
    "exit.png",
    "side_kick.png",
    "deadhero.png",
    "hero_b.png"
]
tiles_paths: Final[List[str]] = [path.join(tiles_dir, t) for t in tiles_names]
tiles_images = [Image.open(t).convert("RGBA") for t in tiles_paths]
split_images = [[x for x in img.split()] for img in tiles_images]

LENGTH: Final[int] = 20


class CollabGenerator(Generator):
    """Generator class."""

    def __init__(self, grid: Grid, agent: Agent, agent_2: CompanionAgent):
        super().__init__(grid, agent)
        self.companion_agent: Final[CompanionAgent] = agent_2

    def generate(self) -> Image:
        """画像を生成する.

        Returns
        -------
        Image

        """
        img = Image.new("RGB", (self.W * LENGTH, self.H * LENGTH))
        for i in range(self.H):
            for j in range(self.W):
                img.paste(tiles_images[0], (LENGTH * j, i * LENGTH))
                e: int = self.grid[i, j]
                if i == self.agent.y and j == self.agent.x:
                    e = 6 if self.agent.hp > 0 else 7
                img.paste(tiles_images[e], (LENGTH * j, i * LENGTH), split_images[e][3])
                if i == self.companion_agent.y and j == self.companion_agent.x:
                    e = 8 if self.companion_agent.hp > 0 else 7
                img.paste(tiles_images[e], (LENGTH * j, i * LENGTH), split_images[e][3])
        return img
