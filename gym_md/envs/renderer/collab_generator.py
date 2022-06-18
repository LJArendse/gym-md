"""Image Generate module."""
from os import path
from typing import Final, List

from PIL import Image

from gym_md.envs.agent.agent import Agent
from gym_md.envs.agent.companion_agent import CompanionAgent
from gym_md.envs.grid import Grid
from gym_md.envs.renderer.generator import Generator, tiles_dir, LENGTH

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
        img = super().generate()

        side_kick_sprite = Image.open(f"{path.join(tiles_dir, 'side_kick.png')}").convert("RGBA")
        side_kick_death_sprite = Image.open(f"{path.join(tiles_dir, 'deadhero.png')}").convert("RGBA")
        split_images = [[x for x in img.split()] for img in [side_kick_sprite, side_kick_death_sprite]]

        sprite = side_kick_sprite if self.companion_agent.hp > 0 else side_kick_death_sprite
        sprite_split_image = split_images[0] if self.companion_agent.hp > 0 else split_images[1]
        
        img.paste(sprite, (LENGTH * self.companion_agent.x, self.companion_agent.y * LENGTH), sprite_split_image[3])
        return img
