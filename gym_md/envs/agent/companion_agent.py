from random import Random

from gym_md.envs.agent.agent import Agent
from gym_md.envs.point import Point
from gym_md.envs.grid import Grid
from gym_md.envs.setting import Setting


class CompanionAgent(Agent):
    def __init__(self, grid: Grid, setting: Setting, random: Random):
        super().__init__(grid, setting, random)

    def _init_player_pos(self) -> Point:
        """プレイヤーの座標を初期化して座標を返す.
        Initialize the companion agent's coordinates and return the
        coordinates.

        Notes
        -----
        初期座標を表すAを'.'にメソッド内で書き換えていることに注意する．
        The agent symbol 'A' within the respective stage (.txt) file
        represents the initial coordinates of the companion agent and
        is replaced with the free space symbol '.' within the self.grid
        object.

        Returns
        -------
        Point
            初期座標を返す
            Return the initial companion agent coordinates.

        """
        for i in range(self.grid.H):
            for j in range(self.grid.W):
                if self.grid[i, j] == self.setting.CHARACTER_TO_NUM["A"]:
                    self.grid[i, j] = self.setting.CHARACTER_TO_NUM["."]
                    return i, j
