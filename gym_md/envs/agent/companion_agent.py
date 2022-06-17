from gym_md.envs.agent.agent import Agent
from gym_md.envs.point import Point

class CompanionAgent(Agent):
    def _init_player_pos(self) -> Point:
        """プレイヤーの座標を初期化して座標を返す.

        Notes
        -----
        初期座標を表すSを'.'にメソッド内で書き換えていることに注意する．

        Returns
        -------
        Point
            初期座標を返す

        """
        for i in range(self.grid.H):
            for j in range(self.grid.W):
                if self.grid[i, j] == self.setting.CHARACTER_TO_NUM["A"]:
                    self.grid[i, j] = self.setting.CHARACTER_TO_NUM["."]
                    return i, j
