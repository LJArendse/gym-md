from typing import DefaultDict, Final, List, Tuple
from PIL import Image
import numpy

from gym_md.envs.md_env import MdEnvBase
from gym_md.envs.agent.companion_agent import CompanionAgent
from gym_md.envs.renderer.collab_renderer import CollabRenderer


JointActions = [List[float], List[float]]

class MdCollabEnv(MdEnvBase):
    def __init__(self, stage_name: str):
        super.init(stage_name)
        self.c_agent: CompanionAgent = CompanionAgent(self.grid, self.setting, self.random)
        self.c_renderer: Final[CollabRenderer] = CollabRenderer(self.grid, self.agent, self.setting, self.c_agent)

    def reset(self) -> List[int]:
        """環境をリセットする."""
        super().reset()
        self.c_agent.reset()
        return self._get_observation()

    def _get_observation_c_agent(self) -> List[int]:
        """環境の観測を取得する.

        Returns
        -------
        list of int
            エージェントにわたす距離の配列 (len: 8)
        """
        sd, _ = self.c_agent.path.get_distance_and_prev(
            y=self.c_agent.y, x=self.c_agent.x, safe=True
        )
        ud, _ = self.c_agent.path.get_distance_and_prev(
            y=self.c_agent.y, x=self.c_agent.x, safe=False
        )
        sd = self.c_agent.path.get_nearest_distance(sd)
        ud = self.c_agent.path.get_nearest_distance(ud)
        ret = [
            ud["M"],
            ud["T"],
            sd["T"],
            ud["P"],
            sd["P"],
            ud["E"],
            sd["E"],
            self.c_agent.hp,
        ]
        return numpy.array(ret, dtype=numpy.int32)

    def _get_observation(self) -> List[int]:
        """環境の観測を取得する.

        Returns
        -------
        list of int
            エージェントにわたす距離の配列 (len: 9)
        """
        ret = super()._get_observation()
        c_ret = self._get_observation_c_agent()
        return numpy.append(ret, c_ret).astype(numpy.int32)

    def _is_done(self) -> bool:
        """ゲームが終了しているか.

        Returns
        -------
        bool
        """
        return super()._is_done() or self.c_agent.is_exited() or self.c_agent.is_dead()

    def _update_grid(self) -> None:
        """グリッドの状態を更新する.

        Notes
        -----
        メソッド内でグリッドの状態を**直接更新している**ことに注意．

        Returns
        -------
        None
        """
        super()._update_grid()

        agent_y, agent_x = self.c_agent.y, self.c_agent.x
        C = self.setting.CHARACTER_TO_NUM
        if self.c_agent.hp <= 0:
            return
        if (self.grid[agent_y, agent_x] in [C["P"], C["M"], C["T"]]):
            self.grid[agent_y, agent_x] = C["."]

    def _get_companion_reward(self) -> float:
        """報酬を計算する.

        Returns
        -------
        int
            報酬

        """
        R = self.setting.REWARDS
        C = self.setting.CHARACTER_TO_NUM
        companion_agent_reward: float = -R.TURN
        y, x = self.c_agent.y, self.c_agent.x
        if self.c_agent.hp <= 0:
            return companion_agent_reward + R.DEAD
        if (self.grid[y, x] == C["T"]):
            companion_agent_reward += R.TREASURE
        if (self.grid[y, x] == C["E"]):
            companion_agent_reward += R.EXIT
        if (self.grid[y, x] == C["M"]):
            companion_agent_reward += R.KILL
        if (self.grid[y, x] == C["P"]):
            companion_agent_reward += R.POTION

        return companion_agent_reward

    def step(self, actions: JointActions) -> Tuple[List[int], int, bool, DefaultDict[str, int]]:
        """エージェントが1ステップ行動する.

        Attributes
        ----------
        actions: Actions
            list of int
            各行動の値を入力する

        Notes
        -----
        行動列をすべて入力としている
        これはある行動をしようとしてもそのマスがない場合があるため
        その場合は次に大きい値の行動を代わりに行う．

        Returns
        -------
        Tuple of (list of int, int, bool, dict)
        """
        observation, reward_agent_1, done, self.info = super().step(actions[0])

        c_action: Final[str] = self.c_agent.select_action(actions[1])
        self.c_agent.take_action(c_action)
        reward_agent_2: int = self._get_companion_reward()
        done: bool = self._is_done()
        self._update_grid()

        return observation, reward_agent_1+reward_agent_2, done, self.info

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