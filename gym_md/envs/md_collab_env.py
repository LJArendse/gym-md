from collections import defaultdict
from typing import DefaultDict, Dict, Final, List, Tuple
from PIL import Image
import numpy
import gym

from gym_md.envs.md_env import MdEnvBase
from gym_md.envs.agent.companion_agent import CompanionAgent
from gym_md.envs.renderer.collab_renderer import CollabRenderer
from gym_md.envs.agent.actioner import Actions

JointActions = [List[float], List[float]]

class MdCollabEnv(MdEnvBase):
    def __init__(self, stage_name: str):
        super().__init__(stage_name)
        self.observation_space = gym.spaces.Box(
            low=0, high=self.setting.DISTANCE_INF, shape=(9,), dtype=numpy.int32
        )
        self.c_agent: CompanionAgent = CompanionAgent(self.grid, self.setting, self.random)
        self.c_renderer: Final[CollabRenderer] = CollabRenderer(self.grid, self.agent, self.setting, self.c_agent)

    def reset(self) -> List[int]:
        """環境をリセットする."""
        super().reset()
        self.c_agent.reset()
        return self._get_observation()

    def _get_observation(self) -> List[int]:
        """環境の観測を取得する.

        Returns
        -------
        list of int
            エージェントにわたす距離の配列 (len: 9)
        """
        ret = super()._get_observation()
        return numpy.append(ret, self.c_agent.hp).astype(numpy.int32)

    def _is_done(self) -> bool:
        """ゲームが終了しているか.

        Returns
        -------
        bool
        """
        agent_1_end = super()._is_done()
        return agent_1_end or self.c_agent.is_exited() or self.c_agent.is_dead()

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
        observation, reward, done, self.info = super().step(actions[0])

        c_action: Final[str] = self.c_agent.select_action(actions[1])
        self.c_agent.take_action(c_action)
        self._update_grid()

        return observation, reward, done, self.info

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