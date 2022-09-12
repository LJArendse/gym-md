from typing import DefaultDict, Final, List, Tuple
from PIL import Image
import numpy

from gym_md.envs.md_env import MdEnvBase
from gym_md.envs.agent.companion_agent import CompanionAgent
from gym_md.envs.renderer.collab_renderer import CollabRenderer


JointActions = [List[float], List[float]]


class MdCollabEnv(MdEnvBase):
    """
    The MdCollabEnv collaborative environment extends
    MiniDungeons (MdEnvBase) to a two player setting,
    through the inclusion of a secondary companion agent
    (c_agent).
    """

    def __init__(self, stage_name: str):
        """
        Initialise a MdCollabEnv object using the
        stage_name as input.
        """
        super.init(stage_name)
        self.c_agent: CompanionAgent = CompanionAgent(self.grid, self.setting, self.random)
        self.c_renderer: Final[CollabRenderer] = CollabRenderer(self.grid, self.agent, self.setting, self.c_agent)

    def reset(self) -> List[int]:
        """環境をリセットする.
           Reset the environment.
        """
        super().reset()
        self.c_agent.reset()
        return self._get_observation()

    def _get_observation_c_agent(self) -> List[int]:
        """環境の観測を取得する.
           Return the environment observation from the
           companion agent's viewpoint.

        Returns
        -------
        list of int
            エージェントにわたす距離の配列 (len: 8)
            List, of length 8, representing the
            environment observations from the agent's
            perspective.

            each index in the observation list represents the following:
                - index 0: Distance to the monster
                - index 1: Distance to the treasure
                - index 2: Distance to treasure (avoid monsters)
                - index 3: Distance to potion
                - index 4: Distance to potion (avoid monsters)
                - index 5: Distance to the exit
                - index 6: Distance to the exit (avoid monsters)
                - index 7: Agent's physical strength (i.e. Hit Points (HP))
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
           Return the environment observation.

           The environment observation returned
           contains both the primary agent and
           companion agent (c_agent) observations.

        Returns
        -------
        list of int
            エージェントにわたす距離の配列 (len: 16)
            List, of length 16, representing the
            environment observations from the agent's
            perspective.

            each index in the observation list represents the following:
                - index 0: Distance to the monster (primary agent)
                - index 1: Distance to the treasure (primary agent)
                - index 2: Distance to treasure (avoid monsters)(primary agent)
                - index 3: Distance to potion (primary agent)
                - index 4: Distance to potion (avoid monsters) (primary agent)
                - index 5: Distance to the exit (primary agent)
                - index 6: Distance to the exit (avoid monsters)(primary agent)
                - index 7: Primary agent's physical strength
                - index 8: Distance to the monster (c_agent)
                - index 9: Distance to the treasure (c_agent)
                - index 10: Distance to treasure (avoid monsters) (c_agent)
                - index 11: Distance to potion (c_agent)
                - index 12: Distance to potion (avoid monsters) (c_agent)
                - index 13: Distance to the exit (c_agent)
                - index 14: Distance to the exit (avoid monsters) (c_agent)
                - index 15: Companion agent's physical strength
        """
        ret = super()._get_observation()
        c_ret = self._get_observation_c_agent()
        return numpy.append(ret, c_ret).astype(numpy.int32)

    def _is_done(self) -> bool:
        """ゲームが終了しているか.
           Returns a boolean indicating whether the game is over or not.

        Returns
        -------
        bool
        """
        return super()._is_done() or self.c_agent.is_exited() or self.c_agent.is_dead()

    def _update_grid(self) -> None:
        """グリッドの状態を更新する.
           Update the state of the grid.

        Notes
        -----
        メソッド内でグリッドの状態を**直接更新している**ことに注意
        Note that we are **directly updating** the state of the grid
        in the method.

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
           Calculate and return the companion agent's reward.

        Returns
        -------
        int
            報酬
            Reward.

        """
        R = self.setting.REWARDS
        C = self.setting.CHARACTER_TO_NUM
        companion_agent_reward: float = -R.TURN
        y, x = self.c_agent.y, self.c_agent.x
        if self.c_agent.hp <= 0:
            return companion_agent_reward + R.DEAD
        if self.grid[y, x] == C["T"]:
            companion_agent_reward += R.TREASURE
        if self.grid[y, x] == C["E"]:
            companion_agent_reward += R.EXIT
        if self.grid[y, x] == C["M"]:
            companion_agent_reward += R.KILL
        if self.grid[y, x] == C["P"]:
            companion_agent_reward += R.POTION

        return companion_agent_reward

    def step(self, actions: JointActions) -> Tuple[List[int], int, bool, DefaultDict[str, int]]:
        """Perform a environment step using the JointActions of both agents.

        Attributes
        ----------
        actions: JointActions
            list of lists of float
                [List[float], List[float]]

            Where the first list (actions[0]) represents the primary agent's
            actions and the second list (actions[1]) represents the companion
            agent's actions.

            Each index in the respective actions list corresponds to a specific
            action available for the game agent to take:
                - index 0: Head to the monster 
                - index 1: Head to the treasure
                - index 2: Head to the treasure (avoid monsters)
                - index 3: Head to the potion
                - index 4: Head to the potion (avoid monsters)
                - index 5: Head to the exit
                - index 6: Head to the exit (avoid monsters)

            各行動の値を入力する
            A value for each should be entered.

        Notes
        -----
        行動列をすべて入力としている
        これはある行動をしようとしてもそのマスがない場合があるため
        その場合は次に大きい値の行動を代わりに行う．

        All action values are evaluated, starting from the highest action
        value. If the highest action value cannot be performed then the
        next highest values is evaluated next. This action selection process
        is repeated until a valid action can be performed within the given
        state. Furthermore, if the desired values are the same, an action is
        randomly selected.

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
           Render the Minidungeons game world.

        Notes
        -----
        画像自体も取得できるため，保存も可能.
        This method returns the world image,
        which can be saved.

        Returns
        -------
        Image
        """
        return self.c_renderer.render(mode=mode)

    def generate(self, mode="human") -> Image:
        """画像を生成する.
           Generate the world image.

        Notes
        -----
        画像の保存などの処理はgym外で行う.
        Processing such as image saving is performed
        outside the gym environment.

        Returns
        -------
        Image
        """
        return self.c_renderer.generate(mode=mode)