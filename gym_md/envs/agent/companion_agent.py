from typing import List, Tuple
import random

from gym_md.envs.agent.agent import Agent
from gym_md.envs.point import Point
from gym_md.envs.agent.actioner import Actions


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

    def __return_position_based_on_action(self, pos: Point, action: str) -> Point:
        """Returns a new Point position based on the input directional action.

        Attributes
        ----------
        pos: Point
            Tuple[int, int]
            The reference point position to be used.
        action: str
            The directional action taken. Where action is a value
            within ['UP', 'DOWN', 'LEFT', 'RIGHT'].

        Notes
        -----
        A neighbouring position can fall within the following regions:
           NW                      NORTH                     NE
                           | -1  || -1  || +1  |
        WEST               | -1  || pos || +1  |                EAST
                           | -1  || +1  || +1  |
           SW                      SOUTH                     SE

        Direction calculations:
            up = (pos[0]-1, pos[1])
            right = (pos[0], pos[1]+1)
            down = (pos[0]+1, pos[1])
            left = (pos[0], pos[1]-1)

        Returns
        -------
        Point
        """
        if action == 'UP':
            return (pos[0]-1, pos[1])
        elif action == 'DOWN':
            return (pos[0]+1, pos[1])
        elif action == 'LEFT':
            return (pos[0], pos[1]-1)
        elif action == 'RIGHT':
            return (pos[0], pos[1]+1)

    def select_directional_action(self, actions: Actions) -> str:
        """行動を選択する.

        Notes
        -----
        行動を選択したときに，その行動が実行できない可能性がある．
        （マスがない可能性など）

        そのため，行動列すべてを受け取りできるだけ値の大きい実行できるものを選択する．
        **選択する**であり何も影響を及ぼさないことに注意．

        Parameters
        ----------
        actions: Actions
            行動列

        Returns
        -------
        str
            選択した行動IDを返す
        """
        #import random
        #actions = [1.0,0.9,1.0,0.2]
        #from typing import List, Tuple
        actions_idx: List[Tuple[float, int]] = [(actions[i], i) for i in range(len(actions))]
        actions_idx.sort(key=lambda z: (-z[0], -z[1]))

        max_value = max(actions)
        max_actions = [i[1] for i in actions_idx if i[0]==max_value]
        random.shuffle(max_actions)

        #NUM_TO_DIRECTIONAL_ACTION =  {0: 'UP', 1: 'DOWN', 2: 'LEFT', 3: 'RIGHT'}
        action_out = self.setting.NUM_TO_DIRECTIONAL_ACTION[max_actions[0]]
        #print(action_out)
        return action_out


    def take_directional_action(self, action: str) -> None:
        agent_pos = (self.y, self.x)
        new_pos = self.__return_position_based_on_action(agent_pos, action)
        self.y, self.x = new_pos
        self.be_influenced(y=self.y, x=self.x)
