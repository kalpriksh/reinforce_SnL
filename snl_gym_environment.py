import gym
import numpy as np
from gym import spaces
from snl_board_gym import SnlBoard

class SNL_env(gym.Env):
    """Custom Environment that follows gym interface."""

    metadata = {"render.modes": ["human"]}

    def __init__(self, printing=False):
        super().__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.action_space = spaces.Discrete(3)
        
        ## 8 [total tokens] * 100 [positions]
        self.observation_space = spaces.MultiDiscrete([6]+[101 for _ in range(0, 6)])
        self.SNLBoard = SnlBoard(printing)
        

    def step(self, action):
        # player 1 plays
        # player 2 plays
        # new state is observed
        # reward is calculated
        # check if game is completed
        # info is optional
        
        observation, reward, done, info = self.SNLBoard.perform_step(action)
        
        return observation, reward, done, info

    def reset(self):
        # initializing state
        state = np.concatenate(([np.random.randint(0,6)],np.zeros(shape=(2,3)).flatten()))
        
        # reset board | set inital state
        self.SNLBoard.reset(state)
        
        return state
    

    def render(self, mode='console'):
        if mode != 'console':
            raise NotImplementedError()
        
        return self.SNLBoard.game_end_info()
        