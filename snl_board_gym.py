import numpy as np
import pickle

#region Player Class

class Player:
    def __init__(self, symbol):
        self.moves = 18
        
        # player token positions [ 1 - 100 ]
        self.pos_token_array = np.zeros(3,)
        self.symbol = symbol
    
    def get_score(self): 
        score = 0
        for token_position in self.pos_token_array:
            if token_position == 100:
                score += 50
            else:
                score += token_position

        return score

#endregion
class SnlBoard:
    
    def __init__(self,printing=False):
        
        self.print_info = printing
        
        # 100 positions available
        # player 2 is a random bot
        
        self.board = np.zeros(shape=(8,100))
        
        self.die_val = -1
        self.total_positions = 100
        self.ties = 0
        
        self.token_home_reward = 20
        self.invalid_move_reward = -50
        self.game_won_reward = 100
        self.game_lost_reward = 100
        self.game_tie_reward = 50
        
        self.p1_wins = 0
        self.p2_wins = 0
        
        self.opp = {1:2,2:1}
        
        self.info = dict()
        
    def reset(self,state):
        """resets the board to its initial state

        Args:
            state (_type_): sample from the observation_state [gym space defined for the gym environment] 
        """        
        # get initial die value
        self.die_val = state[0]  # values [0 - 5]
        
        # get initial state would always be array(2,3) of zeros
        self.board = state[1:].reshape(2,3)
        
        # board info saved
        self.info['starting_state'] = self.board
        
        self.p1 = Player(1)
        self.p2 = Player(2)
        
        self.info = dict()
        
        if(self.print_info):
            print('################################')
            print('environment state:')
            print('die value :{}\nboard state :\n{}'.format(self.die_val + 1, self.board))
            print('player 1 init: ', self.p1)
            print('player 2 init: ', self.p2)
            print('################################')
        
     # step for gym environment 
    
    def perform_step(self, action):
        """perform one step
        i.e player 1 plays and then player 2
        return board state after this
        """        
        
        # reset die value to -1 after p2 turn
        if self.die_val == -1:
            self.die_val = np.random.randint(0, 6)
            
        ######## player 1 plays
        reward = 0
        observation = self.get_board_state()
        is_game_end = False
        
        # action type [VALID | INVALID]
        action_type = self.player_plays(self.p1, action)
        self.p1.moves += -1
        
        # in case the action is invalid
        if action_type == 'INVALID':                
            # get reward
            reward = self.invalid_move_reward

        if self.print_info:
            print('######P1')
            print('action :',action)
            print('die: ', self.die_val + 1)
            print('board state: ', self.get_board_state())
            print('p1 score: ', self.p1.get_score())
            print('p1 token positions: ', self.p1.pos_token_array)
            print('action type: ', action_type)
            print('\n')
        
        ######## player 2 plays

        # roll die
        self.die_val = np.random.randint(0,6) # [ 0-5 ]
        
        # action type does not matter for p2
        action_type = self.player_plays(self.p2, np.random.randint(0,3))
        self.p2.moves += -1
        
        if(self.print_info):
            print('######P2')
            print('die: ', self.die_val + 1)
            print('board state: ', self.get_board_state())
            print('p2 score: ', self.p2.get_score())
            print('p2 token positions: ', self.p2.pos_token_array)
            print('\n')
        
        ####### setup for gym
        
        # 1. get final state
        self.die_val = np.random.randint(0,6) # [ 0-5 ] die roll for next state
        observation = np.concatenate((np.array([self.die_val]), self.get_gym_state())) # observation for next state
        
        # 2. get final reward
        is_game_end = self.game_finished()

        if(is_game_end): # rewards given at end of game
            reward += self.game_end_rewards()
        
        score_diff = (self.p1.get_score() - self.p2.get_score())/4 # score diff rewards
        
        reward += score_diff
        
        # return step output            
        return (observation,reward,is_game_end,{})
    
    
    
    def get_gym_state(self):
        return np.concatenate((self.p1.pos_token_array,self.p2.pos_token_array))
    
    def game_end_info(self):
        return self.info
        
    def is_invalid_move(self, current_position, new_position, active_player:Player):
        
        # check if new position is out of bounds
        if new_position > 100:
            return True
        
        return False
    
    def game_end_rewards(self):
        p1_won = False
        is_tie = False
        
        if self.p1.get_score() > self.p2.get_score():
            p1_won = True
        elif self.p1.get_score() == self.p2.get_score():
            is_tie = True
        
        if p1_won:
            self.info['p1_won'] = True
            self.info['p2_won'] = False
            self.info['tie'] = False
            
            return self.game_won_reward
        elif is_tie:
            self.info['p1_won'] = False
            self.info['p2_won'] = False
            self.info['tie'] = True
            
            return self.game_tie_reward
        else:
            self.info['p1_won'] = False
            self.info['p2_won'] = True
            self.info['tie'] = False
            
            return self.game_lost_reward
              
    def get_board_state(self):
        """
        get board state
        - combination of state and die_val 
        """
        # (die value - 1) + (board state)
        return np.concatenate((self.p1.pos_token_array.flatten(),self.p2.pos_token_array.flatten()))
    
    def player_plays(self, active_player:Player, action):        
        # player plays turn
        token_to_move = action
        
        # board update state
        return self.board_update_after_turn(active_player, token_to_move)
          
    def board_update_after_turn(self, active_player : Player, token_to_move):
        """ 
        1. get the new position for the current token
        2. check if snakes or ladder
        3. update position if 2. is true
        4. check if enemy token is already present
        5. update enemy position if 4. is true
        6. check if self token is already present
        7. update position accordingly
        Args:
            token_symbol (_type_): symbol of token which require updates
        """
                
        # 1. get current position of the token from board
        
        new_token_position = -1
        current_token_position = active_player.pos_token_array[token_to_move]
        
        # get new possible position
        new_token_position = current_token_position + (self.die_val + 1) # die value [0,5]

        # check if valid position
        if(self.is_invalid_move(current_token_position, new_token_position, active_player)):
            return 'INVALID'
        
        
        # 2. & 3. update position if snakes or ladder
        new_token_position,SnL = self.snake_and_ladder(new_token_position)
        
        # 4. check if enemy is present
        enemy_state, enemy_count = self.enemy_check(new_token_position,active_player)
        
        # update to new position
        active_player.pos_token_array[token_to_move] = new_token_position
        # enemy present ? | number of enemy
        if enemy_state:
            if enemy_count == 1 and new_token_position != 100:
                if self.opp[active_player.symbol] == 2:
                    mod_index = np.min(np.where(self.p2.pos_token_array == new_token_position))
                    self.p2.pos_token_array[mod_index] = 0
                else:
                    mod_index = np.min(np.where(self.p1.pos_token_array == new_token_position))
                    self.p1.pos_token_array[mod_index] = 0
        
        return 'VALID'
         
    def enemy_check(self, position, active_player:Player):
        """checks if an enemy player is present in the position of the moving token

        Args:
            position (_type_): position on board [1-100]
            active_player (Agent): current active player
        """
        enemies = 0
        
        if (active_player.symbol == 1):
            # check if p2 present in position
            for pos in self.p2.pos_token_array:
                if pos == position:
                    enemies += 1
        else:
            # check if p2 present in position
            for pos in self.p1.pos_token_array:
                if pos == position:
                    enemies += 1

        if enemies > 0:
            return (True, enemies)
        
        return (False,enemies)
    
    def game_finished(self):
        """check if game finish condition is met
        condtion 1 : if the number moves for each player is exhausted
        condtion 2 : if any of the player reach 100 before moves are exhausted
        """
        if self.p1.moves == 0 and self.p2.moves == 0:
            return True
        return False
    
    def snake_and_ladder(self,position:int):
        """takes the current position of player and returns the updated position in case of snake or ladder
        """        
        if position in self.get_snakes():
            return (self.get_snakes()[position],'snake')
        if position in self.get_ladders():
            return (self.get_ladders()[position],'ladder')
        return (position,'None')
    
    def get_snakes(self):
        snakes = {
            99:4,
            30:11,
            52:29,
            70:51,
            94:75
        }
        return snakes
    
    def get_ladders(self):
        ladders = {
            3:84,
            7:53,
            15:96,
            21:98,
            54:93
        }
        return ladders