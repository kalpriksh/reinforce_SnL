import numpy as np
import pickle

#region Player Class

class Player:
    def __init__(self, symbol):
        self.moves = 10
        
        # player token positions [ 1 - 100 ]
        self.pos_token_array = np.zeros(4,)
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
    
    def __init__(self):
        
        # 100 positions available
        # player 2 is a random bot
        
        self.board = np.zeros(shape=(100,8))
        
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
        
        
        self.p1 = Player()
        self.p2 = Player()
        
        self.opp = {1:2,2:1}
        
        self.info = dict()
    
    # reset for gym environment
    def reset(self,state):
        
        # get initial die value
        self.die_val = state[0]  # values [0 - 5]
        
        # get initial state would always be array(800,) of zeros
        self.board = state[1:].reshape(100,8)
        
        # board info saved
        self.info['starting_state'] = self.board
        
        
     # step for gym environment 
    
    def perform_step(self, action):
        """perform one step
        i.e player 1 plays and then player 2
        return board state after this
        """        
        
        # reset die value to -1 after p2 turn
        if self.die_val != -1:
            self.die_val = np.random.randint(0, 6)
        
        while not self.game_finished():
            
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


            ######## player 2 plays

            # roll die
            self.die_val = np.random.randint(0,6) # [ 0-5 ]
            
            # action type does not matter for p2
            action_type = self.player_plays(self.p2, np.random.randint(0,4))
            self.p2.moves += -1
            
            
            ####### setup for gym
            
            # 1. get final state
            self.die_val = np.random.randint(0,6) # [ 0-5 ] die roll for next state
            observation = np.concatenate((np.array([self.die_val]), self.get_board_state())) # observation for next state
            
            # 2. get final reward
            is_game_end = self.game_finished()

            if(is_game_end): # rewards given at end of game
                reward += self.game_end_rewards()
            
            score_diff = (self.p1.get_score() - self.p2.get_score())/4 # score diff rewards
            
            reward += score_diff
            
            # return step output            
            return (observation,reward,is_game_end,{})
        
        
    
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
            return self.game_won_reward
        elif is_tie:
            return self.game_tie_reward
        else:
            return self.game_lost_reward
              
    def get_board_state(self):
        """
        get board state
        - combination of state and die_val 
        """
        # (die value - 1) + (board state)
        return np.concatenate((np.array([self.die_val]),self.board.flatten()))
    

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
        
        # if SnL == 'snake':
        #     active_player.snake_cut = True
        # if SnL == 'ladder':
        #     active_player.ladder_climb = True
        
        
        # 4. check if enemy is present
        enemy_state, enemy_count = self.enemy_check(new_token_position,active_player)
        
        # enemy present ? | number of enemy
        if enemy_state:
            if enemy_count > 1:
                # in case of multiple enemy
                pos_array = self.board[new_token_position]
                mod_index = np.min(np.where(pos_array == 0))
                
                self.board[new_token_position][mod_index] = active_player.symbol
            else:
                opp_token = self.opp[active_player.symbol]
                
                pos_array = self.board[new_token_position]
                mod_index = np.where(pos_array == opp_token)
                
                self.board[new_token_position][mod_index] = active_player.symbol
                
        # if no enemy is present     
        else:
            pos_array = self.board[new_token_position]
            mod_index = np.min(np.where(pos_array == 0))
            
            self.board[new_token_position][mod_index] = active_player.symbol
            
        # update score based on new token positions
        self.update_player_scores()
        
        return True
       
    def calculate_reward(self,active_player, original_score):
        """calcualtes reward base on actions taken

        Args:
            active_player (Agent): cuurent active player
        """

        if active_player.symbol == 'P1':
            opponent = self.p2
        else: 
            opponent = self.p1

       
        # score diff reward
        score_diff_reward = active_player.score - opponent.score
        
        # snake or ladder reward
        snl_reward = 0
        if active_player.snake_cut:
            snl_reward += active_player.score - original_score
            active_player.snake_cut = False
        if active_player.ladder_climb:
            snl_reward += active_player.score - original_score
            active_player.ladder_climb = False
        
        # reward on token cut?
        token_cut_reward = 0
        if active_player.has_cut_token:
            # fixed reward of 200
            token_cut_reward = 200
            active_player.has_cut_token = False
        
        return (token_cut_reward + score_diff_reward + snl_reward)/100

    def update_player_scores(self):        
        p1_score = 0
        p2_score = 0
        for idx, position in enumerate(self.board):
            for token in position:
                
                if token in self.p1.player_tokens:
                    p1_score += self.board.index(position) + 1
                    if idx == 99: # if token has reached the end then extra points
                        p1_score += 100
                if token in self.p2.player_tokens:
                    p2_score += self.board.index(position) + 1
                    if idx == 99:
                        p2_score += 100
        
        self.p1.score = p1_score
        self.p2.score = p2_score    
        
    def token_on_board(self, token_symbol):
        for position in self.board:
            if token_symbol in position:
                return (True,self.board.index(position),position.index(token_symbol)) # (true,position_index,token_index)
        return (False,-1,-1)

    def enemy_check(self, position, active_player):
        """checks if an enemy player is present in the position of the moving token

        Args:
            position (_type_): position on board [1-100]
            active_player (Agent): current active player
        """        
        enemy_present = False
        enemy_count = 0
        
        for token in self.board[position - 1]:
            if token != 0 and token != active_player.symbol:
                enemy_count += 1
                enemy_present = True
        return(enemy_present, enemy_count)
    
    def update_enemy_token(self,position, active_player):
        """cuts the enemy player and updates scores

        Args:
            position (_type_): _description_
            active_player (Agent): _description_
        """        
        # update enemy position
        self.board[position].pop()
        # update enemy score
        if active_player.symbol == 'P1':
            self.p2.score -= position + 1
        if active_player.symbol == 'P2':
            self.p1.score -= position + 1
            
        # update that active player has cut enemy token
        active_player.has_cut_token = True

    def game_finished(self):
        """check if game finish condition is met
        condtion 1 : if the number moves for each player is exhausted
        condtion 2 : if any of the player reach 100 before moves are exhausted
        """
        if self.p1.moves == 0 and self.p2.moves == 0:
            if self.p1.score > self.p2.score:
                self.p1_wins += 1
            elif self.p1.score < self.p2.score:
                self.p2_wins += 1
            else:
                self.ties += 1
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
    
    def reset(self):
        self.p1.reset()
        self.p2.reset()
        self.board = [[] for _ in range(self.total_positions)]
    
    def get_board():
        pass

    ####utilities#######################################################
    
    def savePolicy(self,active_player):
        fw = open('./snl_rl/STATE_VALUE IMPLEMENTATION/a_one/policies/policy_' + str(active_player.symbol), 'wb')
        pickle.dump(active_player.Q_val, fw)
        fw.close()

    def loadPolicy(self, file, active_player):
        fr = open(file, 'rb')
        active_player.Q_val = pickle.load(fr)
        fr.close()

    def get_stats(self):
        print('p1 wins : ',self.p1_wins)
        print('p1 token cuts : ',self.p1.number_of_tokens_cut)
        print('p2 wins : ',self.p2_wins)
        print('p1 win/ratio : ',self.p1_wins/self.p2_wins)
        print('ties : ',self.ties)

    ####strats###########################################################
    
    def check_stratergies(self, board, active_player):
        
        strat = BoardStrats(board, active_player,self.die_val)
        ladder_token = strat.best_ladder_token()
        
        enemy_cut_token = strat.best_enemy_cut_token()

        if enemy_cut_token:
            return enemy_cut_token
        
        if ladder_token:
            return ladder_token
        
        return None