# AlphaToe

import numpy as np
import random
import json


class Environment(): # Tic-tac-toe board
    
    def __init__(self):
        """Setup a 3 x 3 board as an array.
        ' ' ~ 0
        'X' ~ 1
        'O' ~ -1
        """
        self.state = np.zeros((3, 3))
        self.turn = 1 # Keep track who's turn it is (X first)
        
    def reset(self):
        """Resets to empty"""
        self.state = np.zeros((3, 3))
        
    def get_state(self):
        """Return the current state"""
        return self.state
        
    def set_state(self, state):
        """Set the current state"""
        self.state = state
    
    def display(self, state=self.state):
        """
            1   2   3
          *---*---*---*
        1 | X | O | X |
          *---*---*---*
        2 |   | X | O |
          *---*---*---*
        3 | X |   | O |
          *---*---*---*
        """
        to_txt = {0: ' ', 1: 'X', -1: 'O'} 
        print('    1   2   3')
        print('  *---*---*---*')
        for row in range(3):
            row_str = f'{i+1} | '
            for col in range(3):
                mark = state[row, col]
                row_str += f'{to_txt[mark]} | '
            print(row_str)
            print('  *---*---*---*')
            
    def get_actions(self, state=self.state):
        """Return a list of actions that can be taken in the state, i.e.
        indices of empty spaces (row, col)
        """
        s = []
        for row in range(3):
            for col in range(3):
                if state[row, col] == 0:
                    s.append((row, col))
        return s

    def get_rewards(self, turn=self.turn, state=self.state):
        """Return the reward for given player at the given state"""
        for i in range(goal):
            this_row_sum = np.sum(state[i, :])
            this_col_sum = np.sum(state[:, i])
            if this_row_sum == goal or this_col_sum == goal:
                return mark
        diag1_sum = np.trace(state)
        diag2_sum = np.trace(np.fliplr(state))
        if diag1_sum == goal or diag2_sum == goal:
            return mark
        return 0
    
    def is_full(self):
        """Checks if the board is full"""
        for row in range(3):
            for col in range(3):
                if self.state[row, col] == 0: # Any empty space
                    return False
        return True
    
    def step(self, action):
        """Udpate our state with an action and return:
        
        next state as numpy.array
        reward
        finished as boolean
        
        Only accepts valid actions on empty board spaces
        """
        self.state[*action] = self.turn
        reward = self.get_rewards(self.turn, self.state)
        finished = False
        if reward or self.is_full():
            finished = True
        self.turn *= -1
        return self.state, reward, finished


class Agent(): # AlphaToe

    def __init__(self, mark, state=None):
        """"""
        if not state:
            state = np.zeros((3, 3))
        self.state = (state, mark)
        self.model = {} # States, actions, q-values
        
    def state_key(self, state=self.state):
        """Return the agent's state as a value for use in a dictionary"""
        return str(tuple(state[0].flatten(), state[1]))
        
    def act(self, env, e=0.1):
        """Take an action in an Environment"""
        a = env.get_actions()
        if e > random.uniform(0, 1):
            return random.choice(a) # Explore
        state = self.state_key()
        if state not in self.model:
            self.model[s] = {}
        for a in avail_actions:
            if a not in self.model[s]:
                self.model[s].update({a: 0}) # Initialize
        Q = self.model[s]
        greedy = max(Q, key=Q.get) # Exploit
        state, reward, is_done = env.step(action)
        self.state = state
        return greedy

    def observe(self, env, y):
        """"""
        
        
class Human(): # For playing against AlphaToe

    def __init__(self):
        """Ask for 'X' or 'O' preference"""
        mark = 0
        mark_str = input("X's or O's? ('X'/'O'): ").upper()
        while mark == 0:
            if mark_str.lower() == 'X': mark = 1
            if mark_str.lower() == 'O': mark = -1
            if mark == 0:
                mark_str = input("Sorry, please enter 'X' or 'O': ").upper()
        self.mark = mark

    def act(self, env):
        """Ask for next move and update the board"""
        env.display()
        print('Your turn:')
        is_invalid = True
        while is_invalid:
            row = int(input('Row #: ')) - 1
            col = int(input('Column #: ')) - 1
            try:
                if env.state[row, col] == 0: # Must exist and be empty
                    is_invalid = False
            except:
                print('Sorry, that spot is taken or out of bounds. Please try again:')
        return (row, col, self.mark)


def play_round(p1, p2):
    env = Environment(3)
    p1_turn = True # Player one goes first

    while True:
        if p1_turn:
            s1, r, d = env.step(p1.act(env))
        else:
            s1, r, d = env.step(p2.act(env))
        p1_turn = not p1_turn

        if env.has_winner():
            env.draw_board()
            if p1_turn:
                p2.score += 1
                X_winner = p2.is_X
            else:
                p1.score += 1
                X_winner = p1.is_X
            if X_winner:
                print('X wins!')
            else:
                print('O wins!')
            break

        if env.is_full():
            env.draw_board()
            print('Tie game!')
            break


def play_set(p1, p2, size=3, n_rounds=3):

    for i in range(n_rounds):
        print('Round {} of {}:'.format(i+1,n_rounds))
        # Alternate going first each round
        if i%2 == 0:
            play_round(p1, p2)
        else:
            play_round(p2, p1)

        if p1.is_X:
            print('X score: {}'.format(p1.score))
            print('O score: {}'.format(p2.score))
        else:
            print('X score: {}'.format(p2.score))
            print('O score: {}'.format(p1.score))
    print('Game over!')


if __name__ == '__main__':
    p1 = Human()
    p2 = Agent(mark=-p1.mark)
    play_set(p1, p2, n_rounds=3)