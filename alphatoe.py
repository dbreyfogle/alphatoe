# AlphaToe

import numpy as np
import random


class Environment():
    """A tic-tac-toe board defined in RL terms"""

    def __init__(self, turn, state=np.zeros((3, 3))):
        """Setup a 3 x 3 board as an array and specify who's going first
        1 ~ X, -1 ~ O, 0 ~ ''
        """
        self.state = state
        self.turn = turn

    def reset(self):
        """Resets to empty"""
        self.state = np.zeros((3, 3))

    def get_state(self):
        """Return the current state"""
        return self.state

    def set_state(self, state):
        """Set the current state"""
        self.state = state

    def get_turn(self):
        """Return who's turn it is (+/- 1 for X/O)"""
        return self.turn

    def set_turn(self, turn):
        """Set who's turn it is"""
        self.turn = turn

    def display(self, state=None):
        """
            0   1   2
          *---*---*---*
        0 | X | O | X |
          *---*---*---*
        1 |   | X | O |
          *---*---*---*
        2 | X |   | O |
          *---*---*---*
        """
        if state is None: state = self.state
        to_txt = {0: ' ', 1: 'X', -1: 'O'}
        print('    0   1   2')
        print('  *---*---*---*')
        for row in range(3):
            row_str = '{} | '.format(row)
            for col in range(3):
                mark = state[row, col]
                row_str += '{} | '.format(to_txt[mark])
            print(row_str)
            print('  *---*---*---*')

    def is_full(self, state=None):
        """Checks if the board is full"""
        if state is None: state = self.state
        for row in range(3):
            for col in range(3):
                if state[row, col] == 0: # Any empty space
                    return False
        return True

    def get_actions(self, state=None):
        """Return a list of actions that can be taken in the given state, i.e.
        the indices of empty spaces (row, col)
        """
        if state is None: state = self.state
        actions = []
        for row in range(3):
            for col in range(3):
                if state[row, col] == 0:
                    actions.append((row, col))
        return actions

    def get_rewards(self, state=None):
        """Return rewards for each player given the state (X reward, O reward)"""
        if state is None: state = self.state
        for i in range(3):
            this_row_sum = np.sum(state[i, :])
            this_col_sum = np.sum(state[:, i])
            if this_row_sum == 3 or this_col_sum == 3: return 1, -1
            if this_row_sum == -3 or this_col_sum == -3: return -1, 1
        diag1_sum = np.trace(state)
        diag2_sum = np.trace(np.fliplr(state))
        if diag1_sum == 3 or diag2_sum == 3: return 1, -1
        if diag1_sum == -3 or diag2_sum == -3: return -1, 1
        return 0, 0

    def step(self, action):
        """Perform an action and return the new state and any rewards"""
        assert self.state[action[0], action[1]] == 0, 'Environment tried to process an invalid action'
        self.state[action[0], action[1]] = self.turn
        rewards = self.get_rewards()
        self.turn *= -1
        return self.state, rewards


class Agent():
    """AlphaToe"""

    def __init__(self, mark, state=np.zeros((3, 3))):
        """Spawn a q-learning agent. Must provide the X/O mark (+/- 1). Current state is
        optional and will default to an empty board
        """
        self.mark = mark
        self.state = state
        self.model = {1: {}, -1: {}} # States, actions, q-values
        
    def reset(self):
        """Resets back to a new game"""
        self.state = np.zeros((3, 3))
    
    def evaluate(self, state=None, mark=None):
        """Return a table of explored actions and q-values for the given state"""
        if state is None: state = self.state
        if mark is None: mark = self.mark
        state_key = tuple(self.state.flatten())
        return self.model[mark][state_key]

    def act(self, env, e=0.1):
        """Take an action in the Environment. Larger epsilon values encourage exploration where random
        actions are taken epsilon percent of the time
        """
        self.state = env.get_state()
        m = self.mark
        s = tuple(self.state.flatten())
        choices = env.get_actions()
        if e > random.uniform(0, 1): # Explore
            a = random.choice(choices)
        else: # Exploit
            try:
                Q = self.evaluate()
                a = max(Q, key=Q.get)
            except: a = random.choice(choices)
        s1, R = env.step(a)
        if m == 1: r = R[0]
        else: r = R[1]
        return s1, a, r

    def observe(self, s1, a, r, lr=0.3, y=0.5):
        """"""
        m = self.mark
        s = tuple(self.state.flatten())
        s1_key = tuple(s1.flatten())
        if s1_key not in self.model[m]:
            self.model[m][s1_key] = {}
        Q = self.evaluate(s1)
        try: q_next = max(Q.values())
        except: q_next = 0
        try: q_curr = self.model[m][s][a]
        except: q_curr = 0
        self.model[m][s][a] = q_curr + lr * (r + (y * q_next) - q_curr)


class Human():
    """For playing against AlphaToe"""

    def __init__(self):
        """"""
        mark = 0 # 'X' or 'O' preference
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
            row = int(input('Row #: '))
            col = int(input('Column #: '))
            try:
                if env.get_state[row, col] == 0: # Must exist and be empty
                    is_invalid = False
            except:
                print('Sorry, that spot is taken or out of bounds. Please try again:')
        return (row, col)


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