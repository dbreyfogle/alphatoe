# AlphaToe

import numpy as np
import random


class Environment(): # Tic-tac-toe board

    def __init__(self, state=np.zeros((3, 3)), turn=1):
        """Setup a 3 x 3 board as an array.
        ' ' ~ 0
        'X' ~ 1
        'O' ~ -1
        """
        self.state = state
        self.turn = turn # X will go first by default

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
        """Return who's turn it is (+/- 1)"""
        return self.turn

    def set_turn(self, turn):
        """Set who's turn it is"""
        self.turn = turn

    def display(self, state=None):
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
        s = []
        for row in range(3):
            for col in range(3):
                if state[row, col] == 0:
                    s.append((row, col))
        return s

    def get_rewards(self, state=None):
        """Return rewards for each player given the state (X reward, O reward)"""
        if state is None: state = self.state
        for i in range(3):
            this_row_sum = np.sum(state[i, :])
            this_col_sum = np.sum(state[:, i])
            if this_row_sum == 3 or this_col_sum == 3:
                return (1, -1)
            if this_row_sum == -3 or this_col_sum == -3:
                return (-1, 1)
        diag1_sum = np.trace(state)
        diag2_sum = np.trace(np.fliplr(state))
        if diag1_sum == 3 or diag2_sum == 3:
            return (1, -1)
        if diag1_sum == -3 or diag2_sum == -3:
            return (-1, 1)
        return (0, 0)

    def step(self, action):
        """Udpate our state with an action and return the new state, any rewards,
        and finished as a boolean
        """
        self.state[action[0], action[1]] = self.turn
        rewards = self.get_rewards()
        self.turn *= -1
        finished = False
        if rewards[0] or self.is_full():
            finished = True
        return self.state, rewards, finished


class Agent(): # AlphaToe

    def __init__(self, mark, state=np.zeros((3, 3))):
        """"""
        self.mark = mark
        self.state = state
        self.model = {} # States, actions, q-values

    def state_key(self, state=None):
        """Return the agent's state as a value for use in a dictionary"""
        if state is None: state = self.state
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