# AlphaToe: Q-Learning practice

import numpy as np
import random
import pickle
import sys
import math
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('-t', '--train', action='store_true', help='train the model')
args = parser.parse_args()


class Environment():
    """A tic-tac-toe board in RL terms"""

    def __init__(self):
        """Setup a 3 x 3 board as an array: 1 ~ X, -1 ~ O, 0 ~ empty"""
        self.state = np.zeros((3, 3))

    def reset(self):
        """Resets to empty"""
        self.state = np.zeros((3, 3))

    def get_state(self):
        """Get the current state"""
        return self.state

    def display(self):
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
        to_txt = {0: ' ', 1: 'X', -1: 'O'}
        print('\n    0   1   2')
        print('  *---*---*---*')
        for row in range(3):
            row_str = '{} | '.format(row)
            for col in range(3):
                mark = self.state[row, col]
                row_str += '{} | '.format(to_txt[mark])
            print(row_str)
            print('  *---*---*---*')

    def get_actions(self, mark):
        """A list of actions that the player can take, i.e. empty spaces (row, col, mark)"""
        actions = []
        for row in range(3):
            for col in range(3):
                if self.state[row, col] == 0:
                    actions.append((row, col, mark))
        return actions

    def rewards(self, mark):
        """Return rewards for the player in the current state"""
        goal = 3 * mark
        for i in range(3):
            row_sum = np.sum(self.state[i, :])
            col_sum = np.sum(self.state[:, i])
            if row_sum == goal or col_sum == goal: return 1
            if row_sum == -goal or col_sum == -goal: return -1
        diag1_sum = np.trace(self.state)
        diag2_sum = np.trace(np.fliplr(self.state))
        if diag1_sum == goal or diag2_sum == goal: return 1
        if diag1_sum == -goal or diag2_sum == -goal: return -1
        return 0

    def step(self, action):
        """Perform an action and return the new state and rewards"""
        row, col, mark = action[0], action[1], action[2]
        assert self.state[row, col] == 0, 'Environment tried to process an invalid action'
        self.state[row, col] = mark
        r = self.rewards(mark)
        return self.state, r

    def is_full(self):
        """Checks if the board is full"""
        for row in range(3):
            for col in range(3):
                if self.state[row, col] == 0: # Any empty space
                    return False
        return True

    def is_done(self):
        """Check if the game is finished"""
        return self.rewards(1) or self.is_full()


class Agent():
    """AlphaToe Q-Learning agent"""

    def __init__(self, mark):
        """Setup an agent and assign it X or O"""
        assert mark in [-1, 1], 'Mark must be +/- 1'
        self.Q = {} # States, actions, q-values
        self.m = mark
        self.s = np.zeros((3, 3))
        self.s_prev = np.zeros((3, 3)) # Previous state
        self.a = None # Previous action

    def reset(self):
        """Reset to a new game"""
        self.s = np.zeros((3, 3))
        self.s_prev = np.zeros((3, 3))
        self.a = None

    def save_model(self, path):
        """Save the model to a pickle file at the path"""
        with open(path, 'wb') as fp:
            pickle.dump(self.Q, fp)

    def load_model(self, path):
        """Load a model from a pickle file at the path"""
        with open(path, 'rb') as fp:
            self.Q = pickle.load(fp)

    def get_mark(self):
        """Get the X/O mark"""
        return self.m

    def set_mark(self, mark):
        """Set X or O"""
        assert mark in [-1, 1], 'Mark must be +/- 1'
        self.m = mark

    def evaluate(self, state=None):
        """Return Q(s) as an array. Filter out unavailable actions"""
        if state is None: state = self.s
        s = tuple(state.flatten())
        try:
            d = {a: q for a, q in self.Q[s].items() if a[2] == self.m}
            Q_s = np.array(d.items())
        except: Q_s = None # No history
        return Q_s

    def act(self, env, e=0.3):
        """Take an action in the Environment and return rewards. Random actions are taken e percent of the time"""
        self.s[:] = env.get_state()
        actions = env.get_actions(self.m)
        a = random.choice(actions) # Explore
        if e < random.uniform(0, 1):
            Q_s = self.evaluate()
            if Q_s is not None:
                np.random.shuffle(Q_s) # In case of multiple maximums
                a = Q_s[np.argmax(Q_s[:, 1])][0] # Enhance
        s1, r = env.step(a)
        s = tuple(self.s.flatten())
        if s not in self.Q: self.Q[s] = {}
        if a not in self.Q[s]: self.Q[s].update({a: 0})
        self.s_prev[:], self.s[:], self.a = self.s, s1, a
        return r

    def observe(self, r, lr=0.2, y=0.5):
        """Observe rewards and update Q. The learning rate determines how heavily recent observations
        impact the model. The discount factor (y) diminishes the value of long term rewards"""
        if self.a is None: return # Nothing to observe
        s = tuple(self.s_prev.flatten())
        q, q1 = self.Q[s][self.a], 0
        Q_s1 = self.evaluate(self.s)
        if Q_s1 is not None: q1 = Q_s1[np.argmax(Q_s1[:, 1])][1]
        self.Q[s][self.a] = q + lr * (r + (y * q1) - q)


class Human():
    """For playing against AlphaToe"""

    def __init__(self, mark):
        """Assign X/O"""
        assert mark in [-1, 1], 'Mark must be +/- 1'
        self.m = mark

    def act(self, env):
        """Ask for the next move and update the Environment"""
        env.display()
        r = None
        while True:
            row = input('Row #: ')
            if row == 'q': sys.exit()
            col = input('Column #: ')
            if col == 'q': sys.exit()
            try:
                _, r = env.step((int(row), int(col), self.m))
                break
            except: print("Sorry, try again. ('q' to exit)")
        return r


if __name__ == '__main__':

    # Training
    if args.train:
        env = Environment()
        a1 = Agent(1)
        a2 = Agent(-1)
        n_games = 100000
        e = 0.3 # Epsilon
        y = 0.5 # Gamma
        a1_wins, a2_wins = 0, 0
        for i in range(n_games):
            if i % 10000 == 0: print('Training run {} of {}'.format(i, n_games))
            lr = 0.7 * math.exp(-1.5e-5 * i) # Learning rate (decaying)
            while True:
                r = a1.act(env, e)
                a1.observe(r, lr, y)
                a2.observe(-r, lr, y)
                if env.is_done():
                    a1_wins += r
                    env.reset()
                    a1.reset()
                    a2.reset()
                    break
                r = a2.act(env, e)
                a2.observe(r, lr, y)
                a1.observe(-r, lr, y)
                if env.is_done():
                    a2_wins += r
                    env.reset()
                    a1.reset()
                    a2.reset()
                    break
        print('a1 wins: {}, a2 wins: {}, Ties: {}'.format(a1_wins, a2_wins, n_games - a1_wins + a2_wins))
        a1.save_model('model.pkl')

    # Interactive
    else:
        env = Environment()
        you = Human(-1)
        atoe = Agent(1)
        atoe.load_model('model.pkl')
        n_games = 5
        print('Best of {} against AlphaToe. Good luck!'.format(n_games))
        e = 0 # Epsilon
        lr = 0.2 # Learning rate
        y = 0.5 # Gamma
        atoe_wins, your_wins = 0, 0
        for i in range(n_games):
            if atoe_wins > (n_games / 2) or your_wins > (n_games / 2): break
            while True:
                print('\nQ(s):')
                print(atoe.evaluate(env.get_state()))
                r = atoe.act(env, e)
                atoe.observe(r, lr)
                if env.is_done():
                    env.display()
                    if r: print('AlphaToe won!')
                    else: print('Tie game!')
                    atoe_wins += r
                    print('Your wins: {}, AlphaToe wins: {}'.format(your_wins, atoe_wins))
                    env.reset()
                    atoe.reset()
                    break
                r = you.act(env)
                atoe.observe(r, lr)
                if env.is_done():
                    env.display()
                    if r: print('You won!')
                    else: print('Tie game!')
                    your_wins += r
                    print('Your wins: {}, AlphaToe wins: {}'.format(your_wins, atoe_wins))
                    env.reset()
                    atoe.reset()
                    break