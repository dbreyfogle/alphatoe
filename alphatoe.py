# AlphaToe: Q-Learning practice

import numpy as np
import random
import math
import pickle


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
        print('    0   1   2')
        print('  *---*---*---*')
        for row in range(3):
            row_str = '{} | '.format(row)
            for col in range(3):
                mark = self.state[row, col]
                row_str += '{} | '.format(to_txt[mark])
            print(row_str)
            print('  *---*---*---*')

    def is_full(self):
        """Checks if the board is full"""
        for row in range(3):
            for col in range(3):
                if self.state[row, col] == 0: # Any empty space
                    return False
        return True

    def get_actions(self, mark):
        """A list of actions that the player can take, i.e. empty spaces (row, col, mark)"""
        assert mark in [1, -1], 'Mark not recognized'
        actions = []
        for row in range(3):
            for col in range(3):
                if self.state[row, col] == 0:
                    actions.append((row, col, mark))
        return actions

    def get_rewards(self, mark):
        """Return rewards for the player in the current state"""
        assert mark in [1, -1], 'Mark not recognized'
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
        assert self.state[row, col] == 0, 'Invalid coordinates'
        assert mark in [1, -1], 'Mark not recognized'
        self.state[row, col] = mark
        reward = self.get_rewards(mark)
        return self.state, reward

    def is_done(self):
        """Check if the game is finished"""
        done = False
        if self.get_rewards(1) or self.is_full():
            done = True
        return done


class Agent():
    """AlphaToe Q-Learning agent"""

    def __init__(self, mark):
        """Setup an agent and assign it X's or O's"""
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
        """Save the model to a .json file at the chosen path"""
        with open(path, 'wb') as fp:
            pickle.dump(self.Q, fp, protocol=pickle.HIGHEST_PROTOCOL)

    def load_model(self, path):
        """Load a model from a .json file at the path"""
        with open(path, 'rb') as fp:
            self.Q = pickle.load(fp)

    def get_mark(self):
        """Get the X/O mark"""
        return self.m

    def set_mark(self, mark):
        """Set X or O"""
        self.m = mark

    def evaluate(self, state=None):
        """Return Q(s). Filter out unavailable actions"""
        if state is None: state = self.s
        s = tuple(state.flatten())
        try: Q_s = {a: q for a, q in self.Q[s].items() if a[2] == self.m}
        except: Q_s = {} # No history
        return Q_s

    def act(self, env, e=0.1):
        """Take an action in the Environment and return rewards. Random actions are taken e percent of the time"""
        self.s[:] = env.get_state()
        actions = env.get_actions(self.m)
        if e > random.uniform(0, 1):
            a = random.choice(actions)  # Explore
        else:
            Q_s = self.evaluate()
            if Q_s:
                a = max(Q_s, key=Q_s.get) # Enhance
            else:
                a = random.choice(actions) # Explore
        s1, r = env.step(a)
        s = tuple(self.s.flatten())
        if s not in self.Q: self.Q[s] = {}
        if a not in self.Q[s]: self.Q[s].update({a: 0})
        self.s_prev[:] = self.s
        self.s[:] = s1
        self.a = a
        return r

    def observe(self, r, lr=0.2, y=0.3):
        """Observe rewards and update Q for the previous action"""
        if self.a is None: return # No action to observe
        Q_s = self.evaluate(self.s_prev)
        Q_s1 = self.evaluate(self.s)
        q, q1 = 0, 0
        if Q_s: q = max(Q_s.values())
        if Q_s1: q1 = max(Q_s1.values())
        s = tuple(self.s_prev.flatten())
        self.Q[s][self.a] = q + lr * (r + (y * q1) - q)


class Human():
    """For playing against AlphaToe"""

    def __init__(self, mark):
        """Assign X/O"""
        self.m = mark

    def act(self, env):
        """Ask for the next move and update the Environment"""
        env.display()
        print('Your turn:')
        valid = False
        while not valid:
            row = int(input('Row #: '))
            col = int(input('Column #: '))
            try:
                if env.get_state()[row, col] == 0: # Must exist and be empty
                    valid = True
            except:
                print('Sorry, that spot is unavailable. Try again:')
        env.step((row, col, self.m))


if __name__ == '__main__':
#    n_games = 500000
#    env = Environment()
#    a1 = Agent(1)
#    a2 = Agent(-1)
#    a1_wins, a2_wins = 0, 0
#    for i in range(n_games):
#        lr = 0.5 * math.exp(-4.5e-6 * i)
#        e = 0.2 * math.exp(-2e-6 * i)
#        if a1_wins <= a2_wins:
#            while True:
#                a1.observe(a1.act(env, e))
#                a2.observe(env.get_rewards(a2.get_mark()), lr)
#                if env.is_done():
#                    if env.get_rewards(1): a1_wins += 1
#                    env.reset()
#                    a1.reset()
#                    a2.reset()
#                    break
#                a2.observe(a2.act(env, e))
#                a1.observe(env.get_rewards(a1.get_mark()), lr)
#                if env.is_done():
#                    if env.get_rewards(1): a2_wins += 1
#                    env.reset()
#                    a1.reset()
#                    a2.reset()
#                    break
#        elif a1_wins > a2_wins:
#            while True:
#                a2.observe(a2.act(env, e))
#                a1.observe(env.get_rewards(a1.get_mark()), lr)
#                if env.is_done():
#                    if env.get_rewards(1): a2_wins += 1
#                    env.reset()
#                    a1.reset()
#                    a2.reset()
#                    break
#                a1.observe(a1.act(env, e))
#                a2.observe(env.get_rewards(a2.get_mark()), lr)
#                if env.is_done():
#                    if env.get_rewards(1): a1_wins += 1
#                    env.reset()
#                    a1.reset()
#                    a2.reset()
#                    break
#    print('a1 wins: {}, a2 wins: {}, ties: {}'.format(a1_wins, a2_wins, (n_games - (a1_wins + a2_wins))))
#    a1.Q.update(a2.Q)
#    a1.save_model('model.pkl')

    # Play against AlphaToe
    n_games = 5
    env = Environment()
    atoe = Agent(1)
    atoe.load_model('model.pkl')
    me = Human(-1)
    atoe_wins, your_wins = 0, 0
    for i in range(n_games):
        while True:
            atoe.observe(atoe.act(env, e=0))
            if env.is_done():
                env.display()
                if env.get_rewards(1):
                    print('AlphaToe won')
                    atoe_wins += 1
                else: print('Tie game')
                env.reset()
                atoe.reset()
                break
            me.act(env)
            atoe.observe(env.get_rewards(atoe.get_mark()))
            if env.is_done():
                env.display()
                if env.get_rewards(1):
                    print('You won')
                    your_wins += 1
                else: print('Tie game')
                env.reset()
                atoe.reset()
                break
    print('Your wins: {}, alphatoe wins: {}, ties: {}'.format(your_wins, atoe_wins, (n_games - (atoe_wins + your_wins))))