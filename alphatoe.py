# AlphaToe: Q-Learning practice

import numpy as np
import random


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
        """Return rewards for the player given the current state"""
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
        """Perform an action and return the new state, reward, and a done boolean"""
        row, col, mark = action[0], action[1], action[2]
        assert self.state[row, col] == 0, 'Invalid coordinates'
        assert mark in [1, -1], 'Mark not recognized'
        self.state[row, col] = mark
        reward = self.get_rewards(mark)
        return self.state, reward

    def is_done(self):
        """Check if the game is completed"""
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
        self.state = np.zeros((3, 3))

    def get_mark(self):
        """Get the agent's X/O mark"""
        return self.m

    def evaluate(self, state=None):
        """Return Q(s) where s is a state. Filter out unavailable actions"""
        if state is None: state = self.s
        s = tuple(state.flatten())
        try: Q_s = {a: q for a, q in self.Q[s].items() if a[2] == self.m}
        except: Q_s = {} # No history
        return Q_s

    def act(self, env, e=0.1):
        """Take an action in the Environment. Random actions are taken e percent of the time"""
        self.s[:] = env.get_state()
        actions = env.get_actions(self.m)
        if e > random.uniform(0, 1):
            print('Randomly exploring other options')
            a = random.choice(actions)  # Explore
        else:
            Q_s = self.evaluate()
            if Q_s:
                print('Taking the greedy route')
                a = max(Q_s, key=Q_s.get) # Enhance
            else:
                print('No history. Taking a random action')
                a = random.choice(actions) # No exp
        s1, r = env.step(a)
        s = tuple(self.s.flatten())
        if s not in self.Q: self.Q[s] = {}
        if a not in self.Q[s]: self.Q[s].update({a: 0})
        self.s_prev[:] = self.s
        self.s[:] = s1
        self.a = a
        return r

    def observe(self, r, lr=0.3, y=0.5):
        """Observe rewards and update Q for the previous move"""
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