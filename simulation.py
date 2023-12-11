import numpy as np
from collections import defaultdict


class MonteCarloTreeSearchNode():
    def __init__(self, state, parent=None, parent_action=None):
        self.state = state
        self.parent = parent
        self.parent_action = parent_action
        self.children = []
        self._number_of_visits = 0
        self._results = defaultdict(int)
        self._results[1] = 0
        self._results[-1] = 0
        self._untried_actions = None
        self._untried_actions = self.untried_actions()
        return

    def untried_actions(self):
        self._untried_actions = self.state.get_legal_actions()
        return self._untried_actions

    def q(self):
        wins = self._results[1]
        loses = self._results[-1]
        return wins - loses

    def n(self):
        return self._number_of_visits

    def expand(self):
        action = self._untried_actions.pop()
        next_state = self.state.move(action)
        child_node = MonteCarloTreeSearchNode(next_state, parent=self, parent_action=action)
        self.children.append(child_node)
        return child_node

    def is_terminal_node(self):
        return self.state.is_game_over()

    def rollout(self):
        current_rollout_state = self.state

        while not current_rollout_state.is_game_over():
            possible_moves = current_rollout_state.get_legal_actions()

            action = self.rollout_policy(possible_moves)
            current_rollout_state = current_rollout_state.move(action)
        return current_rollout_state.game_result()

    def backpropagate(self, result):
        self._number_of_visits += 1.
        self._results[result] += 1.
        if self.parent:
            self.parent.backpropagate(result)

    def is_fully_expanded(self):
        return len(self._untried_actions) == 0

    def best_child(self, c_param=0.1):

        choices_weights = [(c.q() / c.n()) + c_param * np.sqrt((2 * np.log(self.n()) / c.n())) for c in self.children]
        return self.children[np.argmax(choices_weights)]

    def rollout_policy(self, possible_moves):
        return possible_moves[np.random.randint(len(possible_moves))]

    def _tree_policy(self):

        current_node = self
        while not current_node.is_terminal_node():

            if not current_node.is_fully_expanded():
                return current_node.expand()
            else:
                current_node = current_node.best_child()
        return current_node

    def best_action(self):
        simulation_no = 100

        for i in range(simulation_no):
            v = self._tree_policy()
            reward = v.rollout()
            v.backpropagate(reward)

        return self.best_child(c_param=0.)


class State:
    def __init__(self, board, colour, turn):
        self.board = board
        self.colour = colour
        self.turn = turn
        self.winner = None
        self.board_size = 9
        return

    def get_legal_actions(self):
        legal_actions = []
        for i, row in enumerate(self.board):
            for j, column in enumerate(row):
                if column == 0:
                    legal_actions.append([i, j])
        return legal_actions

    '''
    Modify according to your game or
    needs. Constructs a list of all
    possible actions from current state.
    Returns a list.
    '''

    def is_game_over(self):
        for row in range(self.board_size - 5 + 1):
            for col in range(self.board_size - 5 + 1):
                current_grid = self.board[row: row + 5, col: col + 5]
                sum_horizontal = np.sum(current_grid, axis=1)
                sum_vertical = np.sum(current_grid, axis=0)
                sum_diagonal_1 = np.sum(current_grid.diagonal())
                sum_diagonal_2 = np.sum(np.flipud(current_grid).diagonal())
                if 5 in sum_horizontal or 5 in sum_vertical:
                    self.winner = 1
                    return True
                if sum_diagonal_1 == 5 or sum_diagonal_2 == 5:
                    self.winner = 1
                    return True
                if -5 in sum_horizontal or -5 in sum_vertical:
                    self.winner = -1
                    return True
                if sum_diagonal_1 == -5 or sum_diagonal_2 == -5:
                    self.winner = -1
                    return True

            if not any(0 in rows for rows in self.board):
                self.winner = 0
                return True
            else:
                return False

        '''
        Modify according to your game or
        needs. It is the game over condition
        and depends on your game. Returns
        true or false
        '''

    def game_result(self):
        if self.winner == 0:
            return 0
        if self.winner == self.colour:
            return 1
        else:
            return -1

    '''
    Modify according to your game or
    needs. Returns 1 or 0 or -1 depending
    on your state corresponding to win,
    tie or a loss.
    '''

    def move(self, action):
        new_board = self.board.copy()
        if self.turn:
            new_board[(action[0], action[1])] = self.colour
        else:
            new_board[(action[0], action[1])] = self.colour*-1
        return State(new_board, self.colour, not self.turn)

    '''
    Modify according to your game or
    needs. Changes the state of your
    board with a new value. For a normal
    Tic Tac Toe game, it can be a 3 by 3
    array with all the elements of array
    being 0 initially. 0 means the board
    position is empty. If you place x in
    row 2 column 3, then it would be some
    thing like board[2][3] = 1, where 1
    represents that x is placed. Returns
    the new state after making a move.
    '''

'W.I.P. function of main code'

for i in range(100):
    current_board = np.zeros((9,9),dtype=int)
    turn = 0
    while True:
        '''print(current_board)
        print('Turn: {}'.format(turn))
        print('\n\n')'''
        black_state = State(current_board, 1, True)
        black = MonteCarloTreeSearchNode(black_state)
        action = black.best_action()
        current_board = action.state.board
        turn+=1
        '''print(current_board)
        print('Turn: {}'.format(turn))
        print('\n\n')'''
        if action.state.winner is not None:
            if action.state.winner == 1:
                print('Black Wins')
            else:
                print('Draw')
            break
        white_state = State(current_board, -1, True)
        white = MonteCarloTreeSearchNode(white_state)
        action = white.best_action()
        current_board = action.state.board
        turn += 1
        '''print(current_board)
        print('Turn: {}'.format(turn))
        print('\n\n')'''
        if action.state.winner is not None:
            if action.state.winner == -1:
                print('White Wins')
            else:
                print('Draw')
            break
