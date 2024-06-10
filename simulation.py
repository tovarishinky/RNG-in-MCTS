import numpy as np
from collections import defaultdict


class MonteCarloTreeSearchNode:
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

    '''
    Get all possible legal actions that can be used.
    '''
    def untried_actions(self):
        self._untried_actions = self.state.get_legal_actions()
        return self._untried_actions

    def q(self):
        wins = self._results[1]
        loses = self._results[-1]
        return wins - loses

    def n(self):
        return self._number_of_visits

    '''
    Expand the MCTS tree.
    '''
    def expand(self):
        action = self._untried_actions.pop()
        next_state = self.state.move(action)
        child_node = MonteCarloTreeSearchNode(next_state, parent=self, parent_action=action)
        self.children.append(child_node)
        return child_node

    '''
    Check if the game ends in this node.
    '''
    def is_terminal_node(self):
        return self.state.is_game_over()

    '''
    Rollout actions to simulate the game.
    '''
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

    def get_heuristic_score(self):
        # Evaluate the desirability of the current game state using a simple heuristic
        score = 0
        r = self.parent_action[0]
        c = self.parent_action[1]
        board = self.parent.state.board
        # Check rows, columns, and diagonals for consecutive stones
        score += self.evaluate_line(board, r, c, 1, 0)  # Check row right
        score += self.evaluate_line(board, r, c, 0, 1)  # Check column up
        score += self.evaluate_line(board, r, c, 1, 1)  # Check diagonal top right
        score += self.evaluate_line(board, r, c, 1, -1)  # Check diagonal bottom right
        score += self.evaluate_line(board, r, c, -1, 0)  # Check row left
        score += self.evaluate_line(board, r, c, 0, -1)  # Check column down
        score += self.evaluate_line(board, r, c, -1, -1)  # Check diagonal bottom left
        score += self.evaluate_line(board, r, c, -1, 1)  # Check diagonal top left

        return score

    def evaluate_line(self, board, row, col, dr, dc):
        # Evaluate a single line for consecutive stones
        consecutive_count_player = 0  # Player's consecutive nodes
        consecutive_count_opp = 0  # Opponent's consecutive nodes
        empty_count = 0

        for i in range(5):
            r, c = row + i * dr, col + i * dc

            if not (0 <= r < self.state.board_size and 0 <= c < self.state.board_size):
                break

            if board[r, c] == self.state.colour:
                consecutive_count_player += 1
            elif board[r, c] == 0:
                empty_count += 1
            else:
                consecutive_count_opp += 1

        if consecutive_count_player == 4 and empty_count == 1:
            return 100  # One move away from winning
        elif consecutive_count_opp == 4 and empty_count == 1:
            return 50  # One move away from winning
        elif consecutive_count_player == 3 or consecutive_count_opp == 3:
            if self.check_for_winning_move(row, col, dr, dc, False):
                return 100  # One move away from winning
            elif empty_count == 2:
                return 2  # Two moves away from winning
            else:
                return 0
        elif consecutive_count_player == 2 or consecutive_count_opp == 2:
            if self.check_for_winning_move(row, col, dr, dc, True):
                return 100  # One move away from winning
            else:
                return 0
        else:
            return 0

    def check_for_winning_move(self, row, col, dr, dc, check_middle):
        if check_middle:  # check for OO 00
            for i in range(2):
                r, c = row + i * dr, col + i * dc
                if not (0 <= r < self.state.board_size and 0 <= c < self.state.board_size):
                    return False
                if self.state.board[r, c] != self.state.colour:
                    return False
            for i in range(2):
                r, c = row + i * dr * -1, col + i * dc * -1
                if not (0 <= r < self.state.board_size and 0 <= c < self.state.board_size):
                    return False
                if self.state.board[r, c] != self.state.colour:
                    return False
        else:  # check for 0 000 or 000 0
            for i in range(3):
                r, c = row + i * dr, col + i * dc
                if not (0 <= r < self.state.board_size and 0 <= c < self.state.board_size):
                    return False
                if self.state.board[r, c] != self.state.colour:
                    return False
            r, c = row + -1 * dr, col + -1 * dc
            if not (0 <= r < self.state.board_size and 0 <= c < self.state.board_size):
                return False
            if self.state.board[r, c] != self.state.colour:
                return False
        return True

    '''
    Get the best child node using either the UCT algorithm
    or the combination of the UCT algorithm with basic heuristic.
    '''
    def best_child(self, c_param=np.sqrt(2)):
        choices_weights = []
        for c in self.children:
            exploitation = c.q() / c.n()
            # print(exploitation)
            exploration = c_param * np.sqrt((2 * np.log(self.n()) / c.n()))
            # print(exploration)
            heuristic = c.get_heuristic_score()
            # print(heuristic)
            # choices_weights.append(exploration+exploitation)
            choices_weights.append(exploration+exploitation+heuristic)

        # choices_weights = [(c.q() / c.n()) + c_param * np.sqrt((2 * np.log(self.n()) / c.n())) for c in self.children]
        # print(choices_weights)
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

        return self.best_child(c_param=np.sqrt(2))


class State:
    def __init__(self, board, colour, turn):
        self.board = board  # current layout of the state
        self.colour = colour  # the colour of the player, 1=Black, -1=White
        self.turn = turn  # whether it is the player's turn
        self.winner = None  # None until a winner is decided, then it is 1=Black, 0=Tie, -1=White
        self.board_size = 9  # Size of the square board
        return

    '''
    Go through all spaces on the board, if there is an empty space (0),
    add it to the list of legal actions
    '''
    def get_legal_actions(self):
        legal_actions = []
        for i, row in enumerate(self.board):
            for j, column in enumerate(row):
                if column == 0:
                    legal_actions.append([i, j])
        return legal_actions

    '''
    Check the surroundings using a 5x5 grid.
    If the sum of any horizontal/vertical lines or if either of the diagonals
    equal 5 or -5, the game is won by either the Black player (when the value is 5), 
    or the White player (when the value is -5) and the function returns True.
    If no winner was found, go through the rows and check if there are any empty spots left.
    If there aren't, the game ends in a Tie and the function returns True.
    If there are, the game is not over and the function returns False.
    In the case of the game ending, the function saves the result of the match for later use.
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
    Check the value of the result of the game.
    If it is a 0, the game is a Tie.
    If the value of the winning player matches the player's colour, it is a Win.
    If the value of the winning player is the opposite of the player's colour, it is a Loss.
    '''

    def game_result(self):
        if self.winner == 0:
            return 0
        if self.winner == self.colour:
            return 1
        else:
            return -1

    '''
    Create a copy of the state's board. 
    If it is the player's turn, change the spot on the new board decided by the action
    to the player's colour, where 1=Black and -1=White.
    If it is not the player's turn, change the spot to the opposite colour instead.
    Return a new state using the new board, the player's colour and the opposite of the current
    turn value. 
    '''

    def move(self, action):
        new_board = self.board.copy()
        if self.turn:
            new_board[(action[0], action[1])] = self.colour
        else:
            new_board[(action[0], action[1])] = self.colour*-1
        return State(new_board, self.colour, not self.turn)


'''
W.I.P. function of main code

Run 100 simulations of Go played between two players using MCTS.
At the end of each game, print the game's result.
Code for printing out the entire matches is commented out.
'''

white_wins = 0
black_wins = 0
draws = 0
for i in range(100):
    current_board = np.zeros((9,9),dtype=int)
    turn = 0
    while True:
        # print(current_board)
        # print('Turn: {}'.format(turn))
        # print('\n\n')
        black_state = State(current_board, 1, True)
        black = MonteCarloTreeSearchNode(black_state)
        action = black.best_action()
        current_board = action.state.board
        turn+=1
        # print(current_board)
        # print('Turn: {}'.format(turn))
        # print('\n\n')
        if action.state.winner is not None:
            if action.state.winner == 1:
                print('Black Wins')
                black_wins += 1
            else:
                print('Draw')
                draws += 1
            break
        white_state = State(current_board, -1, True)
        white = MonteCarloTreeSearchNode(white_state)
        action = white.best_action()
        current_board = action.state.board
        turn += 1
        # print(current_board)
        # print('Turn: {}'.format(turn))
        # print('\n\n')
        if action.state.winner is not None:
            if action.state.winner == -1:
                print('White Wins')
                white_wins += 1
            else:
                print('Draw')
                draws += 1
            break
print(f"Black won {black_wins} times.")
print(f"White won {white_wins} times.")
print(f"There were {draws} draws.")

# current_board = np.zeros((9, 9), dtype=int)
# turn = 0
# print(current_board)
# print('Turn: {}'.format(turn))
# print('\n\n')
# while True:
#
#     black_state = State(current_board, 1, True)
#     black = MonteCarloTreeSearchNode(black_state)
#     action = black.best_action()
#     current_board = action.state.board
#     turn += 1
#
#     print(current_board)
#     print('Turn: {}'.format(turn))
#     print('\n\n')
#
#     if action.state.winner is not None:
#         if action.state.winner == 1:
#             print('Black Wins')
#         else:
#             print('Draw')
#         break
#
#     white_state = State(current_board, -1, True)
#     white = MonteCarloTreeSearchNode(white_state)
#     action = white.best_action()
#     current_board = action.state.board
#     turn += 1
#
#     print(current_board)
#     print('Turn: {}'.format(turn))
#     print('\n\n')
#
#     if action.state.winner is not None:
#         if action.state.winner == -1:
#             print('White Wins')
#         else:
#             print('Draw')
#         break
