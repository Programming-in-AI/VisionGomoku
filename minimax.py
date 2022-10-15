import utils
import Menu
import pygame, sys
import math

class Minimax(object):
    def __init__(self, gomoku, state, depth) -> None:
        self.gomoku = gomoku
        self.state = state
        self.depth = depth
        self.best_move = None
    
    def get_best_move(self):
        return self.best_move
    
    # Minimax with alpha-beta pruning
    # Alpha is the best value that the maximizer currently can guarantee at that level or above
    # Beta is the best value that the minimizer currently can guarantee at that level or above
    def minimax(self, depth, isMaximizing, alpha = -math.inf, beta = math.inf):
        # Terminal node, return the static evaluation of the board
        if depth == 0:
            return self.evaluate(self.state, self.state)

        # Build up minimax tree
        if isMaximizing:
            maxEval = -math.inf
            # Get all possible moves
            moves = self.get_all_moves()
            # Simulate each move
            for move in moves:
                self.make_move(move, self.state)
                # Recursive call with opponent's turn
                score = self.minimax(depth - 1, False, alpha, beta)
                self.undo_move(move)
                maxEval = max(score, maxEval)
                alpha = max(alpha, maxEval)
                # Prune the tree
                if beta <= alpha:
                    break
            return maxEval
        else:
            minEval = math.inf
            # Get all possible moves
            moves = self.get_all_moves()
            # Simulate each move
            for move in moves:
                self.make_move(move, self.get_opponent_state())
                # Recursive call with opponent's turn
                score = self.minimax(depth - 1, True, alpha, beta)
                self.undo_move(move)
                minEval = min(score, minEval)
                beta = min(beta, minEval)
                # Prune the tree
                if beta <= alpha:
                    break
            return minEval
    
    def calculate_next_move(self):
        self.minimax(self.depth, True)
        return self.best_move
    
    # Look for a move that instantly wins the game
    def lookForWinningMove(self, state):
        moves = self.get_all_moves()
        for move in moves:
            self.make_move(move, state)
            if self.gomoku.check_win(move, state):
                self.undo_move(move)
                return move
            self.undo_move(move)
        # No winning move found 
        return None

    # Calculate the score by evaluating the stones in a row
    def evaluateHorizontal(self, state, playerTurn):
        successive = 0
        blockedEnds = 2
        score = 0

        for i in range(utils.board_size):
            for j in range(utils.board_size):
                # Check if the selected playear has a stone in the current position
                if self.gomoku.board[i][j] == state:
                    successive += 1
                # If the cell is empty
                elif self.gomoku.board[i][j] == utils.empty:
                    # Check if there were any successive stones before this empty cell
                    if successive > 0:
                        # successive set is not blocked by the opponent, decrement the blocks
                        blockedEnds -= 1
                        # Get successive score
                        score += self.getSuccessiveScore(successive, blockedEnds, playerTurn)
                        # Reset successive and blocks
                        successive = 0
                        # Current cell is empty, next successive set will have at most one blocked end
                        blockedEnds = 1
                    else:
                        # Current cell is empty, next successive set will have at most one blocked end
                        blockedEnds = 1
                # If the cell is occupied by the opponent
                elif successive > 0:
                    score += self.getSuccessiveScore(successive, blockedEnds, playerTurn)
                    # Reset successive stone count
                    successive = 0
                    # Current cell is occupied by the opponent, next successive set will have two blocked ends
                    blockedEnds = 2
                else:
                    # Current cell is occupied by the opponent, next successive set will have two blocked ends
                    blockedEnds = 2
                # End of row, check if there were any successive stones before we reached the right border
                if successive > 0:
                    score += self.getSuccessiveScore(successive, blockedEnds, playerTurn)
                # Reset successive stone count
                successive = 0
                # Reset block count
                blockedEnds = 2
        return score
    
    # Calculate the score by evaluating the stones in a column
    def evaluateVertical(self, state, playerTurn):
        successive = 0
        blockedEnds = 2
        score = 0

        for i in range(utils.board_size):
            for j in range(utils.board_size):
                # Check if the selected player has a stone in the current position
                if self.gomoku.board[j][i] == state:
                    successive += 1
                # If the cell is empty
                elif self.gomoku.board[j][i] == utils.empty:
                    # Check if there were any successive stones before this empty cell
                    if successive > 0:
                        # successive set is not blocked by the opponent, decrement the blocks
                        blockedEnds -= 1
                        # Get successive score
                        score += self.getSuccessiveScore(successive, blockedEnds, playerTurn)
                        # Reset successive and blocks
                        successive = 0
                        # Current cell is empty, next successive set will have at most one blocked end
                        blockedEnds = 1
                    else:
                        # Current cell is empty, next successive set will have at most one blocked end
                        blockedEnds = 1
                # If the cell is occupied by the opponent
                elif successive > 0:
                    score += self.getSuccessiveScore(successive, blockedEnds, playerTurn)
                    # Reset successive stone count
                    successive = 0
                    # Current cell is occupied by the opponent, next successive set will have two blocked ends
                    blockedEnds = 2
                else:
                    # Current cell is occupied by the opponent, next successive set will have two blocked ends
                    blockedEnds = 2
                # End of column, check if there were any successive stones before we reached the bottom border
                if successive > 0:
                    score += self.getSuccessiveScore(successive, blockedEnds, playerTurn)
                # Reset successive stone count
                successive = 0
                # Reset block count
                blockedEnds = 2
        return score

    # Calculate the score by evaluating the stones in a diagonal
    def evaluateDiagonal(self, state, playerTurn):
        successive = 0
        blockedEnds = 2
        score = 0

        # Check the diagonals from top left to bottom right
        for k in range(2 * utils.board_size - 1):
            for i in range(utils.board_size):
                j = k - i
                if j < 0 or j >= utils.board_size:
                    continue
                # Check if the selected playear has a stone in the current position
                if self.gomoku.board[i][j] == state:
                    successive += 1
                # If the cell is empty
                elif self.gomoku.board[i][j] == utils.empty:
                    # Check if there were any successive stones before this empty cell
                    if successive > 0:
                        # successive set is not blocked by the opponent, decrement the blocks
                        blockedEnds -= 1
                        # Get successive score
                        score += self.getSuccessiveScore(successive, blockedEnds, playerTurn)
                        # Reset successive and blocks
                        successive = 0
                        # Current cell is empty, next successive set will have at most one blocked end
                        blockedEnds = 1
                    else:
                        # Current cell is empty, next successive set will have at most one blocked end
                        blockedEnds = 1
                # If the cell is occupied by the opponent
                elif successive > 0:
                    score += self.getSuccessiveScore(successive, blockedEnds, playerTurn)
                    # Reset successive stone count
                    successive = 0
                    # Current cell is occupied by the opponent, next successive set will have two blocked ends
                    blockedEnds = 2
                else:
                    # Current cell is occupied by the opponent, next successive set will have two blocked ends
                    blockedEnds = 2
                # End of diagonal, check if there were any successive stones before we reached the bottom border
                if successive > 0:
                    score += self.getSuccessiveScore(successive, blockedEnds, playerTurn)
                successive = 0
                blockedEnds = 2
        # Check the diagonals from bottom left to top right
        for k in range(2 * utils.board_size - 1):
            for i in range(utils.board_size):
                j = k - i
                if j < 0 or j >= utils.board_size:
                    continue
                # Check if the selected playear has a stone in the current position
                if self.gomoku.board[utils.board_size - i - 1][j] == state:
                    successive += 1
                # If the cell is empty
                elif self.gomoku.board[utils.board_size - i - 1][j] == utils.empty:
                    # Check if there were any successive stones before this empty cell
                    if successive > 0:
                        # successive set is not blocked by the opponent, decrement the blocks
                        blockedEnds -= 1
                        # Get successive score
                        score += self.getSuccessiveScore(successive, blockedEnds, playerTurn)
                        # Reset successive and blocks
                        successive = 0
                        # Current cell is empty, next successive set will have at most one blocked end
                        blockedEnds = 1
                    else:
                        # Current cell is empty, next successive set will have at most one blocked end
                        blockedEnds = 1
                # If the cell is occupied by the opponent
                elif successive > 0:
                    score += self.getSuccessiveScore(successive, blockedEnds, playerTurn)
                    # Reset successive stone count
                    successive = 0
                    # Current cell is occupied by the opponent, next successive set will have two blocked ends
                    blockedEnds = 2
                else:
                    # Current cell is occupied by the opponent, next successive set will have two blocked ends
                    blockedEnds = 2
                # End of diagonal, check if there were any successive stones before we reached the top border
                if successive > 0:
                    score += self.getSuccessiveScore(successive, blockedEnds, playerTurn)
                successive = 0
                blockedEnds = 2
        return score

    # Resturns score of a successive stone set
    def getSuccessiveScore(self, successive, blockedEnds, playerTurn):
        winScore = 1000000

        # If both ends are blocked, this set is worthless
        if blockedEnds == 2 and successive < 5:
            return 0
        
        if successive == 5:
            # Five successive stones is a win
            return winScore
        elif successive == 4:
            # Four successive stones in user's turn is a win
            if playerTurn:
                return winScore
            else: # Opponent's turn
                if blockedEnds == 0:
                    # Four successive stones with no blocked ends is a win
                    return winScore / 4
                else:
                    # If only a single end is blocked, four successive stones forces the opponent to block the other end
                    # This is a good state, therefore give a high score
                    return 200
        elif successive == 3:
            if blockedEnds == 0:
                if playerTurn:
                    # If it's Player 0's turn, a win is guaranteed in the next two turns
                    # Since the opponent may win in the next turn, give it a lower score
                    return 50000
                else:
                    # The opponent is forced to block one end
                    return 200
            else:
                if playerTurn:
                    return 10
                else:
                    return 5
        elif successive == 2:
            if blockedEnds == 0:
                if playerTurn:
                    return 7
                else:
                    return 3
        elif successive == 1:
            return 1

    def evaluate(self, state, playerTurn):
        return self.evaluateHorizontal(state, playerTurn) + \
            self.evaluateVertical(state, playerTurn) + \
            self.evaluateDiagonal(state, playerTurn)        


    def get_all_moves(self):
        moves = []
        for i in range(utils.board_size):
            for j in range(utils.board_size):
                if self.gomoku.board[i][j] == utils.empty:
                    moves.append((i, j))
        return moves

    def make_move(self, move, state):
        i, j = move
        self.gomoku.board[i][j] = state

    def undo_move(self, move):
        i, j = move
        self.gomoku.board[i][j] = utils.empty
    
    def get_opponent_state(self):
        if self.state == utils.black_stone:
            return utils.white_stone
        else:
            return utils.black_stone
        
if __name__ == '__main__':
    # initialize
    pygame.init()
    surface = pygame.display.set_mode((utils.window_width, utils.window_height))
    pygame.display.set_caption("Omok game")

    # make instance
    omok = utils.Omok(surface)
    menu = Menu.Menu(surface)

    while True:
        omok.run_game(omok, menu)
        if menu.game_over(omok):
            pygame.quit()
            break
