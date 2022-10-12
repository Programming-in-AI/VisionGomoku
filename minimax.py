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
        self.minimax(self.depth, True)
        return self.best_move
    
    # 
    def minimax(self, depth, is_maximizing):
        if depth == 0 or self.gomoku.is_gameover:
            return self.evaluate()
        
        if is_maximizing:
            best_score = -math.inf
            for move in self.get_all_moves():
                self.make_move(move, self.state)
                score = self.minimax(depth - 1, False)
                self.undo_move(move)
                if score > best_score:
                    best_score = score
                    if depth == self.depth:
                        self.best_move = move
            return best_score
        else:
            best_score = math.inf
            for move in self.get_all_moves():
                self.make_move(move, self.get_opponent_state())
                score = self.minimax(depth - 1, True)
                self.undo_move(move)
                if score < best_score:
                    best_score = score
            return best_score
    
    def lookForWinningMove(self, state):
        for i in range(utils.board_size):
            for j in range(utils.board_size):
                if self.gomoku.board[i][j] == utils.empty:
                    self.gomoku.board[i][j] = state
                    if self.gomoku.is_gameover:
                        self.gomoku.board[i][j] = utils.empty
                        return (i, j)
                    self.gomoku.board[i][j] = utils.empty
        return None

    # Calculate the score by evaluating the stones in a row
    def evaluateHorizontal(self, state, playerTurn):
        consecutive = 0
        blockedEnds = 2
        score = 0

        for i in range(utils.board_size):
            for j in range(utils.board_size):
                # Check if the selected playear has a stone in the current position
                if self.gomoku.board[i][j] == state:
                    consecutive += 1
                # If the cell is empty
                elif self.gomoku.board[i][j] == utils.empty:
                    # Check if there were any consecutive stones before this empty cell
                    if consecutive > 0:
                        # Consecutive set is not blocked by the opponent, decrement the blocks
                        blockedEnds -= 1
                        # Get consecutive score
                        score += self.getConsecutiveScore(consecutive, blockedEnds, playerTurn)
                        # Reset consecutive and blocks
                        consecutive = 0
                        # Current cell is empty, next consecutive set will have at most one blocked end
                        blockedEnds = 1
                    else:
                        # Current cell is empty, next consecutive set will have at most one blocked end
                        blockedEnds = 1
                # If the cell is occupied by the opponent
                elif consecutive > 0:
                    score += self.getConsecutiveScore(consecutive, blockedEnds, playerTurn)
                    # Reset consecutive stone count
                    consecutive = 0
                    # Current cell is occupied by the opponent, next consecutive set will have two blocked ends
                    blockedEnds = 2
                else:
                    # Current cell is occupied by the opponent, next consecutive set will have two blocked ends
                    blockedEnds = 2
                # End of row, check if there were any consecutive stones before we reached the right border
                if consecutive > 0:
                    score += self.getConsecutiveScore(consecutive, blockedEnds, playerTurn)
                # Reset consecutive stone count
                consecutive = 0
                # Reset block count
                blockedEnds = 2
        return score
    
    # Calculate the score by evaluating the stones in a column
    def evaluateVertical(self, state, playerTurn):
        consecutive = 0
        blockedEnds = 2
        score = 0

        for i in range(utils.board_size):
            for j in range(utils.board_size):
                # Check if the selected playear has a stone in the current position
                if self.gomoku.board[j][i] == state:
                    consecutive += 1
                # If the cell is empty
                elif self.gomoku.board[j][i] == utils.empty:
                    # Check if there were any consecutive stones before this empty cell
                    if consecutive > 0:
                        # Consecutive set is not blocked by the opponent, decrement the blocks
                        blockedEnds -= 1
                        # Get consecutive score
                        score += self.getConsecutiveScore(consecutive, blockedEnds, playerTurn)
                        # Reset consecutive and blocks
                        consecutive = 0
                        # Current cell is empty, next consecutive set will have at most one blocked end
                        blockedEnds = 1
                    else:
                        # Current cell is empty, next consecutive set will have at most one blocked end
                        blockedEnds = 1
                # If the cell is occupied by the opponent
                elif consecutive > 0:
                    score += self.getConsecutiveScore(consecutive, blockedEnds, playerTurn)
                    # Reset consecutive stone count
                    consecutive = 0
                    # Current cell is occupied by the opponent, next consecutive set will have two blocked ends
                    blockedEnds = 2
                else:
                    # Current cell is occupied by the opponent, next consecutive set will have two blocked ends
                    blockedEnds = 2
                # End of column, check if there were any consecutive stones before we reached the bottom border
                if consecutive > 0:
                    score += self.getConsecutiveScore(consecutive, blockedEnds, playerTurn)
                # Reset consecutive stone count
                consecutive = 0
                # Reset block count
                blockedEnds = 2
        return score

    # Calculate the score by evaluating the stones in a diagonal
    def evaluateDiagonal(self, state, playerTurn):
        consecutive = 0
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
                    consecutive += 1
                # If the cell is empty
                elif self.gomoku.board[i][j] == utils.empty:
                    # Check if there were any consecutive stones before this empty cell
                    if consecutive > 0:
                        # Consecutive set is not blocked by the opponent, decrement the blocks
                        blockedEnds -= 1
                        # Get consecutive score
                        score += self.getConsecutiveScore(consecutive, blockedEnds, playerTurn)
                        # Reset consecutive and blocks
                        consecutive = 0
                        # Current cell is empty, next consecutive set will have at most one blocked end
                        blockedEnds = 1
                    else:
                        # Current cell is empty, next consecutive set will have at most one blocked end
                        blockedEnds = 1
                # If the cell is occupied by the opponent
                elif consecutive > 0:
                    score += self.getConsecutiveScore(consecutive, blockedEnds, playerTurn)
                    # Reset consecutive stone count
                    consecutive = 0
                    # Current cell is occupied by the opponent, next consecutive set will have two blocked ends
                    blockedEnds = 2
                else:
                    # Current cell is occupied by the opponent, next consecutive set will have two blocked ends
                    blockedEnds = 2
                # End of diagonal, check if there were any consecutive stones before we reached the bottom border
                if consecutive > 0:
                    score += self.getConsecutiveScore(consecutive, blockedEnds, playerTurn)
                # Reset consecutive stone count
                consecutive = 0
                # Reset block count
                blockedEnds = 2
        return score

    # Resturns score of a consecutive stone set
    def getConsecutiveScore(self, consecutive, blockedEnds, playerTurn):
        winScore = 1000000

        # If both ends are blocked, this set is worthless
        if blockedEnds == 2 and consecutive < 5:
            return 0
        
        if consecutive == 5:
            # Five consecutive stones is a win
            return winScore
        elif consecutive == 4:
            # Four consecutive stones in user's turn is a win
            if playerTurn:
                return winScore
            else: # Opponent's turn
                if blockedEnds == 0:
                    # Four consecutive stones with no blocked ends is a win
                    return winScore / 4
                else:
                    # If only a single end is blocked, four consecutive stones forces the opponent to block the other end
                    # This is a good state, therefore give a high score
                    return 200
        elif consecutive == 3:
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
        elif consecutive == 2:
            if blockedEnds == 0:
                if playerTurn:
                    return 7
                else:
                    return 3
        elif consecutive == 1:
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
