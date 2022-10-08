import utils
import Menu
import pygame, sys
import math

class gomokuAI(object):
    def __init__(self, gomoku, state, depth) -> None:
        self.gomoku = gomoku
        self.state = state
        self.depth = depth
        self.best_move = None
    
    def get_best_move(self):
        self.minimax(self.depth, True)
        return self.best_move
    
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
    
    def evaluate(self):
        if self.gomoku.is_gameover:
            if self.gomoku.winner == self.state:
                return math.inf
            else:
                return -math.inf
        return 0
    
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
