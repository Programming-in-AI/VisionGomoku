import utils

class Rule(object):
    def __init__(self, board):
        self.board = board

    def is_invalid(self, x, y):
        return (x < 0 or x >= utils.board_size or y < 0 or y >= utils.board_size)

    # black moves first and ckeck that he puts stone in the middle of the board
    def is_valid_first_move(self, x, y):
        return x == utils.board_size // 2 and y == utils.board_size // 2

    # black's second move must be outside the center 5x5 square
    def is_valid_second_move(self, x, y):
        return (
            x >= utils.board_size // 2 - 2 and x <= utils.board_size // 2 + 2 
            and y >= utils.board_size // 2 - 2 and y <= utils.board_size // 2 + 2)

    def is_gameover(self, x, y, stone):
        x1, y1 = x, y
        list_dx = [-1, 1, -1, 1, 0, 0, 1, -1]
        list_dy = [0, 0, -1, 1, -1, 1, -1, 1]
        for i in range(0, len(list_dx), 2):
            cnt = 1
            for j in range(i, i + 2):
                dx, dy = list_dx[j], list_dy[j]
                x, y = x1, y1
                while True:
                    x, y = x + dx, y + dy
                    if self.is_invalid(x, y) or self.board[y][x] != stone:
                        break;
                    else:
                        cnt += 1
            if cnt >= 5:
                return cnt
        return cnt