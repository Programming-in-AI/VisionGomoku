import pygame
import Menu
import Rule
import gomokuAI as AI

# default value
window_width = 470
window_height = 470
board_width = 470
board_size = 15
grid_size = 30
empty = 0
black_stone = 1
white_stone = 2
tie = 100

board_color1 = (153, 102, 000)
board_color2 = (153, 102, 51)
board_color3 = (204, 153, 000)
board_color4 = (204, 153, 51)
bg_color = (128, 128, 128)
black = (0, 0, 0)
blue = (0, 50, 255)
white = (255, 255, 255)
red = (255, 0, 0)
green = (0, 200, 0)

fps = 60
fps_clock = pygame.time.Clock()


class Gomoku(object):
    def __init__(self, surface):
        self.board = [[0 for i in range(board_size)] for j in range(board_size)]
        self.menu = Menu.Menu(surface)
        self.rule = Rule.Rule(self.board)
        self.surface = surface
        self.pixel_coords = []
        self.set_coords()
        self.set_image_font()
        self.is_show = True
        self.move_count = 0


    def init_game(self):
        self.turn  = black_stone
        self.draw_board()
        self.menu.show_msg(empty)
        self.init_board()
        self.coords = []
        self.redos = []
        self.id = 1
        self.is_gameover = False


    def run_game(self, omok, menu):
        omok.init_game()
        # black plays as AI
        playerBlack = AI.AI(omok.board, black_stone)
        while True:
            if self.turn == black_stone:
                if not self.is_gameover:
                    # playerBlack.play()
                    playerBlack.play()
                    self.turn = 3 - self.turn
            else:
                
                for event in pygame.event.get():
                    if event.type == pygame.QUIT: # close window
                        menu.terminate()
                        pygame.quit()
                    elif event.type == pygame.MOUSEBUTTONUP: # mouse clicked
                        print(f'coord : {event.pos}')
                        if not omok.check_board(event.pos): # did it click board?
                            if menu.check_rect(event.pos, omok):
                                omok.init_game()

            if omok.is_gameover:
                return

            pygame.display.update()
            fps_clock.tick(fps)


    def set_image_font(self):
        white_img = pygame.image.load('./image/white.png')
        self.white_img = pygame.transform.scale(white_img, (grid_size, grid_size))
        black_img = pygame.image.load('./image/black.png')
        self.black_img = pygame.transform.scale(black_img, (grid_size, grid_size))
        # self.last_w_img = pygame.image.load('./image/white_last.png')
        # self.last_b_img = pygame.image.load('./image/black_last.png')
        self.board_img = pygame.image.load('./image/table.jpg')
        self.font = pygame.font.Font("freesansbold.ttf", 14)

    def init_board(self):
        for y in range(board_size):
            for x in range(board_size):
                self.board[y][x] = 0
        # center = board_size // 2
        # center_coord = center * grid_size + 25
        # self.draw_stone((center_coord, center_coord), black_stone, 1)

    def draw_board(self):
        self.surface.blit(self.board_img, (0, 0))

    def draw_image(self, img_index, x, y):
        img = [self.black_img, self.white_img]
        self.surface.blit(img[img_index], (x-grid_size/2, y-grid_size/2))

    def show_number(self, x, y, stone, number):
        colors = [white, black, red, red]
        color = colors[stone]
        self.make_text(self.font, str(number), color, x + 15, y + 15, 'center')

    def hide_numbers(self):
        for i in range(len(self.coords)):
            x, y = self.coords[i]
            self.draw_image(i % 2, x, y)
        # if self.coords:
        #     x, y = self.coords[-1]
        #     self.draw_image(i % 2 + 2, x, y)

    def show_numbers(self):
        for i in range(len(self.coords)):
            x, y = self.coords[i]
            self.show_number(x, y, i % 2, i + 1)
        if self.coords:
            x, y = self.coords[-1]
            self.draw_image(i % 2, x, y)
            self.show_number(x, y, i % 2 + 2, i + 1)

    def set_coords(self):
        for y in range(board_size):
            for x in range(board_size):
                self.pixel_coords.append((x * grid_size + 25, y * grid_size + 25))

    def get_coord(self, pos):
        for coord in self.pixel_coords:
            x, y = coord
            e = 10 # error value (becuz board is made by hands we need to compensate the pixel value)
            rect = pygame.Rect(x-e, y-e, grid_size, grid_size)
            if rect.collidepoint(pos):
                return coord
        return None

    def get_point(self, coord):
        x, y = coord
        x = (x - 25) // grid_size
        y = (y - 25) // grid_size
        return x, y

    def check_board(self, pos):
        coord = self.get_coord(pos)
        if not coord:
            return False
        x, y = self.get_point(coord)
        if self.board[y][x] != empty: # if already stone exists => return True
            return True

        self.coords.append(coord)
        self.draw_stone(coord, self.turn, 1)
        if self.check_gameover(coord, 3 - self.turn):
            self.is_gameover = True

        return True

    def check_gameover(self, coord, stone):
        x, y = self.get_point(coord)
        if self.id > board_size * board_size:
            self.show_winner_msg(stone)
            return True
        elif 5 <= self.rule.is_gameover(x, y, stone):
            self.show_winner_msg(stone)
            return True
        return False

    def show_winner_msg(self, stone):
        for i in range(3):
            self.menu.show_msg(stone)
            pygame.display.update()
            pygame.time.delay(200)
            self.menu.show_msg(empty)
            pygame.display.update()
            pygame.time.delay(200)
        self.menu.show_msg(stone)

    def draw_stone(self, coord, stone, increase):
        x, y = self.get_point(coord)
        # if self.is_valid_move(x, y, self.move_count + 1, stone) or stone == white_stone or self.move_count > 3:
        self.board[y][x] = stone
        self.hide_numbers()
        #self.show_numbers()
        self.id += increase
        self.turn = 3 - self.turn
        self.move_count += 1
        print("Number of moves: ", self.move_count)
        print("ID: ", self.id)

    def is_valid_move(self, x, y, move_count, stone):
        center = board_size // 2
        if move_count == 1 and x == center and y == center and stone == black_stone:
            return True
        elif move_count == 3 and (center - 2 <= x <= center + 2) and (center - 2 <= y <= center + 2) and stone == black_stone:
            return True
        else:
            return False