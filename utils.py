import pygame
import Menu
import Rule
import random

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

fps = 60000000
fps_clock = pygame.time.Clock()


class Omok(object):
    def __init__(self, surface):
        self.board = [[0 for i in range(board_size)] for j in range(board_size)]
        self.menu = Menu.Menu(surface)
        self.rule = Rule.Rule(self.board)
        self.surface = surface
        self.pixel_coords = []
        self.set_coords()
        self.set_image_font()
        self.is_show = True
        self.black_win_time = 0
        self.white_win_time = 0


    def init_game(self):
        self.turn  = black_stone
        self.draw_board()
        self.menu.show_msg(empty)
        self.init_board()
        self.coords = []
        self.id = 1
        self.is_gameover = False

    @staticmethod
    def run_game(omok, menu):
        # randomly select who is black
        you_first = random.randint(1,2)  # if you first 1, black is you, vice versa


        omok.init_game()
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT: # close window
                    menu.terminate()
                    pygame.quit()

                elif event.type == pygame.MOUSEBUTTONUP:  # mouse clicked
                    print(f'coord: {event.pos}')
                    if not omok.check_board(event.pos): # 1. did it click board? 2. check that the game is over or not
                        omok.init_game()  # if anybody wins, initialize game

            if omok.is_gameover:  # if game is over break while loop
                break

            pygame.display.update()
            fps_clock.tick(fps)

    def set_image_font(self):
        white_img = pygame.image.load('./image/white.png')
        self.white_img = pygame.transform.scale(white_img, (grid_size, grid_size))
        black_img = pygame.image.load('./image/black.png')
        self.black_img = pygame.transform.scale(black_img, (grid_size, grid_size))
        self.board_img = pygame.image.load('./image/table.jpg')
        self.font = pygame.font.Font("freesansbold.ttf", 14)

    def init_board(self):
        for y in range(board_size):
            for x in range(board_size):
                self.board[y][x] = 0

    def draw_board(self):
        self.surface.blit(self.board_img, (0, 0))

    def draw_image(self, img_index, x, y):
        img = [self.black_img, self.white_img]
        self.surface.blit(img[img_index], (x-grid_size/2, y-grid_size/2))

    def hide_numbers(self):
        for i in range(len(self.coords)):
            x, y = self.coords[i]
            self.draw_image(i % 2, x, y)

    def set_coords(self):
        for y in range(board_size):
            for x in range(board_size):
                self.pixel_coords.append((x * grid_size + 25, y * grid_size + 25))

    def get_coord(self, pos):
        for coord in self.pixel_coords:
            x, y = coord
            e = 10 # error value (becuz board is made by the hand, we need to compensate the pixel value)
            rect = pygame.Rect(x-e, y-e, grid_size, grid_size)
            if rect.collidepoint(pos):  # if included in the rectangular then return coord
                return coord
        return None

    @staticmethod
    def get_point(coord):
        x, y = coord
        x = (x - 25) // grid_size
        y = (y - 25) // grid_size
        return x, y

    def check_board(self, pos):
        coord = self.get_coord(pos)
        if not coord:
            return False
        x, y = self.get_point(coord)
        if self.board[y][x] != empty: # stone already exists => return True
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
        elif 5 <= self.rule.is_gameover(x, y, stone):  # checking how many times win
            if stone == 1:  # black
                self.black_win_time += 1
            else:  # white
                self.white_win_time += 1
            self.show_winner_msg(stone)
            return True
        return False

    def show_winner_msg(self, stone):
        if self.black_win_time < 2 and self.white_win_time < 2:
            self.menu.show_msg(stone)
            pygame.display.update()
            pygame.time.delay(2000)
        else:
            self.menu.show_msg(stone, True)
            pygame.display.update()
            pygame.time.delay(2000)

    def draw_stone(self, coord, stone, increase):
        x, y = self.get_point(coord)
        self.board[y][x] = stone
        self.hide_numbers()
        self.id += increase
        self.turn = 3 - self.turn

