import pygame
import Menu
import Rule
import random
from Opago.SimpleNet import *
import torch
import numpy as np

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
        self.computer_win_time = 0
        self.human_win_time = 0
        self.quit = False

    def init_game(self):
        self.turn = black_stone
        self.draw_board()
        self.menu.show_msg(empty)
        self.init_board()
        self.coords = []
        self.id = 1
        self.is_gameover = False
        self.start = True
        # randomly select who is black
        self.list = ['human', 'computer']
        self.who_is_black = random.choice(self.list)  # 1 = human, 2 = computer / black means first also

    def run_game(self, omok, menu):
        omok.init_game()
        while True:
            # default action
            if self.who_is_black == 'computer' and self.start:  # is it first time?
                omok.check_board((237, 237), self.who_is_black, computer_input = None)
                self.start = False


            # computer action
            if (self.turn  == black_stone and self.who_is_black == 'computer') or (self.turn  == white_stone and self.who_is_black == 'human'):
                # using AI model
                path = './Opago/models/model_27.pth'
                net = SimpleNet()
                model = torch.load(path)
                net.load_state_dict(model, strict=False)
                with torch.no_grad():
                    net.eval()
                result = net(torch.unsqueeze(torch.unsqueeze(torch.from_numpy(np.array(self.board)).float(), 0),0))
                computer_input= np.unravel_index(torch.argmax(result[0]), (15, 15))
                omok.check_board(None, self.who_is_black, computer_input)

            # human action
            for event in pygame.event.get():
                if event.type == pygame.QUIT: # close window
                    pygame.quit()
                    self.quit = True
                    return
                elif event.type == pygame.MOUSEBUTTONUP:  # mouse clicked
                    # print(f'coord: {event.pos}')
                    if not omok.check_board(event.pos, self.who_is_black, computer_input = None):  # 1. did it click board? 2. check that the game is over or not
                        omok.init_game()  # if anybody wins, initialize game

            pygame.display.update()
            fps_clock.tick(fps)

            if omok.is_gameover:  # if game is over break while loop
                return

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

    def drawing_img(self):
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

    def check_board(self, pos, who_is_black, computer_input):
        # human action
        if pos is not None:
            coord = self.get_coord(pos)
            if not coord: # but if clicked strange spot
                return False
            x, y = self.get_point(coord)

            # special rule
            if self.id == 3 and self.is_3rd_black_in_middle(x, y, 'human'):  # nullity the action
                return True

        # computer action
        if pos == None:
            coord = (computer_input[1]*grid_size+25, computer_input[0]*grid_size+25)
            x, y = self.get_point(coord)

            # special rule
            if self.id == 3:
                x,y = self.is_3rd_black_in_middle(x, y, 'computer')
                coord = (x * grid_size + 25, y * grid_size + 25)

        # update board information
        if self.board[y][x] != empty:
            return True
        self.coords.append(coord)

        # draw stone
        self.draw_stone(coord, self.turn, 1, who_is_black)

        # check whether game is over
        if self.check_gameover(coord, 3 - self.turn, who_is_black):
            self.is_gameover = True

        return True

    def check_gameover(self, coord, turn, who_is_black):
        x, y = self.get_point(coord)
        if self.id > board_size * board_size:
            self.show_winner_msg(turn)
            return True
        elif 5 <= self.rule.is_gameover(x, y, turn, who_is_black):  # checking how many times win
            if (turn == black_stone and who_is_black == 'computer') or (turn == white_stone and who_is_black == 'human'):  # black
                self.computer_win_time += 1
            elif (turn == white_stone and who_is_black == 'computer') or (turn == black_stone and who_is_black == 'human'):  # white
                self.human_win_time += 1
            self.show_winner_msg(turn)
            return True
        return False

    def show_winner_msg(self, stone):
        if self.computer_win_time < 2 and self.human_win_time < 2:
            self.menu.show_msg(stone, final_victory=None)
            pygame.display.update()
            pygame.time.delay(2000)
        else:
            if self.computer_win_time == 2 :
                self.menu.show_msg(stone, final_victory='Computer')
                pygame.display.update()
                pygame.time.delay(2000)
            else :
                self.menu.show_msg(stone, final_victory='You')
                pygame.display.update()
                pygame.time.delay(2000)

    def draw_stone(self, coord, stone, increase, who_is_black):

        if stone == 1 and who_is_black == 'computer':  # computer is black(1) and it is black's turn
            x, y = self.get_point(coord)
            self.board[y][x] = 1

        elif stone == 2 and who_is_black == 'computer':  # computer is black(1) and it is white's turn
            x, y = self.get_point(coord)
            self.board[y][x] = -1

        elif stone == 1 and who_is_black == 'human':  # human is black(-1) and it is black's turn
            x, y = self.get_point(coord)
            self.board[y][x] = -1

        elif stone == 2 and who_is_black == 'human':  # human is black(-1) and it is white's turn
            x, y = self.get_point(coord)
            self.board[y][x] = 1

        self.drawing_img()
        self.id += increase
        self.turn = 3 - self.turn
        print('coord：', tuple(int((elem-25)/30) for elem in coord))

    def is_3rd_black_in_middle(self, x, y, turn):
        if turn == 'computer':
            if 5<=x<=9 and 5<=y<=9: # nullity the stone

                while not (x in [4,10]) and  not (y in [4,10]): # let computer place a stone at the nearest stone
                    x = random.choice([4,5,6,7,8,9,10])
                    y = random.choice([4,5,6,7,8,9,10])

                return x, y
            else : return x, y
        elif turn == 'human':
            if 5<=x<=9 and 5<=y<=9: # nullity the stone
                return True


