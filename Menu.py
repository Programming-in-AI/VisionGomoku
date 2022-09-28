import utils
import pygame


class Menu(object):
    def __init__(self, surface):
        print('hi')
        self.surface = surface

    def is_continue(self, omok):
        return

    def terminate(self):
        return

    def check_rect(self, pos, omok):
        return False

    def show_msg(self, stone):
        if stone == utils.black_stone:
            font1 = pygame.font.SysFont('Times New Roman Italic.ttf', 50)
            img = font1.render('Game over! Black wins', True, (255,0,0))
            self.surface.blit(img,(52,210))
        elif stone == utils.white_stone:
            font1 = pygame.font.SysFont('Times New Roman Italic.ttf', 50)
            img = font1.render('Game over! White wins', True, (255, 0, 0))
            self.surface.blit(img, (52, 210))


