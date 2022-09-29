import utils
import pygame


class Menu(object):
    def __init__(self, surface):
        print('hi')
        self.surface = surface

    @staticmethod
    def game_over(omok):
        if omok.white_win_time == 2 or omok.black_win_time == 2:
            return True
        return False

    def terminate(self):
        return

    def check_rect(self, pos, omok):
        return False

    def show_msg(self, stone, final_victory = False):
        if stone == utils.black_stone:
            font1 = pygame.font.SysFont('Times New Roman Italic.ttf', 40)
            if final_victory:
                img = font1.render('Final Winner is Black', True, (255, 0, 0))
                self.surface.blit(img, (105, 210))
            else:
                img = font1.render('Black wins', True, (255,0,0))
                self.surface.blit(img,(160,210))

        elif stone == utils.white_stone:
            font1 = pygame.font.SysFont('Times New Roman Italic.ttf', 40)
            if final_victory:
                img = font1.render('Final Winner is White', True, (255, 0, 0))
                self.surface.blit(img, (105, 210))
            else:
                img = font1.render('White wins', True, (255, 0, 0))
                self.surface.blit(img, (160, 210))

