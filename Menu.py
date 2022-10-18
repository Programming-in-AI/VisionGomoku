import utils
import pygame


class Menu(object):
    def __init__(self, surface):
        self.surface = surface

    @staticmethod
    def game_over(omok):
        if omok.human_win_time == 2 or omok.computer_win_time == 2:
            return True
        return False

    def show_msg(self, stone, final_victory=None):

        if stone == utils.black_stone:
            font1 = pygame.font.SysFont('Times New Roman Italic.ttf', 40)
            if final_victory is not None:
                img = font1.render('Final Winner is '+final_victory, True, (255, 0, 0))
                self.surface.blit(img, (70, 210))

            else:
                img = font1.render('Black wins', True, (255, 0, 0))
                self.surface.blit(img, (160, 210))

        elif stone == utils.white_stone:
            font1 = pygame.font.SysFont('Times New Roman Italic.ttf', 40)
            if final_victory is not None:
                img = font1.render('Final Winner is '+final_victory, True, (255, 0, 0))
                self.surface.blit(img, (70, 210))

            else:
                img = font1.render('White wins', True, (255, 0, 0))
                self.surface.blit(img, (160, 210))




