import pygame, sys
import utils
import Menu

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
        menu.is_continue(omok)

