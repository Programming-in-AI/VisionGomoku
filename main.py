import pygame
import utils
import Menu

if __name__ == '__main__':

    # TODO 0 : let computer play game, who play black, white is totally random. [done]
    # TODO 1 : whoever start, black first. [done]
    # TODO 2 : black should start from center. [done]
    # TODO 3 : white randomly chooses the spot which is empty. [done]
    # TODO 4 : balck's third stone must be out of center area 5x5. [yet]
    ##
    # TODO 5 : must play by taking turn. [done]
    # TODO 6 : visualize the board [done]
    # TODO 7 : win twice, get the win [done]
    # TODO 8 : let it available to input opponent's coordinate and react in 5 sec. [done]
    ##
    # initialize
    pygame.init()
    surface = pygame.display.set_mode((utils.window_width, utils.window_height))
    pygame.display.set_caption("Omok game")

    # make instance
    omok = utils.Omok(surface)
    menu = Menu.Menu(surface)

    print('[Game start!]')
    # run game
    while True:
        omok.run_game(omok, menu)
        # if x button clicked then break
        if omok.quit:
            break
        # if anyside win twice quit the window
        if menu.game_over(omok):
            pygame.quit()
            break

