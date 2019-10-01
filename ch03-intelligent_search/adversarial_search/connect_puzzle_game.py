from adversarial_search import connect_ai_alpha_beta_pruning as caiab
from adversarial_search import connect_ai as cai
import connect_puzzle as cp

DEPTH = 10
game = cp.Connect()
while game.has_winner() == 0:
    game.print_turn()
    game.play_move(caiab.choose_move(game, DEPTH))
    game.print_board()
    print(game.has_winner())

    game.print_turn()
    human_move_result = False
    while human_move_result is False:
        print('Make your move: ')
        human_move = int(input())
        human_move_result = game.play_move(human_move)
    game.print_board()
    print(game.has_winner())
