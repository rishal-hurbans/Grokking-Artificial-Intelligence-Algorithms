import connect_ai_alpha_beta_pruning as caab
import connect_puzzle as cp

connect_main = cp.Connect()
connect_main.play_move(0)
connect_main.print_board()
connect_main.play_move(4)
connect_main.print_board()
print(connect_main.get_score_for_ai())

connect_main.play_move(3)
connect_main.print_board()
connect_main.play_move(2)
connect_main.print_board()
print(connect_main.get_score_for_ai())

connect_main.play_move(2)
connect_main.print_board()
connect_main.play_move(1)
connect_main.print_board()
print(connect_main.get_score_for_ai())

connect_main.play_move(3)
connect_main.print_board()
connect_main.play_move(4)
connect_main.print_board()
print(connect_main.get_score_for_ai())

connect_main.play_move(0)
connect_main.print_board()
connect_main.play_move(1)
connect_main.print_board()
print(connect_main.get_score_for_ai())

connect_main.play_move(1)
connect_main.print_board()
connect_main.play_move(4)
connect_main.print_board()
print(connect_main.get_score_for_ai())

connect_main.play_move(4)
connect_main.print_board()
connect_main.play_move(0)
connect_main.print_board()
print(connect_main.get_score_for_ai())

chosen_move = caab.choose_move(connect_main, 100)
print('MOVE: ', chosen_move)
connect_main.play_move(chosen_move)
print(connect_main.get_score_for_ai())
connect_main.print_board()
print(connect_main.has_winner())
