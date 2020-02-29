import connect_ai_alpha_beta_pruning as caab
import connect_puzzle as cp

# Initialize a Connect game with default board size
connect = cp.Connect()

connect.play_move(0)
connect.print_board()
connect.play_move(4)
connect.print_board()
print(connect.get_score_for_ai())

connect.play_move(3)
connect.print_board()
connect.play_move(2)
connect.print_board()
print(connect.get_score_for_ai())

connect.play_move(2)
connect.print_board()
connect.play_move(1)
connect.print_board()
print(connect.get_score_for_ai())

connect.play_move(3)
connect.print_board()
connect.play_move(4)
connect.print_board()
print(connect.get_score_for_ai())

connect.play_move(0)
connect.print_board()
connect.play_move(1)
connect.print_board()
print(connect.get_score_for_ai())

connect.play_move(1)
connect.print_board()
connect.play_move(4)
connect.print_board()
print(connect.get_score_for_ai())

connect.play_move(4)
connect.print_board()
connect.play_move(0)
connect.print_board()
print(connect.get_score_for_ai())

chosen_move = caab.choose_move(connect, 100)
print('MOVE: ', chosen_move)
connect.play_move(chosen_move)
print(connect.get_score_for_ai())
connect.print_board()
print(connect.has_winner())
