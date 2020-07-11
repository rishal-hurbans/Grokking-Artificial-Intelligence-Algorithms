
# Symbols to represent human or AI players
PLAYER_HUMAN = 'H'
PLAYER_AI = 'A'
BOARD_EMPTY_SLOT = '_'
WINNING_SEQUENCE_COUNT = 4

# Tuple encoding human player as -1, and AI player as 1
PLAYERS = {PLAYER_HUMAN: -1,
           PLAYER_AI: 1}


# This class encompasses the logic to play a game of connect.
class Connect:

    # The game board is initialized with a board size for x and y.
    def __init__(self, board_size_x=5, board_size_y=4):
        self.board_size_x = board_size_x
        self.board_size_y = board_size_y
        self.player_turn = PLAYERS[PLAYER_AI]
        self.board = self.generate_board(board_size_x, board_size_y)

    # Reset the game with an empty board
    def reset(self):
        self.board = self.generate_board(self.board_size_x, self.board_size_y)

    # Generate an empty board to begin on reset the game
    def generate_board(self, board_size_x, board_size_y):
        board = []
        for x in range(board_size_x):
            row = BOARD_EMPTY_SLOT * board_size_y
            board.append(row)
        return board

    # Print the board to console
    def print_board(self):
        result = ''
        for y in range(0, self.board_size_y):
            for x in range(0, self.board_size_x):
                result += self.board[x][y]
            result += '\n'
        print(result)

    # Print which player's turn it is
    def print_turn(self):
        if self.player_turn == PLAYERS[PLAYER_HUMAN]:
            print('It is Human to play')
        else:
            print('It is AI to play')

    # Determine if the game has a winner between the human and AI
    def has_winner(self):
        if self.has_a_row(PLAYER_HUMAN, WINNING_SEQUENCE_COUNT):
            return "Human won"
        elif self.has_a_row(PLAYER_AI, WINNING_SEQUENCE_COUNT):
            return "AI won"
        return 0

    # Get the score for the AI
    def get_score_for_ai(self):
        if self.has_a_row(PLAYER_HUMAN, 4):
            return -10
        if self.has_a_row(PLAYER_AI, 4):
            return 10
        return 0

    # Determine if a player has a row
    def has_a_row(self, player, row_count):
        for x in range(self.board_size_x):
            for y in range(self.board_size_y):
                if self.has_row_of_x_from_point(player, row_count, x, y, 1, 0):  # Horizontal row
                    return True
                elif self.has_row_of_x_from_point(player, row_count, x, y, 0, 1):  # Vertical row
                    return True
                elif self.has_row_of_x_from_point(player, row_count, x, y, 1, 1):  # Diagonal row
                    return True
        return False

    # Determine if a player has a row given a starting point and offset
    def has_row_of_x_from_point(self, player, row_count, x, y, offset_x, offset_y):
        total = 0
        for i in range(row_count):
            target_x = x + (i * offset_x)
            target_y = y + (i * offset_y)
            if self.is_within_bounds(target_x, target_y):
                if self.board[target_x][target_y] == player:
                    total += 1
        if total == row_count:
            return True
        return False

    # Determine if a specific x,y pair is within bounds of the board
    def is_within_bounds(self, x, y):
        if 0 <= x < self.board_size_x and 0 <= y < self.board_size_y:
            return True
        return False

    # Determine if the entire board is filled with disks
    def is_board_full(self):
        for x in range(self.board_size_x):
            if BOARD_EMPTY_SLOT in self.board[x]:
                return False
        return True

    # Determine if a slot is full
    def is_slot_full(self, slot_number):
        if BOARD_EMPTY_SLOT in self.board[slot_number]:
            return False
        return True

    # Determine if a slot number is empty
    def is_slot_empty(self, slot_number):
        count = 0
        for i in range(self.board_size_y):
            if self.board[slot_number][i] == BOARD_EMPTY_SLOT:
                count += 1
        if count == self.board_size_y:
            return True
        return False

    # Execute a move for a player
    def execute_move(self, player, slot_number):
        row = self.board[slot_number]
        # Place the disk at the bottom if the slot number is empty
        if self.is_slot_empty(slot_number):
            self.board[slot_number] = row[0:self.board_size_y - 1] + player
        else:
            # Place the disk at the next empty slot if the slot number is not empty
            for i in range(0, self.board_size_y - 1):
                if row[i + 1] != BOARD_EMPTY_SLOT:
                    self.board[slot_number] = row[0:i] + player + row[i + 1:]
                    break

    # Execute a move for a player if there's space in the slot and choose the player based on whose turn it is
    def play_move(self, slot):
        if 0 <= slot <= 4:
            if not self.is_slot_full(slot):
                if self.player_turn == PLAYERS[PLAYER_AI]:
                    self.execute_move(PLAYER_AI, slot)
                else:
                    self.execute_move(PLAYER_HUMAN, slot)
                self.player_turn *= -1
                return True
            return False
        return False
