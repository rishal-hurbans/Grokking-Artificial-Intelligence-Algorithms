
PLAYER_HUMAN = 'H'
PLAYER_AI = 'A'

PLAYERS = {PLAYER_HUMAN: -1,
           PLAYER_AI: 1}


class Connect:

    def __init__(self, board_size_x=5, board_size_y=4):
        self.board_size_x = board_size_x
        self.board_size_y = board_size_y
        self.player_turn = PLAYERS[PLAYER_AI]
        self.board = ['____',
                      '____',
                      '____',
                      '____',
                      '____']

    def reset(self):
        self.board = ['____',
                      '____',
                      '____',
                      '____',
                      '____']

    def print_board(self):
        result = ''
        for y in range(0, self.board_size_y):
            for x in range(self.board_size_x - 1, -1, -1):
                result += self.board[x][y]
            result += '\n'
        print(result)

    def print_turn(self):
        if self.player_turn == PLAYERS[PLAYER_HUMAN]:
            print('It is Human to play')
        else:
            print('It is AI to play')

    def has_winner(self):
        if self.has_a_row(PLAYER_HUMAN, 4):
            return "Human won"
        elif self.has_a_row(PLAYER_AI, 4):
            return "AI won"
        return 0

    def get_score_for_ai(self):
        if self.has_a_row(PLAYER_HUMAN, 4):
            return -10
        if self.has_a_row(PLAYER_AI, 4):
            return 10
        return 0

    def has_a_row(self, player, row_count):
        for x in range(self.board_size_x):
            for y in range(self.board_size_y):
                if self.has_row_of_x_from_point(player, row_count, x, y, 1, 0):
                    # Horizontal row
                    return True
                elif self.has_row_of_x_from_point(player, row_count, x, y, 0, 1):
                    # Vertical row
                    return True
                elif self.has_row_of_x_from_point(player, row_count, x, y, 1, 1):
                    # Diagonal row
                    return True
        return False

    def has_row_of_x_from_point(self, player, row_count, x, y, offset_x, offset_y):
        total = 0
        for i in range(row_count):
            target_x = x + (i * offset_x)
            target_y = y + (i * offset_y)
            if self.is_in_range(target_x, target_y):
                if self.board[target_x][target_y] == player:
                    total += 1
        if total == row_count:
            return True
        return False

    def is_board_full(self):
        for x in range(self.board_size_x):
            if '_' in self.board[x]:
                return False
        return True

    def is_in_range(self, x, y):
        if 0 <= x < self.board_size_x and 0 <= y < self.board_size_y:
            return True
        return False

    def is_slot_full(self, slot_number):
        if '_' in self.board[slot_number]:
            return False
        return True

    def is_slot_empty(self, slot_number):
        count = 0
        for i in range(self.board_size_y):
            if self.board[slot_number][i] == '_':
                count += 1
        if count == self.board_size_y:
            return True
        return False

    def execute_move(self, player, slot_number):
        row = self.board[slot_number]
        if self.is_slot_empty(slot_number):
            self.board[slot_number] = row[0:self.board_size_y - 1] + player
        else:
            for i in range(0, self.board_size_y - 1):
                if row[i + 1] != '_':
                    self.board[slot_number] = row[0:i] + player + row[i + 1:]
                    break

    def play_move(self, slot):
        if not self.is_slot_full(slot):
            if self.player_turn == PLAYERS[PLAYER_AI]:
                self.execute_move(PLAYER_AI, slot)
            else:
                self.execute_move(PLAYER_HUMAN, slot)
            self.player_turn *= -1
            return True
        return False
