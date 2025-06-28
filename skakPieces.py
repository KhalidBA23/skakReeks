from typing import List, Tuple, Optional

# Base klasse for alle skakbrikker.
class Piece:
    def __init__(self, color: str, name: str):
        # Initialiserer en skakbrik med farve og navn.
        self.color = color
        self.name = name
        self.image = f"{color}{name}.png"
        self.has_moved = False

    def get_possible_moves(self, board: list, row: int, col: int, include_attacks: bool = False) -> List[Tuple[int, int]]:
        # Returnerer mulige træk for brikken (skal overskrives i subklasser).
        return []

    def __str__(self) -> str:
        # Returnerer tekstrepræsentation af brikken.
        return f"{self.color}{self.name}"

    def is_valid_position(self, row: int, col: int) -> bool:
        # Tjekker om positionen er på brættet.
        return 0 <= row < 8 and 0 <= col < 8

class Pawn(Piece):
    def __init__(self, color: str):
        # Initialiserer en bonde.
        super().__init__(color, "P")
        self.en_passant_vulnerable = False

    def get_possible_moves(self, board: list, row: int, col: int, include_attacks: bool = False) -> List[Tuple[int, int]]:
        # Returnerer alle mulige træk for en bonde (inkl. en passant og angreb).
        moves = []
        direction = -1 if self.color == 'w' else 1

        if include_attacks:
            # Returnerer kun angrebsfelter.
            if self.is_valid_position(row + direction, col - 1):
                moves.append((row + direction, col - 1))
            if self.is_valid_position(row + direction, col + 1):
                moves.append((row + direction, col + 1))
            return moves

        # Ét felt fremad
        if self.is_valid_position(row + direction, col) and board[row + direction][col] is None:
            moves.append((row + direction, col))
            # Dobbelttræk fra startposition
            start_row = 6 if self.color == 'w' else 1
            if row == start_row and self.is_valid_position(row + direction * 2, col) and board[row + direction * 2][col] is None:
                moves.append((row + direction * 2, col))

        # Diagonale angreb og en passant
        for c_offset in [-1, 1]:
            new_col = col + c_offset
            new_row = row + direction
            if self.is_valid_position(new_row, new_col):
                # Normalt angreb
                if isinstance(board[new_row][new_col], Piece) and board[new_row][new_col].color != self.color:
                    moves.append((new_row, new_col))
                # En passant
                elif board[new_row][new_col] is None and row == (3 if self.color == 'w' else 4):
                    # Tjek om der står en modstanderbonde ved siden af, som er sårbar
                    if self.is_valid_position(row, new_col):
                        side_pawn = board[row][new_col]
                        if (
                            isinstance(side_pawn, Pawn)
                            and side_pawn.color != self.color
                            and getattr(side_pawn, "en_passant_vulnerable", False)
                        ):
                            moves.append((new_row, new_col))
        return moves

class Rook(Piece):
    def __init__(self, color: str):
        # Initialiserer et tårn.
        super().__init__(color, "R")

    def get_possible_moves(self, board: list, row: int, col: int, include_attacks: bool = False) -> List[Tuple[int, int]]:
        # Returnerer alle mulige træk for et tårn (vandret/lodret).
        moves = []
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for d_row, d_col in directions:
            r, c = row, col
            while True:
                r += d_row
                c += d_col
                if not self.is_valid_position(r, c):
                    break
                if board[r][c] is None:
                    moves.append((r, c))
                elif isinstance(board[r][c], Piece):
                    if include_attacks or board[r][c].color != self.color:
                        moves.append((r, c))
                    break
                else:
                    break
        return moves

class Knight(Piece):
    def __init__(self, color: str):
        # Initialiserer en springer.
        super().__init__(color, "N")

    def get_possible_moves(self, board: list, row: int, col: int, include_attacks: bool = False) -> List[Tuple[int, int]]:
        # Returnerer alle mulige træk for en springer (L-form).
        moves = []
        knight_moves = [(-2, -1), (-1, -2), (1, -2), (2, -1), (2, 1), (1, 2), (-1, 2), (-2, 1)]
        for d_row, d_col in knight_moves:
            new_row, new_col = row + d_row, col + d_col
            if not self.is_valid_position(new_row, new_col):
                continue
            if board[new_row][new_col] is None or include_attacks or (
                isinstance(board[new_row][new_col], Piece) and board[new_row][new_col].color != self.color
            ):
                moves.append((new_row, new_col))
        return moves

class Bishop(Piece):
    def __init__(self, color: str):
        # Initialiserer en løber.
        super().__init__(color, "B")

    def get_possible_moves(self, board: list, row: int, col: int, include_attacks: bool = False) -> List[Tuple[int, int]]:
        # Returnerer alle mulige træk for en løber (diagonalt).
        moves = []
        directions = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
        for d_row, d_col in directions:
            r, c = row, col
            while True:
                r += d_row
                c += d_col
                if not self.is_valid_position(r, c):
                    break
                if board[r][c] is None:
                    moves.append((r, c))
                elif isinstance(board[r][c], Piece):
                    if include_attacks or board[r][c].color != self.color:
                        moves.append((r, c))
                    break
                else:
                    break
        return moves

class Queen(Piece):
    def __init__(self, color: str):
        # Initialiserer en dronning.
        super().__init__(color, "Q")

    def get_possible_moves(self, board: list, row: int, col: int, include_attacks: bool = False) -> List[Tuple[int, int]]:
        # Returnerer alle mulige træk for en dronning (tårn + løber).
        rook_moves = Rook(self.color).get_possible_moves(board, row, col, include_attacks)
        bishop_moves = Bishop(self.color).get_possible_moves(board, row, col, include_attacks)
        return rook_moves + bishop_moves

class King(Piece):
    def __init__(self, color: str):
        # Initialiserer en konge.
        super().__init__(color, "K")

    def get_possible_moves(self, board: list, row: int, col: int, include_attacks: bool = False) -> List[Tuple[int, int]]:
        # Returnerer alle mulige træk for en konge (inkl. rokade).
        if include_attacks:
            moves = []
            king_moves = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
            for d_row, d_col in king_moves:
                new_row, new_col = row + d_row, col + d_col
                if not self.is_valid_position(new_row, new_col):
                    continue
                moves.append((new_row, new_col))
            return moves

        moves = []
        king_moves = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        for d_row, d_col in king_moves:
            new_row, new_col = row + d_row, col + d_col
            if not self.is_valid_position(new_row, new_col):
                continue
            if board[new_row][new_col] is None or (
                isinstance(board[new_row][new_col], Piece) and board[new_row][new_col].color != self.color
            ):
                moves.append((new_row, new_col))
        # Rokade
        if self._check_kingside_castling(board, row, col):
            moves.append((row, col + 2))
        if self._check_queenside_castling(board, row, col):
            moves.append((row, col - 2))
        return moves

    def _check_kingside_castling(self, board: list, row: int, col: int) -> bool:
        # Tjekker om kort rokade er mulig.
        if self.has_moved:
            return False
        kingside_rook_col = 7
        if not (0 <= kingside_rook_col < 8):
            return False
        rook = board[row][kingside_rook_col]
        if not isinstance(rook, Rook) or rook.color != self.color or rook.has_moved:
            return False
        if board[row][col + 1] is not None or board[row][col + 2] is not None:
            return False
        return self._verify_castling_path(board, row, col, col + 2)

    def _check_queenside_castling(self, board: list, row: int, col: int) -> bool:
        # Tjekker om lang rokade er mulig.
        if self.has_moved:
            return False
        queenside_rook_col = 0
        if not (0 <= queenside_rook_col < 8):
            return False
        rook = board[row][queenside_rook_col]
        if not isinstance(rook, Rook) or rook.color != self.color or rook.has_moved:
            return False
        if board[row][col - 1] is not None or board[row][col - 2] is not None or board[row][col - 3] is not None:
            return False
        return self._verify_castling_path(board, row, col, col - 2)

    def _verify_castling_path(self, board: list, row: int, start_col: int, end_col: int) -> bool:
        # Tjekker at ingen felter kongen passerer er truet.
        step = 1 if end_col > start_col else -1
        for current_col in range(start_col, end_col + step, step):
            if self._is_square_under_attack(board, row, current_col):
                return False
        return True

    def _is_square_under_attack(self, board: list, target_row: int, target_col: int) -> bool:
        # Tjekker om et felt er truet af modstanderen.
        for row in range(8):
            for col in range(8):
                piece = board[row][col]
                if isinstance(piece, Piece) and piece.color != self.color:
                    attack_moves = piece.get_possible_moves(board, row, col, include_attacks=True)
                    if (target_row, target_col) in attack_moves:
                        return True
        return False

    @staticmethod
    def perform_castling(board: list, king_row: int, king_col: int, dest_col: int) -> list:
        # Udfører rokaden på brættet.
        rook_col = 7 if dest_col > king_col else 0
        rook_new_col = dest_col - 1 if dest_col > king_col else dest_col + 1
        king = board[king_row][king_col]
        rook = board[king_row][rook_col]
        board[king_row][dest_col] = king
        board[king_row][king_col] = None
        king.has_moved = True
        board[king_row][rook_new_col] = rook
        board[king_row][rook_col] = None
        rook.has_moved = True
        return board