import pygame
import os
import sys
import time
import threading
from alphabeta import ChessAI
from skakPieces import Piece, Pawn, Rook, Knight, Bishop, Queen, King
from copy import deepcopy
import sys
sys.setrecursionlimit(10000)

# Skærmindstillinger
WIDTH, HEIGHT = 650, 650
ROWS, COLS = 8, 8
SQUARE_SIZE = WIDTH // COLS

# Farver
LIGHT = (238, 238, 210)
DARK = (118, 150, 86)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
HIGHLIGHT = (255, 255, 0, 100)
BLUE = (65, 105, 225)

STATE_MENU = 0
STATE_SETTINGS = 1
STATE_GAME = 2
STATE_GAME_OVER = 3

class ChessGame:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Skak GUI med pygame")
        self.clock = pygame.time.Clock()
        
        self.piece_images = {}
        self.load_images()
        self.move_log = []
        self.is_paused = False
        self.player_color = None
        self.ai_color = None

        self.difficulty_settings = {
            "let": 1,
            "mellem": 2,
            "svær": 3
        }

        self.board = None
        self.ai = None
        self.ai_depth = None
        self.selected_piece = None
        self.possible_moves = []
        self.human_turn = True
        self.game_over = False
        self.winner_text = ""
        self.last_move = None
        self.ai_thinking = False
        self.move_log = []

        self.state = STATE_MENU
        
        self.large_font = pygame.font.SysFont("Arial", 48)
        self.medium_font = pygame.font.SysFont("Arial", 36)
        self.small_font = pygame.font.SysFont("Arial", 24)
        self.tiny_font = pygame.font.SysFont("Arial", 14)
        self.undo_button = pygame.Rect(WIDTH - 110, HEIGHT - 50, 100, 40)
        self.undo_text = self.small_font.render("Undo", True, (255,255,255))

        # 50-move rule: counter for half-moves since last pawn move or capture
        self.halfmove_clock = 0

    def undo_move(self):
        for i in range(2):
            if not self.move_log:
                break
            r1, c1, r2, c2, moved, captured, prev_halfmove = self.move_log.pop()
            self.board[r1][c1] = moved
            self.board[r2][c2] = captured
            self.halfmove_clock = prev_halfmove
            self.human_turn = (i == 1)
        self.selected_piece = None
        self.possible_moves = []
        self.last_move = None

    def load_images(self):
        pieces = ['wP', 'wR', 'wN', 'wB', 'wQ', 'wK', 'bP', 'bR', 'bN', 'bB', 'bQ', 'bK']
        for piece in pieces:
            path = os.path.join("assets", f"{piece}.png")
            try:
                image = pygame.image.load(path)
                self.piece_images[piece] = pygame.transform.scale(image, (SQUARE_SIZE, SQUARE_SIZE))
            except pygame.error:
                print(f"Kunne ikke indlæse billedfil: {path}")
                sys.exit(1)

    def initialize_board(self):
        return [
            [Rook('b'), Knight('b'), Bishop('b'), Queen('b'), King('b'), Bishop('b'), Knight('b'), Rook('b')],
            [Pawn('b') for _ in range(8)],
            [None] * 8,
            [None] * 8,
            [None] * 8,
            [None] * 8,
            [Pawn('w') for _ in range(8)],
            [Rook('w'), Knight('w'), Bishop('w'), Queen('w'), King('w'), Bishop('w'), Knight('w'), Rook('w')],
        ]

    def draw_board(self):
        for row in range(ROWS):
            for col in range(COLS):
                color = LIGHT if (row + col) % 2 == 0 else DARK
                pygame.draw.rect(self.screen, color, (col * SQUARE_SIZE, row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))
        if self.is_paused:
            pause_overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
            pause_overlay.fill((0, 0, 0, 150))
            self.screen.blit(pause_overlay, (0, 0))
            pause_text = self.medium_font.render("Spillet er på pause", True, (255, 255, 255))
            self.screen.blit(pause_text, (WIDTH // 2 - pause_text.get_width() // 2, HEIGHT // 2 - pause_text.get_height() // 2))

    def draw_coordinates(self):
        for col in range(COLS):
            label = self.tiny_font.render(chr(97 + col), True, (0, 0, 0))
            self.screen.blit(label, (col * SQUARE_SIZE + SQUARE_SIZE - 15, HEIGHT - 15))
        for row in range(ROWS):
            label = self.tiny_font.render(str(8 - row), True, (0, 0, 0))
            self.screen.blit(label, (5, row * SQUARE_SIZE + 5))

    def draw_pieces(self):
        for row in range(8):
            for col in range(8):
                piece = self.board[row][col]
                if piece:
                    image_key = f"{piece.color}{piece.name}"
                    if image_key in self.piece_images:
                        self.screen.blit(self.piece_images[image_key], (col * SQUARE_SIZE, row * SQUARE_SIZE))
        if self.selected_piece:
            row, col, _ = self.selected_piece
            pygame.draw.rect(self.screen, BLUE, (col * SQUARE_SIZE, row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE), 5)

    def draw_possible_moves(self):
        for row, col in self.possible_moves:
            if self.board[row][col] is None:
                pygame.draw.circle(self.screen, GREEN, (col * SQUARE_SIZE + SQUARE_SIZE // 2, row * SQUARE_SIZE + SQUARE_SIZE // 2), 10)
            else:
                pygame.draw.rect(self.screen, RED, (col * SQUARE_SIZE, row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE), 3)

    def highlight_selected(self):
        if self.selected_piece:
            row, col, _ = self.selected_piece
            s = pygame.Surface((SQUARE_SIZE, SQUARE_SIZE), pygame.SRCALPHA)
            s.fill(HIGHLIGHT)
            self.screen.blit(s, (col * SQUARE_SIZE, row * SQUARE_SIZE))

    def highlight_last_move(self):
        if self.last_move:
            r1, c1, r2, c2 = self.last_move
            pygame.draw.rect(self.screen, BLUE, (c1 * SQUARE_SIZE, r1 * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE), 3)
            pygame.draw.rect(self.screen, BLUE, (c2 * SQUARE_SIZE, r2 * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE), 3)

    def highlight_check(self, king_pos):
        if king_pos:
            row, col = king_pos
            pygame.draw.rect(self.screen, RED, (col * SQUARE_SIZE, row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE), 3)

    def get_square_under_mouse(self):
        mouse_x, mouse_y = pygame.mouse.get_pos()
        row = mouse_y // SQUARE_SIZE
        col = mouse_x // SQUARE_SIZE
        if 0 <= row < 8 and 0 <= col < 8:
            return row, col
        return None

    def show_thinking_indicator(self):
        if self.ai_thinking:
            thinking_text = self.small_font.render("Let him cook", True, (0, 0, 0))
            text_bg = pygame.Rect(WIDTH // 2 - 100, 10, 200, 40)
            pygame.draw.rect(self.screen, (200, 200, 200), text_bg)
            pygame.draw.rect(self.screen, (0, 0, 0), text_bg, 2)
            text_x = WIDTH // 2 - thinking_text.get_width() // 2
            self.screen.blit(thinking_text, (text_x, 15))
            pygame.display.update(text_bg)

    def draw_slider(self, x, y, width, value, min_val, max_val, label):
        pos = x + ((value - min_val) / (max_val - min_val)) * width
        pygame.draw.circle(self.screen, (0, 128, 0), (int(pos), y + 5), 10)
        value_text = self.medium_font.render(f"{label}: {value}", True, (0, 0, 0))
        self.screen.blit(value_text, (x, y - 30))

    def handle_slider(self, x, y, width, value, min_val, max_val, mouse_pos, mouse_pressed):
        if mouse_pressed and y - 10 <= mouse_pos[1] <= y + 20:
            if x <= mouse_pos[0] <= x + width:
                new_value = min_val + ((mouse_pos[0] - x) / width) * (max_val - min_val)
                return max(min_val, min(max_val, round(new_value)))
        return value

    def show_difficulty_settings(self):
        title = self.large_font.render("Sværhedsgrad Indstillinger", True, (0, 0, 0))
        title_rect = title.get_rect(center=(WIDTH // 2, 80))
        easy_depth = self.difficulty_settings["let"]
        medium_depth = self.difficulty_settings["mellem"]
        hard_depth = self.difficulty_settings["svær"]
        slider_width = 400
        slider_x = WIDTH // 2 - slider_width // 2
        easy_y = 200
        medium_y = 300
        hard_y = 400
        start_btn = pygame.Rect(WIDTH // 2 - 100, 550, 200, 60)
        btn_text = self.medium_font.render("Start Spil", True, (255, 255, 255))
        running = True
        while running:
            self.screen.fill((200, 200, 200))
            self.screen.blit(title, title_rect)
            mouse_pos = pygame.mouse.get_pos()
            mouse_pressed = pygame.mouse.get_pressed()[0]
            self.draw_slider(slider_x, easy_y, slider_width, easy_depth, 1, 5, "Let")
            self.draw_slider(slider_x, medium_y, slider_width, medium_depth, 1, 5, "Mellem")
            self.draw_slider(slider_x, hard_y, slider_width, hard_depth, 1, 5, "Svær")
            easy_depth = self.handle_slider(slider_x, easy_y, slider_width, easy_depth, 1, 5, mouse_pos, mouse_pressed)
            medium_depth = self.handle_slider(slider_x, medium_y, slider_width, medium_depth, 1, 5, mouse_pos, mouse_pressed)
            hard_depth = self.handle_slider(slider_x, hard_y, slider_width, hard_depth, 1, 5, mouse_pos, mouse_pressed)
            if easy_depth > medium_depth:
                medium_depth = easy_depth
            if medium_depth > hard_depth:
                hard_depth = medium_depth
            pygame.draw.rect(self.screen, (0, 128, 0), start_btn)
            self.screen.blit(btn_text, (start_btn.x + 30, start_btn.y + 10))
            pygame.display.flip()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if start_btn.collidepoint(event.pos):
                        running = False
        self.difficulty_settings = {
            "let": easy_depth,
            "mellem": medium_depth,
            "svær": hard_depth
        }
        self.state = STATE_MENU

    def show_difficulty_menu(self):
        title = self.large_font.render("Vælg Sværhedsgrad", True, (0, 0, 0))
        title_rect = title.get_rect(center=(WIDTH // 2, HEIGHT // 2 - 100))
        difficulty_options = [
            ("Let", self.difficulty_settings["let"]),
            ("Mellem", self.difficulty_settings["mellem"]),
            ("Svær", self.difficulty_settings["svær"]),
            ("Tilpas indstillinger", 0)
        ]
        buttons = []
        for i, (level, depth) in enumerate(difficulty_options):
            btn = pygame.Rect(WIDTH // 2 - 150, HEIGHT // 2 + i * 70, 300, 60)
            if i < 3:
                text = self.medium_font.render(f"{level} (Dybde {depth})", True, (255, 255, 255))
            else:
                text = self.medium_font.render(level, True, (255, 255, 255))
            buttons.append((btn, text, depth))
        running = True
        while running:
            self.screen.fill((200, 200, 200))
            self.screen.blit(title, title_rect)
            for btn, text, _ in buttons:
                pygame.draw.rect(self.screen, (0, 128, 0), btn)
                self.screen.blit(text, (btn.x + 25, btn.y + 10))
            pygame.display.flip()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.MOUSEBUTTONDOWN:
                    for i, (btn, _, depth) in enumerate(buttons):
                        if btn.collidepoint(event.pos):
                            if depth == 0:
                                self.state = STATE_SETTINGS
                                return
                            self.ai_depth = depth
                            self.start_new_game()
                            return

    def show_start_menu(self):
        title = self.large_font.render("Velkommen til Skak!", True, (0, 0, 0))
        title_rect = title.get_rect(center=(WIDTH // 2, HEIGHT // 2 - 150))
        white_btn = pygame.Rect(WIDTH // 2 - 150, HEIGHT // 2 - 50, 300, 60)
        black_btn = pygame.Rect(WIDTH // 2 - 150, HEIGHT // 2 + 50, 300, 60)
        white_text = self.medium_font.render("Spil som Hvid", True, (255, 255, 255))
        black_text = self.medium_font.render("Spil som Sort", True, (255, 255, 255))
        while self.state == STATE_MENU:
            self.screen.fill((200, 200, 200))
            self.screen.blit(title, title_rect)
            pygame.draw.rect(self.screen, (0, 128, 0), white_btn)
            pygame.draw.rect(self.screen, (0, 128, 0), black_btn)
            self.screen.blit(white_text, (white_btn.x + 65, white_btn.y + 15))
            self.screen.blit(black_text, (black_btn.x + 65, black_btn.y + 15))
            pygame.display.flip()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if white_btn.collidepoint(event.pos):
                        self.player_color = 'w'
                        self.ai_color = 'b'
                        self.show_difficulty_menu()
                        return
                    elif black_btn.collidepoint(event.pos):
                        self.player_color = 'b'
                        self.ai_color = 'w'
                        self.show_difficulty_menu()
                        return

    def show_game_over_menu(self):
        text = self.large_font.render(self.winner_text, True, (200, 0, 0))
        text_rect = text.get_rect(center=(WIDTH // 2, HEIGHT // 2 - 100))
        replay_btn = pygame.Rect(WIDTH // 2 - 110, HEIGHT // 2, 100, 50)
        quit_btn = pygame.Rect(WIDTH // 2 + 10, HEIGHT // 2, 100, 50)
        replay_text = self.small_font.render("Spil igen", True, (255, 255, 255))
        quit_text = self.small_font.render("Afslut", True, (255, 255, 255))
        overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 128))
        self.screen.blit(overlay, (0, 0))
        pygame.draw.rect(self.screen, (255, 255, 255), pygame.Rect(WIDTH//2 - 150, HEIGHT//2 - 150, 300, 250))
        self.screen.blit(text, text_rect)
        pygame.draw.rect(self.screen, (0, 128, 0), replay_btn)
        pygame.draw.rect(self.screen, (128, 0, 0), quit_btn)
        self.screen.blit(replay_text, (replay_btn.x + 10, replay_btn.y + 12))
        self.screen.blit(quit_text, (quit_btn.x + 20, quit_btn.y + 12))
        pygame.display.flip()
        waiting_for_input = True
        while waiting_for_input:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if replay_btn.collidepoint(event.pos):
                        self.state = STATE_MENU
                        waiting_for_input = False
                    elif quit_btn.collidepoint(event.pos):
                        pygame.quit()
                        sys.exit()

    def start_new_game(self):
        self.board = self.initialize_board()
        self.ai = ChessAI(depth=self.ai_depth)
        self.selected_piece = None
        self.possible_moves = []
        self.human_turn = self.player_color == 'w'
        self.game_over = False
        self.winner_text = ""
        self.last_move = None
        self.ai_thinking = False
        self.state = STATE_GAME
        self.halfmove_clock = 0
        self.move_log = []

    def get_valid_moves(self, row, col, piece):
        all_moves = piece.get_possible_moves(self.board, row, col)
        valid_moves = []
        for move in all_moves:
            temp_board = deepcopy(self.board)
            temp_board[move[0]][move[1]] = piece
            temp_board[row][col] = None
            king_pos = None
            if piece.name == 'K':
                king_pos = (move[0], move[1])
            else:
                king_pos = self.ai.find_king(temp_board, piece.color)
            if king_pos and not self.ai.is_in_check(temp_board, piece.color, king_pos):
                valid_moves.append(move)
        return valid_moves

    def handle_game_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.is_paused:
                square = self.get_square_under_mouse()
                if square:
                    row, col = square
                    piece = self.board[row][col]
                    if self.selected_piece:
                        r1, c1, p = self.selected_piece
                        if (row, col) != (r1, c1):
                            self.board[row][col] = p
                            self.board[r1][c1] = None
                            self.selected_piece = None
                            self.possible_moves = []
                    else:
                        if piece:
                            self.selected_piece = (row, col, piece)
                            self.possible_moves = self.get_valid_moves(row, col, piece)
                return
            if self.human_turn and not self.game_over:
                square = self.get_square_under_mouse()
                if square:
                    row, col = square
                    piece = self.board[row][col]
                    if self.selected_piece:
                        r1, c1, p = self.selected_piece
                        en_passant_capture = False
                        if isinstance(p, Pawn) and (row, col) in self.possible_moves:
                            if self.board[row][col] is None and abs(col - c1) == 1 and abs(row - r1) == 1:
                                side_pawn = self.board[r1][col]
                                if (
                                    isinstance(side_pawn, Pawn)
                                    and side_pawn.color != p.color
                                    and getattr(side_pawn, "en_passant_vulnerable", False)
                                ):
                                    en_passant_capture = True
                        if (row, col) in self.possible_moves:
                            r1, c1, p = self.selected_piece
                            captured = self.board[row][col]
                            prev_halfmove = self.halfmove_clock
                            if p.name == 'P' or captured is not None:
                                self.halfmove_clock = 0
                            else:
                                self.halfmove_clock += 1
                            if isinstance(p, Pawn):
                                for r in range(8):
                                    for c in range(8):
                                        piece_ = self.board[r][c]
                                        if isinstance(piece_, Pawn):
                                            piece_.en_passant_vulnerable = False
                                if abs(row - r1) == 2 and c1 == col:
                                    p.en_passant_vulnerable = True
                                if en_passant_capture:
                                    captured = self.board[r1][col]
                                    self.board[r1][col] = None
                            self.move_log.append((r1, c1, row, col, p, captured, prev_halfmove))
                            self.board[row][col] = p
                            self.board[r1][c1] = None
                            self.last_move = (r1, c1, row, col)
                            if p.name == 'P' and (row == 0 or row == 7):
                                self.board[row][col] = Queen(p.color)
                            if p.name == 'K' or p.name == 'R':
                                p.has_moved = True
                            self.selected_piece = None
                            self.possible_moves = []
                            if self.ai.is_game_over(self.board):
                                self.game_over = True
                                self.winner_text = "Sort vinder!" if self.ai.find_king(self.board, 'w') is None else "Hvid vinder!"
                                self.state = STATE_GAME_OVER
                            else:
                                if self.halfmove_clock >= 100:
                                    self.game_over = True
                                    self.winner_text = "Remis (50-træksregel)"
                                    self.state = STATE_GAME_OVER
                                    return
                                self.human_turn = False
                        elif piece and piece.color == self.player_color:
                            self.selected_piece = (row, col, piece)
                            self.possible_moves = self.get_valid_moves(row, col, piece)
                        else:
                            self.selected_piece = None
                            self.possible_moves = []
                    elif piece and piece.color == self.player_color:
                        self.selected_piece = (row, col, piece)
                        self.possible_moves = self.get_valid_moves(row, col, piece)
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_p:
                self.is_paused = not self.is_paused
                self.selected_piece = None
                self.possible_moves = []

    def _verify_castling_path(self, board, row, start_col, end_col, color):
        step = 1 if end_col > start_col else -1
        for current_col in range(start_col, end_col + step, step):
            if self._is_square_under_attack(board, row, current_col, color):
                return False
        return True

    def _is_square_under_attack(self, board, target_row, target_col, defending_color):
        for row in range(8):
            for col in range(8):
                piece = board[row][col]
                if isinstance(piece, Piece) and piece.color != defending_color:
                    attack_moves = piece.get_possible_moves(board, row, col, include_attacks=True)
                    if (target_row, target_col) in attack_moves:
                        return True
        return False

    def ai_move_callback(self, best_move):
        if best_move:
            r1, c1, r2, c2 = best_move
            moved = self.board[r1][c1]
            captured = self.board[r2][c2]
            prev_halfmove = self.halfmove_clock
            if moved.name == 'P' or captured is not None:
                self.halfmove_clock = 0
            else:
                self.halfmove_clock += 1
            if isinstance(moved, Pawn):
                for r in range(8):
                    for c in range(8):
                        piece_ = self.board[r][c]
                        if isinstance(piece_, Pawn):
                            piece_.en_passant_vulnerable = False
                if abs(r2 - r1) == 2 and c1 == c2:
                    moved.en_passant_vulnerable = True
                if self.board[r2][c2] is None and abs(c2 - c1) == 1 and abs(r2 - r1) == 1:
                    captured = self.board[r1][c2]
                    self.board[r1][c2] = None
            self.move_log.append((r1, c1, r2, c2, moved, captured, prev_halfmove))
            self.board[r2][c2] = moved
            self.board[r1][c1] = None
            self.last_move = best_move
            if self.board[r2][c2].name == 'P' and (r2 == 0 or r2 == 7):
                self.board[r2][c2] = Queen(self.ai_color)
        self.ai_thinking = False
        self.human_turn = True
        if self.ai.is_game_over(self.board):
            self.game_over = True
            self.winner_text = "Hvid vinder!" if self.ai.find_king(self.board, 'b') is None else "Sort vinder!"
            self.state = STATE_GAME_OVER
        if self.halfmove_clock >= 100:
            self.game_over = True
            self.winner_text = "Remis (50-træksregel)"
            self.state = STATE_GAME_OVER
            return
        self.human_turn = True

    def update_game(self):
        if not self.human_turn and not self.ai_thinking:
            self.ai_thinking = True
            pygame.display.flip()
            self.ai.calculate_best_move_async(self.board, self.ai_color, self.ai_move_callback)

    def render_game(self):
        self.draw_board()
        self.draw_coordinates()
        self.highlight_last_move()
        self.draw_pieces()
        self.highlight_selected()
        self.draw_possible_moves()
        self.show_thinking_indicator()
        white_king_pos = self.ai.find_king(self.board, 'w')
        black_king_pos = self.ai.find_king(self.board, 'b')
        if white_king_pos and self.ai.is_in_check(self.board, 'w', white_king_pos):
            self.highlight_check(white_king_pos)
            check_text = self.small_font.render("Skak til hvid!", True, (255, 0, 0))
            self.screen.blit(check_text, (WIDTH - 150, 10))
        if black_king_pos and self.ai.is_in_check(self.board, 'b', black_king_pos):
            self.highlight_check(black_king_pos)
            check_text = self.small_font.render("Skak til sort!", True, (255, 0, 0))
            self.screen.blit(check_text, (WIDTH - 150, 40))
        if self.human_turn and not self.game_over:
            turn_text = self.small_font.render(f"Din tur ({'hvid' if self.player_color == 'w' else 'sort'})", True, (0, 0, 0))
            self.screen.blit(turn_text, (10, 10))
            pygame.draw.rect(self.screen, (50,50,50), self.undo_button, border_radius=5)
            self.screen.blit(
                self.undo_text,
                (
                    self.undo_button.x + (self.undo_button.width - self.undo_text.get_width())//2,
                    self.undo_button.y + (self.undo_button.height - self.undo_text.get_height())//2
                )
            )

    def run(self):
        while True:
            if self.state == STATE_MENU:
                self.show_start_menu()
            elif self.state == STATE_SETTINGS:
                self.show_difficulty_settings()
            elif self.state == STATE_GAME:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        sys.exit()
                    elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                        if self.undo_button.collidepoint(event.pos):
                            self.undo_move()
                            continue
                    self.handle_game_event(event)
                self.update_game()
                self.render_game()
                pygame.display.flip()
                self.clock.tick(60)
            elif self.state == STATE_GAME_OVER:
                self.show_game_over_menu()
                

if __name__ == "__main__":
    game = ChessGame()
    game.run()