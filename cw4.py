import tkinter as tk
from tkinter import ttk
import copy
import math

class TicTacToe:
    def __init__(self, root):
        self.root = root
        self.root.title("Kółko i krzyżyk")
        control_frame = ttk.Frame(root, padding=10)
        control_frame.grid(row=0, column=0, sticky="w")
        ttk.Label(control_frame, text="Rozmiar planszy:").grid(row=0, column=0, padx=5)
        self.size_var = tk.IntVar(value=3)
        self.size_spin = ttk.Spinbox(control_frame, from_=3, to=10, width=5, textvariable=self.size_var)
        self.size_spin.grid(row=0, column=1, padx=5)
        ttk.Label(control_frame, text="Maks. głębokość Minimax:").grid(row=0, column=2, padx=5)
        self.depth_var = tk.IntVar(value=6)
        self.depth_spin = ttk.Spinbox(control_frame, from_=1, to=20, width=5, textvariable=self.depth_var)
        self.depth_spin.grid(row=0, column=3, padx=5)
        ttk.Label(control_frame, text="Pierwszy gracz:").grid(row=0, column=4, padx=5)
        self.first_var = tk.StringVar(value="user")
        ttk.Radiobutton(control_frame, text="Użytkownik", variable=self.first_var, value="user").grid(row=0, column=5, padx=2)
        ttk.Radiobutton(control_frame, text="Komputer", variable=self.first_var, value="computer").grid(row=0, column=6, padx=2)
        ttk.Button(control_frame, text="Nowa gra", command=self.new_game).grid(row=0, column=7, padx=10)
        self.status = ttk.Label(root, text="Wybierz ustawienia i kliknij Nowa gra", padding=10)
        self.status.grid(row=1, column=0, sticky="w")
        self.board_frame = ttk.Frame(root, padding=10)
        self.board_frame.grid(row=2, column=0)
        self.buttons = []
        self.board = []
        self.size = 3
        self.max_depth = 6
        self.current_player = 'X'
        self.human = 'X'
        self.computer = 'O'
        self.game_over = False
        self.new_game()

    def new_game(self):
        try:
            self.size = int(self.size_var.get())
        except:
            self.size = 3
        try:
            self.max_depth = int(self.depth_var.get())
        except:
            self.max_depth = 6
        self.human = 'X'
        self.computer = 'O'
        self.current_player = self.human if self.first_var.get() == "user" else self.computer
        self.game_over = False
        self.board = [['' for _ in range(self.size)] for _ in range(self.size)]
        for widget in self.board_frame.winfo_children():
            widget.destroy()
        self.buttons = [[None]*self.size for _ in range(self.size)]
        for r in range(self.size):
            for c in range(self.size):
                b = ttk.Button(self.board_frame, text='', command=lambda rr=r, cc=c: self.on_click(rr, cc), width=4)
                b.grid(row=r, column=c, ipadx=10, ipady=10, padx=2, pady=2)
                self.buttons[r][c] = b
        self.update_status()
        if self.current_player == self.computer:
            self.root.after(200, self.make_computer_move)

    def on_click(self, r, c):
        if self.game_over:
            return
        if self.current_player != self.human:
            return
        if self.board[r][c] != '':
            return
        self.board[r][c] = self.human
        self.buttons[r][c]['text'] = self.human
        self.buttons[r][c]['state'] = 'disabled'
        winner = self.check_winner(self.board)
        if winner or self.is_full(self.board):
            self.end_game(winner)
            return
        self.current_player = self.computer
        self.update_status()
        self.root.after(100, self.make_computer_move)

    def make_computer_move(self):
        if self.game_over:
            return
        move = self.find_best_move(self.board, self.max_depth)
        if move:
            r, c = move
            self.board[r][c] = self.computer
            self.buttons[r][c]['text'] = self.computer
            self.buttons[r][c]['state'] = 'disabled'
        winner = self.check_winner(self.board)
        if winner or self.is_full(self.board):
            self.end_game(winner)
            return
        self.current_player = self.human
        self.update_status()

    def update_status(self):
        if self.game_over:
            return
        if self.current_player == self.human:
            self.status['text'] = "Ruch: Użytkownik (X)"
        else:
            self.status['text'] = "Ruch: Komputer (O)"

    def end_game(self, winner):
        self.game_over = True
        if winner == self.human:
            self.status['text'] = "Koniec gry: Wygrał użytkownik (X)"
        elif winner == self.computer:
            self.status['text'] = "Koniec gry: Wygrał komputer (O)"
        else:
            self.status['text'] = "Koniec gry: Remis"
        for r in range(self.size):
            for c in range(self.size):
                self.buttons[r][c]['state'] = 'disabled'

    def is_full(self, board):
        for r in range(self.size):
            for c in range(self.size):
                if board[r][c] == '':
                    return False
        return True

    def check_winner(self, board):
        target = 3

        for r in range(self.size):
            for c in range(self.size - target + 1):
                segment = board[r][c:c+target]
                if all(cell == self.human for cell in segment):
                    return self.human
                if all(cell == self.computer for cell in segment):
                    return self.computer

        for r in range(self.size - target + 1):
            for c in range(self.size):
                segment = [board[r+i][c] for i in range(target)]
                if all(cell == self.human for cell in segment):
                    return self.human
                if all(cell == self.computer for cell in segment):
                    return self.computer

        for r in range(self.size - target + 1):
            for c in range(self.size - target + 1):
                segment = [board[r+i][c+i] for i in range(target)]
                if all(cell == self.human for cell in segment):
                    return self.human
                if all(cell == self.computer for cell in segment):
                    return self.computer

        for r in range(self.size - target + 1):
            for c in range(target - 1, self.size):
                segment = [board[r+i][c-i] for i in range(target)]
                if all(cell == self.human for cell in segment):
                    return self.human
                if all(cell == self.computer for cell in segment):
                    return self.computer

        return None


    def heuristic(self, board):
        score = 0
        lines = []
        for r in range(self.size):
            lines.append([board[r][c] for c in range(self.size)])
        for c in range(self.size):
            lines.append([board[r][c] for r in range(self.size)])
        lines.append([board[i][i] for i in range(self.size)])
        lines.append([board[i][self.size-1-i] for i in range(self.size)])
        for line in lines:
            if all(cell != self.human for cell in line):
                cnt = sum(1 for cell in line if cell == self.computer)
                score += cnt*cnt
            if all(cell != self.computer for cell in line):
                cnt = sum(1 for cell in line if cell == self.human)
                score -= cnt*cnt
        return score

    def minimax(self, board, depth, maximizing):
        winner = self.check_winner(board)
        if winner == self.computer:
            return 1000 - depth, None
        if winner == self.human:
            return -1000 + depth, None
        if self.is_full(board):
            return 0, None
        if depth >= self.max_depth:
            return self.heuristic(board), None
        if maximizing:
            best_score = -math.inf
            best_move = None
            for r in range(self.size):
                for c in range(self.size):
                    if board[r][c] == '':
                        board[r][c] = self.computer
                        score, _ = self.minimax(board, depth+1, False)
                        board[r][c] = ''
                        if score > best_score:
                            best_score = score
                            best_move = (r, c)
            return best_score, best_move
        else:
            best_score = math.inf
            best_move = None
            for r in range(self.size):
                for c in range(self.size):
                    if board[r][c] == '':
                        board[r][c] = self.human
                        score, _ = self.minimax(board, depth+1, True)
                        board[r][c] = ''
                        if score < best_score:
                            best_score = score
                            best_move = (r, c)
            return best_score, best_move

    def find_best_move(self, board, max_depth):
        self.max_depth = max_depth
        _, move = self.minimax(board, 0, True)
        return move

if __name__ == "__main__":
    root = tk.Tk()
    app = TicTacToe(root)
    root.mainloop()
