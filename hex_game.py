class HexGame:
    def __init__(self, board_size=BOARD_SIZE):
        self.board_size = board_size
        self.board = self.create_board()
        self.current_player = PLAYER1
        self.moves_made = 0
        self.move_history = []  # Record of moves

        # Center cell coordinates
        center = self.board_size - 1
        self.center_cell = (center, center)

        # Precompute edge cells (excluding center cell)
        self.edge_cells = self.get_edge_cells()

    def create_board(self):
        size = 2 * self.board_size - 1
        board = np.full((size, size), -1, dtype=int)
        for i in range(size):
            for j in range(size):
                if self.is_valid_position_static(i, j, self.board_size):
                    board[i, j] = EMPTY
        return board

    @staticmethod
    def is_valid_position_static(x, y, board_size):
        size = 2 * board_size - 1
        return 0 <= x < size and 0 <= y < size and \
               max(abs(x - (board_size - 1)), abs(y - (board_size - 1)), abs((x + y) - (board_size - 1) * 2)) < board_size

    def get_edge_cells(self):
        size = 2 * self.board_size - 1
        edge_cells = []
        for i in range(size):
            for j in range(size):
                if self.is_valid_position_static(i, j, self.board_size):
                    if self.is_edge_cell(i, j):
                        edge_cells.append((i, j))
        # Ensure center cell is not in edge cells
        edge_cells = [cell for cell in edge_cells if cell != self.center_cell]
        return edge_cells

    def get_valid_moves(self):
        moves = []
        indices = np.argwhere(self.board == EMPTY)
        for idx in indices:
            moves.append((idx[0], idx[1]))
        return moves

    def make_move(self, x, y):
        if self.board[x, y] != EMPTY:
            return False
        self.board[x, y] = self.current_player
        self.moves_made += 1
        # Record the move
        self.move_history.append({'player': self.current_player, 'position': (x, y)})
        # Switch player
        self.current_player = PLAYER1 if self.current_player == PLAYER2 else PLAYER2
        return True

    def is_game_over(self):
        return self.moves_made == np.sum(self.board != -1)

    def get_winner(self):
        scores = self.calculate_scores()
        if scores['total'][PLAYER1] > scores['total'][PLAYER2]:
            return PLAYER1
        elif scores['total'][PLAYER2] > scores['total'][PLAYER1]:
            return PLAYER2
        else:
            return 0  # Draw

    def calculate_scores(self):
        """
        Calculate the scores based on the iterative flipping of dead groups.
        """
        # Step 1: Initialize score_state board
        max_player = PLAYER1
        min_player = PLAYER2
        score_state = np.where(self.board == min_player, min_player, max_player)

        done = False
        iterations = 0
        max_iterations = 1000  # To prevent infinite loops

        while not done and iterations < max_iterations:
            done = True
            iterations += 1

            # Step 2: Assign groups based on score_state
            groups, num_groups = self.identify_groups_score_state(score_state)

            # Step 3: Count edge cells for each group
            num_edge_nodes = {max_player: [0] * num_groups[max_player],
                            min_player: [0] * num_groups[min_player]}

            size = score_state.shape[0]
            for x in range(size):
                for y in range(size):
                    if score_state[x, y] in (max_player, min_player):
                        if self.is_edge_cell(x, y):
                            player = score_state[x, y]
                            group_id = groups[player][(x, y)]
                            num_edge_nodes[player][group_id] += 1

            # Step 4: Flip dead groups
            for player in [max_player, min_player]:
                opponent = min_player if player == max_player else max_player
                for (x, y), group_id in groups[player].items():
                    if num_edge_nodes[player][group_id] < 2:
                        score_state[x, y] = opponent
                        done = False  # We made a change, need another iteration

        # Step 5: Calculate edge cells and center cell
        scores = {
            'edge_cells': {PLAYER1: 0, PLAYER2: 0},
            'center_cell': {PLAYER1: 0, PLAYER2: 0},
            'bonus': {PLAYER1: 0, PLAYER2: 0},
            'total': {PLAYER1: 0, PLAYER2: 0}
        }

        num_edges = {PLAYER1: 0, PLAYER2: 0}

        size = score_state.shape[0]
        num_edge_cells = 0
        for x in range(size):
            for y in range(size):
                player = score_state[x, y]
                if player in (PLAYER1, PLAYER2):
                    if self.is_edge_cell(x, y):
                        num_edge_cells += 1
                        num_edges[player] += 1
                    if (x, y) == self.center_cell:
                        scores['center_cell'][player] = 1  # Center cell bonus
        print("num edge cells", num_edge_cells)
        assert(num_edge_cells == 6 * (self.board_size - 1))
        assert(num_edges[PLAYER1] + num_edges[PLAYER2] == 6 * (self.board_size - 1))
        scores['edge_cells'] = num_edges

        # Step 6: Calculate group counts for bonus
        groups, num_groups = self.identify_groups_score_state(score_state)

        # Calculate group bonus
        group_bonus_p1 = 2 * (num_groups[PLAYER2] - num_groups[PLAYER1])
        group_bonus_p2 = 2 * (num_groups[PLAYER1] - num_groups[PLAYER2])
        scores['bonus'][PLAYER1] += group_bonus_p1
        scores['bonus'][PLAYER2] += group_bonus_p2

        # Step 7: Calculate total scores
        for player in [PLAYER1, PLAYER2]:
            scores['total'][player] = (
                scores['edge_cells'][player] +
                scores['center_cell'][player] +
                scores['bonus'][player]
            )

        # Assert total score matches expected total
        edge_cells_count = 6 * (self.board_size - 1)
        total_possible_points = edge_cells_count + 1  # +1 for the center cell

        total_score = scores['total'][PLAYER1] + scores['total'][PLAYER2]
        assert total_score == total_possible_points, f"Total score {total_score} does not match expected {total_possible_points}"

        return scores


    def identify_groups_score_state(self, score_state):
        """
        Identify groups in the score_state board.
        Returns:
            groups: {PLAYER1: {(x, y): group_id}, PLAYER2: {(x, y): group_id}}
            num_groups: {PLAYER1: int, PLAYER2: int}
        """
        visited = set()
        groups = {PLAYER1: {}, PLAYER2: {}}
        group_ids = {PLAYER1: 0, PLAYER2: 0}
        size = score_state.shape[0]

        for x in range(size):
            for y in range(size):
                player = score_state[x, y]
                if player in (PLAYER1, PLAYER2) and (x, y) not in visited:
                    group_id = group_ids[player]
                    stack = [(x, y)]
                    while stack:
                        cx, cy = stack.pop()
                        if (cx, cy) in visited:
                            continue
                        if score_state[cx, cy] == player:
                            visited.add((cx, cy))
                            groups[player][(cx, cy)] = group_id
                            neighbors = self.get_neighbors(cx, cy, score_state)
                            for nx, ny in neighbors:
                                if score_state[nx, ny] == player and (nx, ny) not in visited:
                                    stack.append((nx, ny))
                    group_ids[player] += 1

        num_groups = {PLAYER1: group_ids[PLAYER1], PLAYER2: group_ids[PLAYER2]}
        return groups, num_groups

    def is_edge_cell(self, x, y):
        n = self.board_size
        size = 2 * n - 1
        return (
            x == 0 or
            y == 0 or
            x == size - 1 or
            y == size - 1 or
            (x + y) == n - 1 or
            (x + y) == (3 * n - 3)
        )

    def get_neighbors(self, x, y, board):
        # Hexagonal grid neighbors
        deltas = [(-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0)]
        neighbors = []
        for dx, dy in deltas:
            nx, ny = x + dx, y + dy
            if 0 <= nx < board.shape[0] and 0 <= ny < board.shape[1]:
                if board[nx, ny] != -1:
                    neighbors.append((nx, ny))
        return neighbors

    def copy(self):
        return deepcopy(self)

    def reset(self):
        self.board = self.create_board()
        self.current_player = PLAYER1
        self.moves_made = 0
        self.move_history = []

    def get_state(self):
        state = np.copy(self.board)
        state[state == -1] = EMPTY
        return state

    def render(self, save_path=None, show=False):
        """
        Visualize the board using Matplotlib.
        """
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_aspect('equal')
        ax.axis('off')

        size = self.board.shape[0]
        hex_radius = 1  # Adjust the size of the hexagons

        for i in range(size):
            for j in range(size):
                if self.board[i, j] != -1:
                    # Calculate the positions for a hexagonal grid
                    x_offset = (i - (self.board_size - 1)) * 1.5 * hex_radius
                    y_offset = (j - (self.board_size - 1)) * np.sqrt(3) * hex_radius + (i - (self.board_size - 1)) * np.sqrt(3)/2 * hex_radius

                    hex_patch = patches.RegularPolygon(
                        (x_offset, y_offset),
                        numVertices=6,
                        radius=hex_radius,
                        orientation=np.radians(0),
                        facecolor='lightgray',
                        edgecolor='black'
                    )

                    # Highlight center cell
                    if (i, j) == self.center_cell:
                        hex_patch.set_facecolor('yellow')

                    # Color cells based on ownership
                    if self.board[i, j] == PLAYER1:
                        hex_patch.set_facecolor('blue')
                    elif self.board[i, j] == PLAYER2:
                        hex_patch.set_facecolor('red')

                    ax.add_patch(hex_patch)

        # Calculate scores
        scores = self.calculate_scores()
        player1_score = scores['total'][PLAYER1]
        player2_score = scores['total'][PLAYER2]

        # Display detailed scores on the plot
        score_text = (
            f"Player 1 (Blue): {player1_score} "
            f"(Edges: {scores['edge_cells'][PLAYER1]}, "
            f"Center: {scores['center_cell'][PLAYER1]}, "
            f"Bonus: {scores['bonus'][PLAYER1]})\n"
            f"Player 2 (Red): {player2_score} "
            f"(Edges: {scores['edge_cells'][PLAYER2]}, "
            f"Center: {scores['center_cell'][PLAYER2]}, "
            f"Bonus: {scores['bonus'][PLAYER2]})"
        )
        plt.title(score_text, fontsize=12)

        ax.relim()
        ax.autoscale_view()

        if save_path:
            plt.savefig(save_path)
        if show:
            plt.show()
        plt.close(fig)

    def save_move_history(self, filename):
        with open(filename, 'w') as f:
            for move in self.move_history:
                player = 'Player 1' if move['player'] == PLAYER1 else 'Player 2'
                position = move['position']
                f.write(f"{player}: {position}\n")

    def replay_game(self):
        # Optional: Implement replay functionality if needed
        pass
