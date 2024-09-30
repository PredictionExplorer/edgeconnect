import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Game Constants
BOARD_SIZE = 5  # Radius of the hexagonal board
NUM_PLAYERS = 2
EMPTY = 0
PLAYER1 = 1
PLAYER2 = 2

# Neural Network Parameters
LEARNING_RATE = 0.001
GAMMA = 0.99  # Discount factor
EPSILON_START = 1.0
EPSILON_END = 0.1
EPSILON_DECAY = 0.995
BATCH_SIZE = 32
MEMORY_SIZE = 10000
NUM_EPISODES = 1000

class HexGame:
    def __init__(self, board_size=BOARD_SIZE):
        self.board_size = board_size
        self.board = self.create_board()
        self.current_player = PLAYER1
        self.moves_made = 0
        self.move_history = []  # Record of moves
        self.fig = None  # For visualization
        self.ax = None

        # Center cell coordinates
        center = self.board_size - 1
        self.center_cell = (center, center)

    def create_board(self):
        size = 2 * self.board_size - 1
        board = np.full((size, size), -1, dtype=int)
        for i in range(size):
            for j in range(size):
                if self.is_valid_position(i, j):
                    board[i, j] = EMPTY
        return board

    def is_valid_position(self, x, y):
        size = 2 * self.board_size - 1
        return 0 <= x < size and 0 <= y < size and \
               max(abs(x - (self.board_size - 1)), abs(y - (self.board_size - 1)), abs((x + y) - (self.board_size - 1) * 2)) < self.board_size

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
        if scores[PLAYER1] > scores[PLAYER2]:
            return PLAYER1
        elif scores[PLAYER2] > scores[PLAYER1]:
            return PLAYER2
        else:
            return 0  # Draw

    def calculate_scores(self):
        # Implementing detailed scoring logic

        # Initialize scores and group counts
        scores = {PLAYER1: 0, PLAYER2: 0}
        group_counts = {PLAYER1: 0, PLAYER2: 0}

        # Find connected groups for each player
        visited = np.zeros_like(self.board, dtype=bool)
        for i in range(self.board.shape[0]):
            for j in range(self.board.shape[1]):
                player = self.board[i, j]
                if player in [PLAYER1, PLAYER2] and not visited[i, j]:
                    group_size = self.flood_fill(i, j, player, visited)
                    group_counts[player] += 1

        # Penalty for more groups
        group_diff = group_counts[PLAYER1] - group_counts[PLAYER2]
        if group_diff > 0:
            scores[PLAYER1] -= group_diff
        elif group_diff < 0:
            scores[PLAYER2] += group_diff  # Negative penalty for PLAYER1

        # Center cell bonus
        center_x, center_y = self.center_cell
        center_owner = self.board[center_x, center_y]
        if center_owner in [PLAYER1, PLAYER2]:
            scores[center_owner] += 1  # Center cell bonus

        # Total stones as base score
        scores[PLAYER1] += np.sum(self.board == PLAYER1)
        scores[PLAYER2] += np.sum(self.board == PLAYER2)

        return scores

    def flood_fill(self, x, y, player, visited):
        stack = [(x, y)]
        visited[x, y] = True
        group_size = 0

        while stack:
            cx, cy = stack.pop()
            group_size += 1
            neighbors = self.get_neighbors(cx, cy)
            for nx, ny in neighbors:
                if not visited[nx, ny] and self.board[nx, ny] == player:
                    visited[nx, ny] = True
                    stack.append((nx, ny))

        return group_size

    def get_neighbors(self, x, y):
        # Hexagonal grid neighbors
        deltas = [(-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0)]
        neighbors = []
        for dx, dy in deltas:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.board.shape[0] and 0 <= ny < self.board.shape[1]:
                if self.board[nx, ny] != -1:
                    neighbors.append((nx, ny))
        return neighbors

    def reset(self):
        self.board = self.create_board()
        self.current_player = PLAYER1
        self.moves_made = 0
        self.move_history = []
        self.fig = None
        self.ax = None

    def get_state(self):
        state = np.copy(self.board)
        state[state == -1] = EMPTY
        return state

    def render(self, save_path=None, show=False):
        # Visualize the board using Matplotlib
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_aspect('equal')
        ax.axis('off')

        size = self.board.shape[0]
        for i in range(size):
            for j in range(size):
                if self.board[i, j] != -1:
                    # Position calculation for hex grid
                    x_offset = (i - (self.board_size - 1)) * 0.75
                    y_offset = (j - (self.board_size - 1)) + 0.5 * (i - (self.board_size - 1))

                    hex_patch = patches.RegularPolygon(
                        (x_offset, y_offset),
                        numVertices=6,
                        radius=0.5,
                        orientation=np.radians(0),
                        facecolor='lightgray',
                        edgecolor='black'
                    )
                    ax.add_patch(hex_patch)

                    # Highlight center cell
                    if (i, j) == self.center_cell:
                        hex_patch.set_facecolor('yellow')

                    # Add stones
                    if self.board[i, j] == PLAYER1:
                        ax.text(x_offset, y_offset, 'X', ha='center', va='center', fontsize=14, color='blue')
                    elif self.board[i, j] == PLAYER2:
                        ax.text(x_offset, y_offset, 'O', ha='center', va='center', fontsize=14, color='red')

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
        # Replay the game using Matplotlib animation
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_aspect('equal')
        ax.axis('off')

        ims = []
        temp_board = self.create_board()
        size = temp_board.shape[0]

        for move in self.move_history:
            player = move['player']
            x, y = move['position']
            temp_board[x, y] = player

            # Plot the board
            ax.clear()
            ax.set_aspect('equal')
            ax.axis('off')
            for i in range(size):
                for j in range(size):
                    if temp_board[i, j] != -1:
                        x_offset = (i - (self.board_size - 1)) * 0.75
                        y_offset = (j - (self.board_size - 1)) + 0.5 * (i - (self.board_size - 1))

                        hex_patch = patches.RegularPolygon(
                            (x_offset, y_offset),
                            numVertices=6,
                            radius=0.5,
                            orientation=np.radians(0),
                            facecolor='lightgray',
                            edgecolor='black'
                        )
                        ax.add_patch(hex_patch)

                        # Highlight center cell
                        if (i, j) == self.center_cell:
                            hex_patch.set_facecolor('yellow')

                        # Add stones
                        if temp_board[i, j] == PLAYER1:
                            ax.text(x_offset, y_offset, 'X', ha='center', va='center', fontsize=14, color='blue')
                        elif temp_board[i, j] == PLAYER2:
                            ax.text(x_offset, y_offset, 'O', ha='center', va='center', fontsize=14, color='red')

            # Capture the frame
            ims.append([ax])

        ani = animation.ArtistAnimation(fig, ims, interval=500, blit=True, repeat_delay=1000)
        plt.show()

class HexAgent(nn.Module):
    def __init__(self, board_size):
        super(HexAgent, self).__init__()
        self.board_size = board_size
        conv_channels = 64
        self.conv1 = nn.Conv2d(2, conv_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(conv_channels, conv_channels, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(conv_channels * (2 * board_size - 1) * (2 * board_size - 1), 512)
        self.fc2 = nn.Linear(512, (2 * board_size - 1) * (2 * board_size - 1))
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.to(device)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = x.view(-1, self.num_flat_features(x))
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # Exclude batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

class ReplayMemory:
    def __init__(self, capacity=MEMORY_SIZE):
        self.memory = deque(maxlen=capacity)

    def push(self, transition):
        self.memory.append(transition)

    def sample(self, batch_size=BATCH_SIZE):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

def state_to_tensor(state, current_player):
    # Convert the board state to a tensor
    # Create two channels: one for the current player's stones, one for the opponent's stones
    state_tensor = np.zeros((2, state.shape[0], state.shape[1]), dtype=np.float32)
    state_tensor[0] = (state == current_player).astype(np.float32)
    state_tensor[1] = (state == (3 - current_player)).astype(np.float32)
    return torch.from_numpy(state_tensor).unsqueeze(0)  # Add batch dimension

def select_action(agent, state, valid_moves, epsilon):
    if random.random() < epsilon:
        # Explore: choose a random valid move
        return random.choice(valid_moves)
    else:
        # Exploit: choose the best move according to the policy network
        with torch.no_grad():
            q_values = agent(state.to(device))
            q_values = q_values.cpu().numpy().flatten()
            # Mask invalid moves
            move_scores = []
            for move in valid_moves:
                index = move[0] * state.shape[2] + move[1]
                move_scores.append((q_values[index], move))
            # Choose the move with the highest Q-value
            move_scores.sort(reverse=True)
            return move_scores[0][1]

def optimize_model(agent, optimizer, memory):
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample()
    batch = list(zip(*transitions))

    state_batch = torch.cat(batch[0]).to(device)
    action_batch = batch[1]
    reward_batch = torch.tensor(batch[2], dtype=torch.float32).to(device)
    next_state_batch = torch.cat(batch[3]).to(device)
    done_batch = torch.tensor(batch[4], dtype=torch.float32).to(device)

    # Compute Q(s_t, a)
    q_values = agent(state_batch)
    action_indices = torch.tensor([a[0] * state_batch.shape[2] + a[1] for a in action_batch]).unsqueeze(1).to(device)
    q_values = q_values.gather(1, action_indices)

    # Compute V(s_{t+1})
    next_q_values = agent(next_state_batch).max(1)[0].detach()
    expected_q_values = (next_q_values * GAMMA * (1 - done_batch)) + reward_batch

    # Compute loss
    loss = nn.functional.mse_loss(q_values.squeeze(), expected_q_values)

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def train_agent():
    game = HexGame()
    agent = HexAgent(game.board_size).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=LEARNING_RATE)
    memory = ReplayMemory()

    epsilon = EPSILON_START
    for episode in range(NUM_EPISODES):
        game.reset()
        state = state_to_tensor(game.get_state(), game.current_player)
        total_reward = 0
        done = False

        while not done:
            valid_moves = game.get_valid_moves()
            if not valid_moves:
                break

            action = select_action(agent, state, valid_moves, epsilon)
            x, y = action
            game.make_move(x, y)
            reward = 0  # Placeholder for reward calculation

            # Check if the game has ended
            done = game.is_game_over()
            if done:
                winner = game.get_winner()
                if winner == game.current_player:
                    reward = 1.0
                elif winner == 0:
                    reward = 0.5  # Draw
                else:
                    reward = -1.0
            else:
                reward = 0.0

            next_state = state_to_tensor(game.get_state(), game.current_player)
            memory.push((state, (x, y), reward, next_state, done))
            state = next_state
            total_reward += reward

            optimize_model(agent, optimizer, memory)

            if done:
                # Save the move history and board visualization every 100 episodes
                if (episode + 1) % 10 == 0:
                    moves_filename = f"game_{episode + 1}_moves.txt"
                    game.save_move_history(moves_filename)
                    # Visualize the final board
                    image_filename = f"game_{episode + 1}_board.png"
                    game.render(save_path=image_filename)
                    print(f"Saved game {episode + 1} moves to {moves_filename} and board to {image_filename}")
                break

        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)
        print(f"Episode {episode + 1}/{NUM_EPISODES}, Total Reward: {total_reward}, Epsilon: {epsilon:.3f}")

    # Save the trained model
    torch.save(agent.state_dict(), "hex_agent.pth")

    # Optionally, replay the last game
    print("Replaying the last game...")
    game.replay_game()

if __name__ == "__main__":
    train_agent()
