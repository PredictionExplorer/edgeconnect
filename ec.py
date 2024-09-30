from collections import deque, defaultdict
from copy import deepcopy
import glob
import logging
import math
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import warnings

# =========================
# Configuration and Constants
# =========================

# Configure logging
logging.basicConfig(
    filename='training.log',
    filemode='a',
    format='%(asctime)s %(levelname)s: %(message)s',
    level=logging.INFO
)

# Suppress FutureWarnings from torch.load
warnings.simplefilter('ignore', category=FutureWarning)

# Set the device for PyTorch
if torch.backends.mps.is_available():
    device = torch.device('mps')
    print("Using MPS device")
elif torch.cuda.is_available():
    device = torch.device('cuda')
    print("Using CUDA device")
else:
    device = torch.device('cpu')
    print("Using CPU device")

# Game Constants
BOARD_SIZE = 5  # Radius of the hexagonal board
NUM_PLAYERS = 2
EMPTY = 0
PLAYER1 = 1
PLAYER2 = 2

# Training Parameters
LEARNING_RATE = 1e-2
BATCH_SIZE = 256
MEMORY_SIZE = 100000  # Increased for more training data
NUM_EPISODES = 1000
MCTS_SIMULATIONS = 800  # Similar to AlphaZero's approach
CPUCT = 1.0  # Exploration constant
NUM_RES_BLOCKS = 5  # Reduced depth for optimization
NUM_FILTERS = 64  # Reduced width for optimization
MOMENTUM = 0.9

# Learning Rate Scheduler Parameters
T_MAX = NUM_EPISODES

# =========================
# Helper Functions
# =========================

def move_to_index(move, board_size=BOARD_SIZE):
    x, y = move
    size = 2 * board_size - 1
    index = 0
    for i in range(size):
        for j in range(size):
            if HexGame.is_valid_position_static(i, j, board_size):
                if (i, j) == (x, y):
                    return index
                index += 1
    return -1  # Invalid move

def index_to_move(index, board_size=BOARD_SIZE):
    size = 2 * board_size - 1
    idx = 0
    for i in range(size):
        for j in range(size):
            if HexGame.is_valid_position_static(i, j, board_size):
                if idx == index:
                    return (i, j)
                idx += 1
    return None  # Invalid index

def state_to_tensor(state, current_player):
    """
    Convert the board state to a tensor.
    Create two channels:
        - Channel 0: Current player's stones
        - Channel 1: Opponent's stones
    """
    state_tensor = np.zeros((2, state.shape[0], state.shape[1]), dtype=np.float32)
    state_tensor[0] = (state == current_player).astype(np.float32)
    state_tensor[1] = (state == (3 - current_player)).astype(np.float32)
    return torch.from_numpy(state_tensor).unsqueeze(0)  # Shape: [1, 2, size, size]

def augment_data(state, pi, board_size=BOARD_SIZE):
    """
    Apply rotational and reflectional transformations to the state and policy.
    """
    augmented_data = []
    state_np = state.squeeze(0).numpy()
    size = 2 * board_size - 1

    # Create a full board representation of pi
    pi_full = np.full((size, size), -1, dtype=np.float32)
    index = 0
    for i in range(size):
        for j in range(size):
            if HexGame.is_valid_position_static(i, j, board_size):
                pi_full[i, j] = pi[index]
                index += 1

    for k in range(4):
        # Rotate state and pi
        rotated_state = np.rot90(state_np, k, axes=(1, 2))
        rotated_pi_full = np.rot90(pi_full, k)
        # Flatten pi back to valid positions
        pi_aug = []
        for i in range(size):
            for j in range(size):
                if HexGame.is_valid_position_static(i, j, board_size):
                    pi_aug.append(rotated_pi_full[i, j])
        augmented_data.append((torch.from_numpy(rotated_state.copy()), np.array(pi_aug)))

        # Flip state horizontally
        flipped_state = np.flip(rotated_state, axis=2)
        flipped_pi_full = np.flip(rotated_pi_full, axis=1)
        # Flatten pi back to valid positions
        pi_aug = []
        for i in range(size):
            for j in range(size):
                if HexGame.is_valid_position_static(i, j, board_size):
                    pi_aug.append(flipped_pi_full[i, j])
        augmented_data.append((torch.from_numpy(flipped_state.copy()), np.array(pi_aug)))

    return augmented_data

# =========================
# HexGame Class
# =========================

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

    def is_edge_cell(self, x, y):
        size = 2 * self.board_size - 1
        return x == 0 or y == 0 or x == size - 1 or y == size - 1 or \
               (x + y) == (self.board_size - 1) * 2 or \
               (x + y) == 0

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
        Implement the scoring logic as per your game rules.
        For simplicity, let's assume the player with more stones on the board wins.
        """
        player1_score = np.sum(self.board == PLAYER1)
        player2_score = np.sum(self.board == PLAYER2)
        scores = {
            'total': {
                PLAYER1: player1_score,
                PLAYER2: player2_score
            }
        }
        return scores

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
        hex_height = np.sqrt(3) * hex_radius  # Height of a hexagon
        hex_width = 2 * hex_radius  # Width of a hexagon

        for i in range(size):
            for j in range(size):
                if self.board[i, j] != -1:
                    # Calculate the offset positions for a hexagonal grid
                    x_offset = (j - (self.board_size - 1)) * (hex_width * 0.75)
                    y_offset = -(i - (self.board_size - 1)) * (hex_height) + (j - (self.board_size - 1)) * (hex_height / 2)

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

        # Display scores on the plot
        score_text = f"Player 1 (Blue): {player1_score}    Player 2 (Red): {player2_score}"
        plt.title(score_text, fontsize=16)

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

# =========================
# Neural Network Classes
# =========================

class HexNet(nn.Module):
    def __init__(self, board_size, num_res_blocks=NUM_RES_BLOCKS, num_filters=NUM_FILTERS):
        """
        HexNet Neural Network with residual blocks.
        Args:
            board_size (int): Size of the board.
            num_res_blocks (int): Number of residual blocks.
            num_filters (int): Number of convolutional filters.
        """

    def __init__(self, board_size, num_res_blocks=NUM_RES_BLOCKS, num_filters=NUM_FILTERS):
        super(HexNet, self).__init__()
        self.board_size = board_size
        self.input_channels = 2
        self.num_actions = np.sum([HexGame.is_valid_position_static(i, j, board_size)
                                   for i in range(2 * board_size - 1)
                                   for j in range(2 * board_size - 1)])

        # Initial convolutional layer
        self.conv1 = nn.Conv2d(self.input_channels, num_filters, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_filters)

        # Residual blocks
        self.res_blocks = nn.ModuleList([ResidualBlock(num_filters) for _ in range(num_res_blocks)])

        # Policy head
        self.policy_head = PolicyHead(num_filters, self.num_actions, board_size)

        # Value head
        self.value_head = ValueHead(num_filters, board_size)

    def forward(self, x):
        x = x.to(device)
        x = F.silu(self.bn1(self.conv1(x)))  # Using SiLU activation function
        for res_block in self.res_blocks:
            x = res_block(x)
        # Policy and value heads
        policy = self.policy_head(x)
        value = self.value_head(x)
        return policy, value

class ResidualBlock(nn.Module):
    def __init__(self, num_filters):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.dropout = nn.Dropout2d(p=0.3)
        self.conv2 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_filters)

    def forward(self, x):
        residual = x
        out = F.silu(self.bn1(self.conv1(x)))  # Using SiLU activation function
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out += residual
        return F.silu(out)

class PolicyHead(nn.Module):
    def __init__(self, num_filters, num_actions, board_size):
        super(PolicyHead, self).__init__()
        self.conv = nn.Conv2d(num_filters, 2, kernel_size=1)
        self.bn = nn.BatchNorm2d(2)
        self.fc = nn.Linear(2 * (2 * board_size - 1) * (2 * board_size - 1), num_actions)

    def forward(self, x):
        x = F.silu(self.bn(self.conv(x)))  # Using SiLU activation function
        x = x.view(-1, self.num_flat_features(x))
        x = self.fc(x)
        return F.log_softmax(x, dim=1)  # Log probabilities

    def num_flat_features(self, x):
        size = x.size()[1:]  # Exclude batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

class ValueHead(nn.Module):
    def __init__(self, num_filters, board_size):
        super(ValueHead, self).__init__()
        self.conv = nn.Conv2d(num_filters, 1, kernel_size=1)
        self.bn = nn.BatchNorm2d(1)
        self.fc1 = nn.Linear((2 * board_size - 1) * (2 * board_size - 1), 256)
        self.fc2 = nn.Linear(256, 1)

    def forward(self, x):
        x = F.silu(self.bn(self.conv(x)))  # Using SiLU activation function
        x = x.view(-1, self.num_flat_features(x))
        x = F.silu(self.fc1(x))
        x = torch.tanh(self.fc2(x))  # Output between -1 and 1
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # Exclude batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

# =========================
# Monte Carlo Tree Search (MCTS)
# =========================

class MCTSNode:
    def __init__(self, game_state, parent=None, move=None):
        self.game_state = game_state  # HexGame instance
        self.parent = parent  # Parent node
        self.move = move  # Move that led to this node
        self.children = {}  # Dictionary: move -> child_node
        self.visit_count = 0
        self.total_value = 0
        self.prior = 0  # Prior probability from policy network

class MCTS:
    def __init__(self, neural_net, num_simulations=MCTS_SIMULATIONS, cpuct=CPUCT):
        self.neural_net = neural_net
        self.num_simulations = num_simulations
        self.cpuct = cpuct

    def evaluate(self, game_states):
        """
        Evaluate a batch of game states using the neural network.
        Returns:
            policies (list): List of policy probabilities for each game state.
            values (list): List of value estimates for each game state.
        """
        state_tensors = []
        for game in game_states:
            state_tensor = state_to_tensor(game.get_state(), game.current_player)
            state_tensors.append(state_tensor)
        state_tensors = torch.cat(state_tensors).to(device)
        with torch.no_grad():
            policies, values = self.neural_net(state_tensors)
            policies = policies.exp().cpu().numpy()
            values = values.cpu().numpy().flatten()
        return policies, values

    def get_action_probs(self, game, temp=1):
        """
        Run MCTS simulations and return action probabilities.
        """
        root = MCTSNode(game.copy())

        # Collect leaf nodes for batch evaluation
        game_states = []
        nodes = []

        for _ in range(self.num_simulations):
            node = root
            state = game.copy()

            # Selection
            while node.children:
                move, node = self.select_child(node)
                state.make_move(*move)

            nodes.append(node)
            game_states.append(state)

        # Batch evaluation
        policies, values = self.evaluate(game_states)

        for node, policy, value in zip(nodes, policies, values):
            state = node.game_state
            if not state.is_game_over():
                valid_moves = state.get_valid_moves()
                policy_probs = {}
                for move in valid_moves:
                    index = move_to_index(move, state.board_size)
                    policy_probs[move] = policy[index]
                policy_sum = sum(policy_probs.values())
                if policy_sum > 0:
                    for move in policy_probs:
                        policy_probs[move] /= policy_sum
                else:
                    # All valid moves have zero probability, assign equal probability
                    num_moves = len(valid_moves)
                    for move in policy_probs:
                        policy_probs[move] = 1 / num_moves
                for move in valid_moves:
                    child_game = state.copy()
                    child_game.make_move(*move)
                    child_node = MCTSNode(child_game, parent=node, move=move)
                    child_node.prior = policy_probs[move]
                    node.children[move] = child_node
            else:
                # Terminal state
                winner = state.get_winner()
                if winner == state.current_player:
                    value = 1
                elif winner == 0:
                    value = 0
                else:
                    value = -1

            # Backpropagation
            self.backpropagate(node, value)

        # Aggregate visit counts
        visit_counts = defaultdict(int)
        for move, child in root.children.items():
            visit_counts[move] = child.visit_count

        # Apply temperature
        if temp == 0:
            best_move = max(visit_counts.items(), key=lambda x: x[1])[0]
            probs = {best_move: 1.0}
        else:
            visits = np.array(list(visit_counts.values()), dtype=np.float64)
            moves = list(visit_counts.keys())
            visits = visits ** (1 / temp)
            visits_sum = np.sum(visits)
            if visits_sum > 0:
                probs = {move: visit / visits_sum for move, visit in zip(moves, visits)}
            else:
                # Assign equal probability if visits_sum is zero
                num_moves = len(moves)
                probs = {move: 1 / num_moves for move in moves}

        return probs

    def select_child(self, node):
        """
        Select a child node with the highest UCB value.
        """
        best_move, best_node = None, None
        max_ucb = -float('inf')

        for move, child in node.children.items():
            ucb = self.compute_ucb(child, node.visit_count)
            if ucb > max_ucb:
                max_ucb = ucb
                best_move = move
                best_node = child

        return best_move, best_node

    def compute_ucb(self, child, parent_visits):
        if child.visit_count == 0:
            ucb = float('inf')
        else:
            ucb = (child.total_value / child.visit_count) + \
                  self.cpuct * child.prior * math.sqrt(parent_visits) / (1 + child.visit_count)
        return ucb

    def backpropagate(self, node, value):
        """
        Backpropagate the value up the tree.
        """
        while node is not None:
            node.visit_count += 1
            node.total_value += value
            value = -value  # Switch perspective
            node = node.parent

# =========================
# Training Loop
# =========================

def get_latest_model_path():
    model_files = glob.glob('hex_net_episode_*.pth')
    if model_files:
        # Extract episode numbers and find the latest
        episodes = [int(f.split('_episode_')[1].split('.pth')[0]) for f in model_files]
        latest_episode = max(episodes)
        latest_model_path = f'hex_net_episode_{latest_episode}.pth'
        return latest_model_path
    else:
        return None

def evaluate_model(neural_net, num_games=10):
    """
    Evaluate the current model by playing games against a random opponent.
    """
    wins = 0
    for _ in range(num_games):
        game = HexGame()
        mcts = MCTS(neural_net, num_simulations=MCTS_SIMULATIONS, cpuct=CPUCT)
        while not game.is_game_over():
            if game.current_player == PLAYER1:
                # Use MCTS to select move
                action_probs = mcts.get_action_probs(game, temp=0)
                moves = list(action_probs.keys())
                probs = list(action_probs.values())
                move = moves[np.argmax(probs)]
            else:
                # Random move for the opponent
                valid_moves = game.get_valid_moves()
                move = random.choice(valid_moves)
            game.make_move(*move)
        winner = game.get_winner()
        if winner == PLAYER1:
            wins += 1
    win_rate = wins / num_games
    return win_rate

def train_agent():
    # Initialize neural network
    neural_net = HexNet(BOARD_SIZE, num_res_blocks=NUM_RES_BLOCKS, num_filters=NUM_FILTERS).to(device)

    # Optimizer and Scheduler
    optimizer = optim.SGD(neural_net.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM, nesterov=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_MAX)

    # Load the latest model and optimizer states if available
    latest_model_path = get_latest_model_path()
    if latest_model_path and os.path.exists(latest_model_path):
        checkpoint = torch.load(latest_model_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            neural_net.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_episode = checkpoint['episode'] + 1
            logging.info(f"Resuming training from episode {start_episode}")
        else:
            start_episode = 1
            logging.info("Starting training from scratch")
    else:
        start_episode = 1
        logging.info("Starting training from scratch")

    # Replay memory
    memory = deque(maxlen=MEMORY_SIZE)

    # MCTS instance
    mcts = MCTS(neural_net, num_simulations=MCTS_SIMULATIONS, cpuct=CPUCT)

    # Initialize variables for logging
    total_policy_loss = 0
    total_value_loss = 0
    total_loss = 0
    episodes_since_last_log = 0
    start_time = time.time()

    for episode in range(start_episode, NUM_EPISODES + 1):
        game = HexGame()
        training_examples = []

        while not game.is_game_over():
            # Adjust MCTS simulations dynamically
            total_moves = len(game.get_valid_moves())
            simulations = int(MCTS_SIMULATIONS * (total_moves / ((2 * BOARD_SIZE - 1) ** 2)))
            mcts.num_simulations = max(simulations, 100)

            # Get action probabilities from MCTS
            action_probs = mcts.get_action_probs(game, temp=1)

            # Choose action according to probabilities
            moves = list(action_probs.keys())
            probs = list(action_probs.values())
            if any(np.isnan(probs)) or sum(probs) == 0:
                # Assign equal probability if probs are invalid
                probs = [1 / len(probs)] * len(probs)
            move = random.choices(moves, weights=probs)[0]

            # Record training data
            state_tensor = state_to_tensor(game.get_state(), game.current_player)
            pi = np.zeros(neural_net.num_actions, dtype=np.float32)
            for move, prob in action_probs.items():
                index = move_to_index(move, game.board_size)
                pi[index] = prob
            training_examples.append((state_tensor, pi, game.current_player))

            # Make the move
            game.make_move(*move)

        for state, pi, player in training_examples:
            if winner == player:
                reward = 1
            elif winner == 0:
                reward = 0
            else:
                reward = -1

            augmented_data = augment_data(state, pi, game.board_size)
            for aug_state, aug_pi in augmented_data:
                memory.append((aug_state.unsqueeze(0), aug_pi, reward))

        # Training step
        if len(memory) >= BATCH_SIZE:
            # Sample a batch
            batch = random.sample(memory, BATCH_SIZE)
            states = torch.cat([item[0] for item in batch]).to(device)

            # Prepare target policy and value
            target_pis = torch.zeros((BATCH_SIZE, neural_net.num_actions), dtype=torch.float32).to(device)
            target_vs = torch.zeros(BATCH_SIZE, dtype=torch.float32).to(device)

            for idx, (state, pi, reward) in enumerate(batch):
                target_pis[idx] = torch.from_numpy(pi).to(device)
                target_vs[idx] = reward

            # Forward pass
            out_pis, out_vs = neural_net(states)

            # Compute loss
            loss_pis = -torch.mean(torch.sum(target_pis * out_pis, dim=1))
            loss_vs = F.mse_loss(out_vs.view(-1), target_vs)
            loss = loss_pis + loss_vs

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(neural_net.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            # Accumulate losses for logging
            total_policy_loss += loss_pis.item()
            total_value_loss += loss_vs.item()
            total_loss += loss.item()
            episodes_since_last_log += 1

        # Periodic logging
        if episode % 10 == 0:
            avg_policy_loss = total_policy_loss / episodes_since_last_log if episodes_since_last_log else 0
            avg_value_loss = total_value_loss / episodes_since_last_log if episodes_since_last_log else 0
            avg_total_loss = total_loss / episodes_since_last_log if episodes_since_last_log else 0
            elapsed_time = time.time() - start_time

            logging.info(
                f"Episode: {episode}, "
                f"Avg Policy Loss: {avg_policy_loss:.4f}, "
                f"Avg Value Loss: {avg_value_loss:.4f}, "
                f"Avg Total Loss: {avg_total_loss:.4f}, "
                f"Elapsed Time: {elapsed_time:.2f}s"
            )

            # Reset accumulators
            total_policy_loss = 0
            total_value_loss = 0
            total_loss = 0
            episodes_since_last_log = 0
            start_time = time.time()

        # Periodic evaluation
        if episode % 50 == 0:
            win_rate = evaluate_model(neural_net)
            logging.info(f"Episode: {episode}, Win Rate against Random: {win_rate:.2f}")

        # Periodic saving
        if episode % 100 == 0:
            model_path = f"hex_net_episode_{episode}.pth"
            torch.save({
                'episode': episode,
                'model_state_dict': neural_net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, model_path)

            # Save move history and board visualization
            moves_filename = f"game_episode_{episode}_moves.txt"
            game.save_move_history(moves_filename)

            image_filename = f"game_episode_{episode}_board.png"
            game.render(save_path=image_filename)

            logging.info(f"Episode {episode}/{NUM_EPISODES} completed. Model saved to {model_path}.")

    # Save the final model
    torch.save({
        'episode': NUM_EPISODES,
        'model_state_dict': neural_net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, "hex_net_final.pth")
    logging.info("Training completed. Final model saved.")

# =========================
# Main Execution
# =========================

if __name__ == "__main__":
    # Start training
    train_agent()

