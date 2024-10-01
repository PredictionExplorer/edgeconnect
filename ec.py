# Import necessary modules
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from collections import deque, defaultdict
from copy import deepcopy
import numpy as np
import random
import time
import math
import logging
import glob
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import warnings
from hex_game import HexGame
from hex_game import PLAYER1, PLAYER2

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

# Suppress warnings
warnings.simplefilter('ignore', category=UserWarning)
warnings.simplefilter('ignore', category=FutureWarning)

# Game Constants
BOARD_SIZE = 3 # Radius of the hexagonal board

# Training Parameters
LEARNING_RATE = 1e-2
BATCH_SIZE = 256  # Increased batch size
MEMORY_SIZE = 1000
NUM_EPISODES = 1000
MCTS_SIMULATIONS = 50  # You can increase this number to make MCTS more intensive
CPUCT = 1.0  # Exploration constant
NUM_RES_BLOCKS = 10  # Increased number of residual blocks
NUM_FILTERS = 128  # Increased number of filters
MOMENTUM = 0.9

# Learning Rate Scheduler Parameters
T_MAX = NUM_EPISODES

# Adjust the number of self-play processes
NUM_CPUS = os.cpu_count()
NUM_SELF_PLAYERS = min(NUM_CPUS, 8)  # Adjust based on CPU availability

# Adjust saving frequency
SAVE_INTERVAL = 10  # Save model every 10 episodes

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
    return torch.from_numpy(state_tensor)  # Shape: [2, size, size]

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
        #print(f"Input tensor shape in forward: {x.shape}")  # Debugging
        assert x.dim() == 4, f"Input tensor should be 4D, got {x.shape}"
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
    def __init__(self, neural_net, num_simulations=MCTS_SIMULATIONS, cpuct=CPUCT, device='cpu'):
        self.neural_net = neural_net
        self.num_simulations = num_simulations
        self.cpuct = cpuct
        self.device = device
    
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
        state_tensors = torch.stack(state_tensors).to(self.device)
        #print(f"state_tensors shape in evaluate: {state_tensors.shape}")  # Debugging
        assert state_tensors.dim() == 4, f"state_tensors should be 4D, got {state_tensors.shape}"
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

        for _ in range(self.num_simulations):
            node = root
            state = game.copy()

            # Selection
            while node.children:
                move, node = self.select_child(node)
                state.make_move(*move)

            # Expansion and Evaluation
            if not state.is_game_over():
                policy, value = self.evaluate([state])
                policy = policy[0]
                value = value[0]

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
            self.backpropagate(node, -value)  # Note: value is from the perspective of the next player

        # Aggregate visit counts
        visit_counts = defaultdict(int)
        for move, child in root.children.items():
            visit_counts[move] = child.visit_count

        # After computing probs
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

        # Add assertions to check probs
        assert all(v >= 0 for v in probs.values()), f"Negative probabilities in probs: {probs}"
        total_prob = sum(probs.values())
        assert abs(total_prob - 1.0) < 1e-6, f"Probabilities in probs do not sum to 1, sum: {total_prob}"


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

def evaluate_model(neural_net, num_games=10, device='cpu'):
    """
    Evaluate the current model by playing games against a random opponent.
    """
    neural_net.eval()
    wins = 0
    for _ in range(num_games):
        game = HexGame(board_size=BOARD_SIZE)
        mcts = MCTS(neural_net, num_simulations=MCTS_SIMULATIONS, cpuct=CPUCT, device=device)
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

def play_game(neural_net_state_dict, device, return_data_queue):
    """
    Play a single self-play game and return training data.
    """
    # Ensure device is set correctly
    device = torch.device(device)
    if device.type == 'cuda':
        torch.cuda.set_device(device)

    # Initialize neural network and load state dict
    neural_net = HexNet(BOARD_SIZE, num_res_blocks=NUM_RES_BLOCKS, num_filters=NUM_FILTERS)
    # Adjust the state_dict keys if necessary
    state_dict = neural_net_state_dict
    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = k.replace('module.', '')  # Remove 'module.' prefix if present
        new_state_dict[new_key] = v
    neural_net.load_state_dict(new_state_dict)
    neural_net.to(device)
    neural_net.eval()

    game = HexGame(board_size=BOARD_SIZE)
    mcts = MCTS(neural_net, num_simulations=MCTS_SIMULATIONS, cpuct=CPUCT, device=device)
    training_examples = []

    while not game.is_game_over():
        action_probs = mcts.get_action_probs(game, temp=1)
        
        # Check that action_probs are valid
        total_prob = sum(action_probs.values())
        assert total_prob > 0, "Total probability from action_probs is zero or negative."
        action_probs = {k: v / total_prob for k, v in action_probs.items()}
        assert all(v >= 0 for v in action_probs.values()), f"Negative probabilities in action_probs: {action_probs}"
        assert abs(sum(action_probs.values()) - 1.0) < 1e-6, f"Probabilities in action_probs do not sum to 1, sum: {sum(action_probs.values())}"
        
        # Proceed with choosing the move
        moves = list(action_probs.keys())
        probs = list(action_probs.values())
        move = random.choices(moves, weights=probs)[0]
        
        state_tensor = state_to_tensor(game.get_state(), game.current_player)
        pi = np.zeros(neural_net.num_actions, dtype=np.float32)
        for move_prob, prob in action_probs.items():
            index = move_to_index(move_prob, game.board_size)
            pi[index] = prob
        
        # Add assertions to check pi
        assert np.all(pi >= 0), f"Negative probabilities in pi: {pi[pi < 0]}"
        assert np.isclose(np.sum(pi), 1.0), f"Probabilities in pi do not sum to 1, sum: {np.sum(pi)}"
        
        training_examples.append((state_tensor, pi, game.current_player))
        
        # Make the move
        game.make_move(*move)

    winner = game.get_winner()

    # After the game ends
    data = []
    for state, pi, player in training_examples:
        if winner == player:
            reward = 1
        elif winner == 0:
            reward = 0
        else:
            reward = -1
        data.append((state, pi, reward))

    return_data_queue.put(data)

def train():
    # Check if we're in distributed mode
    distributed = False
    if 'WORLD_SIZE' in os.environ and int(os.environ['WORLD_SIZE']) > 1:
        distributed = True
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])

        # Initialize process group with appropriate backend
        if torch.cuda.is_available():
            dist_backend = "nccl"
        else:
            dist_backend = "gloo"
        dist.init_process_group(backend=dist_backend, rank=rank, world_size=world_size)

        # Set device for this process
        if torch.cuda.is_available():
            device_index = rank % torch.cuda.device_count()
            torch.cuda.set_device(device_index)
            device = torch.device(f'cuda:{device_index}')
        else:
            device = torch.device('cpu')  # For CPUs or MPS devices
    else:
        # Single-process mode
        rank = 0
        world_size = 1
        if torch.backends.mps.is_available():
            device = torch.device('mps')
            print("Using MPS device")
        elif torch.cuda.is_available():
            device = torch.device('cuda')
            print("Using CUDA device")
        else:
            device = torch.device('cpu')
            print("Using CPU device")

    # Prepare device string to pass to child processes
    if device.type == 'cuda':
        device_index = device.index
        device_str = f'cuda:{device_index}'
    else:
        device_str = device.type

    # Initialize neural network
    neural_net = HexNet(BOARD_SIZE, num_res_blocks=NUM_RES_BLOCKS, num_filters=NUM_FILTERS).to(device)

    if distributed:
        # Wrap model with DDP
        ddp_model = DDP(
            neural_net,
            device_ids=[device.index] if device.type == 'cuda' else None,
            output_device=device.index if device.type == 'cuda' else None
        )
    else:
        ddp_model = neural_net

    # Optimizer and Scheduler
    optimizer = optim.SGD(ddp_model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM, nesterov=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_MAX)

    # Load the latest model and optimizer states if available
    latest_model_path = get_latest_model_path()
    if latest_model_path and os.path.exists(latest_model_path):
        checkpoint = torch.load(latest_model_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            ddp_model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_episode = checkpoint['episode'] + 1
            if rank == 0:
                logging.info(f"Resuming training from episode {start_episode}")
        else:
            start_episode = 1
            if rank == 0:
                logging.info("Starting training from scratch")
    else:
        start_episode = 1
        if rank == 0:
            logging.info("Starting training from scratch")

    # Replay memory (use a local deque in each process)
    memory = deque(maxlen=MEMORY_SIZE)

    # Initialize variables for logging
    total_policy_loss = 0
    total_value_loss = 0
    total_loss = 0
    episodes_since_last_log = 0
    start_time = time.time()

    # Set up multiprocessing for self-play
    mp.set_start_method('spawn', force=True)
    manager = mp.Manager()
    return_data_queue = manager.Queue()

    for episode in range(start_episode, NUM_EPISODES + 1):
        # Generate self-play games in parallel
        processes = []

        # Move the model state_dict to CPU before passing
        state_dict_cpu = {k: v.cpu() for k, v in ddp_model.state_dict().items()}

        for _ in range(NUM_SELF_PLAYERS):
            p = mp.Process(target=play_game, args=(state_dict_cpu, device_str, return_data_queue))
            p.start()
            processes.append(p)

        # Collect training data
        for _ in range(NUM_SELF_PLAYERS):
            data = return_data_queue.get()
            memory.extend(data)

        # Ensure all processes have finished
        for p in processes:
            p.join()

        # Training step
        if len(memory) >= BATCH_SIZE:
            # Sample a batch
            batch = random.sample(memory, BATCH_SIZE)
            states = torch.stack([item[0] for item in batch]).to(device)

            # Prepare target policy and value
            target_pis = torch.zeros((BATCH_SIZE, neural_net.num_actions), dtype=torch.float32).to(device)
            target_vs = torch.zeros(BATCH_SIZE, dtype=torch.float32).to(device)

            for idx, (state, pi, reward) in enumerate(batch):
                target_pis[idx] = torch.from_numpy(pi).to(device)
                target_vs[idx] = reward

            # Add assertions to check target_pis
            assert torch.all(target_pis >= 0), f"Negative probabilities found in target_pis: {target_pis[target_pis < 0]}"
            assert torch.all(target_pis <= 1), f"Probabilities in target_pis exceed 1: {target_pis[target_pis > 1]}"
            row_sums = target_pis.sum(dim=1)
            assert torch.allclose(row_sums, torch.ones(BATCH_SIZE).to(device), atol=1e-6), f"Probabilities in target_pis do not sum to 1, sums: {row_sums}"

            # Optional: Print min, max, and sum of target_pis for debugging
            #print(f"target_pis min: {target_pis.min().item()}, max: {target_pis.max().item()}, sum per sample (should be 1): {row_sums}")


            # Forward pass
            ddp_model.train()
            out_pis, out_vs = ddp_model(states)

            # Since out_pis are log probabilities, exponentiate to get probabilities
            out_pis_exp = out_pis.exp()

            # Add assertions to check out_pis_exp
            assert torch.all(out_pis_exp >= 0), f"Negative probabilities found in out_pis_exp: {out_pis_exp[out_pis_exp < 0]}"
            row_sums_out = out_pis_exp.sum(dim=1)
            assert torch.allclose(row_sums_out, torch.ones(BATCH_SIZE).to(device), atol=1e-6), f"Probabilities in out_pis_exp do not sum to 1, sums: {row_sums_out}"

            # Optional: Print min, max, and sum of out_pis_exp for debugging
            #print(f"out_pis_exp min: {out_pis_exp.min().item()}, max: {out_pis_exp.max().item()}, sum per sample (should be 1): {row_sums_out}")

            # Compute loss
            loss_pis = torch.mean(-torch.sum(target_pis * out_pis, dim=1))
            loss_vs = F.mse_loss(out_vs.view(-1), target_vs)
            loss = loss_pis + loss_vs

            # Add assertion to ensure losses are positive
            assert loss_pis.item() >= 0, f"Policy loss is negative: {loss_pis.item()}"
            assert loss_vs.item() >= 0, f"Value loss is negative: {loss_vs.item()}"
            assert loss.item() >= 0, f"Total loss is negative: {loss.item()}"

            # Optional: Print loss values for debugging
            #print(f"loss_pis: {loss_pis.item()}, loss_vs: {loss_vs.item()}, total_loss: {loss.item()}")

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            # Accumulate losses for logging
            total_policy_loss += loss_pis.item()
            total_value_loss += loss_vs.item()
            total_loss += loss.item()
            episodes_since_last_log += 1
        else:
            logging.info(f"Episode {episode}: Not enough data to train. Memory size: {len(memory)}")

        # Periodic logging
        if episode % SAVE_INTERVAL == 0 and rank == 0:
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

            # Save the model
            model_path = f"hex_net_episode_{episode}.pth"
            torch.save({
                'episode': episode,
                'model_state_dict': ddp_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, model_path)

            logging.info(f"Episode {episode}/{NUM_EPISODES} completed. Model saved to {model_path}.")

        # Periodic evaluation
        if episode % 50 == 0 and rank == 0:
            eval_net = HexNet(BOARD_SIZE, num_res_blocks=NUM_RES_BLOCKS, num_filters=NUM_FILTERS).to(device)
            eval_net.load_state_dict(ddp_model.state_dict())
            win_rate = evaluate_model(eval_net, device=device)
            logging.info(f"Episode: {episode}, Win Rate against Random: {win_rate:.2f}")

    # Save the final model
    if rank == 0:
        torch.save({
            'episode': NUM_EPISODES,
            'model_state_dict': ddp_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, "hex_net_final.pth")
        logging.info("Training completed. Final model saved.")

    if distributed:
        dist.destroy_process_group()

# =========================
# Main Execution
# =========================

if __name__ == "__main__":
    # Detect device and set up
    if torch.backends.mps.is_available():
        print("Using MPS device")
    elif torch.cuda.is_available():
        NUM_GPUS = torch.cuda.device_count()
        print(f"Using CUDA device with {NUM_GPUS} GPUs")
    else:
        print("Using CPU device")

    # Check if we're being launched with torchrun (distributed mode)
    if 'WORLD_SIZE' in os.environ and int(os.environ['WORLD_SIZE']) > 1:
        # Distributed mode
        train()
    else:
        # Single-process mode
        train()
