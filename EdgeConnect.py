import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import multiprocessing as mp
from tqdm import trange
import random
import math
import os

# Base Game class
class Game:
    def get_initial_state(self):
        raise NotImplementedError

    def get_next_state(self, state, action, player):
        raise NotImplementedError

    def get_valid_moves(self, state):
        raise NotImplementedError

    def get_value_and_terminated(self, state, action):
        raise NotImplementedError

    def get_opponent(self, player):
        raise NotImplementedError

    def change_perspective(self, state, player):
        raise NotImplementedError

    def get_encoded_state(self, state):
        raise NotImplementedError

    def action_size(self):
        raise NotImplementedError

    def __repr__(self):
        return self.__class__.__name__

# EdgeConnect Game
class EdgeConnect(Game):
    EMPTY = 0
    PLAYER1 = 1
    PLAYER2 = -1  # Changed to -1 for consistency with AlphaZero

    def __init__(self, board_size):
        self.board_size = board_size
        self.size = 2 * self.board_size - 1
        self.center_cell = (self.board_size - 1, self.board_size - 1)
        self.valid_positions = []
        self.position_to_action = {}
        self.action_to_position = {}
        self._action_size = 0
        self.initialize_positions()
        self.edge_cells = self.get_edge_cells()  # Added get_edge_cells method
        self.num_edge_cells = len(self.edge_cells)

    def initialize_positions(self):
        action_idx = 0
        for i in range(self.size):
            for j in range(self.size):
                if self.is_valid_position_static(i, j):
                    self.valid_positions.append((i, j))
                    self.position_to_action[(i, j)] = action_idx
                    self.action_to_position[action_idx] = (i, j)
                    action_idx += 1
        self._action_size = action_idx

    def get_initial_state(self):
        board = np.full((self.size, self.size), -1, dtype=int)
        for i, j in self.valid_positions:
            board[i, j] = self.EMPTY
        return board

    @staticmethod
    def is_valid_position_static(x, y, board_size=None):
        if board_size is None:
            # Approximate board_size based on x and y
            board_size = (max(x, y) + 1 + 1) // 2
        size = 2 * board_size - 1
        return 0 <= x < size and 0 <= y < size and \
               max(abs(x - (board_size - 1)), abs(y - (board_size - 1)), abs((x + y) - (board_size - 1) * 2)) < board_size

    def get_valid_moves(self, state):
        valid_moves = np.zeros(self.action_size(), dtype=np.uint8)
        for idx, (x, y) in enumerate(self.valid_positions):
            if state[x, y] == self.EMPTY:
                valid_moves[idx] = 1
        return valid_moves

    def get_next_state(self, state, action, player):
        x, y = self.action_to_position[action]
        if state[x, y] != self.EMPTY:
            raise ValueError("Invalid move")
        next_state = state.copy()
        next_state[x, y] = player
        return next_state

    def get_value_and_terminated(self, state, action):
        if action is None:
            return 0, False
        if np.all(state != self.EMPTY):
            # Game over, calculate winner
            scores = self.calculate_scores(state)
            winner = self.get_winner(scores)
            if winner == self.PLAYER1:
                return 1, True
            elif winner == self.PLAYER2:
                return -1, True
            else:
                return 0, True  # Draw
        else:
            return 0, False

    def get_opponent(self, player):
        return -player

    def change_perspective(self, state, player):
        return state * player

    def get_encoded_state(self, state):
        encoded_state = np.zeros((2, state.shape[0], state.shape[1]), dtype=np.float32)
        encoded_state[0] = (state == 1)
        encoded_state[1] = (state == -1)
        return encoded_state

    def action_size(self):
        return self._action_size

    def __repr__(self):
        return f"EdgeConnect{self.board_size}"

    def get_edge_cells(self):
        size = self.size
        edge_cells = []
        for i in range(size):
            for j in range(size):
                if self.is_valid_position_static(i, j, self.board_size):
                    if self.is_edge_cell(i, j):
                        edge_cells.append((i, j))
        # Ensure center cell is not in edge cells
        edge_cells = [cell for cell in edge_cells if cell != self.center_cell]
        return edge_cells

    # Additional methods adapted to accept state as parameter
    def calculate_scores(self, state):
        max_player = self.PLAYER1
        min_player = self.PLAYER2

        done = False
        iterations = 0
        max_iterations = 1000  # To prevent infinite loops

        score_state = state.copy()

        self.fill_empty_cells(score_state)

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
            'edge_cells': {self.PLAYER1: 0, self.PLAYER2: 0},
            'center_cell': {self.PLAYER1: 0, self.PLAYER2: 0},
            'bonus': {self.PLAYER1: 0, self.PLAYER2: 0},
            'total': {self.PLAYER1: 0, self.PLAYER2: 0}
        }

        num_edges = {self.PLAYER1: 0, self.PLAYER2: 0}

        size = score_state.shape[0]
        for x in range(size):
            for y in range(size):
                player = score_state[x, y]
                if player in (self.PLAYER1, self.PLAYER2):
                    if self.is_edge_cell(x, y):
                        num_edges[player] += 1
                    if (x, y) == self.center_cell:
                        scores['center_cell'][player] = 1  # Center cell bonus
        scores['edge_cells'] = num_edges

        # Step 6: Calculate group counts for bonus
        groups, num_groups = self.identify_groups_score_state(score_state)

        # Calculate group bonus
        group_bonus_p1 = 2 * (num_groups[self.PLAYER2] - num_groups[self.PLAYER1])
        group_bonus_p2 = 2 * (num_groups[self.PLAYER1] - num_groups[self.PLAYER2])
        scores['bonus'][self.PLAYER1] += group_bonus_p1
        scores['bonus'][self.PLAYER2] += group_bonus_p2

        # Step 7: Calculate total scores
        for player in [self.PLAYER1, self.PLAYER2]:
            scores['total'][player] = (
                scores['edge_cells'][player] +
                scores['center_cell'][player] +
                scores['bonus'][player]
            )

        return scores

    def fill_empty_cells(self, state):
        size = state.shape[0]
        for x in range(size):
            for y in range(size):
                if state[x, y] == self.EMPTY:
                    state[x, y] = self.PLAYER1

    def identify_groups_score_state(self, state):
        visited = set()
        groups = {self.PLAYER1: {}, self.PLAYER2: {}}
        group_ids = {self.PLAYER1: 0, self.PLAYER2: 0}
        size = state.shape[0]

        for x in range(size):
            for y in range(size):
                player = state[x, y]
                if player in (self.PLAYER1, self.PLAYER2) and (x, y) not in visited:
                    group_id = group_ids[player]
                    stack = [(x, y)]
                    while stack:
                        cx, cy = stack.pop()
                        if (cx, cy) in visited:
                            continue
                        if state[cx, cy] == player:
                            visited.add((cx, cy))
                            groups[player][(cx, cy)] = group_id
                            neighbors = self.get_neighbors(cx, cy, state)
                            for nx, ny in neighbors:
                                if state[nx, ny] == player and (nx, ny) not in visited:
                                    stack.append((nx, ny))
                    group_ids[player] += 1

        num_groups = {self.PLAYER1: group_ids[self.PLAYER1], self.PLAYER2: group_ids[self.PLAYER2]}
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

    def get_neighbors(self, x, y, state):
        # Hexagonal grid neighbors
        deltas = [(-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0)]
        neighbors = []
        for dx, dy in deltas:
            nx, ny = x + dx, y + dy
            if 0 <= nx < state.shape[0] and 0 <= ny < state.shape[1]:
                if state[nx, ny] != -1:
                    neighbors.append((nx, ny))
        return neighbors

    def get_winner(self, scores):
        if scores['total'][self.PLAYER1] > scores['total'][self.PLAYER2]:
            return self.PLAYER1
        elif scores['total'][self.PLAYER2] > scores['total'][self.PLAYER1]:
            return self.PLAYER2
        else:
            return 0  # Draw

# Neural Network Architecture
class ResidualCNN(nn.Module):
    def __init__(self, game, args, device):
        super(ResidualCNN, self).__init__()
        self.game = game
        self.args = args
        self.device = device  # Set the device attribute

        self.input_channels = 2  # As per get_encoded_state
        self.board_x, self.board_y = self.game.get_initial_state().shape
        self.action_size = self.game.action_size()

        self.conv1 = nn.Conv2d(self.input_channels, args['num_channels'], kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(args['num_channels'])

        self.res_layers = nn.ModuleList(
            [ResidualBlock(args['num_channels']) for _ in range(args['num_res_blocks'])]
        )

        # Policy head
        self.policy_conv = nn.Conv2d(args['num_channels'], 2, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * self.board_x * self.board_y, self.action_size)

        # Value head
        self.value_conv = nn.Conv2d(args['num_channels'], 1, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(self.board_x * self.board_y, args['value_head_hidden'])
        self.value_fc2 = nn.Linear(args['value_head_hidden'], 1)

        # Move the model to the specified device
        self.to(self.device)

    def forward(self, s):
        s = s.to(self.device)
        s = F.relu(self.bn1(self.conv1(s)))

        for res_block in self.res_layers:
            s = res_block(s)

        # Policy head
        p = F.relu(self.policy_bn(self.policy_conv(s)))
        p = p.view(-1, 2 * self.board_x * self.board_y)
        p = self.policy_fc(p)

        # Value head
        v = F.relu(self.value_bn(self.value_conv(s)))
        v = v.view(-1, self.board_x * self.board_y)
        v = F.relu(self.value_fc1(v))
        v = torch.tanh(self.value_fc2(v))

        return p, v

class ResidualBlock(nn.Module):
    def __init__(self, num_channels):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_channels)

        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, s):
        identity = s
        out = F.relu(self.bn1(self.conv1(s)))
        out = self.bn2(self.conv2(out))
        out += identity
        out = F.relu(out)
        return out

# MCTS Worker Process
def mcts_worker(game, args, model_weights, task_queue, result_queue):
    # Load model in each process
    device = torch.device("cpu")  # Use CPU in worker processes
    model = ResidualCNN(game, args, device)
    model.load_state_dict(model_weights)
    model.eval()

    while True:
        task = task_queue.get()
        if task is None:
            break  # Shutdown signal

        state, player = task
        mcts = MCTS(game, model, args)
        root = mcts.search(state, player)
        result_queue.put(root)

# Node for MCTS
class Node:
    def __init__(self, game, state, player, parent=None, prior=0):
        self.game = game
        self.state = state
        self.player = player
        self.parent = parent
        self.prior = prior
        self.children = {}
        self.visit_count = 0
        self.value_sum = 0
        self.is_expanded = False

    def expanded(self):
        return self.is_expanded

    def value(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count

# MCTS with batched neural network evaluations
class MCTS:
    def __init__(self, game, model, args):
        self.game = game
        self.model = model
        self.args = args

    def search(self, initial_state, player):
        root = Node(self.game, initial_state, player)

        for _ in range(self.args['num_simulations']):
            node = root
            search_path = [node]

            # Selection
            while node.expanded():
                action, node = self.select_child(node)
                search_path.append(node)

            # Expansion and Evaluation
            value = self.evaluate(node)

            # Backpropagation
            self.backpropagate(search_path, value)

        return root

    def select_child(self, node):
        best_ucb = -float('inf')
        best_action = None
        best_child = None

        for action, child in node.children.items():
            ucb = self.ucb_score(node, child)
            if ucb > best_ucb:
                best_ucb = ucb
                best_action = action
                best_child = child

        return best_action, best_child

    def ucb_score(self, parent, child):
        c_puct = self.args['c_puct']
        prior = child.prior
        q_value = 0 if child.visit_count == 0 else child.value()
        ucb = q_value + c_puct * prior * math.sqrt(parent.visit_count) / (1 + child.visit_count)
        return ucb

    def evaluate(self, node):
        state = node.state
        value, is_terminal = self.game.get_value_and_terminated(state, None)
        if is_terminal:
            return value

        # Neural network evaluation
        state_tensor = torch.tensor(self.game.get_encoded_state(state), dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            policy_logits, value = self.model(state_tensor)
        policy = torch.softmax(policy_logits, dim=1).cpu().numpy()[0]
        value = value.item()

        # Mask invalid moves
        valid_moves = self.game.get_valid_moves(state)
        policy *= valid_moves
        policy_sum = np.sum(policy)
        if policy_sum > 0:
            policy /= policy_sum
        else:
            policy = valid_moves / np.sum(valid_moves)

        node.is_expanded = True

        for action in range(self.game.action_size()):
            if valid_moves[action]:
                next_state = self.game.get_next_state(state, action, node.player)
                node.children[action] = Node(self.game, next_state, player=-node.player, parent=node, prior=policy[action])

        return value

    def backpropagate(self, search_path, value):
        for node in reversed(search_path):
            node.visit_count += 1
            node.value_sum += value if node.player == search_path[-1].player else -value
            value = -value  # Switch perspective

# Self-Play Game Instance
class SelfPlayGame:
    def __init__(self, game):
        self.game = game
        self.state = game.get_initial_state()
        self.history = []
        self.is_done = False
        self.value = 0
        self.player = 1  # Start with player 1

# AlphaZero Training Loop
class AlphaZero:
    def __init__(self, game, model, args):
        self.game = game
        self.model = model
        self.args = args
        self.replay_buffer = ReplayBuffer(args['replay_buffer_size'])
        self.device = model.device

        self.optimizer = torch.optim.Adam(model.parameters(), lr=args['learning_rate'], weight_decay=args['weight_decay'])

    def self_play(self):
        # Set up multiprocessing
        num_processes = self.args['num_processes']
        manager = mp.Manager()
        task_queue = manager.Queue()
        result_queue = manager.Queue()

        # Start worker processes
        workers = []
        model_weights = self.model.state_dict()
        for _ in range(num_processes):
            worker = mp.Process(target=mcts_worker, args=(self.game, self.args, model_weights, task_queue, result_queue))
            worker.start()
            workers.append(worker)

        games = [SelfPlayGame(self.game) for _ in range(self.args['num_parallel_games'])]

        while any(not game.is_done for game in games):
            active_games = [game for game in games if not game.is_done]

            # Submit tasks to the queue
            for game in active_games:
                task_queue.put((game.state, game.player))

            # Collect results
            for game in active_games:
                root = result_queue.get()

                action_probs = np.zeros(self.game.action_size())
                for action, child in root.children.items():
                    action_probs[action] = child.visit_count
                action_probs /= np.sum(action_probs)

                # Temperature
                temperature = self.args['temperature']
                if temperature == 0:
                    action = np.argmax(action_probs)
                else:
                    action_probs = action_probs ** (1 / temperature)
                    action_probs /= np.sum(action_probs)
                    action = np.random.choice(self.game.action_size(), p=action_probs)

                # Store history
                game.history.append((self.game.get_encoded_state(game.state), action_probs, game.player))

                # Play the action
                game.state = self.game.get_next_state(game.state, action, game.player)

                value, is_terminal = self.game.get_value_and_terminated(game.state, action)
                if is_terminal:
                    game.is_done = True
                    game.value = value

                # Switch player
                game.player = self.game.get_opponent(game.player)

        # Shutdown worker processes
        for _ in workers:
            task_queue.put(None)
        for worker in workers:
            worker.join()

        # Collect data
        data = []
        for game in games:
            for state, policy, player in game.history:
                reward = game.value if player == 1 else -game.value
                data.append((state, policy, reward))

        return data

    def train(self):
        for epoch in trange(self.args['num_epochs'], desc='Training'):
            if len(self.replay_buffer) < self.args['batch_size']:
                continue

            batch = self.replay_buffer.sample(self.args['batch_size'])
            states, policies, rewards = zip(*batch)

            # Convert lists to numpy arrays to avoid warnings
            states = np.array(states)
            policies = np.array(policies)
            rewards = np.array(rewards)

            states = torch.tensor(states, dtype=torch.float32).to(self.device)
            policies = torch.tensor(policies, dtype=torch.float32).to(self.device)
            rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)

            self.model.train()
            self.optimizer.zero_grad()

            out_policies, out_values = self.model(states)

            # Policy loss
            policy_loss = torch.mean(torch.sum(-policies * F.log_softmax(out_policies, dim=1), dim=1))

            # Value loss
            out_values = out_values.view(-1)
            value_loss = F.mse_loss(out_values, rewards)

            loss = policy_loss + self.args['value_loss_weight'] * value_loss

            loss.backward()
            self.optimizer.step()

    def learn(self):
        for iteration in range(self.args['num_iterations']):
            print(f"\nIteration {iteration + 1}/{self.args['num_iterations']}")

            # Self-play to collect data
            data = self.self_play()
            self.replay_buffer.extend(data)

            # Training the neural network
            self.train()

            # Save model
            torch.save(self.model.state_dict(), f"model_{iteration}_{self.game}.pt")

# Replay Buffer with fixed size
class ReplayBuffer:
    def __init__(self, max_size):
        self.buffer = []
        self.max_size = max_size

    def extend(self, data):
        self.buffer.extend(data)
        if len(self.buffer) > self.max_size:
            self.buffer = self.buffer[-self.max_size:]

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

# Main execution
if __name__ == "__main__":
    # Use 'spawn' start method for multiprocessing
    mp.set_start_method('spawn')

    # Game instance
    game = EdgeConnect(board_size=3)

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Training arguments
    args = {
        'num_channels': 128,
        'num_res_blocks': 9,
        'value_head_hidden': 256,
        'num_simulations': 400,  # Adjusted for efficiency
        'num_iterations': 10,
        'num_epochs': 5,
        'batch_size': 256,
        'learning_rate': 0.001,
        'weight_decay': 1e-4,
        'temperature': 1.0,
        'c_puct': 1.0,
        'value_loss_weight': 1.0,
        'num_parallel_games': 8,
        'replay_buffer_size': 10000,
        'num_processes': os.cpu_count(),  # Automatically get number of CPUs
    }

    # Neural network and optimizer
    model = ResidualCNN(game, args, device)

    # AlphaZero instance
    alpha_zero = AlphaZero(game, model, args)

    # Start learning
    alpha_zero.learn()