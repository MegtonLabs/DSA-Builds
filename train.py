# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import os
from collections import deque, namedtuple
from enum import Enum
import pygame
import matplotlib.pyplot as plt
from IPython import display # Optional: for cleaner plotting in some environments
import traceback # For printing detailed error messages

# --- Pygame Initialization ---
pygame.init()
# Use default font or specify 'arial.ttf' path if available
try:
    font = pygame.font.Font(None, 25)
except FileNotFoundError:
    print("Default font not found, using pygame's default.")
    font = pygame.font.SysFont("arial", 25) # Fallback font
# --------------------------

# --- Plotting Setup ---
plt.ion() # Interactive mode on
# ----------------------

# --- Configuration & Constants ---

# Game Settings
BLOCK_SIZE = 20
DEFAULT_SPEED = 20 # Speed used only if not overridden by train/play settings
WIDTH = 640
HEIGHT = 480

# Colors
WHITE = (255, 255, 255)
RED = (200, 0, 0)
GREEN_HEAD = (0, 255, 0)   # Bright green for the head
GREEN_TAIL = (0, 100, 0)   # Darker green for the tail end
BLACK = (0, 0, 0)
BORDER_COLOR = BLACK # Color for the snake segment border

# === >> START RECOMMENDED VALUES FOR FASTER LEARNING / HIGHER SCORE << ===
# RL Hyperparameters
MODEL_PATH = 'model/model_ddqn_enhanced_state.pth' # New model name for this version
MAX_GAMES = 5000       # << Moderate number of games for faster results goal
VISUALIZE_TRAINING = True # << Set to False for faster training (NO GAME/PLOT WINDOWS)
GAME_SPEED_TRAIN = 10000 # << Max speed when headless
GAME_SPEED_PLAY = 15    # Speed when watching a trained agent play

MAX_MEMORY = 100_000    # << Reduced memory for potentially faster initial learning phase
BATCH_SIZE = 512        # << Adjusted batch size
LR = 0.0005            # << Slightly higher LR for faster initial learning
GAMMA = 0.99           # << Keep high for long-term focus
TARGET_UPDATE_FREQ = 50  # << Update target network more frequently

# Exploration Settings (Epsilon-Greedy)
MIN_EPSILON = 0.01      # << Minimum exploration rate
INITIAL_EPSILON = 1.0   # << Start with full exploration
# Decay adjusted for MAX_GAMES=5000 (reaches ~min after ~3960 games)
EPSILON_DECAY_DIVISOR = 4000 # << Faster Epsilon Decay for shorter training goal

# Reward Shaping (Highly Recommended)
ENABLE_REWARD_SHAPING = True
REWARD_CLOSER = 0.1
REWARD_FARTHER = -0.15

# Model Hyperparameters
INPUT_SIZE = 18         # << INCREASED STATE SIZE (11 + 3 + 4 = 18 Features)
HIDDEN_SIZE = 256       # (Keep 256, or try 512 if 256 plateaus)
OUTPUT_SIZE = 3
# === >> END RECOMMENDED VALUES << ===

# --- Device Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
# -------------------

# --- Game Definitions ---
class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

Point = namedtuple('Point', 'x, y')
# ----------------------

# --- Snake Game Class ---
class SnakeGameAI:
    def __init__(self, w=WIDTH, h=HEIGHT, visualize=True):
        self.w = w
        self.h = h
        self.visualize = visualize # Store visualize flag passed from train/play
        self.font = font # Use the globally initialized font
        self.display = None
        self.speed = DEFAULT_SPEED
        self._check_create_display() # Create display now if visualize is True
        self.clock = pygame.time.Clock()
        self.reset()

    def _check_create_display(self):
        # Only creates display if self.visualize is True
        if self.visualize and self.display is None:
            try:
                self.display = pygame.display.set_mode((self.w, self.h))
                pygame.display.set_caption('Snake RL (Enhanced State + Double DQN)')
            except pygame.error as e:
                print(f"Error initializing Pygame display: {e}. Disabling visualization.")
                self.visualize = False # Ensure visualization is off if display fails
                self.display = None
        elif not self.visualize:
            # Ensure display is None if visualization is explicitly off
            if self.display:
                pygame.display.quit() # Properly close existing display if turning off
            self.display = None

    def reset(self):
        self.direction = Direction.RIGHT
        self.head = Point(self.w/2, self.h/2)
        self.snake = [self.head,
                      Point(self.head.x-BLOCK_SIZE, self.head.y),
                      Point(self.head.x-(2*BLOCK_SIZE), self.head.y)]
        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0
        # Don't recreate display here, rely on constructor or external setting change
        # self._check_create_display()

    def _place_food(self):
        x = random.randint(0, (self.w-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE
        y = random.randint(0, (self.h-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE
        self.food = Point(x, y)
        # Ensure food doesn't spawn inside snake
        while self.food in self.snake:
            x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
            y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
            self.food = Point(x, y)


    def play_step(self, action):
        dist_before = 0
        if ENABLE_REWARD_SHAPING and self.food:
            dist_before = np.sqrt((self.head.x - self.food.x)**2 + (self.head.y - self.food.y)**2)

        self.frame_iteration += 1
        # Handle quit events even if not visualizing
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                print("Quit event received. Shutting down.")
                pygame.quit()
                quit() # Exit the entire script

        # Perform Move
        self._move(action)
        self.snake.insert(0, self.head)

        dist_after = 0
        if ENABLE_REWARD_SHAPING and self.food:
             dist_after = np.sqrt((self.head.x - self.food.x)**2 + (self.head.y - self.food.y)**2)

        # Check Game Over Conditions & Calculate Reward
        reward = 0
        game_over = False
        if self.is_collision() or self.frame_iteration > 100 * len(self.snake):
             game_over = True
             reward = -10 # Big penalty for dying
             # Update UI one last time on death if visualizing
             if self.display: self._update_ui()
             # Control speed even on death frame (prevents hanging)
             self.clock.tick(self.speed)
             return reward, game_over, self.score

        # Check Food Eaten
        if self.head == self.food:
            self.score += 1
            reward = 10 # Big reward for eating
            self._place_food()
            self.frame_iteration = 0 # Reset timeout counter on eating
        else:
            self.snake.pop() # Remove tail segment if no food eaten
            # Reward Shaping for Distance
            if ENABLE_REWARD_SHAPING:
                if dist_after < dist_before: reward += REWARD_CLOSER
                else: reward += REWARD_FARTHER

        # Update UI and Clock
        # Update UI only if display exists (set during init based on visualize flag)
        if self.display:
            self._update_ui()
        # Tick clock regardless of visualization to maintain game logic timing consistency
        self.clock.tick(self.speed)

        return reward, game_over, self.score

    def is_collision(self, pt=None):
        if pt is None: pt = self.head
        # Check boundaries (use >= for width/height as coordinates are 0-based)
        if pt.x >= self.w or pt.x < 0 or pt.y >= self.h or pt.y < 0: return True
        # Check self collision (excluding the exact head position if pt is head)
        if pt in self.snake[1:]: return True
        return False

    def _update_ui(self):
        # This function should only be called if self.display is not None
        if not self.display: return

        try:
            self.display.fill(BLACK)

            # --- Gradient Snake Drawing with Border ---
            num_segments = len(self.snake)
            for i, pt in enumerate(self.snake):
                # Calculate gradient color
                interp_factor = i / (num_segments - 1) if num_segments > 1 else 0
                r = int(GREEN_HEAD[0] + (GREEN_TAIL[0] - GREEN_HEAD[0]) * interp_factor)
                g = int(GREEN_HEAD[1] + (GREEN_TAIL[1] - GREEN_HEAD[1]) * interp_factor)
                b = int(GREEN_HEAD[2] + (GREEN_TAIL[2] - GREEN_HEAD[2]) * interp_factor)
                segment_color = (max(0, min(255, r)), max(0, min(255, g)), max(0, min(255, b)))

                # Draw main segment rect
                main_rect = pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE)
                pygame.draw.rect(self.display, segment_color, main_rect)

                # Draw border using the outline feature of draw.rect
                border_thickness = 1 # Adjust thickness if needed
                pygame.draw.rect(self.display, BORDER_COLOR, main_rect, border_thickness)
            # --- End Snake Drawing ---

            # Draw Food
            pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))

            # Draw Score
            text = self.font.render("Score: " + str(self.score), True, WHITE)
            self.display.blit(text, [0, 0])

            # Update the full display Surface to the screen
            pygame.display.flip()
        except pygame.error as e:
            # Handle cases where the display might have been closed unexpectedly
            print(f"Pygame error during UI update: {e}")
            self.visualize = False # Disable visualization if display fails
            self.display = None


    def _move(self, action):
        # action: [1,0,0] -> straight, [0,1,0] -> right, [0,0,1] -> left
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)

        if np.array_equal(action, [1, 0, 0]): new_dir = clock_wise[idx] # No change
        elif np.array_equal(action, [0, 1, 0]): new_dir = clock_wise[(idx + 1) % 4] # Right turn
        else: new_dir = clock_wise[(idx - 1) % 4] # Left turn
        self.direction = new_dir

        x, y = self.head.x, self.head.y
        if self.direction == Direction.RIGHT: x += BLOCK_SIZE
        elif self.direction == Direction.LEFT: x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN: y += BLOCK_SIZE
        elif self.direction == Direction.UP: y -= BLOCK_SIZE
        self.head = Point(x, y)

    def set_speed(self, speed):
        self.speed = speed
# ----------------------

# --- Plotting Function ---
def plot(scores, mean_scores, filename='training_progress.png'):
    # Plotting function should only be called if visualization is intended
    # Added check at call site in train() function
    if not plt or not plt.figure: return

    try:
        # Attempt IPython display update if available
        display.clear_output(wait=True)
        display.display(plt.gcf()) # Get current figure
    except Exception:
        # Fallback if not in IPython environment or display fails
        pass

    try:
        plt.clf() # Clear the current figure before drawing new data
        plt.title('Training Progress (Enhanced State + DDQN)')
        plt.xlabel('Number of Games')
        plt.ylabel('Score')
        plt.plot(scores, label='Score per Game', linewidth=0.8, alpha=0.7, color='blue')
        plt.plot(mean_scores, label=f'Mean Score (Avg over {len(mean_scores)} games)', linewidth=1.5, color='orange')
        plt.ylim(ymin=0) # Ensure Y axis starts at 0
        # Add text annotations for last score and mean score
        if scores: plt.text(len(scores)-1, scores[-1], f" {scores[-1]}", va='bottom')
        if mean_scores: plt.text(len(mean_scores)-1, mean_scores[-1], f" {mean_scores[-1]:.2f}", va='bottom', color='orange')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout() # Adjust plot to prevent labels overlapping
        plt.show(block=False) # Show plot non-blockingly
        plt.pause(.1) # Pause allows plot window to update

        # Save the plot
        plot_folder_path = './plots'
        if not os.path.exists(plot_folder_path): os.makedirs(plot_folder_path)
        # Save plot with name derived from model path for correlation
        plot_filename = os.path.basename(filename).replace(".pth", ".png") if filename else "training_progress.png"
        filepath = os.path.join(plot_folder_path, plot_filename)
        plt.savefig(filepath)
    except Exception as e:
        # Catch potential errors during plotting/saving
        print(f"Warning: Could not update/save plot. Error: {e}")
# ------------------------

# --- Neural Network Model ---
class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = x.to(device) # Ensure tensor is on the correct device
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    def save(self, file_name=MODEL_PATH): # Default to constant MODEL_PATH
        model_folder_path = os.path.dirname(file_name) # Get directory from path
        # Create directory if it doesn't exist
        if model_folder_path and not os.path.exists(model_folder_path):
             os.makedirs(model_folder_path)
        try:
            # Save the model state dictionary
            torch.save(self.state_dict(), file_name)
        except Exception as e:
            print(f"Error saving model to {file_name}: {e}")


    def load(self, file_name=MODEL_PATH):
        if os.path.exists(file_name):
            try:
                # Load state dict, mapping to the current device
                self.load_state_dict(torch.load(file_name, map_location=device))
                self.to(device) # Ensure model is on the correct device after loading
                self.eval() # Set to evaluation mode after loading
                print(f"Model loaded successfully from {file_name}")
                return True
            except Exception as e:
                # Handle potential errors during loading (e.g., corrupted file, architecture mismatch)
                print(f"Error loading model from {file_name}: {e}. Starting/Continuing with untrained model.")
                return False
        else:
            # File doesn't exist, return False (handled by Agent init logic)
            return False
# ---------------------------

# --- Q-Trainer (Implementing Double DQN) ---
class QTrainer:
    def __init__(self, model, target_model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model.to(device)
        self.target_model = target_model.to(device)
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss() # Mean Squared Error Loss for Q-learning
        # Initialize target network weights to match the policy network
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval() # Target network is only for inference, not training

    def train_step(self, state, action, reward, next_state, done):
        # Convert inputs to tensors and move to the appropriate device (GPU/CPU)
        state = torch.tensor(np.array(state), dtype=torch.float).to(device)
        next_state = torch.tensor(np.array(next_state), dtype=torch.float).to(device)
        action = torch.tensor(np.array(action), dtype=torch.long).to(device) # Actions used for indexing later
        reward = torch.tensor(np.array(reward), dtype=torch.float).to(device)
        done = torch.tensor(np.array(done), dtype=torch.bool).to(device) # Boolean mask for terminal states

        # Handle case where inputs are single samples (not batches)
        if len(state.shape) == 1:
            # Add a batch dimension (shape becomes [1, num_features])
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = torch.unsqueeze(done, 0)

        # 1. Get predicted Q-values from the policy network for the current state
        pred = self.model(state) # Shape: (batch_size, num_actions)

        # 2. Calculate target Q-values using the Double DQN logic
        target = pred.clone() # Start with predictions, modify targets for taken actions
        for idx in range(len(done)): # Iterate through batch samples
            Q_new = reward[idx] # Target starts with the immediate reward
            if not done[idx]: # Only add discounted future reward if not a terminal state
                with torch.no_grad(): # No gradient calculation needed for target selection/evaluation
                    # --- Double DQN Target Calculation ---
                    # Get the next state for the current sample
                    next_state_sample = next_state[idx].unsqueeze(0) if len(next_state.shape) > 1 else next_state[idx]

                    # 1. Select the best action 'a' for the next state using the POLICY network (self.model)
                    # .detach() might not be strictly needed due to no_grad context, but reinforces intent
                    next_action = self.model(next_state_sample).argmax(dim=1, keepdim=True).detach() # Shape: [1, 1]

                    # 2. Evaluate the Q-value of that selected action 'a' using the TARGET network (self.target_model)
                    next_q_value = self.target_model(next_state_sample).gather(1, next_action).detach() # Shape: [1, 1]
                    # --- End Double DQN ---

                    # Bellman equation: Q_target = r + gamma * Q_target_network(s', argmax_a(Q_policy_network(s', a)))
                    Q_new = reward[idx] + self.gamma * next_q_value.item()

            # Find the index corresponding to the action actually taken in the original transition
            action_idx = torch.argmax(action[idx]).item()
            # Update the target Q-value for the action that was taken
            target[idx][action_idx] = Q_new

        # 3. Perform optimization step
        self.optimizer.zero_grad() # Reset gradients
        loss = self.criterion(target, pred) # Calculate loss between prediction and target
        loss.backward() # Backpropagate the loss
        self.optimizer.step() # Update policy network weights

        return loss.item() # Return loss value for monitoring
# -----------------

# --- RL Agent (Using Enhanced State) ---
class Agent:
    def __init__(self, load_model_path=None):
        self.n_games = 0
        self.epsilon = INITIAL_EPSILON # Initialize epsilon
        self.gamma = GAMMA
        self.memory = deque(maxlen=MAX_MEMORY) # Experience replay buffer

        # Initialize policy and target networks
        self.model = Linear_QNet(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE).to(device)
        self.target_model = Linear_QNet(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE).to(device)

        # Attempt to load pre-trained model if path is provided and exists
        self.model_loaded = False
        if load_model_path and os.path.exists(load_model_path):
            print(f"Attempting to load model from: {load_model_path}")
            self.model_loaded = self.model.load(load_model_path) # .load handles internal logic

        if not self.model_loaded:
            print("Initializing new model or continuing training without loaded weights.")

        # Synchronize target network weights initially and initialize trainer
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval() # Target network is always in evaluation mode
        self.trainer = QTrainer(self.model, self.target_model, lr=LR, gamma=self.gamma)

    # --- Enhanced State Representation ---
    def get_state(self, game):
        head = game.snake[0]
        head_x, head_y = head.x, head.y
        snake_body = game.snake # Cache snake body for efficient checks

        # Helper to check collision for a point
        def is_collision(pt):
            # Check boundaries
            if pt.x >= game.w or pt.x < 0 or pt.y >= game.h or pt.y < 0:
                return True
            # Check collision with snake body (excluding the neck segment, index 1)
            # Important: If pt is head, only check against snake[1:]. If pt is not head (e.g. lookahead), check against full snake.
            # Simpler: check against snake[1:] is usually sufficient for lookahead checks too.
            if pt in snake_body[1:]:
                return True
            return False

        # Points relative to current direction (1 block)
        point_l_dir = Point(head_x - BLOCK_SIZE, head_y) if game.direction == Direction.UP else \
                      Point(head_x + BLOCK_SIZE, head_y) if game.direction == Direction.DOWN else \
                      Point(head_x, head_y - BLOCK_SIZE) if game.direction == Direction.LEFT else \
                      Point(head_x, head_y + BLOCK_SIZE) # RIGHT (moving up, left is x-1)
        point_r_dir = Point(head_x + BLOCK_SIZE, head_y) if game.direction == Direction.UP else \
                      Point(head_x - BLOCK_SIZE, head_y) if game.direction == Direction.DOWN else \
                      Point(head_x, head_y + BLOCK_SIZE) if game.direction == Direction.LEFT else \
                      Point(head_x, head_y - BLOCK_SIZE) # RIGHT (moving up, right is x+1)
        point_s_dir = Point(head_x + BLOCK_SIZE, head_y) if game.direction == Direction.RIGHT else \
                      Point(head_x - BLOCK_SIZE, head_y) if game.direction == Direction.LEFT else \
                      Point(head_x, head_y - BLOCK_SIZE) if game.direction == Direction.UP else \
                      Point(head_x, head_y + BLOCK_SIZE) # DOWN

        # Calculate points 2 steps ahead based on 1 step point and direction
        dx, dy = 0, 0
        if game.direction == Direction.RIGHT: dx = BLOCK_SIZE
        elif game.direction == Direction.LEFT: dx = -BLOCK_SIZE
        elif game.direction == Direction.UP: dy = -BLOCK_SIZE
        elif game.direction == Direction.DOWN: dy = BLOCK_SIZE

        point_s2_dir = Point(point_s_dir.x + dx, point_s_dir.y + dy)
        # Calculate L2 and R2 based on L1/R1 and current direction - Careful logic needed
        # Example for L2: If moving RIGHT, L1 is UP, so L2 is UP again.
        # Example for R2: If moving RIGHT, R1 is DOWN, so R2 is DOWN again.
        # This needs to be correct for all 4 directions.
        # Simpler approximation: Just check the point 2 blocks away in the relative L/R direction?
        # Let's use the simpler approximation: check 2 blocks relative left/right
        point_l2_dir = Point(point_l_dir.x + dx, point_l_dir.y + dy) # Approx: 2 steps in relative L direction
        point_r2_dir = Point(point_r_dir.x + dx, point_r_dir.y + dy) # Approx: 2 steps in relative R direction

        # Points absolute adjacent to head
        point_l_abs = Point(head_x - BLOCK_SIZE, head_y)
        point_r_abs = Point(head_x + BLOCK_SIZE, head_y)
        point_u_abs = Point(head_x, head_y - BLOCK_SIZE)
        point_d_abs = Point(head_x, head_y + BLOCK_SIZE)

        # Current direction booleans
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        # State features (18 total)
        state = [
            # 1-3: Danger 1 step ahead (Straight, Right, Left relative to current direction)
            is_collision(point_s_dir),
            is_collision(point_r_dir),
            is_collision(point_l_dir),

            # 4-7: Current direction (One-hot)
            dir_l, dir_r, dir_u, dir_d,

            # 8-11: Food location (Relative to head)
            game.food.x < head_x, # Food left
            game.food.x > head_x, # Food right
            game.food.y < head_y, # Food up
            game.food.y > head_y, # Food down

            # === NEW FEATURES ===
            # 12-14: Danger 2 steps ahead (Straight, Approx Right, Approx Left relative to current direction)
            is_collision(point_s2_dir),
            is_collision(point_r2_dir), # Using simplified L2/R2 calc
            is_collision(point_l2_dir), # Using simplified L2/R2 calc

            # 15-18: Adjacent body parts (Absolute Left, Right, Up, Down from head)
            # Check if the absolute adjacent square contains part of the snake body *other than the neck*
            # (The neck check isn't strictly needed if `is_collision` checks against snake[1:])
            is_collision(point_l_abs), # Check collision (wall or body[1:])
            is_collision(point_r_abs),
            is_collision(point_u_abs),
            is_collision(point_d_abs)
        ]
        # Convert booleans to integers (0 or 1)
        return np.array(state, dtype=int)
    # --- End Enhanced State ---

    def remember(self, state, action, reward, next_state, done):
        # Store experience tuple in the replay memory
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        # Train on a random batch from the replay memory
        if len(self.memory) < BATCH_SIZE: # Check if enough samples for a full batch
             if len(self.memory) > 10: # Use full memory if < BATCH_SIZE but has some samples
                 mini_sample = self.memory
             else:
                 return # Not enough memory to train effectively
        else:
             # Sample a random batch
             mini_sample = random.sample(self.memory, BATCH_SIZE)

        # Unzip the batch into separate lists/tuples
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        # Perform a training step on the batch
        loss = self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        # Train on the single, most recent experience
        loss = self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state, is_training=True):
        # Epsilon-greedy action selection
        if is_training:
            # Calculate current epsilon based on decay schedule
            self.epsilon = max(MIN_EPSILON, INITIAL_EPSILON - self.n_games / EPSILON_DECAY_DIVISOR)
        else:
             # No exploration during evaluation/playback
             self.epsilon = 0

        final_move = [0, 0, 0] # Initialize action as "go straight"
        if random.random() < self.epsilon and is_training:
            # Explore: choose a random action (0: straight, 1: right, 2: left)
            move_idx = random.randint(0, 2)
            final_move[move_idx] = 1
        else:
            # Exploit: choose the best action based on the current Q-network prediction
            with torch.no_grad(): # No need to track gradients for inference
                state_tensor = torch.tensor(state, dtype=torch.float).to(device)
                # Add batch dimension if needed (for single state prediction)
                if len(state_tensor.shape) == 1: state_tensor = torch.unsqueeze(state_tensor, 0)

                self.model.eval() # Set model to evaluation mode
                prediction = self.model(state_tensor)
                self.model.train() # Set model back to training mode

                # Get the index (0, 1, or 2) of the action with the highest Q-value
                move_idx = torch.argmax(prediction).item()
                final_move[move_idx] = 1

        return final_move

    def update_target_network(self):
        # Copy weights from the policy network to the target network
        print(f"--- Updating target network (Game {self.n_games}) ---")
        self.target_model.load_state_dict(self.model.state_dict())
# ---------------

# --- Training Loop Function ---
def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    # Agent init attempts to load MODEL_PATH if continuing training
    agent = Agent(load_model_path=MODEL_PATH if os.path.exists(MODEL_PATH) else None)
    # Pass visualize flag based on global config
    # Important: Game uses this flag to decide whether to create the display surface
    game = SnakeGameAI(visualize=VISUALIZE_TRAINING)

    if agent.model_loaded:
        print("Model loaded, resetting record score tracking for this session.")
        # If you want persistent record across sessions, load/save record to a separate file
        record = 0

    # Set game speed based on visualization mode
    game.set_speed(GAME_SPEED_TRAIN) # Set speed even if headless for consistency maybe?

    if VISUALIZE_TRAINING:
        print(f"Starting training (up to {MAX_GAMES} games) with visualization...")
    else:
        print(f"Starting training (up to {MAX_GAMES} games) headless...")

    # --- Main Training Loop ---
    while agent.n_games < MAX_GAMES:
        # 1. Get current state
        state_old = agent.get_state(game)

        # 2. Get action from agent
        final_move = agent.get_action(state_old, is_training=True)

        # 3. Perform action in game and get reward, next state, done flag
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        # 4. Train agent on the recent step (short-term memory)
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # 5. Store experience in replay memory
        agent.remember(state_old, final_move, reward, state_new, done)

        # 6. If game is over, train on batch from memory, update counters, etc.
        if done:
            game.reset() # Reset game environment
            agent.n_games += 1 # Increment game counter
            agent.train_long_memory() # Experience replay

            # Periodically update the target network
            if agent.n_games > 0 and agent.n_games % TARGET_UPDATE_FREQ == 0:
                agent.update_target_network()

            # Check for new record score and save model
            if score > record:
                record = score
                agent.model.save(MODEL_PATH) # Saves to specified MODEL_PATH
                print(f"--- New Record: {record}! Model Saved. (Game {agent.n_games}) ---")

            # Print progress (adjust frequency if needed)
            if agent.n_games % 10 == 0 or score > record * 0.8 : # Print more often for high scores or every 10 games
                print(f'Game: {agent.n_games}/{MAX_GAMES}, Score: {score}, Record: {record}, Epsilon: {agent.epsilon:.4f}')

            # Update plotting data
            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)

            # --- CORRECTED PLOTTING CALL ---
            # Only call plot if visualization is enabled for training
            if VISUALIZE_TRAINING:
                plot(plot_scores, plot_mean_scores, filename=MODEL_PATH)
            # --------------------------------

    # --- End of Training ---
    print(f"--- Training finished after {agent.n_games} games. Final Record: {record} ---")
    # Keep plot window open only if it was displayed during training
    if VISUALIZE_TRAINING:
        print("Close the plot window to exit.")
        plt.ioff()
        plt.show(block=True) # Block execution until plot window is closed
# --------------------------

# --- Playback Function ---
def play_trained_agent(model_path_to_load):
    print(f"--- Starting Playback Mode ---")
    # Agent needs the path to load the specific model for playback
    agent = Agent(load_model_path=model_path_to_load)

    if not agent.model_loaded:
        print(f"Cannot start playback: Model failed to load from {model_path_to_load}")
        return # Exit if model didn't load

    # Force visualization ON for playback
    game = SnakeGameAI(visualize=True)
    game.set_speed(GAME_SPEED_PLAY)

    total_score = 0
    game_count = 0
    max_playback_games = 20 # Limit number of games in one playback session

    while game_count < max_playback_games:
        # 1. Get current state
        state = agent.get_state(game)
        # 2. Get action from agent (no exploration)
        final_move = agent.get_action(state, is_training=False)
        # 3. Perform action
        reward, done, score = game.play_step(final_move)

        if done:
            game_count += 1
            total_score += score
            print(f"--- Game {game_count}/{max_playback_games} Over --- Score: {score}, Average: {total_score/game_count:.2f}")
            # Pause and reset for the next game if not finished
            if game_count < max_playback_games:
                pygame.time.wait(1500) # Pause for 1.5 seconds
                game.reset()
            else:
                print("--- Playback session finished. ---")

# -------------------------

# --- Main Execution Block ---
if __name__ == '__main__':
    mode = 'train' # Default mode if no interaction possible
    # --- Automatic Mode Selection ---
    if os.path.exists(MODEL_PATH):
        print(f"--- Found existing model at '{MODEL_PATH}'. ---")
        while True:
            try:
                 # Ask user how to proceed
                 user_choice = input("Enter 'P' to Play, 'T' to Continue Training, or 'D' to Delete and Train New: ").upper()
                 if user_choice == 'P':
                     mode = 'play'
                     break
                 elif user_choice == 'T':
                     mode = 'train' # Agent init will handle loading the model
                     print("--- Continuing training (loading existing model weights)... ---")
                     break
                 elif user_choice == 'D':
                     try:
                         # Delete the existing model file
                         os.remove(MODEL_PATH)
                         # Also delete associated plot if it exists
                         plot_file = os.path.join('./plots', os.path.basename(MODEL_PATH).replace(".pth", ".png"))
                         if os.path.exists(plot_file): os.remove(plot_file)
                         print(f"--- Deleted existing model and plot. Starting new training. ---")
                         mode = 'train'
                         break
                     except OSError as e:
                         print(f"Error deleting model/plot: {e}. Please delete manually if needed.")
                         # Stay in loop to ask again
                         continue
                 else:
                     # Handle invalid input
                     print("Invalid input. Please enter 'P', 'T', or 'D'.")
            except EOFError:
                 # Handle non-interactive execution (e.g., piping input)
                 print("Input stream closed, defaulting to training.")
                 mode = 'train'
                 break
            except KeyboardInterrupt:
                 print("\nUser cancelled selection.")
                 exit() # Exit script if selection is cancelled

    else:
        # Model file doesn't exist, must train
        print(f"--- Model not found at '{MODEL_PATH}'. Starting new training session. ---")
        mode = 'train'
    # ---------------------------------

    # --- Run selected mode ---
    try:
        if mode == 'train':
            train()
        elif mode == 'play':
            play_trained_agent(MODEL_PATH)

    except KeyboardInterrupt:
        # Handle Ctrl+C gracefully during train/play
        print("\n--- Operation interrupted by user (Ctrl+C). ---")
    except Exception as e:
         # Catch any other unexpected errors
         print(f"\n--- An unexpected error occurred: {e} ---")
         traceback.print_exc() # Print detailed traceback for debugging
    finally:
        # --- Cleanup ---
        print("--- Quitting Pygame and closing plots. ---")
        pygame.quit()
        # Attempt to close matplotlib plots if they exist
        try:
            if plt and plt.close:
                 plt.close('all') # Close all open plot windows
        except NameError:
            # Matplotlib might not have been fully initialized if error occurred early
            pass
# --- End Main Execution Block ---