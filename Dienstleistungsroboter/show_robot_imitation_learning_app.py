import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import time
from collections import deque

# -------------------------------
# Environment & Learning Settings
# -------------------------------

# Define grid actions.
actions = {
    'N': (0, 1),
    'S': (0, -1),
    'E': (1, 0),
    'W': (-1, 0)
}

# Constant reward matrix.
# Cells with nonzero reward are considered absorbing.
reward_matrix = {
    (1, 1): 25,
    (2, 1): -100,
    (2, 3): -80,
    (3, 1): 80,
    (3, 3): 100
}

def is_absorbing(state):
    """Return True if the given state is absorbing (i.e. has a reward)."""
    return state in reward_matrix

def step_environment(state, action):
    """Compute the next state given the current state and action (staying within the 3x3 grid)."""
    dx, dy = actions[action]
    new_state = (state[0] + dx, state[1] + dy)
    # Stay within bounds.
    if new_state[0] < 1 or new_state[0] > 3 or new_state[1] < 1 or new_state[1] > 3:
        new_state = state
    return new_state

def epsilon_greedy_action(state, Q, epsilon):
    """Select an action from the Q-table using an ε-greedy strategy."""
    if random.random() < epsilon:
        return random.choice(list(actions.keys()))
    else:
        q_vals = {a: Q.get((state, a), 0) for a in actions.keys()}
        max_val = max(q_vals.values())
        best_actions = [a for a, v in q_vals.items() if v == max_val]
        return random.choice(best_actions)

def expert_policy(state):
    """
    A simple expert that guides the Show-Robot toward the best absorbing state ((3,3) with +100)
    while avoiding other absorbing states.
    Uses a breadth-first search for a safe (shortest) path.
    """
    target = (3, 3)
    if state == target:
        return None
    queue = deque([(state, [])])
    visited = set()
    while queue:
        s, path = queue.popleft()
        if s == target:
            return path[0] if path else None
        for a, (dx, dy) in actions.items():
            ns = (s[0] + dx, s[1] + dy)
            # Stay in bounds.
            if ns[0] < 1 or ns[0] > 3 or ns[1] < 1 or ns[1] > 3:
                continue
            # Avoid absorbing states unless it's the target.
            if ns in reward_matrix and ns != target:
                continue
            if ns not in visited:
                visited.add(ns)
                queue.append((ns, path + [a]))
    return random.choice(list(actions.keys()))

def get_escape_state(absorbing_state):
    """
    Given an absorbing state, return one of its neighboring nonabsorbing states.
    If multiple exist, one is chosen at random. If none exist, return a default (1,3).
    """
    nonabsorbing_neighbors = []
    for a in actions.keys():
        candidate = step_environment(absorbing_state, a)
        if not is_absorbing(candidate):
            nonabsorbing_neighbors.append(candidate)
    if nonabsorbing_neighbors:
        return random.choice(nonabsorbing_neighbors)
    else:
        return (1, 3)  # Fallback

# Learning parameters.
alpha = 0.5      # SARSA learning rate.
gamma = 0.5      # Discount factor (updated to 0.5)
beta = 0.5       # Supervised update (demonstration) learning rate.
demo_target = 10 # Target Q-value for a demonstrated (good) action.

# ----------------------------------
# Initialize simulation in session_state
# ----------------------------------

if 'Q' not in st.session_state:
    st.session_state.Q = {}  # Q-table stored as a dict keyed by (state, action)
if 'current_state' not in st.session_state:
    st.session_state.current_state = (1, 3)  # Start at (1,3)
if 'path' not in st.session_state:
    st.session_state.path = [st.session_state.current_state]
if 'visited_states' not in st.session_state:
    st.session_state.visited_states = {st.session_state.current_state}
if 'demonstration_set' not in st.session_state:
    st.session_state.demonstration_set = set()  # States with demonstration provided
if 'shift_history' not in st.session_state:
    st.session_state.shift_history = []  # List of (step, shift) pairs
if 'step_count' not in st.session_state:
    st.session_state.step_count = 0
if 'episode_count' not in st.session_state:
    st.session_state.episode_count = 1
if 'episode_active' not in st.session_state:
    # Flag indicating if the current episode is active.
    st.session_state.episode_active = True
if 'auto_run' not in st.session_state:
    st.session_state.auto_run = False
if 'episode_rewards' not in st.session_state:
    st.session_state.episode_rewards = []
if 'current_episode_reward' not in st.session_state:
    st.session_state.current_episode_reward = 0

# ----------------------------------
# Simulation Step Function
# ----------------------------------

def simulate_step(epsilon, auto_supervision=True):
    """
    Run one simulation step.
    If auto_supervision is False, the demonstration update is skipped.
    """
    Q = st.session_state.Q
    state = st.session_state.current_state

    # --- If previous episode ended, perform an "escape" step ---
    if not st.session_state.episode_active:
        escape_state = get_escape_state(state)
        st.info(f"Escaping from absorbing state {state} to nonabsorbing state {escape_state} for a new episode.")
        st.session_state.path = [escape_state]  # Clear previous episode's path
        st.session_state.current_state = escape_state
        st.session_state.episode_active = True
        return

    # --- Normal episode step ---
    # If the current state is absorbing, end the episode.
    if is_absorbing(state):
        st.session_state.episode_active = False
        st.session_state.episode_count += 1
        # Record the cumulative reward for this episode.
        st.session_state.episode_rewards.append(st.session_state.current_episode_reward)
        st.session_state.current_episode_reward = 0
        st.info(f"Episode ended at absorbing state {state}. Click 'Next Step' to escape and start a new episode.")
        return

    # Select an action via ε-greedy.
    action = epsilon_greedy_action(state, Q, epsilon)

    # --- Supervisor Demonstration (DAgger) ---
    if auto_supervision:  # Only update with demonstration if supervision is enabled.
        demo_action = expert_policy(state)
        if demo_action is not None:
            if state not in st.session_state.demonstration_set:
                st.session_state.demonstration_set.add(state)
            key_demo = (state, demo_action)
            current_demo_val = Q.get(key_demo, 0)
            Q[key_demo] = current_demo_val + beta * (demo_target - current_demo_val)

    # --- Environment Interaction ---
    next_state = step_environment(state, action)
    reward = reward_matrix.get(next_state, 0)  # Only nonzero for absorbing states.
    
    # Accumulate reward for the episode.
    st.session_state.current_episode_reward += reward

    # For on-policy SARSA, if next state is not absorbing, select a next action.
    if not is_absorbing(next_state):
        next_action = epsilon_greedy_action(next_state, Q, epsilon)
        q_next = Q.get((next_state, next_action), 0)
    else:
        q_next = 0

    # SARSA update.
    key_sa = (state, action)
    current_q = Q.get(key_sa, 0)
    Q[key_sa] = current_q + alpha * (reward + gamma * q_next - current_q)
    
    # Update visited states.
    st.session_state.visited_states.add(state)
    # Compute distributional shift: states visited without demonstration.
    shift = len(st.session_state.visited_states - st.session_state.demonstration_set)
    st.session_state.shift_history.append((st.session_state.step_count, shift))
    
    # Append next_state to the path and update state.
    st.session_state.path.append(next_state)
    st.session_state.current_state = next_state
    st.session_state.step_count += 1

    # If the next state is absorbing, then end the episode.
    if is_absorbing(next_state):
        st.session_state.episode_active = False
        st.session_state.episode_count += 1
        st.session_state.episode_rewards.append(st.session_state.current_episode_reward)
        st.session_state.current_episode_reward = 0
        st.info(f"Episode ended at absorbing state {next_state}. Click 'Next Step' to escape and start a new episode.")

# ----------------------------------
# Streamlit UI: Sidebar Controls
# ----------------------------------

st.sidebar.header("Simulation Controls")

# Adjust exploration rate.
epsilon = st.sidebar.slider("Epsilon (exploration rate)",
                            min_value=0.0,
                            max_value=1.0,
                            value=0.7,
                            step=0.05)

# Button to manually take one simulation step.
if st.sidebar.button("Next Step"):
    simulate_step(epsilon)

# Buttons to control auto-run.
if st.sidebar.button("Start Auto-Run"):
    st.session_state.auto_run = True

if st.sidebar.button("Stop Auto-Run"):
    st.session_state.auto_run = False

# Button to reset simulation.
if st.sidebar.button("Reset Simulation"):
    st.session_state.Q = {}
    st.session_state.current_state = (1, 3)
    st.session_state.path = [(1, 3)]
    st.session_state.visited_states = {(1, 3)}
    st.session_state.demonstration_set = set()
    st.session_state.shift_history = []
    st.session_state.step_count = 0
    st.session_state.episode_count = 1
    st.session_state.current_episode_reward = 0
    st.session_state.episode_rewards = []
    st.session_state.auto_run = False
    st.session_state.episode_active = True
    st.rerun()

# ----------------------------------
# Main Display: Q-table, Show-Robot Path, Distributional Shift, and Rewards Plot
# ----------------------------------

st.header("Show-Robot Simulation with DAgger (Learning from Demonstrations)")
st.write(f"**Current State:** {st.session_state.current_state}")
st.write(f"**Episode:** {st.session_state.episode_count}   **Step:** {st.session_state.step_count}")

# Display Q-table.
states = [(x, y) for x in range(1, 4) for y in range(1, 4)]
q_data = {}
for s in states:
    row = {}
    for a in actions.keys():
        row[a] = st.session_state.Q.get((s, a), 0)
    q_data[str(s)] = row
df_q = pd.DataFrame(q_data).T
st.subheader("Q-Table")
st.dataframe(df_q)

# Plot the current episode's Show-Robot path.
fig, ax = plt.subplots()
ax.set_title("Show-Robot Path (Current Episode)")
ax.set_xlim(0.5, 3.5)
ax.set_ylim(0.5, 3.5)
ax.set_xticks(range(1, 4))
ax.set_yticks(range(1, 4))
ax.grid(True)
path = st.session_state.path
xs = [s[0] for s in path]
ys = [s[1] for s in path]
ax.plot(xs, ys, marker='o', color='blue')
# Mark absorbing states.
for state, rew in reward_matrix.items():
    ax.scatter(state[0], state[1], marker='s', color='red', s=200)
    ax.text(state[0], state[1], f"{rew}", color='white', ha='center', va='center')
st.pyplot(fig)

# Plot distributional shift.
if st.session_state.shift_history:
    steps, shifts = zip(*st.session_state.shift_history)
    fig2, ax2 = plt.subplots()
    ax2.plot(steps, shifts, marker='o')
    ax2.set_title("Distributional Shift")
    ax2.set_xlabel("Step")
    ax2.set_ylabel("States visited without demonstration")
    st.pyplot(fig2)

# Plot rewards over episodes.
if st.session_state.episode_rewards:
    fig3, ax3 = plt.subplots()
    episodes = range(1, len(st.session_state.episode_rewards) + 1)
    rewards = st.session_state.episode_rewards
    ax3.plot(episodes, rewards, marker='o')
    ax3.set_title("Rewards Over Episodes")
    ax3.set_xlabel("Episode")
    ax3.set_ylabel("Reward")
    st.pyplot(fig3)

# ----------------------------------
# Auto-Run Loop: Continue simulation automatically if enabled.
# ----------------------------------
if st.session_state.auto_run:
    # Run one simulation step (without supervision to focus on Q-learning updates)
    simulate_step(epsilon, auto_supervision=False)
    time.sleep(0.5)
    st.rerun()
