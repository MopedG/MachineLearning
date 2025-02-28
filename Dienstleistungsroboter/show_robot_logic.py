from collections import deque

import pandas as pd
import random


class ShowEnvironment:
    services = {
        "QR-Code": [(1, 4)],
        "Survey feedback": [(14, 6), (14, 2), (14, 3), (14, 4), (14, 5)],
        "Customer profile enrichment": [(4, 1), (2, 1), (3, 1)],
        "Exchange scheduling": [(4, 7), (2, 7), (3, 7)],
        "gallery enquiry": [(9, 4), (10, 4)],
        "Concierge service": [(5, 6), (5, 5)]
    }

    block_list = [
        (1, 7),
        (15, 7),
        (1, 6),
        (3, 6),
        (4, 6),
        (6, 6),
        (7, 6),
        (9, 6),
        (10, 6),
        (12, 6),
        (13, 6),
        (15, 6),
        (1, 5),
        (3, 5),
        (4, 5),
        (6, 5),
        (7, 5),
        (9, 5),
        (10, 5),
        (12, 5),
        (13, 5),
        (15, 5),
        (1, 3),
        (3, 3),
        (4, 3),
        (6, 3),
        (7, 3),
        (9, 3),
        (10, 3),
        (12, 3),
        (13, 3),
        (15, 3),
        (1, 2),
        (3, 2),
        (4, 2),
        (6, 2),
        (7, 2),
        (9, 2),
        (10, 2),
        (12, 2),
        (13, 2),
        (15, 2),
        (1, 1),
        (15, 1),
    ]

    reward_matrix = {
        (2, 7): 5,
        (3, 7): 5,
        (4, 7): 5,
        (12, 7): 5,
        (13, 7): 5,
        (14, 7): 5,
        (2, 6): 5,
        (5, 6): 5,
        (14, 6): 5,
        (2, 5): 5,
        (5, 5): 5,
        (14, 5): 5,
        (1, 4): 1,
        (2, 4): 5,
        (6, 4): 1,
        (7, 4): 1,
        (9, 4): 5,
        (10, 4): 5,
        (14, 4): 5,
        (15, 4): 1,
        (2, 3): 5,
        (14, 3): 5,
        (2, 2): 5,
        (14, 2): 5,
        (2, 1): 5,
        (3, 1): 5,
        (4, 1): 5,
        (9, 1): 10,
        (10, 1): 10,
        (12, 1): 1,
        (13, 1): 1,
        (14, 1): 1,
    }

    def __init__(self, dim_x: int, dim_y: int, start_pos: tuple[int, int]):
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.start_pos = start_pos
        self.actions = {
            'N': (0, 1),
            'S': (0, -1),
            'E': (1, 0),
            'W': (-1, 0)
        }

    def is_in_bounds(self, x: int, y: int) -> bool:
        return 1 <= x <= self.dim_x and 1 <= y <= self.dim_y and (x, y) not in ShowEnvironment.block_list

    @staticmethod
    def is_absorbing(state: tuple[int, int], chosen_service: str):
        """Return True if the given state is absorbing (i.e. has a reward)."""
        return state in ShowEnvironment.services[chosen_service]

    def step_environment(self, state: tuple[int, int], action: str) -> tuple[int, int]:
        """Compute the next state given the current state and action (staying within the 3x3 grid)."""
        dx, dy = self.actions[action]
        new_state = (state[0] + dx, state[1] + dy)
        # Stay within bounds.
        if not self.is_in_bounds(new_state[0], new_state[1]):
            return state
        return new_state


class Q_Learning:
    def __init__(self, service: str, alpha: float = 0.5, gamma: float = 0.5, beta: float = 0.5, demo_target: int = 10):
        self.service = service
        self.show_environment = ShowEnvironment(15, 7, (8, 7))
        self.alpha = alpha  # SARSA learning rate.
        self.gamma = gamma  # Discount factor (updated to 0.5)
        self.beta = beta  # Supervised update (demonstration) learning rate.
        self.demo_target = demo_target  # Target Q-value for a demonstrated (good) action.
        self.q_table = self.init_q_table()
        self.current_state = self.show_environment.start_pos
        self.path = [self.show_environment.start_pos]
        self.episode_active = True
        self.episode_count = 0
        self.episode_rewards = []
        self.current_episode_reward = 0
        self.current_action = ""
        self.demonstration_set = set()
        self.visited_states = {self.show_environment.start_pos}
        self.step_count = 0
        self.shift_history = []

    def init_q_table(self):
        tables = {}
        data = {}
        for x in range(1, self.show_environment.dim_x + 1):
            for y in range(1, self.show_environment.dim_y + 1):
                row = {"N": 0.0, "S": 0.0, "E": 0.0, "W": 0.0}
                data[str((x, y))] = row
        return pd.DataFrame(data).T

    def save_to_q_table(self, state: tuple[int, int], action: str,
                        value: int | float) -> None:
        self.q_table.loc[str(state), action] = value

    def epsilon_greedy_action(self, state, Q, epsilon) -> str:
        """Select an action from the Q-table using an ε-greedy strategy."""
        if random.random() < epsilon:
            return random.choice(list(self.show_environment.actions.keys()))
        else:
            q_vals = {a: Q.get(a, {}).get(f"{state}", 0) for a in self.show_environment.actions.keys()}
            max_val = max(q_vals.values())
            best_actions = [a for a, v in q_vals.items() if v == max_val]
            return random.choice(best_actions)

    def expert_policy(self, state) -> str:  # TODO: Funktioniert nicht für einige Ziele (bspw. QR-Code)
        """
        A simple expert that guides the Show-Robot toward the best absorbing state ((3,3) with +100)
        while avoiding other absorbing states.
        Uses a breadth-first search for a safe (shortest) path.
        """
        target = self.show_environment.services[self.service][
            0]  # Der kürzeste Weg von der Startposition zum angeforderten Service
        if state == target:
            return None
        queue = deque([(state, [])])
        visited = set()
        visited.add(state)
        while queue:
            s, path = queue.popleft()
            if s == target:
                return path[0] if path else None
            for a, (dx, dy) in self.show_environment.actions.items():
                ns = (s[0] + dx, s[1] + dy)
                # Stay in bounds.
                if not self.show_environment.is_in_bounds(ns[0], ns[1]):
                    continue
                # Avoid absorbing states unless it's the target.
                if ns in self.show_environment.services[self.service] and ns != target:
                    continue
                if ns not in visited:
                    visited.add(ns)
                    queue.append((ns, path + [a]))
        print("No Path found")
        return random.choice(list(self.show_environment.actions.keys()))

    def save_q_table(self, path: str = "./data/", auto_supervision=False):
        extension = ""
        if auto_supervision:
            extension = "_DAgger"
        self.q_table.to_csv(path + f"{self.service}" + extension + "_Q_Table.csv")

    def load_q_table_from_csv(self, path: str, extension: str) -> None:
        self.q_table = pd.read_csv(f"{path}/{self.service}" + extension + "_Q_Table.csv", index_col=0)

    def simulate_step(self, epsilon, auto_supervision=True) -> str | None:
        """
        Run one simulation step.
        If auto_supervision is False, the demonstration update is skipped.
        """
        q = self.q_table.to_dict()
        state = self.current_state

        # --- If previous episode ended, perform an "escape" step ---
        if not self.episode_active:
            self.current_state = self.show_environment.start_pos
            self.path = [self.show_environment.start_pos]  # Clear previous episode's path
            self.current_state = self.show_environment.start_pos
            self.episode_active = True
            return f"Escaping from absorbing state {self.current_state} to nonabsorbing state {self.show_environment.start_pos} for a new episode."

        # --- Normal episode step ---
        # If the current state is absorbing, end the episode.
        if self.show_environment.is_absorbing(state, self.service):
            self.episode_active = False
            self.episode_count += 1
            # Record the cumulative reward for this episode.
            self.episode_rewards.append(self.current_episode_reward)
            self.current_episode_reward = 0
            return f"Episode ended at absorbing state {state}. Click 'Next Step' to escape and start a new episode."

        # Select an action via ε-greedy.
        action = self.epsilon_greedy_action(state, q, epsilon)
        self.current_action = action
        # --- Supervisor Demonstration (DAgger) ---
        if auto_supervision:  # Only update with demonstration if supervision is enabled.
            demo_action = self.expert_policy(state)
            if demo_action is not None:
                if state not in self.demonstration_set:
                    self.demonstration_set.add(state)
                current_demo_val = q.get(demo_action, {}).get(f"{state}", 0)
                value = current_demo_val + self.beta * (self.demo_target - current_demo_val)
                self.save_to_q_table(state, demo_action, value)

        # --- Environment Interaction ---
        next_state = self.show_environment.step_environment(state, action)
        reward = 0
        if next_state in self.show_environment.services[self.service]:
            reward = self.show_environment.reward_matrix.get(next_state, 0)  # Only nonzero for absorbing states.
        # if state == next_state: # TODO: Brauchen wir theoretisch nicht mehr, sollte aber besprochen werden
        #    reward = -100  # Wenn er gegen Wände läuft, sollte er bestraft werden
        # Accumulate reward for the episode.
        self.current_episode_reward += reward

        # For on-policy SARSA, if next state is not absorbing, select a next action.
        if not self.show_environment.is_absorbing(next_state, self.service):
            next_action = self.epsilon_greedy_action(next_state, q, epsilon)
            q_next = q.get(next_action, {}).get(f"{next_state}", 0)  # TODO: Müssen schauen was hier passiert
        else:
            q_next = 0

        # SARSA update.
        current_q = q.get(action, {}).get(f"{state}", 0)
        value = current_q + self.alpha * (reward + self.gamma * q_next - current_q)
        self.save_to_q_table(state, action, value)

        # Update visited states.
        self.visited_states.add(state)
        # Compute distributional shift: states visited without demonstration.
        shift = len(self.visited_states - self.demonstration_set)
        self.shift_history.append((self.step_count, shift))

        # Append next_state to the path and update state.
        self.path.append(next_state)
        self.current_state = next_state
        self.step_count += 1

        # If the next state is absorbing, then end the episode.
        if self.show_environment.is_absorbing(next_state, self.service):
            self.episode_active = False
            self.episode_count += 1
            self.episode_rewards.append(self.current_episode_reward)
            self.current_episode_reward = 0
            return f"Episode ended at absorbing state {next_state}. Click 'Next Step' to escape and start a new episode."

        return None

    def get_q_table_sum(self):
        return self.q_table.sum().sum()

    def get_optimal_path(self):
        state = self.show_environment.start_pos
        optimal_path = []
        visited_states = [state]
        while not self.show_environment.is_absorbing(state, self.service):
            row = self.q_table.loc[str(state)]
            if row.max() == row.min():
                return None
            best_action = row.idxmax()
            optimal_path.append(best_action)
            state = self.show_environment.step_environment(state, best_action)
            visited_states.append(state)

        return {"path": optimal_path, "states": visited_states}
