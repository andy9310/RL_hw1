import numpy as np
from gridworld import GridWorld


class DynamicProgramming:
    """Base class for dynamic programming algorithms"""

    def __init__(self, grid_world: GridWorld, discount_factor: float = 1.0):
        """Constructor for DynamicProgramming

        Args:
            grid_world (GridWorld): GridWorld object
            discount_factor (float, optional): Discount factor gamma. Defaults to 1.0.
        """
        self.grid_world = grid_world
        self.discount_factor = discount_factor
        self.threshold = 1e-4  # default threshold for convergence
        self.values = np.zeros(grid_world.get_state_space())  # V(s)
        self.policy = np.zeros(grid_world.get_state_space(), dtype=int)  # pi(s)

    def set_threshold(self, threshold: float) -> None:
        """Set the threshold for convergence

        Args:
            threshold (float): threshold for convergence
        """
        self.threshold = threshold

    def get_policy(self) -> np.ndarray:
        """Return the policy

        Returns:
            np.ndarray: policy
        """
        return self.policy

    def get_values(self) -> np.ndarray:
        """Return the values

        Returns:
            np.ndarray: values
        """
        return self.values

    def get_q_value(self, state: int, action: int) -> float:
        """Get the q-value for a state and action

        Args:
            state (int)
            action (int)

        Returns:
            float
        """
        # TODO: Get reward from the environment and calculate the q-value
        next_state, reward, done = self.grid_world.step(state, action)
        done_flag = 1
        if done:
            done_flag = 0
        ## since step is 100% to the next_state, transition probability is 1
        return (reward + self.discount_factor * self.values[next_state] * done_flag)*1
    
    def get_q_value_next_state(self, state: int, action: int):
        next_state, reward, done = self.grid_world.step(state, action)
        done_flag = 1
        if done:
            done_flag = 0
            return (reward + self.discount_factor * self.values[next_state] * done_flag)*1 , -1
        ## since step is 100% to the next_state, transition probability is 1
        return (reward + self.discount_factor * self.values[next_state] * done_flag)*1 , next_state

class IterativePolicyEvaluation(DynamicProgramming):
    def __init__(
        self, grid_world: GridWorld, policy: np.ndarray, discount_factor: float
    ):
        """Constructor for IterativePolicyEvaluation

        Args:
            grid_world (GridWorld): GridWorld object
            policy (np.ndarray): policy (probability distribution state_spacex4)
            discount (float): discount factor gamma
        """
        super().__init__(grid_world, discount_factor)
        self.policy = policy

    def get_state_value(self, state: int) -> float:
        """Get the value for a state

        Args:
            state (int)

        Returns:
            float: value
        """
        # TODO: Get the value for a state by calculating the q-values
        # 
        action_space = self.grid_world.get_action_space()
        state_value = 0.0
        for action in range(action_space):
            action_probability = self.policy[state][action] 
            q_value = self.get_q_value(state, action) 
            state_value += action_probability * q_value

        return state_value

    def evaluate(self):
        """Evaluate the policy and update the values for one iteration"""
        # TODO: Implement the policy evaluation step
        ## since the value function needs to update synchronously
        new_values = np.zeros_like(self.values) 
        for state in range(self.grid_world.get_state_space()):
            new_values[state] = self.get_state_value(state)
        self.values = new_values

    def run(self) -> None:
        """Run the algorithm until convergence."""
        # TODO: Implement the iterative policy evaluation algorithm until convergence
        while True:
            old_values = self.values.copy() 
            self.evaluate() 
            delta = np.max(np.abs(self.values - old_values))
            if delta < self.threshold:
                break


class PolicyIteration(DynamicProgramming):
    def __init__(self, grid_world: GridWorld, discount_factor: float = 1.0):
        """Constructor for PolicyIteration

        Args:
            grid_world (GridWorld): GridWorld object
            discount_factor (float, optional): Discount factor gamma. Defaults to 1.0.
        """
        super().__init__(grid_world, discount_factor)

    def get_state_value(self, state: int) -> float:
        """Get the value for a state

        Args:
            state (int)

        Returns:
            float
        """
        # TODO: Get the value for a state by calculating the q-values
        return self.get_q_value(state, self.policy[state]) 

    def policy_evaluation(self):
        """Evaluate the policy and update the values"""
        # TODO: Implement the policy evaluation step
        while True:
            ## since the value function needs to update synchronously
            new_values = np.zeros_like(self.values)
            for state in range(self.grid_world.get_state_space()):
                new_values[state] = self.get_state_value(state)
            delta = np.max(np.abs(new_values - self.values))
            self.values = new_values
            if delta < self.threshold:
                break

    def policy_improvement(self):
        """Improve the policy based on the evaluated values"""
        # TODO: Implement the policy improvement step
        policy_stable = True
        for state in range(self.grid_world.get_state_space()):
            old_action = self.policy[state]  # Remember the old action
            argmax_act = 0
            max_q_value = -100000
            for action in range( self.grid_world.get_action_space() ):
                q_value = self.get_q_value(state, action)
                if q_value > max_q_value:
                    max_q_value = q_value
                    argmax_act = action
            self.policy[state] = argmax_act
            if old_action != argmax_act:
                policy_stable = False
        return policy_stable

    def run(self) -> None:
        """Run the algorithm until convergence"""
        # TODO: Implement the policy iteration algorithm until convergence
        while True:
            self.policy_evaluation()
            if self.policy_improvement():
                break


class ValueIteration(DynamicProgramming):
    def __init__(self, grid_world: GridWorld, discount_factor: float = 1.0):
        """Constructor for ValueIteration

        Args:
            grid_world (GridWorld): GridWorld object
            discount_factor (float, optional): Discount factor gamma. Defaults to 1.0.
        """
        super().__init__(grid_world, discount_factor)

    def get_state_value(self, state: int) -> float:
        """Get the value for a state

        Args:
            state (int)

        Returns:
            float
        """
        # TODO: Get the value for a state by calculating the q-values
        action_space = self.grid_world.get_action_space()
        max = -10000
        for action in range(action_space):
            q_value = self.get_q_value(state, action)
            if q_value > max:
                max = q_value
        return max

    def policy_evaluation(self):
        """Evaluate the policy and update the values"""
        # TODO: Implement the policy evaluation step
        """Perform one iteration of value iteration, updating all state values."""
        new_values = np.zeros_like(self.values) 
        for state in range(self.grid_world.get_state_space()):
            new_values[state] = self.get_state_value(state) 
        self.values = new_values

    def policy_improvement(self):
        """Improve the policy based on the evaluated values"""
        # TODO: Implement the policy improvement step
        # find the argmax action
        for state in range(self.grid_world.get_state_space()):
            action_space = self.grid_world.get_action_space()
            max = -10000
            max_action = 0
            for action in range(action_space):
                q_value = self.get_q_value(state, action)
                if q_value > max:
                    max = q_value
                    max_action = action
            self.policy[state] = max_action

    def run(self) -> None:
        """Run the algorithm until convergence"""
        # TODO: Implement the value iteration algorithm until convergence
        while True:
            old_values = self.values.copy()
            self.policy_evaluation() 
            delta = np.max(np.abs(self.values - old_values))
            if delta < self.threshold: 
                break
        self.policy_improvement()


class AsyncDynamicProgramming(DynamicProgramming):
    def __init__(self, grid_world: GridWorld, discount_factor: float = 1.0):
        """Constructor for ValueIteration

        Args:
            grid_world (GridWorld): GridWorld object
            discount_factor (float, optional): Discount factor gamma. Defaults to 1.0.
        """
        super().__init__(grid_world, discount_factor)
        self.predecessors = []
        self.priority_queue = []
        self.best_transition = []

    def get_state_value(self, state: int) -> float:
        action_space = self.grid_world.get_action_space()
        max = -10000
        for action in range(action_space):
            q_value = self.get_q_value(state, action)
            if q_value > max:
                max = q_value
        return max
    
    def get_state_value_next_state(self, state: int):
        action_space = self.grid_world.get_action_space()
        max = -10000
        tmp_state = -1
        for action in range(action_space):
            q_value,next_state = self.get_q_value_next_state(state, action)
            if q_value > max:
                max = q_value
                tmp_state = next_state
        return max, tmp_state

    def policy_improvement(self):
        """Improve the policy based on the evaluated values"""
        # TODO: Implement the policy improvement step
        # find the argmax action
        for state in range(self.grid_world.get_state_space()):
            action_space = self.grid_world.get_action_space()
            max = -10000
            max_action = 0
            for action in range(action_space):
                q_value = self.get_q_value(state, action)
                if q_value > max:
                    max = q_value
                    max_action = action
            self.policy[state] = max_action
    
    def InplaceDP(self): ## based on value iteration that only stores one copy of value function
        while True:
            delta = 0
            for state in range(self.grid_world.get_state_space()):
                old_value = self.values[state]
                self.values[state] = self.get_state_value(state)
                delta = max(delta, abs(old_value - self.values[state]))
            if delta < self.threshold:
                break
        self.policy_improvement()
    
    def update_state(self, state): ## for priority sweeping 
        old_value = self.values[state]
        self.values[state] = self.get_state_value(state)
        bellman_error = abs(old_value - self.values[state])
        if bellman_error > self.threshold:
            self.priority_queue.append((bellman_error, state))
            self.priority_queue.sort(reverse=True, key=lambda x: x[0])
            for pred_state in range(self.grid_world.get_state_space()):
                if pred_state != state:
                    old_value = self.values[pred_state]
                    self.values[pred_state] = self.get_state_value(pred_state)
                    bellman_error = abs(old_value - self.values[pred_state])
                    if bellman_error > self.threshold:
                        self.priority_queue = [(bellman_error, pred_state) if item[1] == pred_state else item for item in self.priority_queue]
                        # self.priority_queue.append((bellman_error, pred_state))
                        self.priority_queue.sort(reverse=True, key=lambda x: x[0])

    
    def PrioritizedSweeping(self): 
        
        for state in range(self.grid_world.get_state_space()):
            self.priority_queue.append((0, state))
        
        while len(self.priority_queue) != 0:
            state = self.priority_queue.pop(0)[1]
            self.update_state(state)
        self.policy_improvement()

    def realTimeDP(self): ## best : 240 max_steps
        
        index = 0
        while index < self.grid_world.get_state_space():
            delta = 0
            state = index
            while state != -1:
                old_value = self.values[state]
                self.values[state], next = self.get_state_value_next_state(state)
                delta = max(delta, abs(old_value - self.values[state]))
                state = next
            if delta < self.threshold:
                index = index + 1
                    
        
        # Extract the optimal policy
        self.policy_improvement()
    def run(self) -> None:
        """Run the algorithm until convergence"""
        # TODO: Implement the async dynamic programming algorithm until convergence
        self.realTimeDP()
