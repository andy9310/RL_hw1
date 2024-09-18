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

       # Step into the environment to get the next state, reward, and done flag
        next_state, reward, done = self.grid_world.step(state, action)
        # If the next state is terminal, there's no future value to consider
        if done:
            return reward
        # Otherwise, return the immediate reward plus the discounted future value
        return reward + self.discount_factor * self.values[next_state]

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
        action_space = self.grid_world.get_action_space()
        state_value = 0.0
        # Sum over all possible actions, weighted by the probability of selecting each action
        for action in range(action_space):
            action_probability = self.policy[state, action]  # Probability of taking this action in the given state
            q_value = self.get_q_value(state, action)  # Compute the Q-value for the state-action pair
            state_value += action_probability * q_value  # Weight by the action probability

        return state_value
        # raise NotImplementedError

    def evaluate(self):
        """Evaluate the policy and update the values for one iteration"""
        # TODO: Implement the policy evaluation step
        new_values = np.zeros_like(self.values)  # Initialize a new value array to store updated values
        
        # Loop over all states and update the values based on the policy
        for state in range(self.grid_world.get_state_space()):
            new_values[state] = self.get_state_value(state)
        
        # Update the values with the new values
        self.values = new_values
        # raise NotImplementedError

    def run(self) -> None:
        """Run the algorithm until convergence."""
        # TODO: Implement the iterative policy evaluation algorithm until convergence
        while True:
            old_values = self.values.copy()  # Store the old values to compare for convergence
            self.evaluate()  # Perform one iteration of policy evaluation
            delta = np.max(np.abs(self.values - old_values)) # Calculate the maximum change in value across all states
            if delta < self.threshold: # Check for convergence (if the maximum change is smaller than the threshold)
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
        action = self.policy[state]  # Get the action from the current policy
        return self.get_q_value(state, action)  # Return the Q-value for this state-action pair
        # raise NotImplementedError

    def policy_evaluation(self):
        """Evaluate the policy and update the values"""
        # TODO: Implement the policy evaluation step
        while True:
            new_values = np.zeros_like(self.values)  # Initialize new values for the next iteration
            # Evaluate the policy for each state
            for state in range(self.grid_world.get_state_space()):
                new_values[state] = self.get_state_value(state)
            # Calculate the difference between old values and new values
            delta = np.max(np.abs(new_values - self.values))
            # Update values
            self.values = new_values
            # Stop when the change in values is smaller than the threshold (convergence)
            if delta < self.threshold:
                break
        raise NotImplementedError

    def policy_improvement(self):
        """Improve the policy based on the evaluated values"""
        # TODO: Implement the policy improvement step
        policy_stable = True
        # Iterate over all states to update the policy
        for state in range(self.grid_world.get_state_space()):
            old_action = self.policy[state]  # Remember the old action
            action_space = self.grid_world.get_action_space()
            # Find the action that maximizes the Q-value (greedy improvement)
            q_values = [self.get_q_value(state, action) for action in range(action_space)]
            new_action = np.argmax(q_values)
            # Update the policy with the greedy action
            self.policy[state] = new_action
            # Check if the policy has changed
            if old_action != new_action:
                policy_stable = False
        return policy_stable
        raise NotImplementedError

    def run(self) -> None:
        """Run the algorithm until convergence"""
        # TODO: Implement the policy iteration algorithm until convergence
        while True:
            # Step 1: Policy Evaluation
            self.policy_evaluation()
            # Step 2: Policy Improvement
            if self.policy_improvement():
                # Step 3: Stop if the policy is stable (i.e., no further changes)
                break
        raise NotImplementedError


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
        raise NotImplementedError

    def policy_evaluation(self):
        """Evaluate the policy and update the values"""
        # TODO: Implement the policy evaluation step
        raise NotImplementedError

    def policy_improvement(self):
        """Improve the policy based on the evaluated values"""
        # TODO: Implement the policy improvement step
        raise NotImplementedError

    def run(self) -> None:
        """Run the algorithm until convergence"""
        # TODO: Implement the value iteration algorithm until convergence
        raise NotImplementedError


class AsyncDynamicProgramming(DynamicProgramming):
    def __init__(self, grid_world: GridWorld, discount_factor: float = 1.0):
        """Constructor for ValueIteration

        Args:
            grid_world (GridWorld): GridWorld object
            discount_factor (float, optional): Discount factor gamma. Defaults to 1.0.
        """
        super().__init__(grid_world, discount_factor)

    def run(self) -> None:
        """Run the algorithm until convergence"""
        # TODO: Implement the async dynamic programming algorithm until convergence
        raise NotImplementedError
