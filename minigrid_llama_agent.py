import warnings
import logging

# Suppress all UserWarning messages
warnings.filterwarnings("ignore", category=UserWarning)
# Suppress UserWarning messages specifically from gymnasium.envs.registration
warnings.filterwarnings("ignore", category=UserWarning, module="gymnasium.envs.registration")
# Set the logger level for gymnasium.envs.registration to ERROR
logging.getLogger("gymnasium.envs.registration").setLevel(logging.ERROR)

import gymnasium as gym
import numpy as np
from Minigrid.minigrid.wrappers import FullyObsWrapper
from Minigrid.minigrid.core.actions import Actions
import Minigrid.minigrid.core.constants as mini_cons
from utils import tile_encoding_to_description, query_llm, parse_llm_response, interact_with_llm

class MinigridLlamaAgent:
    """
    An agent for interacting with Minigrid environments using an LLM (Large Language Model) 
    for decision-making.

    Attributes:
        env (gym.Env): The wrapped Minigrid environment.
        llm_server_url (str): URL of the LLM server for querying.
        obs (dict): Current observation from the environment.
        messages (list): List of messages exchanged with the LLM.
        action_map (dict): Mapping of action descriptions to environment action identifiers.
    """
    def __init__(self, environment_name, llm_server_url):
        """
        Initialize the agent with a specific environment and LLM server.

        Args:
            environment_name (str): Name of the Minigrid environment.
            llm_server_url (str): URL of the LLM server for querying.
        """
        self.env = FullyObsWrapper(gym.make(environment_name, render_mode='human'))
        self.llm_server_url = llm_server_url
        self.obs = None
        self.messages = []
        self.action_map = {
            'turn left': Actions.left,
            'turn right': Actions.right,
            'move forward': Actions.forward,
            'pickup': Actions.pickup,
            'drop': Actions.drop,
            'toggle': Actions.toggle,
            'task done': Actions.done,
        }

    def reset_environment(self):
        """
        Reset the environment and update the initial observation.
        """
        self.obs, _ = self.env.reset()

    def describe_grid(self):
        """
        Describe the gridworld environment in human-readable language for LLM based on the agent's current observation.

        Returns:
            np.ndarray: A 2D array containing descriptions of each grid tile.
        """
        # Extract grid dimensions and reshape for processing
        obs_x, obs_y, _ = self.obs['image'].shape
        obs_long = self.obs['image'].reshape(obs_x * obs_y, 3)

        # Generate descriptions for each tile
        descriptions = [
            tile_encoding_to_description((x[0], x[1], x[2])) for x in obs_long
        ]
        descriptions = np.array(descriptions).reshape(obs_x, obs_y)

        # Create a formatted description with grid indices
        grid_description = []
        for i, column in enumerate(descriptions):
            for j, tile in enumerate(column):
                grid_description.append(f"({i}, {j}): {tile}")
        return np.array(grid_description).reshape(obs_x, obs_y)

    def interact(self):
        """
        Start the interaction loop where the agent uses the LLM to decide actions 
        and executes them in the environment.
        """
        done = False
        while not done:
            # Generate a description of the grid and mission
            grid_description = self.describe_grid()
            mission = self.obs['mission']
            prompt = (
                "You are in a gridworld. Each tile is described below with its index.\n"
                f"# Grid\n```python\n{grid_description}\n```\n"
                f"Your mission: {mission}.\n"
                "The agent is represented by 'the red agent.' Based on the grid, what step should the agent take to get closer to the objective?\n"
                "Available actions: 'Turn left', 'Turn right', 'Move forward', 'Pickup object', 'Drop object', 'Toggle (open door)', 'Task done'. "
                "Provide an action and a brief explanation for your choice. Explicitly identify the action at the end of your message (e.g., Action: 'Turn left')."
            )
            # Query the LLM for the next action
            response = query_llm(prompt, self.messages)
            print(f"[Llama Response] {response}")

            # Parse the action from the LLM's response
            action = parse_llm_response(response)
            action_id = self.action_map.get(action)

            # Respond to LLM's message and execute action if correct
            user_decision, action = interact_with_llm(self.messages, action)
            action_id = self.action_map.get(action)

            if user_decision == "yes" and action_id is not None:
                # Step the environment with the chosen action
                self.obs, reward, terminated, truncated, _ = self.env.step(action_id)
                print(f"Action: {action}, Reward: {reward}")
                done = terminated or truncated
                self.env.render()

        print("Episode completed.")
        self.env.close()
