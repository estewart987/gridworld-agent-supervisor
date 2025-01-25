import gymnasium as gym
from minigrid_llama_agent import MinigridLlamaAgent

if __name__ == "__main__":
    print("Please choose a MiniGrid environment from those listed in the MiniGrid documentation:")
    print("https://minigrid.farama.org/environments/minigrid/")

    env_name = None
    while env_name is None:
        user_input = input("Enter the name of the MiniGrid environment you want to use: ")

        try:
            gym.make(user_input)
            env_name = user_input
        except gym.error.UnregisteredEnv:
            # Handle invalid environments
            print(f"Error: '{user_input}' is not a valid MiniGrid environment.")
            print("Please check the link and enter a correct environment name.")

    llm_server_url = "http://localhost:11434/api/v1/chat"

    # Activate environment and chat with Llama
    agent = MinigridLlamaAgent(env_name, llm_server_url)
    agent.reset_environment()
    agent.interact()
