from minigrid_llama_agent import MinigridLlamaAgent

if __name__ == "__main__":
    # Set Minigrid environment you want to use
    environment_name = 'MiniGrid-GoToDoor-8x8-v0'
    llm_server_url = "http://localhost:11434/api/v1/chat"

    # Activate environment and chat with Llama
    agent = MinigridLlamaAgent(environment_name, llm_server_url)
    agent.reset_environment()
    agent.interact()
