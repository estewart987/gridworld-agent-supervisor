from minigrid_llama_agent import MinigridLlamaAgent

if __name__ == "__main__":
    environment_name = 'MiniGrid-GoToDoor-8x8-v0'
    llm_server_url = "http://localhost:11434/api/v1/chat"

    agent = MinigridLlamaAgent(environment_name, llm_server_url)
    agent.reset_environment()
    agent.interact()
