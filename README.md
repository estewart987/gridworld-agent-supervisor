# Minigrid Llama Agent

This repository provides a Python-based framework for interacting with a Minigrid environment using an LLM (Llama 3.2:3b). The system allows users to control an agent in a gridworld environment, leveraging the capabilities of the LLM to generate human-readable instructions and make informed decisions for agent actions.

## Features
- **Interactive Gridworld Environment:** Navigate a Minigrid environment (`MiniGrid-GoToDoor-8x8-v0`) rendered in real-time.
- **LLM-Driven Decision Making:** Use Llama 3.2:3b to interpret grid descriptions, generate actions, and provide reasoning for its choices.
- **Flexible User Interaction:** Users can interact dynamically with the LLM to ask clarifying questions, provide corrections, or accept/reject the suggested actions.
- **Reusable and Modular Code:** Core logic is encapsulated in reusable classes and utility functions, making it easy to extend or adapt the framework for other environments or LLMs.

## Installation

### Prerequisites
- Python 3.8 or higher
- [Gymnasium](https://github.com/Farama-Foundation/Gymnasium) and [Minigrid](https://github.com/Farama-Foundation/Minigrid)
- Llama 3.2:3b server accessible at `http://localhost:11434/api/v1/chat`

### Steps
1. Clone this repository:
   ```bash
   git clone https://github.com/estewart987/minigrid_llama.git
   cd minigrid_llama
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

3. Ensure your Llama server is running and accessible at `http://localhost:11434/api/v1/chat`.

4. Install Minigrid:
   ```bash
   pip install gym-minigrid
   ```

5. Verify installation:
   ```bash
   python run_minigrid_agent.py
   ```

## Usage

### Run the Agent
To start the interaction with the Minigrid environment, execute:
```bash
python run_minigrid_agent.py
```

### How It Works
1. **Environment Initialization:** The agent is placed in a Minigrid environment with a random initial state.
2. **Grid Description:** The environment is translated into a human-readable format, which includes descriptions of each tile (e.g., `A red agent that is facing right`).
3. **LLM Interaction:** The agent queries the LLM to determine the next action based on the current grid state and mission.
4. **Dynamic User Interaction:** Users can interact with the LLM to clarify or modify the suggested action before it is executed in the environment.
5. **Action Execution:** The chosen action is executed, and the grid is updated in real time.
6. **Completion:** The process repeats until the task is completed or the environment terminates.

### User Input Commands
- **`yes`:** Accept the LLM's suggested action and proceed.
- **`continue`:** Skip further interaction and move forward.
- **Any other input:** Engage in conversation with the LLM to ask clarifying questions or provide feedback.

## Repository Structure
```
<your-repo-name>/
├── minigrid_llama_agent.py  # Main class handling agent logic
├── utils.py                 # Utility functions for LLM interaction and grid description
├── run_minigrid_agent.py    # Script for running the agent
├── requirements.txt         # Python dependencies
└── README.md                # Documentation
```

## Future Enhancements
- Support BabyAI environments.
- Integrate other LLMs for broader functionality.
- Add logging and metrics for tracking LLM performance and decision accuracy.

## License
This project is licensed under the MIT License. See `LICENSE` for details.

## Acknowledgments
- [Minigrid](https://github.com/Farama-Foundation/Minigrid)
- [Llama 3.2:3b](https://ollama.ai/)

