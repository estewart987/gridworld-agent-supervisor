import re
from litellm import completion
import Minigrid.minigrid.core.constants as mini_cons

DIR_TO_VEC = ["right", "down", "left", "up"]
IDX_TO_STATE = dict(zip(mini_cons.STATE_TO_IDX.values(), mini_cons.STATE_TO_IDX.keys()))

def tile_encoding_to_description(tile_encoding: tuple[int, int, int]) -> str:
    """
    Convert a tile's encoding into a human-readable description.

    Args:
        tile_encoding (tuple): A tuple containing three integers:
            - object_idx (int): Index of the object on the tile.
            - color_idx (int): Index of the color of the object.
            - state_idx (int): Index representing the state of the object.

    Returns:
        str: A description of the tile, including its object type, color, and state.
    """
    object_idx, color_idx, state_idx = tile_encoding
    object_name = mini_cons.IDX_TO_OBJECT[object_idx]
    color_name = mini_cons.IDX_TO_COLOR[color_idx]
    # Handle specific cases for agent, empty tiles, and walls
    if object_idx == mini_cons.OBJECT_TO_IDX["agent"]:
        state_name = "facing " + DIR_TO_VEC[state_idx]
    elif object_idx == mini_cons.OBJECT_TO_IDX["empty"]:
        return "An open, empty tile."
    elif object_idx == mini_cons.OBJECT_TO_IDX["wall"]:
        return "A wall tile."
    else:
        state_name = IDX_TO_STATE[state_idx]
    return f"A {color_name} {object_name} that is {state_name}."

def query_llm(prompt: str, messages: list) -> str:
    """
    Send a prompt to the LLM and update the message history.

    Args:
        prompt (str): The user's prompt to the LLM.
        messages (list): A list of message dictionaries representing the conversation history.
            Each dictionary has keys "content" (str) and "role" (str).

    Returns:
        str: The LLM's response as a string.
    """
    messages.append({"content": prompt, "role": "user"})
    response = completion(model="ollama_chat/llama3.2:3b", messages=messages)
    response = response.choices[0].message.content
    messages.append({"content": response, "role": "assistant"})
    return response

def parse_llm_response(response: str) -> str:
    """
    Extract an action from the LLM's response using a specific format.

    Args:
        response (str): The LLM's response containing the action.

    Returns:
        str or None: The extracted action in lowercase if found, otherwise None.
    """
    try:
        pattern = r"Action:\s*(.*)"
        match = re.search(pattern, response)
        return match.group(1).lower().strip()
    except AttributeError:
        return None

def interact_with_llm(messages: list, action: str) -> tuple:
    """
    Facilitate an interactive conversation with the LLM.

    Args:
        messages (list): A list of message dictionaries representing the conversation history.
            Each dictionary has keys "content" (str) and "role" (str).
        action (str): The current action extracted from the LLM's responses.

    Returns:
        tuple: A tuple containing:
            - user_input (str): User's final input ("continue" or "yes").
            - action (str): The most recent action parsed from the LLM's response.
    """
    while True:
        user_input = input("[You] Enter a message (or type 'continue' to proceed or 'yes' to accept action): ").strip()
        # Exit the loop if the user decides to proceed or accept the action
        if user_input.lower() in ['continue', 'yes']:
            return user_input.lower(), action
        messages.append({"content": user_input, "role": "user"})
        response = completion(model="ollama_chat/llama3.2:3b", messages=messages)
        response = response.choices[0].message.content
        messages.append({"content": response, "role": "assistant"})
        print(f"[Llama] {response}")
        # Update the action based on the LLM's response
        action = parse_llm_response(response)
