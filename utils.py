import re
from litellm import completion
import Minigrid.minigrid.core.constants as mini_cons

DIR_TO_VEC = ["right", "down", "left", "up"]
IDX_TO_STATE = dict(zip(mini_cons.STATE_TO_IDX.values(), mini_cons.STATE_TO_IDX.keys()))

def tile_encoding_to_description(tile_encoding):
    object_idx, color_idx, state_idx = tile_encoding
    object_name = mini_cons.IDX_TO_OBJECT[object_idx]
    color_name = mini_cons.IDX_TO_COLOR[color_idx]
    if object_idx == mini_cons.OBJECT_TO_IDX["agent"]:
        state_name = "facing " + DIR_TO_VEC[state_idx]
    elif object_idx == mini_cons.OBJECT_TO_IDX["empty"]:
        return "An open, empty tile."
    elif object_idx == mini_cons.OBJECT_TO_IDX["wall"]:
        return "A wall tile."
    else:
        state_name = IDX_TO_STATE[state_idx]
    return f"A {color_name} {object_name} that is {state_name}."

def query_llm(prompt, messages):
    messages.append({"content": prompt, "role": "user"})
    response = completion(model="ollama_chat/llama3.2:3b", messages=messages)
    response = response.choices[0].message.content
    messages.append({"content": response, "role": "assistant"})
    return response

def parse_llm_response(response):
    try:
        pattern = r"Action:\s*(.*)"
        match = re.search(pattern, response)
        return match.group(1).lower().strip()
    except AttributeError:
        return None

def interact_with_llm(messages):
    while True:
        user_input = input("[You] Enter a message (or type 'continue' to proceed or 'yes' to accept action): ").strip()
        if user_input.lower() in ['continue', 'yes']:
            return user_input.lower()
        messages.append({"content": user_input, "role": "user"})
        response = completion(model="ollama_chat/llama3.2:3b", messages=messages)
        response = response.choices[0].message.content
        messages.append({"content": response, "role": "assistant"})
        print(f"[Llama] {response}")
