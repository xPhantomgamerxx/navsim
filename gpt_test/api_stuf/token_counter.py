import tiktoken
import json

# Choose the right encoding for the model you're using
# For GPT-4o or GPT-4 Turbo models, use `cl100k_base`
encoding = tiktoken.get_encoding("cl100k_base")

def count_tokens_in_message(message, system_message=None):
    """
    Count tokens in a single OpenAI API call message with optional system message.
    """
    total_tokens = 0
    full_payload = []

    # Add the system message if present
    if system_message:
        full_payload.append({"role": "system", "content": system_message})

    full_payload.extend(message)

    for msg in full_payload:
        # Token count for message structure
        total_tokens += 4  # role + content overhead

        # Count content tokens (can be str or list of dicts for images + text)
        if isinstance(msg["content"], str):
            total_tokens += len(encoding.encode(msg["content"]))
        elif isinstance(msg["content"], list):
            for item in msg["content"]:
                if item["type"] == "input_text":
                    total_tokens += len(encoding.encode(item["text"]))
                elif item["type"] == "input_image":
                    # Estimate 250 tokens per image (adjust as needed)
                    total_tokens += 1105
                # Include type + image_url overhead
                total_tokens += 2

    # Add assistant priming token overhead
    total_tokens += 2

    return total_tokens

with open("jsons/prompt.json", "r") as f:
    loaded_data = json.load(f)

print("Estimated total tokens:", count_tokens_in_message(loaded_data))
