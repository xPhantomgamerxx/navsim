from openai import OpenAI
import os

myapi_key = os.environ.get("OPENAI_API_KEY")
if myapi_key is None:
    print(f"Please set OPENAI_API_KEY in your environment variables.")
    exit()
client = OpenAI(api_key=myapi_key)

message = {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Hello, how are you?"},
                    ],
                },

response = client.chat.completions.create(
    model = "ft:gpt-4.1-2025-04-14:scania-eearp:av-finetune-7:BNKqGQNC", #"gpt-4.1"
    messages = message,
    max_completion_tokens = 100,
    store = True,
    metadata={"token": "test_token"}
)
print(response)