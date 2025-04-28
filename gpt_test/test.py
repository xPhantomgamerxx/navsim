from openai import OpenAI
import os

myapi_key = os.environ.get("OPENAI_API_KEY")
if myapi_key is None:
    print(f"Please set OPENAI_API_KEY in your environment variables.")
    exit()
client = OpenAI(api_key=myapi_key)

completions = client.chat.completions.list()
print(completions)
for completion in completions:
    print(completion)