from openai import OpenAI
client = OpenAI()

response = client.files.create(
  file=open("/home/ubuntu/project_ws/navsim/gpt_test/splits/finetune_val_data_fixed_fixed.jsonl", "rb"),
  purpose="fine-tune"
)

print(response)