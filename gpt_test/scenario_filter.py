import json

threshold = 5
used_tokens = []

with open("/home/ubuntu/project_ws/navsim/gpt_test/jsons/difficulties_filtered.jsonl", 'r') as f:
    for line in f:
        existing_entry = json.loads(line)
        used_tokens.append(existing_entry.get("token"))

with open("/home/ubuntu/project_ws/navsim/gpt_test/jsons/difficulties.jsonl", 'r') as infile, open("/home/ubuntu/project_ws/navsim/gpt_test/jsons/difficulties_filtered.jsonl", 'a') as outfile, open("/home/ubuntu/project_ws/navsim/gpt_test/jsons/difficult_tokens.jsonl", "a") as tokenfile:
    for line in infile:
        entry = json.loads(line)
        if entry.get("difficulty", 0) > threshold and entry.get("token") not in used_tokens:
            json.dump(entry, outfile)
            json.dump({"token": entry.get("token")}, tokenfile)
            outfile.write('\n')
            tokenfile.write('\n')
      