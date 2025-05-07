import json
import numpy as np

with open("/home/ubuntu/project_ws/navsim/navsim/planning/script/datasplit_builder/data2.txt", "r") as f:
    lines = f.readlines()


random_list = []
for i in range(100):
    token = np.random.choice(lines)
    random_list.append(token.strip())
    lines.remove(token)

with open("/home/ubuntu/project_ws/navsim/navsim/planning/script/datasplit_builder/output2.txt", "w") as f:
    for token in random_list:
        f.write(token.strip() + "\n")
