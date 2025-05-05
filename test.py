# message = []
# message.append({
#     "role": "developer",
#     "content": f"system_message"}
# )

imgs = [[1,2,3], [4,5,6], [7,8,9], [10,11,12]] 


# message.append({
#     "role": "user",
#     "content": [
#         {"type": "text", "text": "These are the images at timestep t-3 in order front-left, front, front-right."},
#         {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{imgs[0][0]}"}},
#         {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{imgs[0][1]}"}},
#         {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{imgs[0][2]}"}},
#         {"type": "text", "text": "These are the images at timestep t-2 in order front-left, front, front-right."},
#         {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{imgs[1][0]}"}},
#         {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{imgs[1][1]}"}},
#         {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{imgs[1][2]}"}},
#         {"type": "text", "text": "These are the images at timestep t-1 in order front-left, front, front-right."},
#         {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{imgs[2][0]}"}},
#         {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{imgs[2][1]}"}},
#         {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{imgs[2][2]}"}},
#         {"type": "text", "text": "These are the images at timestep t-0 in order front-left, front, front-right."},
#         {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{imgs[3][0]}"}},
#         {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{imgs[3][1]}"}},
#         {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{imgs[3][2]}"}},
#         {"type": "text", "text": f"""Using the provided images, you need to complete these  following instructions and questions.
# ---
# 1.scene_description_prompt

# ---
# 2.object_description_prompt

# ---
# 3.commandintent_description_prompt

# ---
# 4. The historical waypoints of the ego car of the last 2 seconds at an interval of 0.5s up until the present are: past_waypoints. prediction_prompt_waypoints"""},
#         ]
#     }
# )

message = []

message.append({
    "role": "developer",
    "content": "system_message"
})

content = []
timesteps = ["t-3", "t-2", "t-1", "t-0"]

for i, timestep in enumerate(timesteps):
    content.append({
        "type": "text",
        "text": f"These are the images at timestep {timestep} in order front-left, front, front-right."
    })
    for img in imgs[i]:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{img}"}
        })

content.append({
    "type": "text",
    "text": """Using the provided images, you need to complete these  following instructions and questions.
---
1.scene_description_prompt

---
2.object_description_prompt

---
3.commandintent_description_prompt

---
4. The historical waypoints of the ego car of the last 2 seconds at an interval of 0.5s up until the present are: past_waypoints. prediction_prompt_waypoints"""
})

# Add user message
message.append({
    "role": "user",
    "content": content
})

print(message)