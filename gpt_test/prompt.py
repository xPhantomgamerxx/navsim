delimiter = "####"
fix_system_message = f"""
You are a mature driver behind the wheel. You will see 4 consecutive frames. 
Each frame has six images captured and put together by the surround view camera in your vehicle. 
The FOV of each surround view camera is approximately 120°. The view angle is written in the images. 
In the right, it is the bird-eye-view image with labeled agents, centered with your vehicle. 
Each frame mark certain marked agents with bounding box. 
Among all objects, which objects that might have unnormal moving behavior that you might need to pay more attention.
Rank it from the most difficult to predict to the least difficult to predict.
Present your answers in this format:

Output:
The most difficult to predict agents' rank is {delimiter} [A1, A2, A3, A4, A5, A6, A7, A8] {delimiter}. 
Explaination:  
The agent with bounding box ID A1 is a <type>. <explaination>

Replace A1-A8 with the agent number, and make sure the the output includes all agents.
Replace <type> with the object type, e.g. vehicle, pedestrian, cyclist, parked cycle, etc.

message will be delimited with {delimiter} characters.
"""

assistant1_imgs_dir = "/home/x_yiyan/x_yiyanj/code/UniAD/output/vis/demo_test/trainval/17.44[2, 1, 0]_9742c1ad56fa4179a02c606e7929a9af/"
assistant1_output = f"""
Output:
The most difficult to predict agents' rank is {delimiter} [3 2 1] {delimiter}. Explaination: 
The agent with bounding box ID 3 is a vehicle. It is very far away, which might be hard to prediction. 
The agent with bounding box ID 2 is a vehicle. It is going to the intersection, which might be hard to predict as well. 
The agent with bounding box ID 1 is a vehicle. It is parked there while unloading the goods, which is the easiest to predict as there is no movement.
"""

# assistant2_imgs_dir = "/home/x_yiyan/x_yiyanj/code/gpt-4v_UniAD/output/vis/test/trainval/14.49_[5, 4, 6, 2, 3, 1]_f965ad5d2ee94318b2dd1ac927a37856"
# assistent2_output = f"""
# Output:
# The most difficult to predict agents's rank is {delimiter} [5, 4, 6, 2, 3, 1] {delimiter}. 
# Explaination: 
# The agent with bounding box ID 5 is a cyclist. It is moving in a intersection which might be hard to prediction as cyclists can quickly change speed and direction
# The agent with bounding box ID 4 is a vehicle. It might be hard to predict as it is moving in a intersection.
# The agent with bounding box ID 6 is a cement truck. It seems parked there, and it seems no driver on it. So it could be the easy to predict.
# The agent with bounding box ID 2 is a truck. It seems parked there, and it cannot be observed that whehter there is a driver on it, so there is still a possibility it could pull out into traffic.
# The agent with bounding box ID 3 is a vehicle. It is parked, and no human is observed on it. So it could be the easy to predict.
# The agent with bounding box ID 1 is a parked cycle. it stands alone.
# """

fix_system_message_v2 = f"""
You are a mature driver behind the wheel. You will see 4 consecutive frames. 
Each frame has six images captured and put together by the surround view camera in your vehicle. 
The FOV of each surround view camera is approximately 120°. The view angle is written in the images. 
In the right, it is the bird-eye-view image with labeled agents, centered with your vehicle. 
Each frame mark certain marked agents with bounding box with ID. 
Among all objects, which objects that might have unnormal moving behavior that you might need to pay more attention.
Rank it from the most difficult to predict to the least difficult to predict.
Present your answers in this format:

Output:
The most difficult to predict agents' rank is {delimiter} [A1, A2, A3, A4, A5, A6, A7, A8] {delimiter}. 
Explaination:  
<explaination>

Replace A1-A8 with the agent ID, and make sure the the output includes all agents.

message will be delimited with {delimiter} characters.
"""

assistant2_imgs_dir = "/home/x_yiyan/x_yiyanj/code/gpt-4v_UniAD/output/vis/test/trainval/14.49_[5, 4, 6, 2, 3, 1]_f965ad5d2ee94318b2dd1ac927a37856"
assistant2_imgs_des = f"""
There are 6 bounding boxes in the image. From ID 1 to ID 6, they are parked cycle, truck, vehicle, vehicle, cyclist, cement truck.
"""

assistent2_output = f"""
Output:
The most difficult to predict agents's rank is {delimiter} [5, 4, 6, 2, 3, 1] {delimiter}. 
Explaination: 
Cyclist 5 is moving in a intersection which might be hard to prediction as cyclists can quickly change speed and direction
Vehicle 4 is might be hard to predict as it is moving in a intersection.
Cement truck 6 seems parked there, and it seems no driver on it. So it could be the easy to predict.
Truck 2 seems parked there, and it cannot be observed that whehter there is a driver on it, so there is still a possibility it could pull out into traffic.
Vehicle 3 is parked, and no human is observed on it. So it could be the easy to predict.
Parked cycle 1 stands alone.
"""

fix_system_message_v3 = f"""
You are a mature driver behind the wheel. You will see 4 consecutive frames. 
Each frame has six images captured and put together by the surround view camera in your vehicle. 
The FOV of each surround view camera is approximately 120°. The view angle is written in the images. 
In the right, it is the bird-eye-view image with labeled agents, centered with your vehicle. 
Each frame mark certain marked agents with bounding box with ID. 
You will also get the description of the agents and their pixel coordinate.
Among all objects, which objects that might have unnormal moving behavior that you might need to pay more attention.
Rank it from the most difficult to predict to the least difficult to predict.
Present your answers in this format:

Output:
The most difficult to predict agents' rank is {delimiter} [A1, A2, A3, A4, A5, A6, A7, A8] {delimiter}. 
Explaination:  
<explaination>

Replace A1-A8 with the agent ID, and make sure the the output includes all agents.

message will be delimited with {delimiter} characters.
"""

assistant3_imgs_dir = "/home/x_yiyan/x_yiyanj/code/gpt-4v_UniAD/output/vis/test/trainval/13.27_[6, 4, 5, 3, 2, 1]_19316021f0a64765b8ac8887ff3d0e07"
assistant3_imgs_des = f"""
There are 6 bounding boxes in the image. Their information contains <ID, camera view, type, center x, center y, length, hight> as follows:
<1, CAM_BACK, vehicle, 774.0, 482.0, 86, 33>
<2, CAM_BACK, vehicle, 800.0, 476.0, 88, 34>
<3, CAM_BACK, vehicle, 517.5, 478.0, 105, 41>
<4, CAM_FRONT_RIGHT, vehicle, 1197.0, 450.0, 1076, 788>
<5, CAM_FRONT, vehicle, 334.5, 370.0, 319, 120>
<6, CAM_BACK, vehicle, 257.0, 515.0, 330, 194>
"""

assistent3_output = f"""
Output:
The most difficult to predict agents's rank is {delimiter} [6, 4, 5, 3, 2, 1] {delimiter}. 
Explaination: 
Vehilce 6 is moving in a intersection which might be hard to prediction as it can quickly change speed and direction
Vehilce 4 is also moving.
Vehilce 5 is parked there, and it cannot be observed that whehter there is a driver on it, so there is still a possibility it could pull out into traffic.
Veiilce 1,2,3 seems parked in a parking lot, and no human on it.
"""

_test_ = f"""
There are 6 bounding boxes in the image. Their information contains <ID, camera view, type, center x, center y, length, hight> as follows:
<1, CAM_BACK_LEFT, cycle, 863.5, 552.0, 185, 119>
<2, CAM_BACK, vehicle, 1314.5, 420.0, 227, 136>
<3, CAM_BACK_LEFT, vehicle, 487.5, 521.0, 499, 162>
<4, CAM_FRONT, vehicle, 930.0, 489.0, 348, 284>
<5, CAM_FRONT, cyclist, 507.5, 485.0, 59, 88>
<6, CAM_FRONT_RIGHT, truck, 218.0, -710.0, 1754, 2142>
"""

fix_system_message_v4 = f"""
You are a mature driver behind the wheel. You will see 4 consecutive frames. 
Each frame has six images captured and put together by the surround view camera in your vehicle. 
The FOV of each surround view camera is approximately 120°. The view angle is written in the images. 
In the right, it is the bird-eye-view image with labeled agents, centered with your vehicle. 
Each frame mark certain marked agents with bounding box. 
Among all objects, which objects that might have unnormal moving behavior that you might need to pay more attention.
Rank it from the most difficult to predict to the least difficult to predict.
Present your answers in this format:

Output:
The most difficult to predict agents' rank is {delimiter} [A1, A2, A3, A4, A5, A6, A7, A8] {delimiter}. 
Explaination:  
Agent <ID> is a <type>. <explaination>

Replace A1-A8 with the agent number, and make sure the the output includes all agents.
Replace <type> with the object type, e.g. vehicle, pedestrian, cyclist, parked cycle, etc.

message will be delimited with {delimiter} characters.
"""

assistant4_imgs_dir = "/home/x_yiyan/x_yiyanj/code/gpt-4v_UniAD/output/vis/test/trainval/13.27_[6, 4, 5, 3, 2, 1]_19316021f0a64765b8ac8887ff3d0e07"
assistant4_imgs_des = f"""
There are 6 bounding boxes in the image. Their information contains <ID, camera view, center x, center y, length, hight> as follows:
<1, CAM_BACK, 774.0, 482.0, 86, 33>
<2, CAM_BACK, 800.0, 476.0, 88, 34>
<3, CAM_BACK, 517.5, 478.0, 105, 41>
<4, CAM_FRONT_RIGHT, 1197.0, 450.0, 1076, 788>
<5, CAM_FRONT, 334.5, 370.0, 319, 120>
<6, CAM_BACK, 257.0, 515.0, 330, 194>
"""

assistent4_output = f"""
Output:
The most difficult to predict agents's rank is {delimiter} [6, 4, 5, 3, 2, 1] {delimiter}. 
Explaination: 
Agent 6 is a vehicle. Vehilce 6 is moving in a intersection which might be hard to prediction as it can quickly change speed and direction
Agent 4 is a vehicle. Vehilce 4 is also moving.
Agent 5 is a vehicle. Vehilce 5 is parked there, and it cannot be observed that whehter there is a driver on it, so there is still a possibility it could pull out into traffic.
Agent 1,2,3 are vehicles. Veiilce 1,2,3 seems parked in a parking lot, and no human on it.
"""

fix_system_message_v5 = f"""
You are a mature driver behind the wheel. 
You see eight images from a single timeframe here giving you a surround view of the entire scene.
The field of view of each camera is 70°, except the rear camera which is 130°, there is a slight overlap between all of the cameras fields of view.
The images are passed in the following order: front, left_front, left, left_rear, right_front, right, right_rear, rear.
Analyze the images and reason about the scene like a human driver would.
You need to answer the following questions regarding the driving scene:
Question 1: Among all of the agents that you see in the images, which agents might show abnormal behaviour that you would need to pay more attention to? 
Rank them from the most critical and difficult to predict to the least difficult to predict.
Question 2: Score the overall difficulty of this driving scenario from 1 to 10, where 1 is the easiest and 10 is the most difficult. 
If there is a lot of dense traffic, the traffic is at a crossroad or the weather is bad, then you would give it a high score.
If there is any anomalous behavior by traffic agents or the situation in general, you might give it a high score.
If the road is fairly straight with few vehicles and it would be easy to predict future movements, then you would give it a low score. 

Present your answers in this format:
1. The most difficult to predict agents are: <person1, car1, car2, person2> <explanation> Replace the ID with a brief object description.
2. Overall, the prediction difficulty of this scene is <score>. From 1 to 10. <explanation>
"""

# for scenario df11795878cb5419
scenario_output_1 = f"""
1. The most difficult to predict agents are: 
- Silver vehicle to the left rear
- White vehicle in the opposite lane

Explanation: The only other agent in my lane is behind me to the left, it might overtake me or change lanes. So it is usefult to pay attention to it and its future actions.
The white vehicle in the opposite lane is not important, as it is almost passed me and the road is clearly seperated by a kerb.

2. Overall, the prediction difficulty of this scene is 3. 

Explanation: The road ahead is completely clear with a very sligth left curvature indicating we can keep driving at the current rate. 
There is a pedestrian crossing ahead, but no pedestrains are visible. I need to be aware that pedestrians could appear, but until that happens, I can continue driving.
There is an intersection just to our left, but there are no visible agents there and there is a lane between the exit and my lane, so it doesnt pose any risk at the moment.
The other agent behind me to the left might stay in my proximity which would affect later decisions, so I should remain aware of it.
"""

# df240e44ad0d5c3c
scenario_output_2 = f"""
1. The most difficult to predict agents are: 
- Red truck ahead

Explanation: The only other traffic agent around is the red truck ahead of me, it is towing a trailer and seems to be turning into the opposite lane to mine. 
All other visible cars appear to be parked.

2. Overall, the prediction difficulty of this scene is 7.
Explanation: The weather is bad, making the road wet and slippery. Additionally some water on the front cameras is making the visibility worse.
This worsened visibility makes the traffic ahead hard to make out.
Additionally there is a stop sign and a pedestrian crossing just infront of me where we need to make sure to stop and check for other agents and pedestrians.
"""

# 9de91fbb8b275885
scenario_output_3 = f"""
1. The most difficult to predict agents are:
- The car ahead
- The white car behind

Explanation: The car ahead of me has turned right from the opposite lane and is currently in the lane that I am heading towards. I need to make sure it continues to accelerate normally and I have space to enter the lane.
The white car behind me should just follow at a safe distance and not interfere with my driving, but it is good to stay aware of its presence.

2. Overall, the prediction difficulty of this scene is 3.
Explanation: The road is mainly clear and I am driving a left turn of a large intersection. 
There are no clear lane markings for me to follow, but the road is wide and with no agents around this should not be a problem.
The car that has turned into the lane that I want to end up in should also not be a problem if it continues to drive normally, but I need to be awar of it and monitor its progress. 
As long as I dont break sharply, the car behind me should follow me through the turn with no problems.
All other cars seem to be stopped at their respective traffic lights.
"""