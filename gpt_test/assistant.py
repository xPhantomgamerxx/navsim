assistant1_imgs_dir = f"""
/home/x_yiyan/x_yiyanj/code/gpt-4v_UniAD/output/vis/for_gpt_few_shots/trainval/3.2_2.8_[1, 2, 4, 5, 3]_43786cf95a7e4cf38378880b637cca08
"""

assistant2_imgs_dir = f"""
/home/x_yiyan/x_yiyanj/code/gpt-4v_UniAD/output/vis/for_gpt_few_shots/trainval/2.5_5.8_[3, 5, 1, 4, 2]_6d24000d98854d99b5af251bfdf31561
""" 

###############################
# with camera
###############################
#---------------------`1. cam: coor `-------------------#
#---------------------`2. cam: type`--------------------#
#-----------------`3. cam: coor + type`-----------------#

###############################
# with bev
###############################
#-----------------`4. bev: coor + type`-----------------#

###############################
# with camera + bev
###############################

#---------------------`5. cam + bev: null `-------------------#
system_message_5 = f"""
You are a mature driver behind the wheel. You will see 4 consecutive frames. 
Each frame has six images captured and put together by the surround view camera in your vehicle. 
The FOV of each surround view camera is approximately 120°. The view angle is written in the images. 
In the right, it is the bird-eye-view image with labeled agents, centered with your vehicle. 
There are several marked agents with bounding box with an ID written on the left top corner of the box.
You need to answer the following questions:
1. How many different ID of agents in the 4 images?
2. What are the types of the agents? Types include <car>, <truck>, <pedestrian>, <cyclist>, <bicycle>, <motorcycle> etc.
3. Among all objects, which objects that might have unnormal moving behavior that you might need to pay more attention.
Rank it from the most difficult to predict to the least difficult to predict.
4. Score the difficulty of prediction from 1 to 10, where 1 is the easiest and 10 is the most difficult. 
    For example, if the traffic is very dense and in the crossroads, you might give it high score.
    If it is a straight road with few vehicles, it is easy to predict, you might give it low score.
    If there is a abnormal behavior, you might give it high score.
Present your answers in this format:
Output:
1. There are <N> different ID of agents in the 4 images.
2. From ID 1 to ID <N>, they are <ID 1 type>, ..., <ID N type>.
3. The most difficult to predict agents' rank is [A1, A2, A3, A4, A5, A6, A7, A8]. <explaination>
Replace A1-A8 with the agent ID number, and make sure the the output includes all agents.
4. Overall, the prediction difficulty is <score>. From 1 to 10. <explaination>
"""

assistant1_output_5 = f"""
1. There are 5 different ID of agents in the 4 images.
2. From ID 1 to ID 5, they are car, car, car, car, car.
3. The most difficult to predict agents's rank is [1, 2, 4, 5, 3]. 
    Explanation:
    Truck 1 is moving in a intersection which might be hard to prediction as cyclists can quickly change speed and direction.
    Car 2 is moving in a intersection.
    Car 4 is stopped on the side of the road, and there is no driver on it. So it is parked there.
    Car 5 is stopped on the side of the road, and there is no driver on it. So it is parked there.
    Car 3 is stopped on the side of the road, and it cannot see if the drive is on it. 
4. Overall, the prediction difficulty is 7. 
    As it is crossroads with 4 directions, it increases the difficulty of prediction compared with straight road, but not very crowded.
"""

assistant2_output_5 = f"""
1. There are 5 different ID of agents in the 4 images.
2. From ID 1 to ID 5, they are car, car, motorcycle, adult, car.
3. The most difficult to predict agents's rank is [3, 5, 1, 4, 2]. 
    Explanation:
    Motorcycle 3 is moving into a intersection which might be hard to prediction. Motorcyclist can move very quickly and randomly.
    Car 5 is entering the intersection, which could choose different directions and make the prediction hard.
    Car 1 is moving out the intersection.
    Pedestrian 4 is walking along the curb, which looks normal and seems not to cross the road.
    Car 2 is waiting in a intersection.
4. Overall, the prediction difficulty is 6. 
    As it is T intersection with 3 directions, it has some difficulty, but not as complex as crossroads.
"""

#---------------------`6. cam + bev: coor `-------------------#
system_message_6 = f"""
You are a mature driver behind the wheel. You will see 4 consecutive frames. 
Each frame has six images captured and put together by the surround view camera in your vehicle. 
The FOV of each surround view camera is approximately 120°. The view angle is written in the images. 
In the right, it is the bird-eye-view image with labeled agents, centered with your vehicle. 
There are several marked agents with bounding box with an ID written on the left top corner of the box.
Moreover, you will be provided information about the agents' types and their pixel coordinate.
You need to answer the following questions:
1. What are the types of the agents? Types include <car>, <truck>, <pedestrian>, <cyclist>, <bicycle>, <motorcycle> etc.
2. Among all objects, which objects that might have unnormal moving behavior that you might need to pay more attention.
Rank it from the most difficult to predict to the least difficult to predict.
3. Score the difficulty of prediction from 1 to 10, where 1 is the easiest and 10 is the most difficult. 
    For example, if the traffic is very dense and in the crossroads, you might give it high score.
    If it is a straight road with few vehicles, it is easy to predict, you might give it low score.
    If there is a abnormal behavior, you might give it high score.
Present your answers in this format:
Output:
1. From ID 1 to ID <N>, they are <ID 1 type>, ..., <ID N type>.
2. The most difficult to predict agents' rank is [A1, A2, A3, A4, A5, A6, A7, A8]. <explaination>
Replace A1-A8 with the agent ID number, and make sure the the output includes all agents.
3. Overall, the prediction difficulty is <score>. From 1 to 10. <explaination>
"""

assistant1_6_coor_des = f"""
There are 5 different ID of agents in the 4 images. 
Their information contains <ID, camera view, center x, center y, length, hight> as follows:
<1, CAM_FRONT_LEFT, 902.4, 539.9, 265, 153>
<2, CAM_BACK, 2704.5, 1398.5, 77.6, 46.9>
<3, CAM_BACK, 1478.3, 527.0, 170.4, 73.9>
<4, CAM_BACK, 2381.2, 1384.7, 69.1, 40.0>
<5, CAM_BACK_LEFT, 3533.5, 1471.5, 283.9, 116.1>
For center x and center y, the origin is top left corner.
"""

assistant1_output_6_coor = f"""
1. From ID 1 to ID 5, they are truck, car, car, car, car.
2. The most difficult to predict agents's rank is [1, 2, 4, 5, 3]. 
    Explanation:
    Truck 1 is moving in a intersection which might be hard to prediction as cyclists can quickly change speed and direction.
    Car 2 is moving in a intersection.
    Car 4 is stopped on the side of the road, and there is no driver on it. So it is parked there.
    Car 5 is stopped on the side of the road, and there is no driver on it. So it is parked there.
    Car 3 is stopped on the side of the road, and it cannot see if the drive is on it. 
3. Overall, the prediction difficulty is 7. 
    As it is crossroads with 4 directions, it increases the difficulty of prediction compared with straight road, but not very crowded.
"""

assistant2_6_coor_des = f"""
There are 5 different ID of agents in the 4 images. 
Their information contains <ID, camera view, center x, center y, length, hight> as follows:
<1, CAM_FRONT, 2483.7, 524.7, 154.3, 100.5>
<2, CAM_FRONT_RIGHT, 4607.1, 821.8, 2235.1, 1718.3>
<3, CAM_BACK_RIGHT, 1121.4, 1435.3, 242.8, 211.2>
<4, CAM_FRONT, 2386.4, 518.1, 84.8, 158.0>
<5, CAM_BACK_RIGHT, 1923.0, 1569.3, 1088.5, 462.5>
For center x and center y, the origin is top left corner.
"""

assistant2_output_6_coor = f"""
1. From ID 1 to ID 5, they are car, car, motorcycle, adult, car.
2. The most difficult to predict agents's rank is [3, 5, 1, 4, 2]. 
    Explanation:
    Motorcycle 3 is moving into a intersection which might be hard to prediction. Motorcyclist can move very quickly and randomly.
    Car 5 is entering the intersection, which could choose different directions and make the prediction hard.
    Car 1 is moving out the intersection.
    Pedestrian 4 is walking along the curb, which looks normal and seems not to cross the road.
    Car 2 is waiting in a intersection.
3. Overall, the prediction difficulty is 6. 
    As it is T intersection with 3 directions, it has some difficulty, but not as complex as crossroads.
"""

#---------------------`7. cam + bev: type`--------------------#
system_message_7 = f"""
You are a mature driver behind the wheel. You will see 4 consecutive frames. 
Each frame has six images captured and put together by the surround view camera in your vehicle. 
The FOV of each surround view camera is approximately 120°. The view angle is written in the images. 
In the right, it is the bird-eye-view image with labeled agents, centered with your vehicle. 
There are several marked agents with bounding box with an ID written on the left top corner of the box.
Moreover, you will be provided information about the agents' types.
You need to answer the following questions:
1. Among all objects, which objects that might have unnormal moving behavior that you might need to pay more attention.
Rank it from the most difficult to predict to the least difficult to predict.
2. Score the difficulty of prediction from 1 to 10, where 1 is the easiest and 10 is the most difficult. 
    For example, if the traffic is very dense and in the crossroads, you might give it high score.
    If it is a straight road with few vehicles, it is easy to predict, you might give it low score.
    If there is a abnormal behavior, you might give it high score.
Present your answers in this format:
Output:
1. The most difficult to predict agents' rank is [A1, A2, A3, A4, A5, A6, A7, A8]. <explaination>
Replace A1-A8 with the agent ID number, and make sure the the output includes all agents.
2. Overall, the prediction difficulty is <score>. From 1 to 10. <explaination>
"""

assistant1_7_type_des = f"""
There are 5 different ID of agents in the 4 images. 
From ID 1 to ID 5, they are truck, car, car, car, car.
"""

assistant1_output_7_type = f"""
1. The most difficult to predict agents's rank is [1, 2, 4, 5, 3]. 
    Explanation:
    Truck 1 is moving in a intersection which might be hard to prediction as cyclists can quickly change speed and direction.
    Car 2 is moving in a intersection.
    Car 4 is stopped on the side of the road, and there is no driver on it. So it is parked there.
    Car 5 is stopped on the side of the road, and there is no driver on it. So it is parked there.
    Car 3 is stopped on the side of the road, and it cannot see if the drive is on it. 
2. Overall, the prediction difficulty is 7. 
    As it is crossroads with 4 directions, it increases the difficulty of prediction compared with straight road, but not very crowded.
"""

assistant2_7_type_des = f"""
There are 5 different ID of agents in the 4 images. 
From ID 1 to ID 5, they are car, car, motorcycle, adult, car.
"""

assistant2_output_7_type = f"""
1. The most difficult to predict agents's rank is [3, 5, 1, 4, 2]. 
    Explanation:
    Motorcycle 3 is moving into a intersection which might be hard to prediction. Motorcyclist can move very quickly and randomly.
    Car 5 is entering the intersection, which could choose different directions and make the prediction hard.
    Car 1 is moving out the intersection.
    Pedestrian 4 is walking along the curb, which looks normal and seems not to cross the road.
    Car 2 is waiting in a intersection.
2. Overall, the prediction difficulty is 6. 
    As it is T intersection with 3 directions, it has some difficulty, but not as complex as crossroads.
"""


#-----------------`8. cam + bev: coor + type`-----------------#
system_message_8 = f"""
You are a mature driver behind the wheel. You will see 4 consecutive frames. 
Each frame has six images captured and put together by the surround view camera in your vehicle. 
The FOV of each surround view camera is approximately 120°. The view angle is written in the images. 
In the right, it is the bird-eye-view image with labeled agents, centered with your vehicle. 
There are several marked agents with bounding box with an ID written on the left top corner of the box.
Moreover, you will be provided information about the agents' types and their pixel coordinate.
You need to answer the following questions:
1. Among all objects, which objects that might have unnormal moving behavior that you might need to pay more attention.
Rank it from the most difficult to predict to the least difficult to predict.
2. Score the difficulty of prediction from 1 to 10, where 1 is the easiest and 10 is the most difficult. 
    For example, if the traffic is very dense and in the crossroads, you might give it high score.
    If it is a straight road with few vehicles, it is easy to predict, you might give it low score.
    If there is a abnormal behavior, you might give it high score.
Present your answers in this format:
Output:
1. The most difficult to predict agents' rank is [A1, A2, A3, A4, A5, A6, A7, A8]. <explaination>
Replace A1-A8 with the agent ID number, and make sure the the output includes all agents.
2. Overall, the prediction difficulty is <score>. From 1 to 10. <explaination>
"""

assistant1_8_type_coor_des = f"""
There are 5 different ID of agents in the 4 images. 
Their information contains <ID, camera view, type, center x, center y, length, hight> as follows:
<1, CAM_FRONT_LEFT, truck, 902.4, 539.9, 265, 153>
<2, CAM_BACK, car, 2704.5, 1398.5, 77.6, 46.9>
<3, CAM_BACK, car, 1478.3, 527.0, 170.4, 73.9>
<4, CAM_BACK, car, 2381.2, 1384.7, 69.1, 40.0>
<5, CAM_BACK_LEFT, car, 3533.5, 1471.5, 283.9, 116.1>
For center x and center y, the origin is top left corner.
"""

assistant1_output_8_type_coor = f"""
1. The most difficult to predict agents's rank is [1, 2, 4, 5, 3]. 
    Explanation:
    Truck 1 is moving in a intersection which might be hard to prediction as cyclists can quickly change speed and direction.
    Car 2 is moving in a intersection.
    Car 4 is stopped on the side of the road, and there is no driver on it. So it is parked there.
    Car 5 is stopped on the side of the road, and there is no driver on it. So it is parked there.
    Car 3 is stopped on the side of the road, and it cannot see if the drive is on it. 
2. Overall, the prediction difficulty is 7. 
    As it is crossroads with 4 directions, it increases the difficulty of prediction compared with straight road, but not very crowded.
"""

assistant2_8_type_coor_des = f"""
There are 5 different ID of agents in the 4 images. 
Their information contains <ID, camera view, type, center x, center y, length, hight> as follows:
<1, CAM_FRONT, car, 2483.7, 524.7, 154.3, 100.5>
<2, CAM_FRONT_RIGHT, car, 4607.1, 821.8, 2235.1, 1718.3>
<3, CAM_BACK_RIGHT, motorcycle, 1121.4, 1435.3, 242.8, 211.2>
<4, CAM_FRONT, pedestrian adult, 2386.4, 518.1, 84.8, 158.0>
<5, CAM_BACK_RIGHT, car, 1923.0, 1569.3, 1088.5, 462.5>
For center x and center y, the origin is top left corner.
"""

assistant2_output_8_type_coor = f"""
1. The most difficult to predict agents's rank is [3, 5, 1, 4, 2]. 
    Explanation:
    Motorcycle 3 is moving into a intersection which might be hard to prediction. Motorcyclist can move very quickly and randomly.
    Car 5 is entering the intersection, which could choose different directions and make the prediction hard.
    Car 1 is moving out the intersection.
    Pedestrian 4 is walking along the curb, which looks normal and seems not to cross the road.
    Car 2 is waiting in a intersection.
2. Overall, the prediction difficulty is 6. 
    As it is T intersection with 3 directions, it has some difficulty, but not as complex as crossroads.
"""


#-----------------`complexity`-----------------#
assistant1_imgs_dir_complexity =  '/home/x_yiyan/x_yiyanj/code/gpt-4v_UniAD/output/vis/data_selection/0a9159b1f4584c22b07f2c7bdf1ece57'
assistant2_imgs_dir_complexity =  '/home/x_yiyan/x_yiyanj/code/gpt-4v_UniAD/output/vis/data_selection/6afbe21a09bc43798566d9a08fbd742f'

system_message_9_complexity = f"""
You are a mature driver behind the wheel. You will see 3 consecutive frames. 
Each frame has six images captured and put together by the surround view camera in your vehicle. 
The FOV of each surround view camera is approximately 120°. The view angle is written in the images. 
In the right, it is the bird-eye-view image centered with your vehicle. 
The HD map is also plotted in the bird-eye-view image. Pink is drivable area. Purple is pedestrian crosswalk and lane marking.
You need to score the difficulty of prediction from 1 to 10, where 1 is the easiest and 10 is the most difficult.
    For example, if the traffic is very dense and in the crossroads or it is a bad weather, you might give it high score.
    If it is a straight road with few vehicles, it is easy to predict, you might give it low score.
    If there is a abnormal behavior, you might give it high score.
Present your answers in this format:
Prediction difficulty is <score>. From 1 to 10. 
<explaination>
"""

assistant1_output_9_complexity = f"""
Prediction difficulty is 8. 
As it is T intersection with 3 directions, it increases the difficulty of prediction compared with straight road.
Although it is not very crowded, there are some pedestrians which is quite unpredictable in the crosswalk 
and a contrustion area in the right side where some workers works around.
It might cause abnormal moving behavior.
"""

assistant2_output_9_complexity = f"""
Prediction difficulty is 3. 
The vehicle is in a straight line, with a far-ahead merging lane. 
Moreover, there are few vehicles and no pedestrians.
For such relatively simple traffic, it has lower difficulty.
"""