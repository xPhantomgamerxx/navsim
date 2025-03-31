import os, sys
BASE_DIR = os.path.abspath(os.path.join( os.path.dirname( __file__ ), '..' ))
sys.path.append(BASE_DIR)

# prompt variables are inside here:
from for_VLM.utils.prompt import *
from for_VLM.utils import read_imgs, bc
import time, fire

from openai import OpenAI
myapi_key = os.environ.get("OPENAI_API_KEY")
if myapi_key is None:
    print(f"{bc.FAIL}Please set OPENAI_API_KEY in your environment variables.{bc.ENDC}")
    print(f"Check our README.md for more details.")
    exit()
client = OpenAI(api_key=myapi_key)

def get_completion_from_user_input(test_img_dir, imgs_postfix, provide_few_shots=True, temperature=0):
    
    messages =  [  
        {'role':'system', 'content': fix_system_message_v2},
        ]
    print(f"Include images are {imgs_postfix}")
    if provide_few_shots:
        # messages.append({'role':'user', 'content': [delimiter, *map(lambda x: {"image": x}, read_imgs(assistant1_imgs_dir)), delimiter]})
        # messages.append({'role':'assistant', 'content': assistant1_output})
        messages.append({'role':'user', \
                         'content': [assistant2_imgs_des, delimiter, *map(lambda x: {"image": x}, read_imgs(assistant2_imgs_dir, imgs_postfix)), delimiter]})
        messages.append({'role':'assistant', 'content': assistent2_output})
        print(f"We are using {bc.BOLD}few shots{bc.ENDC} to predict {test_img_dir}\n")
    else:
        print(f"we directly input messages to predict {test_img_dir}\n")
    # messages.append({'role':'user', 'content': ["the marked agents are all vehicle in the case", delimiter, *map(lambda x: {"image": x}, read_imgs(test_img_dir)), delimiter]})
    messages.append({'role':'user', 'content': [delimiter, *map(lambda x: {"image": x}, read_imgs(test_img_dir, imgs_postfix)), delimiter]})

    params = {
        "model": "gpt-4-vision-preview",
        "messages": messages,
        # "headers": {"Openai-Version": "2020-11-07"},
        "max_tokens": 1000,
        "temperature": temperature,
    }

    result = client.chat.completions.create(**params)
    return result.choices[0].message.content

def main(
    predata_path: str = '/home/x_yiyan/x_yiyanj/code/gpt-4v_UniAD/output/vis/test/trainval',
    imgs_postfix: list = ['his_-3_cam.jpg', 'his_-2_cam.jpg', 'his_-1_cam.jpg', 'cur_cam.jpg'],
    few_shots: bool = True,
    debug: int = 0, # set -1 to run all data
):
    # all subfolder data inside predata_path
    all_subfolders = sorted(os.listdir(predata_path))
    for idx in range(len(all_subfolders)):
        full_path = os.path.join(predata_path, all_subfolders[idx])
        if not os.path.exists(full_path) or not os.path.isdir(full_path):
            print(f"skip {full_path} as it is not a folder or not existed.")
            continue
        print(get_completion_from_user_input(full_path, imgs_postfix, provide_few_shots=few_shots))
        if idx >= debug and debug != -1:
            break

if __name__ == "__main__":
    start_time = time.time()
    fire.Fire(main)
    print(f"Time used: {time.time() - start_time:.2f} s")