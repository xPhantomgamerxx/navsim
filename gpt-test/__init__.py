import base64
import os, sys
import json


class bc:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

# Function to encode the image to base64
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')
    

def read_imgs(imgs_dir, img_name=['his_-3_cam.jpg', 'his_-2_cam.jpg', 'his_-1_cam.jpg', 'cur_cam.jpg']):
    base64Frames = []
    for img in img_name:
        img_path = os.path.join(imgs_dir, img)
        if not os.path.exists(img_path):
            print(f"skip {img_path} as {bc.FAIL}it is not existed.{bc.ENDC}")
            continue
        # print(f"read {img_path} {bc.OKGREEN}successfully{bc.ENDC}")
        base64Frames.append(encode_image(img_path))
    return base64Frames

def combine_with_same_scene(uniad_path, vip3d_path):
    with open(uniad_path, 'r') as f:
        res_uniad = json.load(f)
    with open(vip3d_path, 'r') as f:
        res_vip3d = json.load(f)
    combine = {}
    for sample_token, uniad_value in res_uniad.items():
        # only look at the the samples that are in both json files
        if sample_token in res_vip3d.keys():
            # find the agents that are both detected
            common_agents = [x for x in uniad_value['ranking'] if x in res_vip3d[sample_token]['ranking']]
            if len(common_agents) == 0:
                print(f"skip {sample_token} as {bc.FAIL}it has no shared detected objects.{bc.ENDC}")
                continue
            
            # combine the two json files
            combine[sample_token] = dict(
                scene_token=uniad_value['scene_tokens'],
            )

            # for ranking, max_ade and key_object_infos, save both uniad and vip3d
            # since uniad and vip3d may have different ranking order, we need to save them separately
            combine[sample_token]['ranking'] = dict(
                uniad=[x for x in uniad_value['ranking'] if x in common_agents],   
                vip3d=[x for x in res_vip3d[sample_token]['ranking'] if x in common_agents],
            )
            combine[sample_token]['key_object_infos'] = dict(
                uniad={key: uniad_value['key_object_infos'][key] for key in common_agents if key in uniad_value['key_object_infos']},
                vip3d={key: res_vip3d[sample_token]['key_object_infos'][key] for key in common_agents if key in res_vip3d[sample_token]['key_object_infos']},
            )
            try:
                combine[sample_token]['max_ade'] = dict(
                    uniad=combine[sample_token]['key_object_infos']['uniad'][combine[sample_token]['ranking']['uniad'][0]]['ade'],     # the first afent in the ranking has the max_ade
                    vip3d=combine[sample_token]['key_object_infos']['vip3d'][combine[sample_token]['ranking']['vip3d'][0]]['ade'],
                )
            except:
                print(f"skip {sample_token} as {bc.FAIL}it has no shared detected objects.{bc.ENDC}")
                continue

    file_path_without_ext, file_ext = uniad_path.rsplit('.', 1)
    output_path = file_path_without_ext + '_combine.' + file_ext
    with open(output_path, 'w') as f:
        json.dump(combine, f, indent=4)
    print(f'combined {uniad_path} and {vip3d_path} to: {output_path}')
    return 

def add_key_info_to_json(target_file, combine_file):
    with open(target_file, 'r') as f:
        target_data  = json.load(f)
    with open(combine_file, 'r') as f:
        combine_data = json.load(f)
    # add key_object_infos object's uniad and vip3d's results into target_file
    for sample_token in target_data:
        for object in target_data[sample_token]['key_object_infos']:
            target_data[sample_token]['key_object_infos'][object]['uniad']= combine_data[sample_token]['key_object_infos']['uniad'][object]
            target_data[sample_token]['key_object_infos'][object]['vip3d']= combine_data[sample_token]['key_object_infos']['vip3d'][object]
    # add max_ade to target_file
    target_data[sample_token]['max_ade'] = combine_data[sample_token]['max_ade']
    # save the target_file
    with open(target_file, 'w') as f:
        json.dump(target_data, f, indent=4)
    print(f'added key informations and saved to {target_file}')

def combine_extra_with_gpt4(json_path_root, gpt4v_paths, algo_path, out_path, 
                            use_samples_file, skip_sample_if_one_of_the_gpt4v_is_invalid=True):
    import json
    # read the json files
    gpt4vs = {}
    for file in gpt4v_paths:
        with open(os.path.join(json_path_root, file), 'r') as f:
            # only get the name of the file that we want to use
            name_without_ext, _ = os.path.basename(file).rsplit('.', 1)
            if name_without_ext == use_samples_file:
                file_key = name_without_ext
            # use the file name without extension as the key
            gpt4vs[name_without_ext] = json.load(f)

    with open(os.path.join(json_path_root, algo_path), 'r') as f:
        algos = json.load(f)
    
    out = {}

    for sample, value in gpt4vs[file_key].items():
        # skip the invalid samples
        if skip_sample_if_one_of_the_gpt4v_is_invalid:
            # Skip if any of the gpt4v files, for this sample, is invalid
            try:
                if any(not _gpt4v[sample]['valid'] for _gpt4v in gpt4vs.values()):
                    continue
            except:
                print(f"skip {sample} as {bc.FAIL}it has no 'valid'.{bc.ENDC}")
                continue
        elif not value['valid']:
            # Skip if the current file, for this sample, is invalid
            continue
        
        out[sample] = algos[sample]

        # add the gpt4v results
        for file, _gpt4v in gpt4vs.items():
            # Use setdefault to ensure 'gpt4v' and 'ranking' keys exist
            out[sample].setdefault('gpt4v', {})[file] = _gpt4v[sample]
            try:
                out[sample]['ranking']['order_gpt4v_' + file] =  _gpt4v[sample]['ranking']
            except:
                print(f"skip {sample} as {file} {bc.FAIL}it has no 'ranking'.{bc.ENDC}")
                continue



    with open(out_path, 'w') as f:
        json.dump(out, f, indent=4)
        
    print(f'save file in {out_path}. done')