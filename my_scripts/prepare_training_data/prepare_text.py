import numpy as np
import os, json
import subprocess
import argparse
parser = argparse.ArgumentParser(description="Prepare text data for motion data.")
parser.add_argument("--dataset_dir", type=str, help="Output directory for text data.")
parser.add_argument("--motion_text_file", type=str, help="Text to motion mapping file.")
args = parser.parse_args()

if __name__ == "__main__":
    
    '''
    This script is used to prepare the texts from arbitrary motion data. (e.g., from video)
    Prerequisites:
    1. motion filename (e.g., 000001.npy, demo_cr7_1.npy)
    2. text to motion mapping file (e.g., demo_cr7_1.txt)
    3. Motion_name_dict.txt with the given structure:
        <motion_filename> | <motion_stype>_<motion_direction>_<cut_idx>.npy | <action_id>
        031279            | Chicken_TR1_08.bvh                              | 15
        M031279           | Chicken_TR1_08.bvh                              | 15
        031280            | Chicken_TR1_09.bvh                              | 15
        M031280           | Chicken_TR1_09.bvh                              | 15
    '''
    
    os.makedirs(args.dataset_dir + '/texts/', exist_ok=True)
    os.makedirs(args.dataset_dir + '/train/', exist_ok=True)
    
    with open(args.motion_text_file, 'r') as f:
        json_data = json.load(f)
        
    motion_name_dict = []
    for key, value in json_data.items():
        motion_name = key
        motion_style = value['motion_style']
        motion_type = value['motion_type']    #NOTE: aka motion_type
        texts = value['texts']
        cut_idx = value['cut_idx']
        action_id = value['action_id']
        filename = key.split(".")[0]
        
        # 1. Create text file for each motion
        with open(args.dataset_dir + '/texts/' + filename + '.txt', 'w') as f:
            for t in texts:
                f.write(t + '\n')
        
        # 2. Create motion name dict
        motion_mapping = f'{filename} {motion_style}_{motion_type}_{cut_idx}.npy {action_id}'
        motion_name_dict.append(motion_mapping)
        
        # 3. Create text file to annotate which motion is used for training
        with open(args.dataset_dir + '/train/' + f'train_{filename}' + '.txt', 'w') as f:
            f.write(f'{filename}\n')
        
    with open(args.dataset_dir + '/MintStyle_name_dict.txt', 'w') as f:
        for line in motion_name_dict:
            f.write(line + '\n')