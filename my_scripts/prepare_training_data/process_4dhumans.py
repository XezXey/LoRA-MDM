import numpy as np
import joblib
import matplotlib.pyplot as plt
import os, glob, tqdm
import torch as th
from smpl.smpl_wrapper import SMPLWrapper

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--input_path', type=str, required=True, help='Path to the 4D-Humans prediction file')
parser.add_argument('--output_path', type=str, required=True, help='Path to the output file')
args = parser.parse_args()

def proc(path, focus_idx=0):
    results = joblib.load(path)
    print("[#INFO] From: ", path)
    print("[#INFO] #N frames: ", len(results))
    tracking_presence = {}
    for fid, data in results.items():
        for pid in data['tracked_ids']:
            if pid not in tracking_presence:
                tracking_presence[pid] = 1
            else:
                tracking_presence[pid] += 1
    longest = max(tracking_presence.values())
    longest_id = [k for k, v in tracking_presence.items() if v == longest][focus_idx]
    print("[#INFO] #N unique ids: ", len(tracking_presence))
    print("[#INFO] Longest tracking id: ", longest_id)
    
    all_j3d = []
    betas = []
    body_pose = []
    global_orient = []
    for fid, data in results.items():
        if len(data['tid']) == 0:
            continue
        pid = data['tid'].index(longest_id)
        # print(data['conf'])
        # print(data['class_name'])
        assert data['class_name'][pid] == 0   # person = 0
        if data['conf'][pid] < 0.85:
            return None
        
        j3d = data['3d_joints'][pid]    # [45, 3]
        all_j3d.append(j3d)
        
        smpl_params = data['smpl'][pid]    # [45, 3]
        body_pose.append(smpl_params['body_pose'])
        global_orient.append(smpl_params['global_orient'])
        betas.append(smpl_params['betas'])
        
        # print(data['pose'][pid].shape)
        # for k, v in data['smpl'][pid].items():
        #     print(k, v.shape)
        
        # assert False
        
    all_j3d = np.stack(all_j3d)  # [T, 45, 3]
    betas = np.stack(betas)  # [T, 10]
    body_pose = np.stack(body_pose)  # [T, 23, 3, 3]
    global_orient = np.stack(global_orient)  # [T, 1, 3, 3]
    
    smpl_params = {
        'betas': th.tensor(betas).float().to("cuda"),
        'body_pose': th.tensor(body_pose).float().to("cuda"),
        'global_orient': th.tensor(global_orient).float().to("cuda"),
    }
    smpl_wrapper = SMPLWrapper()
    smpl_joints = smpl_wrapper(smpl_params)
    print("[#INFO] betas shape: ", betas.shape)
    print("[#INFO] body_pose shape: ", body_pose.shape)
    print("[#INFO] global_orient shape: ", global_orient.shape)
    print("[#INFO] all_j3d shape: ", all_j3d.shape)
    print("[#INFO] smpl_joints shape: ", smpl_joints.shape)
    
    return smpl_joints
    
    
if __name__ == "__main__":
    
    if os.path.isdir(args.input_path):
        file = glob.glob(f"{args.input_path}/*.pkl")
    else:
        file = [args.input_path]
    
    os.makedirs(args.output_path, exist_ok=True)
    
    t = tqdm.tqdm(total=len(file), desc="Processing files")
    for f in file:
        t.set_description(f"Processing files: {f}")
        t.update(1)
        fn = os.path.basename(f)
        
        smpl_joints = proc(f)
        if smpl_joints is None:
            print("[#CAUTION!!!] Jumping tracking id with low confidence. Skipping...")
            continue
        else:
            smpl_joints[..., 1] *= -1
            np.save(f"{args.output_path}/{fn.replace('.pkl', '.npy')}", smpl_joints.cpu().numpy())