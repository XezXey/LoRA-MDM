import numpy as np
import torch as th
import json
import os

def rotation_6d_to_matrix(d6: th.Tensor) -> th.Tensor:
    """
    Converts 6D rotation representation by Zhou et al. [1] to rotation matrix
    using Gram--Schmidt orthogonalisation per Section B of [1].
    Args:
        d6: 6D rotation representation, of size (*, 6)

    Returns:
        batch of rotation matrices of size (*, 3, 3)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """

    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = th.nn.functional.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = th.nn.functional.normalize(b2, dim=-1)
    b3 = th.cross(b1, b2, dim=-1)
    R = th.stack((b1, b2, b3), dim=-2)
    
    # Check orthogonality
    # assert th.allclose(R.transpose(-2, -1) @ R, th.eye(3, device=d6.device).expand_as(R), atol=1e-4)
    # Check determinant
    # assert th.allclose(th.det(R), th.ones(1, device=d6.device))
    return R

def get_lora_mask(x_start, mask, pose_rep='rot6d', mask_type='root'):
    
    import data_loaders.humanml_utils as hml_utils
    """
    Get LoRA mask for the given motion data and mask.
    
    Args:
        x_start: Tensor of shape (B, J, D, T) representing the motion data.
        mask: Tensor of shape (B, J, D, T) representing the mask.
        pose_rep: String indicating the pose representation ('rot6d', 'xyz', etc.).
            - 'rot6d': 6D rotation representation is (263, 1) joints with 6D rotations.
            - 'xyz': 3D position representation is (22, 3) joints with 3D positions.
        mask_type: String indicating the type of mask ('root', 'lower_body', 'upper_body').

    Returns:
        A tensor of shape (B, J, D, T) representing the LoRA mask.
    """
    B, J, D, T = x_start.shape
    valid_mask_types = ['root', 'root_horizontal', 'root_traj']
    if pose_rep == 'rot6d':
        if mask_type in valid_mask_types:
            # Root joint in 6D is x_start[:, 1:3, :, :] (2nd and 3rd dimensions)
            lora_mask = hml_utils.get_inpainting_mask(mask_name=mask_type, shape=x_start.shape)
            lora_mask = 1 - lora_mask
            lora_mask = th.tensor(lora_mask, dtype=x_start.dtype, device=x_start.device)  # Convert to tensor
            lora_mask = th.logical_and(lora_mask, mask.bool())  # Applied time-dimensional mask
        else: 
            raise ValueError(f"[#] Only {valid_mask_types} mask type is currently supported.")
    else:
        raise ValueError(f"[#] Only 'rot6d' pose representation is currently supported, got {pose_rep}.")

    return lora_mask

def save_to_visualizer(data_dict, save_dir, out_name):
    motions = data_dict["motion"]
    if "text" in data_dict:
        texts = data_dict["text"]
    else:
        texts = [""] * len(data_dict["motion"])
    
    B, J, D, L = motions.shape   # B x 22 x 3 x T
    
    R = np.eye(3)[None, None, ...].repeat(B, axis=0).repeat(L, axis=1)    # B x T x 3 x 3
    E = np.eye(4)[None, None, ...].repeat(B, axis=0).repeat(L, axis=1)    # B x T x 4 x 4
    T = np.zeros((B, L, 3))    # B x T x 3
    camera_center = np.zeros((B, 2))    # B x 2
    focal_length = np.ones((B, 1))    # B x 1

    out = {'motions': motions.astype(np.float64).tolist(), # B x 22 x 3 x T
        'R': R.tolist(), # B x T x 3 x 3
        'Rinv': np.linalg.inv(R).tolist(), # B x T x 3 x 3
        'T': T.tolist(), # B x T x 3
        'E': E.tolist(), # B x T x 4 x 4
        'camera_center': camera_center.tolist(), # B x 2
        'focal_length': focal_length.tolist(), # B x 1
        'prompts': texts, # B
        }
        
    os.makedirs(save_dir, exist_ok=True)
    
    with open(f"{save_dir}/{out_name}", "w") as f:
        json.dump(out, f)
    print(f"[#] saved to visualizer: {save_dir}/{out_name}")
    

def training_motion_to_visualizer(data_dict, save_dir, out_name):
    """
    Convert training data dictionary to visualizer format and save it.
    
    Args:
        data_dict: Dictionary containing motion data and other information.
        save_dir: Directory where the output will be saved.
        out_name: Name of the output file.
    """
    if 'motions' not in data_dict:
        raise ValueError("data_dict must contain 'motions' key.")
    
    motions = data_dict['motions']
    if len(motions.shape) == 3:
        motions = motions[None, ...]    # B x T x J x 3
    print(f"[#] motion shape: {motions.shape}")
    assert motions.shape[2] == 22, "Motion data must have 22 joints."
    assert motions.shape[3] == 3, "Motion data must have 3 dimensions (x, y, z)."
    
    B, T, J, D = motions.shape  # B x T x 22 x 3
    
    motions = motions.transpose(0, 2, 3, 1)  # B x J x D x T
    out = {
        'motions': motions.astype(np.float64).tolist(),  # B x 22 x 3 x T
    }
    os.makedirs(save_dir, exist_ok=True)
    
    with open(f"{save_dir}/{out_name}", "w") as f:
        json.dump(out, f)
    print(f"[#] saved to visualizer: {save_dir}/{out_name}")