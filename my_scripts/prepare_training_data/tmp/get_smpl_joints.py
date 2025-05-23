from dataclasses import dataclass, field, asdict
import os
from smplx.lbs import vertices2joints
from typing import Dict, Optional

import hydra
import joblib
from omegaconf import MISSING

import torch as th
from torch import nn

import torch as th
from smpl.smpl_head import SMPLHead  # Default is at phalp.models.heads.smpl_head
from smpl.smpl_utils import SMPL  # Default is at phalp.utils.smpl_utils
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--smpl_input",
    type=str,
    default="smpl_input.pkl",
    help="Path to the SMPL input file",
)
args = parser.parse_args()

# Look the default from "phalp.configs.base import CACHE_DIR"
CACHE_DIR = os.path.join(
    os.environ.get("HOME"), ".cache"
)  # None if the variable does not exist


@dataclass
class SMPLConfig:
    MODEL_PATH: str = f"{CACHE_DIR}/phalp/3D/models/smpl/"
    GENDER: str = "neutral"
    MODEL_TYPE: str = "smpl"
    NUM_BODY_JOINTS: int = 23
    # JOINT_REGRESSOR_EXTRA: str = f"{CACHE_DIR}/phalp/3D/SMPL_to_J19.pkl"
    TEXTURE: str = f"{CACHE_DIR}/phalp/3D/texture.npz"


# Config for HMAR
@dataclass
class SMPLHeadConfig:
    TYPE: str = "basic"
    POOL: str = "max"
    SMPL_MEAN_PARAMS: str = f"{CACHE_DIR}/phalp/3D/smpl_mean_params.npz"
    IN_CHANNELS: int = 2048


@dataclass
class FullConfig:
    device: str = "cuda"

    # Fields
    SMPL: SMPLConfig = field(default_factory=SMPLConfig)


class SMPLWrapper(nn.Module):
    # NOTE: My custom wrapper for SMPL to load any SMPL model and export the 3d joints by using a given joint_regressor
    def __init__(self):
        super(SMPLWrapper, self).__init__()
        cfg = FullConfig()
        smpl_cfg = {k.lower(): v for k, v in asdict(cfg.SMPL).items()}
        print("=" * 100)
        print("[#] SMPL config: ", smpl_cfg)
        print("=" * 100)
        self.smpl = SMPL(**smpl_cfg)

    def forward(
        self,
        smpl_params: Dict[str, th.Tensor],
        force_tpose: bool = False,
        visualize: bool = False,
    ):
        
        T, NJ = smpl_params["body_pose"].shape[:2]

        # NOTE: For forcing a T-pose
        if force_tpose:
            eye = th.eye(3).to(smpl_params["body_pose"].device)
            eye = eye[None, None, ...].repeat(T, NJ, 1, 1)
            smpl_params["body_pose"] = eye

        smpl_output = self.smpl(
            **{k: v.float() for k, v in smpl_params.items()}, pose2rot=False
        )
        smpl_joints = vertices2joints(self.smpl.J_regressor, smpl_output.vertices)

        # smpl_joints = th.einsum(
        #     "bij,bjk->bik", self.smpl.J_regressor[None, ...], smpl_output.vertices
        # )
        

        if visualize:
            import plotly.graph_objects as go

            j3d = smpl_params["3d_joints"].cpu().numpy()
            smpl_joints_op = smpl_output.joints.cpu().numpy()
            smpl_joints = smpl_joints.cpu().numpy()
            print(f"[#] SMPL joints shape: {smpl_joints.shape}")
            print(f"[#] 3D joints shape: {j3d.shape}")
            print(f"[#] SMPL joints shape: {smpl_joints.shape}")
            fig = go.Figure(
                data=[
                    go.Scatter3d(
                        x=j3d[0, :, 0],
                        y=j3d[0, :, 1],
                        z=j3d[0, :, 2],
                        mode="markers",
                        marker=dict(size=5, color="red"),
                        name="3D Joints",
                    ),
                ]
            )
            fig.add_trace(
                go.Scatter3d(
                    x=smpl_joints_op[0, :25, 0],
                    y=smpl_joints_op[0, :25, 1],
                    z=smpl_joints_op[0, :25, 2],
                    mode="markers",
                    marker=dict(size=5, color="blue"),
                    name="SMPL Joints OP",
                )
            )
            fig.add_trace(
                go.Scatter3d(
                    x=smpl_joints[0, :, 0],
                    y=smpl_joints[0, :, 1],
                    z=smpl_joints[0, :, 2],
                    mode="markers",
                    marker=dict(size=5, color="green"),
                    name="SMPL Joints",
                )
            )
            fig.update_layout(
                scene=dict(
                    xaxis=dict(nticks=4, range=[-1, 1]),
                    yaxis=dict(nticks=4, range=[-1, 1]),
                    zaxis=dict(nticks=4, range=[-1, 1]),
                )
            )
            fig.write_html("./test.html")

        return smpl_joints

def load_smpl_input(smpl_input_path: str):
    # Load the SMPL input file
    data = joblib.load(smpl_input_path)
    smpl_data = {
        k: th.tensor(v).float().to("cuda")
        for k, v in data.items()
        if k in ["betas", "body_pose", "global_orient", "3d_joints"]
    }

    print(f"[#] Loaded SMPL input file: {smpl_input_path}")
    print(f"[#] SMPL data keys: {smpl_data.keys()}")
    return smpl_data


if __name__ == "__main__":
    smpl_wrapper = SMPLWrapper()

    assert os.path.exists(
        args.smpl_input
    ), f"[#] SMPL input file {args.smpl_input} does not exist."

    smpl_data = load_smpl_input(args.smpl_input)
    smpl_wrapper.to("cuda")
    smpl_wrapper.eval()
    
    #NOTE: Return the 3D joints in smpl joints convention.
    # smpl_joints = smpl_wrapper(smpl_data, visualize=True)
    smpl_joints = smpl_wrapper(smpl_data, visualize=True, force_tpose=False)
    print("[#] Final SMPL joints shape: ", smpl_joints.shape)
    smpl_data["3d_smpl_joints"] = smpl_joints
    
    
