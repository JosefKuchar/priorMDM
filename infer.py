# This code is based on https://github.com/openai/guided-diffusion
"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""
from priorMDM.model.DoubleTake_MDM import doubleTake_MDM
from priorMDM.utils.fixseed import fixseed
import numpy as np
import torch
from priorMDM.utils.model_util import load_model
from priorMDM.utils import dist_util
from priorMDM.data_loaders.get_data import get_dataset_loader
from priorMDM.data_loaders.humanml.scripts.motion_process import recover_from_ric
from priorMDM.utils.sampling_utils import unfold_sample_arb_len, double_take_arb_len
import logging

import sys

sys.modules["numpy.bool"] = bool
sys.modules["numpy.int"] = int
sys.modules["numpy.float"] = float
sys.modules["numpy.complex"] = complex
sys.modules["numpy.object"] = object
sys.modules["numpy.str"] = str
sys.modules["numpy.unicode"] = str


class dotdict(dict):
    """dot.notation access to dictionary attributes"""

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


config = {
    "cuda": True,
    "device": 0,
    "seed": 10,
    "batch_size": 64,
    "short_db": False,
    "cropping_sampler": False,
    "model_path": "./save/my_humanml_trans_enc_512/model000200000.pt",
    "output_dir": "",
    "num_samples": 10,
    "num_repetitions": 1,
    "guidance_param": 2.5,
    "motion_length": 6.0,
    "input_text": "./assets/dt_text_example.txt",
    "action_file": "",
    "text_prompt": "",
    "action_name": "",
    "sample_gt": False,
    "min_seq_len": 45,
    "max_seq_len": 250,
    "double_take": True,
    "second_take_only": False,
    "handshake_size": 20,
    "blend_len": 10,
    "repaint_rep": 10,
    "repaint": False,
    "debug_double_take": False,
    "skip_steps_double_take": 100,
    "dataset": "humanml",
    "data_dir": "",
    "arch": "trans_enc",
    "emb_trans_dec": False,
    "layers": 8,
    "latent_dim": 512,
    "cond_mask_prob": 0.1,
    "lambda_rcxyz": 0.0,
    "lambda_vel": 0.0,
    "lambda_fc": 0.0,
    "use_tta": False,
    "concat_trans_emb": False,
    "trans_emb": False,
    "noise_schedule": "cosine",
    "diffusion_steps": 1000,
    "sigma_small": True,
}

logger = logging.getLogger(__name__)


def main(texts=[]):
    logger.info("PriorMDM DoubleTake inference")
    args = dotdict(config)
    fixseed(args.seed)
    n_frames = 150
    dist_util.setup_dist(args.device)
    args.num_samples = len(texts)
    args.batch_size = (
        args.num_samples
    )  # Sampling a single batch from the testset, with exactly args.num_samples

    logger.info("Loading dataset")
    data = load_dataset(args, n_frames)
    total_num_samples = args.num_samples * args.num_repetitions

    logger.info("Creating model and diffusion")
    model, diffusion = load_model(
        args, data, dist_util.dev(), ModelClass=doubleTake_MDM
    )

    model_kwargs = {
        "y": {
            "mask": torch.ones(
                (len(texts), 1, 1, 196)
            ),  # 196 is humanml max frames number
            # TODO: This can be variable
            "lengths": torch.tensor([n_frames] * len(texts)),
            "text": texts,
            "tokens": [""],
            "scale": torch.ones(len(texts)) * 2.5,
        }
    }

    all_motions = []
    all_lengths = []
    all_text = []
    all_captions = []

    for rep_i in range(args.num_repetitions):
        logger.info("Sampling")
        if args.guidance_param != 1:
            model_kwargs["y"]["scale"] = (
                torch.ones(args.batch_size, device=dist_util.dev())
                * args.guidance_param
            )
        model_kwargs["y"] = {
            key: val.to(dist_util.dev()) if torch.is_tensor(val) else val
            for key, val in model_kwargs["y"].items()
        }

        max_arb_len = model_kwargs["y"]["lengths"].max()
        min_arb_len = 2 * args.handshake_size + 2 * args.blend_len + 10

        for ii, len_s in enumerate(model_kwargs["y"]["lengths"]):
            if len_s > max_arb_len:
                model_kwargs["y"]["lengths"][ii] = max_arb_len
            if len_s < min_arb_len:
                model_kwargs["y"]["lengths"][ii] = min_arb_len
        samples_per_rep_list, samples_type = double_take_arb_len(
            args, diffusion, model, model_kwargs, max_arb_len
        )

        step_sizes = np.zeros(len(model_kwargs["y"]["lengths"]), dtype=int)
        for ii, len_i in enumerate(model_kwargs["y"]["lengths"]):
            if ii == 0:
                step_sizes[ii] = len_i
                continue
            step_sizes[ii] = step_sizes[ii - 1] + len_i - args.handshake_size

        final_n_frames = step_sizes[-1]

        for sample_i, samples_type_i in zip(samples_per_rep_list, samples_type):

            sample = unfold_sample_arb_len(
                sample_i, args.handshake_size, step_sizes, final_n_frames, model_kwargs
            )

            # Recover XYZ *positions* from HumanML3D vector representation
            if model.data_rep == "hml_vec":
                n_joints = 22 if sample.shape[1] == 263 else 21
                sample = data.dataset.t2m_dataset.inv_transform(
                    sample.cpu().permute(0, 2, 3, 1)
                ).float()
                sample = recover_from_ric(sample, n_joints)
                sample = sample.view(-1, *sample.shape[2:]).permute(0, 2, 3, 1)
            if args.dataset == "babel":
                from data_loaders.amass.transforms import SlimSMPLTransform

                transform = SlimSMPLTransform(
                    batch_size=args.batch_size,
                    name="SlimSMPLTransform",
                    ename="smplnh",
                    normalization=True,
                )

                all_feature = sample  # [bs, nfeats, 1, seq_len]
                all_feature_squeeze = all_feature.squeeze(2)  # [bs, nfeats, seq_len]
                all_feature_permutes = all_feature_squeeze.permute(
                    0, 2, 1
                )  # [bs, seq_len, nfeats]
                splitted = torch.split(
                    all_feature_permutes, all_feature.shape[0]
                )  # [list of [seq_len,nfeats]]
                sample_list = []
                for seq in splitted[0]:
                    all_features = seq
                    Datastruct = transform.SlimDatastruct
                    datastruct = Datastruct(features=all_features)
                    sample = datastruct.joints

                    sample_list.append(sample.permute(1, 2, 0).unsqueeze(0))
                sample = torch.cat(sample_list)
            else:
                rot2xyz_pose_rep = (
                    "xyz" if model.data_rep in ["xyz", "hml_vec"] else model.data_rep
                )
                if args.dataset == "babel":
                    rot2xyz_pose_rep = "rot6d"
                rot2xyz_mask = None

                sample = model.rot2xyz(
                    x=sample,
                    mask=rot2xyz_mask,
                    pose_rep=rot2xyz_pose_rep,
                    glob=True,
                    translation=True,
                    jointstype="smpl",
                    vertstrans=True,
                    betas=None,
                    beta=0,
                    glob_rot=None,
                    get_rotations_back=False,
                )

            text_key = "text" if "text" in model_kwargs["y"] else "action_text"

            all_text += model_kwargs["y"][text_key]
            all_captions += model_kwargs["y"][text_key]

            all_motions.append(sample.cpu().numpy())
            all_lengths.append(model_kwargs["y"]["lengths"].cpu().numpy())

            logger.info(f"Created {len(all_motions) * args.batch_size} samples")

    n_frames = final_n_frames
    num_repetitions = args.num_repetitions

    all_motions = np.concatenate(all_motions, axis=0)
    all_motions = all_motions[:total_num_samples]  # [bs, njoints, 6, seqlen]
    all_text = all_text[:total_num_samples]
    all_lengths = [n_frames] * num_repetitions

    return {
        "motion": all_motions,
        "text": all_text,
        "lengths": all_lengths,
        "num_samples": args.num_samples,
        "num_repetitions": num_repetitions,
    }


def load_dataset(args, n_frames):
    if args.dataset == "babel":
        args.num_frames = (args.min_seq_len, args.max_seq_len)
    else:
        args.num_frames = n_frames
    data = get_dataset_loader(
        name=args.dataset,
        batch_size=args.batch_size,
        num_frames=args.num_frames,
        split="val",
        load_mode="text_only",
        short_db=args.short_db,
        cropping_sampler=args.cropping_sampler,
    )
    data.fixed_length = n_frames
    return data
