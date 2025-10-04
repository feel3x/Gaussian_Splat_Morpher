"""
Gaussian Splat Model Interpolator
================================
Author: Felix Hirt
License: MIT License (see LICENSE file for details)

Note:
This file contains original code by Felix Hirt, licensed under MIT.
"""

from typing import List, Optional, Tuple
import torch
import numpy as np
from tqdm import tqdm
import os

from helpers.PointModel import PointModel


class GaussianInterpolator:
    def __init__(self, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.models: List = []  
        self.max_points = 0
        self.correspondences = {}  

    def load_pointmodels(self, models: List):
        self.models = models
        self.max_points = max([m._xyz.shape[0] for m in models]) if models else 0
    
    def slerp(self, q1, q2, t):
        q1 = q1 / torch.linalg.norm(q1, dim=-1, keepdim=True)
        q2 = q2 / torch.linalg.norm(q2, dim=-1, keepdim=True)
        
        dot = (q1 * q2).sum(dim=-1, keepdim=True)
        
        mask = dot < 0.0
        q2 = torch.where(mask, -q2, q2)
        dot = torch.where(mask, -dot, dot)
        
        DOT_THRESHOLD = 0.9995
        
        linear_interp = q1 + t * (q2 - q1)
        linear_interp = linear_interp / torch.linalg.norm(linear_interp, dim=-1, keepdim=True)
        
        theta_0 = torch.arccos(dot.clamp(-1, 1))  
        sin_theta_0 = torch.sin(theta_0)
        theta = theta_0 * t
        s1 = torch.sin(theta_0 - theta) / sin_theta_0
        s2 = torch.sin(theta) / sin_theta_0
        slerp_interp = s1 * q1 + s2 * q2
        
        result = torch.where(dot > DOT_THRESHOLD, linear_interp, slerp_interp)
        
        return result

    def build_correspondences(self, spatial_weight: float = 0.7, color_weight: float = 0.3, distance_threshold: float = None, force_rebuild: bool = False, show_progress: bool = True):
        """Build pairwise correspondences for consecutive model pairs.
        """
        if not force_rebuild and self.correspondences:
            return

        pairs = list(zip(range(len(self.models) - 1), range(1, len(self.models))))
        iterator = pairs if not show_progress else tqdm(pairs, desc="Building correspondences")
        for i, j in iterator:
            mi = self.models[i]
            mj = self.models[j]
            idx_map = self.correspond_one_to_one(
                                        mi._xyz,
                                        mj._xyz,
                                        spatial_weight=spatial_weight, 
                                        color_weight=color_weight,
                                        a_features_dc=mi._features_dc,
                                        b_features_dc=mj._features_dc,
                                        distance_threshold=distance_threshold,
                                        batch_size=2048)
            self.correspondences[(i, j)] = idx_map


    def correspond_one_to_one(self,
        a_xyz: torch.Tensor,
        b_xyz: torch.Tensor,
        spatial_weight: float = 1.0,
        color_weight: float = 0.5,
        a_features_dc: torch.Tensor = None,
        b_features_dc: torch.Tensor = None,
        batch_size: int = 1024,
        distance_threshold: float = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        one-to-one correspondence ensuring all points in the smaller set
        are matched exactly once.
        """

        a_xyz = a_xyz.to(device)
        b_xyz = b_xyz.to(device)
        if a_features_dc is not None:
            a_features_dc = a_features_dc.to(device)
        if b_features_dc is not None:
            b_features_dc = b_features_dc.to(device)

        # helper function to calculate weight
        def make_feat(xyz, feat, sw, cw):
            if feat is None:
                return xyz * sw
            return torch.cat([xyz * sw, feat.view(feat.shape[0], -1) * cw], dim=1)

        a_feat = make_feat(a_xyz, a_features_dc, spatial_weight, color_weight)
        b_feat = make_feat(b_xyz, b_features_dc, spatial_weight, color_weight)

        N, D = a_feat.shape
        M = b_feat.shape[0]

        # Identify smaller and larger sets
        if N <= M:
            small_feat, large_feat = a_feat, b_feat
            small_is_a = True
        else:
            small_feat, large_feat = b_feat, a_feat
            small_is_a = False

        n_small, n_large = small_feat.shape[0], large_feat.shape[0]

        large_used = torch.zeros(n_large, dtype=torch.bool, device=device)
        small_used = torch.zeros(n_small, dtype=torch.bool, device=device)
        matched_small_idx = []
        matched_large_idx = []

        # Track leftovers from threshold filtering
        skipped_small_idx = []
        skipped_large_idx = []

        for start in tqdm(range(0, n_small, batch_size), total=int(n_small/batch_size+1), desc="Matching points", leave=False):
            end = min(start + batch_size, n_small)
            chunk = small_feat[start:end]
            chunk_size = end - start

            # Compute distances for this chunk
            dists = torch.cdist(chunk, large_feat)
            dists[:, large_used] = float("inf")

            for i in range(chunk_size):
                best_j = torch.argmin(dists[i])
                best_dist = dists[i, best_j]
                if best_dist == float("inf"):
                    # all large set points used
                    break

                if distance_threshold is not None and best_dist > distance_threshold:
                    # Skip this match — mark both as leftovers
                    skipped_small_idx.append(start + i)
                    skipped_large_idx.append(best_j.item())
                    continue

                matched_small_idx.append(start + i)
                matched_large_idx.append(best_j.item())
                large_used[best_j] = True
                small_used[start + i] = True
                dists[:, best_j] = float("inf")  # prevent reuse

            del dists
            torch.cuda.empty_cache()

        matched_small_idx = torch.tensor(matched_small_idx, device=device, dtype=torch.long)
        matched_large_idx = torch.tensor(matched_large_idx, device=device, dtype=torch.long)

        skipped_small_idx = torch.tensor(skipped_small_idx, device=device, dtype=torch.long)
        skipped_large_idx = torch.tensor(skipped_large_idx, device=device, dtype=torch.long)

        # Compute leftovers
        leftover_small_idx = torch.nonzero(~small_used, as_tuple=False).squeeze(1)
        leftover_large_idx = torch.nonzero(~large_used, as_tuple=False).squeeze(1)

        # Combine threshold-skipped points with leftovers
        leftover_small_idx = torch.unique(torch.cat([leftover_small_idx, skipped_small_idx]))
        leftover_large_idx = torch.unique(torch.cat([leftover_large_idx, skipped_large_idx]))

        if small_is_a:
            matched_a_idx = matched_small_idx
            matched_b_idx = matched_large_idx
            leftover_a_idx = leftover_small_idx
            leftover_b_idx = leftover_large_idx
        else:
            matched_a_idx = matched_large_idx
            matched_b_idx = matched_small_idx
            leftover_a_idx = leftover_large_idx
            leftover_b_idx = leftover_small_idx

        print(str(min(leftover_a_idx.shape[0], leftover_b_idx.shape[0])) + " points over distance threshold")

        return matched_a_idx, matched_b_idx, leftover_a_idx, leftover_b_idx

    def interpolate_between(
        self,
        idx_a: int,
        idx_b: int,
        t: float
    ):
        """Interpolate between model idx_a and idx_b at fraction t in [0,1].
        Uses one-to-one correspondences (and handles leftover points).
        - At t=0: exactly model A (+ fade-in points from B with opacity 0).
        - At t=1: exactly model B (+ fade-out points from A with opacity 0).
        - In between: linear interpolation in parameter space.
        """
        assert 0.0 <= t <= 1.0
        a = self.models[idx_a]
        b = self.models[idx_b]
        device = self.device

        key = (idx_a, idx_b)
        
        if key not in self.correspondences:
            raise RuntimeError(f"Failed to build correspondence for pair {key}")
        
        matched_a, matched_b, leftover_a, leftover_b = self.correspondences[key]
        matched_a = matched_a.to(device)
        matched_b = matched_b.to(device)
        leftover_a = leftover_a.to(device)
        leftover_b = leftover_b.to(device)

        # These correspond to "matched" points that will interpolate between A and B
        src_idx = matched_a
        tgt_idx = matched_b

        pos_a = a._xyz[src_idx].to(device)
        pos_b = b._xyz[tgt_idx].to(device)

        fdc_a = a._features_dc[src_idx].to(device)
        fdc_b = b._features_dc[tgt_idx].to(device)

        fret_a = a._features_rest[src_idx].to(device)
        fret_b = b._features_rest[tgt_idx].to(device)

        scale_a = a._scaling[src_idx].to(device)
        scale_b = b._scaling[tgt_idx].to(device)

        rot_a = a._rotation[src_idx].to(device)
        rot_b = b._rotation[tgt_idx].to(device)

        op_a = a._opacity[src_idx].to(device)
        op_b = b._opacity[tgt_idx].to(device)

        # Interpolate
        out_xyz = pos_a * (1.0 - t) + pos_b * t
        out_fdc = fdc_a * (1.0 - t) + fdc_b * t
        out_frest = fret_a * (1.0 - t) + fret_b * t
        out_scale = scale_a * (1.0 - t) + scale_b * t
        out_rot = self.slerp(rot_a, rot_b, t)
        out_op = op_a * (1.0 - t) + op_b * t

        # Handle fade-out points (leftovers from A)
        fade_out_xyz = a._xyz[leftover_a]
        fade_out_fdc = a._features_dc[leftover_a]
        fade_out_frest = a._features_rest[leftover_a]
        fade_out_rot = a._rotation[leftover_a]
        # Interpolate scale and opacity to disappear
        fade_out_scale = a._scaling[leftover_a] * (1.0 - t) + -10.0 * t
        fade_out_op = a._opacity[leftover_a] * (1.0 - t) + -10.0 * t

        # Handle fade-in points (leftovers from B)
        fade_in_xyz = b._xyz[leftover_b]
        fade_in_fdc = b._features_dc[leftover_b]
        fade_in_frest = b._features_rest[leftover_b]
        fade_in_rot = b._rotation[leftover_b]
        # Interpolate scale and opacity to appear
        fade_in_scale = -10.0 * (1.0 - t) + b._scaling[leftover_b] * t
        fade_in_op = -10.0 * (1.0 - t) + b._opacity[leftover_b] * t

        # Concatenate all parts
        out_xyz = torch.cat([out_xyz, fade_out_xyz, fade_in_xyz], dim=0)
        out_fdc = torch.cat([out_fdc, fade_out_fdc, fade_in_fdc], dim=0)
        out_frest = torch.cat([out_frest, fade_out_frest, fade_in_frest], dim=0)
        out_scale = torch.cat([out_scale, fade_out_scale, fade_in_scale], dim=0)
        out_rot = torch.cat([out_rot, fade_out_rot, fade_in_rot], dim=0)
        out_op = torch.cat([out_op, fade_out_op, fade_in_op], dim=0)

        # Construct output model
        out_model = type(a)(sh_degree=getattr(a, "max_sh_degree", None))
        out_model._xyz = out_xyz.contiguous()
        out_model._features_dc = out_fdc.contiguous()
        out_model._features_rest = out_frest.contiguous()
        out_model._scaling = out_scale.contiguous()
        out_model._rotation = out_rot.contiguous()
        out_model._opacity = out_op.contiguous()

        return out_model

    def save_interpolated_ply(self, idx_a: int, idx_b: int, t: float, path: str, keep_count: Optional[int] = None, show_progress: bool = True):
        pm = self.interpolate_between(idx_a, idx_b, t)

        pm.save_ply(path)

if(__name__ == "__main__"):

    directory = "C:/Users/felix/Projects/EisbaerenBerlin/Taping/"

    output_dir = "C:/Users/felix/Projects/EisbaerenBerlin/Taping/Output/"

    ply_files = sorted([f for f in os.listdir(directory) if f.endswith('.ply')])

    # Load models
    point_models = []
    for ply_file in ply_files:
        pm = PointModel()
        pm.load_ply(os.path.join(directory, ply_file))
        point_models.append(pm)
    
    interp = GaussianInterpolator(device='cuda') 
    interp.load_pointmodels(point_models)
    interp.build_correspondences(distance_threshold=3)    

    models_to_create = 10

    for model_idx in tqdm(range(0, len(interp.models)-1), desc="Interpolating"):
    #interpolate
        for i in tqdm(range(0, models_to_create+1), desc="Saving PLYs"):
            interp.save_interpolated_ply(model_idx, model_idx+1, i/models_to_create, os.path.join(output_dir, "interpolated_sequence_frame"+str(i + models_to_create * model_idx)+".ply"))
