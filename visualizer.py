"""Gaussian Splatting Viewer using viser + gsplat rasterization
================================
Author: Felix Hirt
License: MIT License (see LICENSE file for details)

Note:
This file contains original code by Felix Hirt, licensed under MIT.
"""

import argparse
import time
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
import viser
import viser.transforms as vt
import os

from gsplat.rendering import rasterization

from helpers.PointModel import PointModel

from interpolate import GaussianInterpolator


class CameraHelpers:
    @staticmethod
    def c2w_from_camera(camera) -> np.ndarray:
        wxyz = camera.wxyz
        # build 4x4 c2w matrix
        rot = vt.SO3(wxyz).as_matrix()  # 3x3
        pos = camera.position
        c2w = np.concatenate([np.concatenate([rot, pos[:, None]], 1), [[0, 0, 0, 1]]], 0)
        return c2w

    @staticmethod
    def K_from_camera_fov_aspect(fov: float, img_wh: Tuple[int, int]) -> np.ndarray:
        W, H = img_wh
        focal_length = H / 2.0 / np.tan(fov / 2.0)
        K = np.array(
            [
                [focal_length, 0.0, W / 2.0],
                [0.0, focal_length, H / 2.0],
                [0.0, 0.0, 1.0],
            ]
        )
        return K

class GsplatViserViewer: 
    '''Adapted for interpolation'''
    def __init__(self, server: viser.ViserServer, model: PointModel, interpolator: GaussianInterpolator, device: torch.device):
        self.server = server
        self.model = model
        self.device = device
        self.interpolator = interpolator

        # default render resolution shown to clients
        self.render_w = 1920
        self.render_h = 1080

        # GUI
        with self.server.gui.add_folder("Controls"):
            self.interpolation_slider = self.server.gui.add_slider(
                "Interpolation", min=1, max=len(interpolator.models), step=0.001, initial_value=1
            )
            self.interpolation_slider.on_update(self._on_slider_update)

        # register client connect/disconnect
        server.on_client_connect(self._connect_client)
        server.on_client_disconnect(self._disconnect_client)

        self._clients = {}

    def _on_slider_update(self, event):
        value = self.interpolation_slider.value - 1
        idx_a = int(value)
        t = value - idx_a
        idx_b = idx_a + 1
        if(idx_b > len(self.interpolator.models) - 1):
            idx_a -= 1
            idx_b -= 1
            t=1
        self.model = self.interpolator.interpolate_between(idx_a, idx_b, t)
        
        for cid, client in list(self._clients.items()):
            self._render_for_client(client)

    def _connect_client(self, client: viser.ClientHandle):
        self._clients[client.client_id] = client

        # camera movement
        @client.camera.on_update
        def _camera_moved(_: viser.CameraHandle):
            # small debounce could be added
            self._render_for_client(client)

        # when a client connects send render
        self._render_for_client(client)

    def _disconnect_client(self, client: viser.ClientHandle):
        self._clients.pop(client.client_id, None)

    def _prepare_render_inputs(self):
        """Convert PointModel tensors into tensors expected by gsplat.rasterization.
        Returns means, quats, scales, opacities, colors, and sh_degree (or None).
        """
        pm = self.model
        device = self.device

        means = pm._xyz.to(device)

        try:
            quats = pm.rotation_activation(pm._rotation).to(device)
        except Exception:
            # fallback: if rotation already stored normalized
            quats = pm._rotation.to(device)

        try:
            scales = pm.get_scaling.to(device)
        except Exception:
            scales = pm._scaling.to(device)

        try:
            opacities = pm.get_opacity.squeeze(-1).to(device)
        except Exception:
            opacities = pm._opacity.squeeze(-1).to(device)

        # colors / SH coefficients: construct colors as [N, 3, coeffs]
        if hasattr(pm, "_features_dc") and pm._features_dc.numel() > 0:
            parts = [pm._features_dc]
            if hasattr(pm, "_features_rest") and pm._features_rest.numel() > 0:
                parts.append(pm._features_rest)
            colors = torch.cat(parts, dim=1).to(device)
            sh_degree = int(colors.shape[2] ** 0.5) - 1 if colors.shape[2] > 0 else None
        else:
            # fallback: try to use 'colors' attribute if present
            if hasattr(pm, "_colors"):
                colors = pm._colors.to(device)
                sh_degree = None
            else:
                # empty colors -> create a dummy gray
                N = means.shape[0]
                colors = torch.ones((N, 3, 1), device=device) * 0.5
                sh_degree = 0

        return means, quats, scales, opacities, colors, sh_degree

    def _render_for_client(self, client: viser.ClientHandle):
        camera = client.camera
        img_wh = (self.render_w, self.render_h)
        c2w = CameraHelpers.c2w_from_camera(camera)
        K = CameraHelpers.K_from_camera_fov_aspect(camera.fov, img_wh)

        means, quats, scales, opacities, colors, sh_degree = self._prepare_render_inputs()

        c2w_t = torch.from_numpy(c2w).float().to(self.device)

        R_fix = torch.tensor([
            [1,  0,  0, 0],
            [0,  0,  -1, 0],
            [0, 1,  0, 0],
            [0,  0,  0, 1],
        ], dtype=torch.float32, device=self.device)
        c2w_t = R_fix @ c2w_t  # adjust camera orientation
        
        K_t = torch.from_numpy(K).float().to(self.device)
        viewmat = c2w_t.inverse()[None]
        K_in = K_t[None]

        with torch.no_grad():
            render_colors, render_alphas, meta = rasterization(
                means,
                quats,
                scales,
                opacities,
                colors,
                viewmat,
                K_in,
                img_wh[0],
                img_wh[1],
                sh_degree=sh_degree,
                render_mode="RGB",
                radius_clip=3,
            )

        img = render_colors[0, ..., 0:3].cpu().numpy()

        client.scene.set_background_image(
            img,
            format="jpeg",
            jpeg_quality=70,
        )


def main():
    parser = argparse.ArgumentParser(
        description="Interpolates between 3D point cloud models and visualizes them.",
        formatter_class=argparse.RawTextHelpFormatter # For better help text formatting
    )
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--device", type=str, default="cuda:0")
        
    #Input Arguments
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        '-d', '--directory', 
        type=str, 
        help="Path to a directory containing .ply models."
    )
    input_group.add_argument(
        '-m', '--models', 
        nargs='+',  # '+' means one or more arguments
        type=str, 
        help="Paths to two or more individual .ply model files."
    )

    # --- Optional Interpolation Parameters ---
    parser.add_argument(
        '--spatial_weight', 
        type=float, 
        default=0.7, 
        help="Weight for spatial distance in correspondence. Default: 0.7"
    )
    parser.add_argument(
        '--color_weight', 
        type=float, 
        default=0.3, 
        help="Weight for color difference in correspondence. Default: 0.3"
    )
    parser.add_argument(
        '--distance_threshold', 
        type=float, 
        default=None,
        help="Max distance for point correspondences. If not set, no threshold is used."
    )
    parser.add_argument(
        '--batch_size', 
        type=int, 
        default=2048,
        help="Size of point batches to process. Lower for less GPU memory usage."
    )
    parser.add_argument(
        "--disable_recenter_models", 
        action="store_true", 
        help="Disables recentering of all the used models before interpolating")
    parser.add_argument(
        "--disable_normalize_scales", 
        action="store_true", 
        help="Disables the normalization of the scale of all models before interpolating")

    args = parser.parse_args()

    # File Collection
    ply_files = []
    if args.directory:
        print(f"Searching for .ply files in directory: {args.directory}")
        # Check if directory exists
        if not os.path.isdir(args.directory):
            parser.error(f"Directory not found: {args.directory}")
        ply_files = sorted([
            os.path.join(args.directory, f) for f in os.listdir(args.directory) if f.lower().endswith('.ply')
        ])
    elif args.models:
        print("Using provided list of models.")
        ply_files = args.models

    # Ensure we have at least 2 models to work with
    if len(ply_files) < 2:
        parser.error("At least two .ply models are required for interpolation, but found " + str(len(ply_files)))
        
    print(f"\nFound {len(ply_files)} models for processing.")

    # Load models
    point_models = []
    for ply_file in ply_files:
        pm = PointModel()
        pm.load_ply(ply_file)
        if(not args.disable_recenter_models):
            pm.recenter_point_cloud()
        if(not args.disable_normalize_scales):
            pm.normalize_scale(10)
        point_models.append(pm)
    
    interp = GaussianInterpolator(device='cuda') 
    interp.load_pointmodels(point_models)
    interp.build_correspondences(spatial_weight=args.spatial_weight, color_weight=args.color_weight, distance_threshold=args.distance_threshold, batch_size=args.batch_size)    

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    # start viser server and viewer
    server = viser.ViserServer(port=args.port, verbose=False)
    viewer = GsplatViserViewer(server=server, model=point_models[0], interpolator=interp, device=device)

    print(f"Viewer server running on port {args.port}. Connect with a viser client.")
    try:
        while True:
            time.sleep(10)
    except KeyboardInterrupt:
        print("Shutting down viewer...")


if __name__ == "__main__":
    main()
