#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from torch import nn
import numpy as np
from utils.graphics_utils import getWorld2View2, getProjectionMatrix, fov2focal, cam_pos_up_forward_to_Rt

class Camera(nn.Module):
    def __init__(self, colmap_id, R, T, FoVx, FoVy, image, gt_alpha_mask,
                 image_name, uid, pl_pos, pl_intensity, 
                 full_width=0, full_height=0, crop_offset_x=0, crop_offset_y=0,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda"
                 ):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name
        
        self.pl_pos = torch.tensor(pl_pos, device=data_device).float() if pl_pos is not None else None
        self.pl_intensity = torch.tensor(pl_intensity, device=data_device).float() if pl_intensity is not None else None
        
        self.full_width = full_width
        self.full_height = full_height
        self.crop_offset_x = crop_offset_x
        self.crop_offset_y = crop_offset_y

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        self.original_image = image.clamp(0.0, 1.0).to(self.data_device)
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]

        if gt_alpha_mask is not None:
            self.original_image *= gt_alpha_mask.to(self.data_device)
        else:
            self.original_image *= torch.ones((1, self.image_height, self.image_width), device=self.data_device)

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        # R is column-major in OpenCV format:
        # W2C is R.transpose(), so it's row-major 
        self.W2C = getWorld2View2(R, T, trans, scale)
        self.C2W = np.linalg.inv(self.W2C)

        # view transform is W2C.transpose() so it's column-major:
        self.world_view_transform = torch.tensor(self.W2C).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

        self.focal = fov2focal(self.FoVx, self.full_width)
        self.camera_rays, self.camera_rays_unnorm = self.gen_rays_from_image(self.full_height, self.full_width, self.focal, self.C2W)
        self.shadow_depth_pts = None ## the depth map of the shadow camera at pl_pos 
        self.distance_pts_pl = None ## distance between the depth_pts of view camera and pl_pos
        
    # TODO: This is OpenCV convention, need to align with Blender world space
    def gen_rays_from_image(self, H, W, focal, c2w):
        u, v = np.meshgrid(np.arange(W), np.arange(H))
        u = u.reshape(-1).astype(np.float32) + 0.5    # add half pixel offset
        v = v.reshape(-1).astype(np.float32) + 0.5    # add half pixel offset
        pixels = np.stack([u, v, np.ones_like(u)], axis=0)  # (3, H*W)

        intrinsics = np.array(
            [[focal, 0, W/2], [0, focal, H/2], [0, 0, 1]], dtype=np.float32)
        rays_d = np.dot(np.linalg.inv(intrinsics[:3, :3]), pixels)
        rays_d = np.dot(c2w[:3, :3], rays_d)  # (3, H*W)
        # rays_d = rays_d.transpose((1, 0))  # (H*W, 3)

        # normalize ray directions to unit length
        rays_d_normalized = rays_d / np.linalg.norm(rays_d, axis=0)

        # convert rays_d to torch tensor in [3, H, W]
        camera_rays = torch.tensor(
            rays_d_normalized, dtype=torch.float32).reshape((3, H, W))
        # the unnormalized version for computing depth points 
        camera_rays_unnorm = torch.tensor(
            rays_d,  dtype=torch.float32).reshape((3, H, W))

        # rays_o = c2w[:3, 3].reshape((1, 3))
        # rays_o = np.tile(rays_o, (rays_d.shape[0], 1))  # (H*W, 3)

        # depth = np.linalg.inv(c2w)[2, 3]
        # depth = depth * np.ones((rays_o.shape[0],), dtype=np.float32)  # (H*W,)
        return camera_rays, camera_rays_unnorm
    
class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height    
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]

