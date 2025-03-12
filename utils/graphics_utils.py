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
import math
import numpy as np
from typing import NamedTuple

class BasicPointCloud(NamedTuple):
    points : np.array
    colors : np.array
    normals : np.array

def geom_transform_points(points, transf_matrix):
    P, _ = points.shape
    ones = torch.ones(P, 1, dtype=points.dtype, device=points.device)
    points_hom = torch.cat([points, ones], dim=1)
    points_out = torch.matmul(points_hom, transf_matrix.unsqueeze(0))

    denom = points_out[..., 3:] + 0.0000001
    return (points_out[..., :3] / denom).squeeze(dim=0)

def getWorld2View(R, t):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0
    return np.float32(Rt)

def getWorld2View2(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)

def cam_pos_up_forward_to_Rt(cam_pos, cam_up, cam_forward):
    '''
        inputs: row vectors [1, 4] (p' = p * R + t)
        return R and T in column-major format
    '''
    cam_forward = cam_forward / np.linalg.norm(cam_forward)
    cam_right = np.cross(cam_up, cam_forward)
    cam_right = cam_right / np.linalg.norm(cam_right)
    cam_up = np.cross(cam_forward, cam_right)
    cam_up = cam_up / np.linalg.norm(cam_up)

    R = np.zeros((3, 3))
    R[:, 0] = cam_right
    R[:, 1] = cam_up
    R[:, 2] = cam_forward

    t = -np.dot(cam_pos, R)
    return R, t

def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P

def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))

def rotation_to_euler(rot_axis, rot_angle_deg):
    # Convert the rotation angle from degrees to radians
    rot_angle_rad = np.deg2rad(rot_angle_deg)
    
    # Compute the rotation matrix using the axis-angle representation
    ux, uy, uz = rot_axis
    c = np.cos(rot_angle_rad)
    s = np.sin(rot_angle_rad)
    one_c = 1 - c
    
    R = np.array([
        [c + ux**2 * one_c,     ux*uy * one_c - uz*s, ux*uz * one_c + uy*s],
        [uy*ux * one_c + uz*s, c + uy**2 * one_c,     uy*uz * one_c - ux*s],
        [uz*ux * one_c - uy*s, uz*uy * one_c + ux*s, c + uz**2 * one_c]
    ])

    # Extract Euler angles from the rotation matrix
    if R[2, 0] < 1:
        if R[2, 0] > -1:
            beta = np.arcsin(-R[2, 0])  # β = arcsin(-r31)
            alpha = np.arctan2(R[2, 1], R[2, 2])  # α = atan2(r32, r33)
            gamma = np.arctan2(R[1, 0], R[0, 0])  # γ = atan2(r21, r11)
        else:  # R[2, 0] = -1
            beta = np.pi / 2
            alpha = 0
            gamma = np.arctan2(-R[1, 2], R[1, 1])
    else:  # R[2, 0] = 1
        beta = -np.pi / 2
        alpha = 0
        gamma = np.arctan2(-R[1, 2], R[1, 1])
    
    # Convert angles from radians to degrees
    alpha_deg = np.rad2deg(alpha)
    beta_deg = np.rad2deg(beta)
    gamma_deg = np.rad2deg(gamma)
    
    return alpha_deg, beta_deg, gamma_deg