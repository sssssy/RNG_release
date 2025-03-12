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

import os
import sys
from PIL import Image
from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal, rotation_to_euler
import numpy as np
import cv2
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    pl_pos: np.array = None
    pl_intensity: np.array = None
    full_width: int = 0
    full_height: int = 0
    crop_offset_x: int = 0
    crop_offset_y: int = 0

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            pl_pos=np.array([0., 0., 0.]), pl_intensity=np.array([0., 0., 0.]), 
                            image_path=image_path, image_name=image_name, width=width, height=height, full_height=height, full_width=width, crop_offset_x=0, crop_offset_y=0)

        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def readColmapSceneInfo(path, images, eval, less=1, max_training_images=300, llffhold=8):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images == None else images
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir))
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    print(f"Train Cameras: {len(train_cam_infos)}, Test Cameras: {len(test_cam_infos)}")
    
    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "random_init_points3d.ply")
    if True: #not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png", less=1, max_training_images=300, max_reso=512):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        '''
            nrhints transforms.json: 
            [
                camera_intrinsics: float x4 (cx, cy, fx, fy)
                camera_far: float
                camera_near: float
                frames: [
                    file_ext: '.png',
                    file_path: str,
                    original_file_path: str,
                    pl_intensity: float x3,
                    pl_pos: float x3,
                    transform_matrix: matrix 4x4,
                ]
            ]
        '''
        try:
            cx_0, cy_0, fx_0, fy_0 = contents["camera_intrinsics"]
            dataset_has_fov = False
            print(f'Found {cx_0=} {cy_0=} {fx_0=} {fy_0=}')
        except KeyError:
            fovx = contents["camera_angle_x"]
            dataset_has_fov = True
            print(f'Found {fovx=}')

        if less > 0:
            contents["frames"] = contents["frames"][:less]
        frames = contents["frames"]
        random_idx = np.random.permutation(len(frames))
        if max_training_images > 0:
            random_idx = random_idx[:max_training_images]
        for idx, frame in enumerate(frames):
            
            # if idx % (len(frames) // 300 + 1) != 0 and len(frames) > 1000:
            #     continue
            
            if idx not in random_idx:
                continue
            
            cam_name = os.path.join(path, frame["file_path"] + (frame["file_ext"] if "file_ext" in frame.keys() else ""))

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            if arr.shape[0] > max_reso and arr.shape[1] > max_reso:
                arr = cv2.resize(arr, (arr.shape[0] // 2, arr.shape[1] // 2), cv2.INTER_AREA)
                cx, cy, fx, fy = cx_0 / 2, cy_0 / 2, fx_0 / 2, fy_0 / 2
                if idx == 0:
                    print(f'Resized to {arr.shape}, {cx=} {cy=} {fx=} {fy=}')
            elif not dataset_has_fov:
                cx, cy, fx, fy = cx_0, cy_0, fx_0, fy_0
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

            # fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            # FovY = fovy 
            # FovX = fovx
            
            if not dataset_has_fov:
                ## may be an off-centered camera (cropped image)
                W, H = image.size
                full_width = int(W / 2 + np.abs(cx - W / 2)) * 2
                full_height = int(H / 2 + np.abs(cy - H / 2)) * 2
                crop_offset_x = 0 if cx >= W / 2 else int(full_width //2 - cx)
                crop_offset_y = 0 if cy >= H / 2 else int(full_height / 2 - cy)
                FovX = np.arctan(full_width / (2 * fx)) * 2
                FovY = np.arctan(full_height / (2 * fy)) * 2
            else:
                FovX = fovx
                FovY = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
                full_width, full_height = W, H = image.size
                crop_offset_x = crop_offset_y = 0
            
            if idx == 0:
                print(f'Image size: {W, H}', f'Full size: {full_width, full_height}', f'Crop offset: {crop_offset_x, crop_offset_y}')
                print(f'FovX (deg): {np.rad2deg(FovX)}, FovY (deg): {np.rad2deg(FovY)}')
            
            pl_pos = np.array(frame["pl_pos"]) if "pl_pos" in frame else np.array([0., 0., 0.])
            pl_intensity = np.array(frame["pl_intensity"]) if "pl_intensity" in frame else np.array([0., 0., 0.])

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1], 
                            pl_pos=pl_pos, pl_intensity=pl_intensity, 
                            full_width=full_width, full_height=full_height, crop_offset_x=crop_offset_x, crop_offset_y=crop_offset_y))
            
    return cam_infos

def readNerfSyntheticInfo(path, white_background, eval, less=1, max_training_images=300, extension=".png", max_reso=512):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension, less, max_training_images, max_reso)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension, less, 0, max_reso)
    
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []
        
    print(f"Train Cameras: {len(train_cam_infos)}, Test Cameras: {len(test_cam_infos)}")

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "random_init_points3d.ply")
    if True: #not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender" : readNerfSyntheticInfo
}