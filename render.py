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
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from utils.system_utils import searchForMaxIteration
from utils.graphics_utils import getWorld2View2, getProjectionMatrix, fov2focal
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
import numpy as np
from copy import deepcopy
from scene.cameras import Camera
from utils.graphics_utils import cam_pos_up_forward_to_Rt
from torchmetrics.functional.image import peak_signal_noise_ratio as psnr, structural_similarity_index_measure as ssim, learned_perceptual_image_patch_similarity as lpips

from color_mlp import ColorMLP
from depth_mlp import DepthMLP
import exr

def render_set(model_path, name, iteration, views, gaussians, pipeline, background, color_mlp = None, depth_mlp=None, dump_samples=False, dump_poses=False, relight_envmap=None, relight_spp=128, relight_intensity_scaling=0.0):
    if relight_envmap:
        raise NotImplementedError
        name += '_relight_' + os.path.basename(relight_envmap)
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    makedirs(render_path, exist_ok=True)
    if not relight_envmap:
        gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
        makedirs(gts_path, exist_ok=True)

    if relight_envmap and relight_envmap != '1':
        raise NotImplementedError
        inten = exr.read(relight_envmap).mean()
        print(f'relight envmap intensity: {inten}')
        if relight_intensity_scaling <= 0:
            raise ValueError('relighting_intensity_scaling <= 0 not supported!')
            relight_intensity_scaling = 0.1 / inten
    else:
        relight_intensity_scaling = 1.0
    
    if dump_poses:
        dump_path = os.path.join(model_path, f'dump_poses_{name}.txt')
        write_views_to_txt(views, dump_path)
            
    mlp_inputs = None
    gts = []
    renderings = []
    psnr_sum = 0
    ssim_sum = 0
    lpips_sum = 0
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        # if idx % (len(views) // 50 + 1) != 0 and len(views) > 100:
        #     continue
        if relight_envmap:
            raise NotImplementedError
            rendering = render_relight_envmap(view, envmap_path=relight_envmap, spps=relight_spp, envmap_scale=relight_intensity_scaling, device='cuda', pc=gaussians, pipe=pipeline, bg_color=background, scaling_modifier=1.0, override_color=None, color_mlp=color_mlp, depth_mlp=depth_mlp, iteration=iteration, dump_samples=dump_samples)
        else:
            res = render(view, gaussians, pipeline, background, color_mlp = color_mlp, depth_mlp=depth_mlp, iteration=iteration, dump_samples=dump_samples)
            rendering = res["render"]
        if dump_samples:
            mlp_input = res["mlp_input"].detach().cpu().numpy() ## [?, in_channels]
            if mlp_inputs is None:
                mlp_inputs = mlp_input
            else:
                mlp_inputs = np.concatenate([mlp_inputs, mlp_input], axis=0)
        if not relight_envmap:
            gt = view.original_image[0:3, :, :]
            torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
        
        if pipeline.output_feature:
            exr.write(rendering.detach().cpu().numpy().transpose(1, 2, 0), os.path.join(render_path, '{0:05d}'.format(idx) + "_feature.exr"))
        elif pipeline.output_shadow:
            if pipeline.output_exr:
                exr.write(rendering.detach().cpu().numpy().transpose(1, 2, 0), os.path.join(render_path, '{0:05d}'.format(idx) + "_shadow.exr"))
            else:
                torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + "_shadow.png"))
        elif pipeline.output_depth:
            if pipeline.output_exr:
                exr.write(rendering.detach().cpu().numpy().transpose(1, 2, 0), os.path.join(render_path, '{0:05d}'.format(idx) + "_depth.exr"))
            else:
                torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + "_depth.png"))
        elif pipeline.output_alpha:
            if pipeline.output_exr:
                exr.write(rendering.detach().cpu().numpy().transpose(1, 2, 0), os.path.join(render_path, '{0:05d}'.format(idx) + "_alpha.exr"))
            else:
                torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + "_alpha.png"))
        else:
            if pipeline.output_exr:
                exr.write(rendering.detach().cpu().numpy().transpose(1, 2, 0), os.path.join(render_path, '{0:05d}'.format(idx) + ".exr"))
            else:
                torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
            if not relight_envmap:
                gts.append(gt)
            renderings.append(rendering)
            
    if len(gts) > 0 and len(renderings) > 0:
        gts = torch.stack(gts, dim=0)
        renderings = torch.stack(renderings, dim=0)
        psnr_sum = psnr(renderings, gts).item()
        ssim_sum = ssim(renderings, gts).item()
        lpips_sum = lpips(renderings, gts).item()
        line = f'PSNR: {psnr_sum:.4f} SSIM: {ssim_sum:.4f} LPIPS: {lpips_sum:.4f}'
        print(line)
        with open(os.path.join(model_path, name, "ours_{}".format(iteration), 'metrics.txt'), 'w') as f:
            f.write(line)
                
    if dump_samples:
        print(mlp_inputs.shape)
        np.save(os.path.join(model_path, name, "ours_{}".format(iteration), "mlp_inputs.npy"), mlp_inputs)
            
def render_video_view(model_path, iteration, views, gaussians, pipeline, background, color_mlp = None, depth_mlp=None, frames=300, y_axis=False, out_name='video_view.mp4', dump_poses=False, bitrate='10M'):
    render_path = os.path.join(model_path, "video_view", "ours_{}".format(iteration), "renders")
    makedirs(render_path, exist_ok=True)

    ## make video views based on the first view from training set
    first_view = views[0]
    video_views = [first_view]
    for idx in range(1, frames):
        if y_axis:
            ## rotate along y-axis
            R = np.array([
                [np.cos(2 * np.pi * idx / frames), 0, np.sin(2 * np.pi * idx / frames)],
                [0, 1, 0],
                [-np.sin(2 * np.pi * idx / frames), 0, np.cos(2 * np.pi * idx / frames)],
            ])
        else:
            ## rotate along z-axis
            R = np.array([
                [np.cos(2 * np.pi * idx / frames), -np.sin(2 * np.pi * idx / frames), 0],
                [np.sin(2 * np.pi * idx / frames), np.cos(2 * np.pi * idx / frames), 0],
                [0, 0, 1],
            ])
        current_view = deepcopy(first_view)
        current_view.R = R @ current_view.R
        current_view.W2C = getWorld2View2(current_view.R, current_view.T, current_view.trans, current_view.scale)
        current_view.C2W = np.linalg.inv(current_view.W2C)

        # view transform is W2C.transpose() so it's column-major:
        current_view.world_view_transform = torch.tensor(current_view.W2C).transpose(0, 1).cuda()
        current_view.projection_matrix = getProjectionMatrix(znear=current_view.znear, zfar=current_view.zfar, fovX=current_view.FoVx, fovY=current_view.FoVy).transpose(0,1).cuda()
        current_view.full_proj_transform = (current_view.world_view_transform.unsqueeze(0).bmm(current_view.projection_matrix.unsqueeze(0))).squeeze(0)
        current_view.camera_center = current_view.world_view_transform.inverse()[3, :3]

        current_view.focal = fov2focal(current_view.FoVx, current_view.full_width)
        current_view.camera_rays, current_view.camera_rays_unnorm = current_view.gen_rays_from_image(current_view.full_height, current_view.full_width, current_view.focal, current_view.C2W)
        current_view.shadow_depth_pts = None
        current_view.distance_pts_pl = None
        video_views.append(current_view)

    if dump_poses:
        dump_path = os.path.join(model_path, 'dump_poses_view.txt')
        write_views_to_txt(video_views, dump_path)

    ## render these views
    for idx, view in enumerate(tqdm(video_views, desc="Rendering video_view")):
        rendering = render(view, gaussians, pipeline, background, color_mlp = color_mlp, depth_mlp=depth_mlp, iteration=iteration)["render"]
        if pipeline.output_exr:
            exr.write(rendering.detach().cpu().numpy().transpose(1, 2, 0), os.path.join(render_path, '{0:05d}'.format(idx) + ".exr"))
        else:
            torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
    
    ## make it a video
    print('generating video...')
    ffmpeg_cmd = f"ffmpeg -y -r {frames//5} -i {os.path.join(render_path, '%05d.png')} -c:v h264 -b:v {bitrate} -vf 'pad=ceil(iw/2)*2:ceil(ih/2)*2,fps={frames//5}' -pix_fmt yuv420p {os.path.join(model_path, out_name)} -hide_banner -loglevel error"
    print(ffmpeg_cmd)
    os.system(ffmpeg_cmd)
                
def render_video_light(model_path, iteration, views, gaussians, pipeline, background, color_mlp = None, depth_mlp=None, frames=300, y_axis=False, out_name='video_light.mp4', dump_poses=False, bitrate='10M'):
    render_path = os.path.join(model_path, "video_light", "ours_{}".format(iteration), "renders")
    makedirs(render_path, exist_ok=True)

    ## make video views based on the first view from training set
    first_view = views[0]
    video_views = [first_view]
    for idx in range(1, frames):
        if y_axis:
            ## rotate along y-axis
            R = torch.tensor([
                [np.cos(2 * np.pi * idx / frames), 0, np.sin(2 * np.pi * idx / frames)],
                [0, 1, 0],
                [-np.sin(2 * np.pi * idx / frames), 0, np.cos(2 * np.pi * idx / frames)],
            ], dtype=torch.float32, device="cuda")
        else:
            ## rotate along z-axis
            R = torch.tensor([
                [np.cos(2 * np.pi * idx / frames), -np.sin(2 * np.pi * idx / frames), 0],
                [np.sin(2 * np.pi * idx / frames), np.cos(2 * np.pi * idx / frames), 0],
                [0, 0, 1],
            ], dtype=torch.float32, device="cuda")
        current_view = deepcopy(first_view)
        current_view.pl_pos = first_view.pl_pos @ R
        current_view.shadow_depth_pts = None
        current_view.distance_pts_pl = None
        video_views.append(current_view)

    if dump_poses:
        dump_path = os.path.join(model_path, 'dump_poses_light.txt')
        write_views_to_txt(video_views, dump_path)
        
    ## render these views
    for idx, view in enumerate(tqdm(video_views, desc="Rendering video_light")):
        rendering = render(view, gaussians, pipeline, background, color_mlp = color_mlp, depth_mlp=depth_mlp, iteration=iteration)["render"]
        if pipeline.output_exr:
            exr.write(rendering.detach().cpu().numpy().transpose(1, 2, 0), os.path.join(render_path, '{0:05d}'.format(idx) + ".exr"))
        else:
            torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
    
    ## make it a video
    print('generating video...')
    ffmpeg_cmd = f"ffmpeg -y -r {frames//5} -i {os.path.join(render_path, '%05d.png')} -c:v h264 -b:v {bitrate} -vf fps={frames//5} -pix_fmt yuv420p {os.path.join(model_path, out_name)} -hide_banner -loglevel error"
    print(ffmpeg_cmd)
    os.system(ffmpeg_cmd)

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, render_videos=False, bitrate='10M', render_light=False, render_view=False, novel_view=None, collocated=False, frame_idx=0, frames=300, y_axis=False, crop_pc=0.0, dump_samples=False, dump_poses=False, relight_envmap=None, relight_spp=128, relight_intensity_scaling=0.0):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False, crop=crop_pc)

        color_mlp = None
        out_light_name = 'video_light.mp4'
        out_view_name = 'video_view.mp4'
        out_name = 'video.mp4'
        if pipeline.color_mlp:
            if iteration == -1:
                iteration = searchForMaxIteration(os.path.join(dataset.model_path, "color_mlp"))
            ckpt_path = os.path.join(dataset.model_path, 'color_mlp', f'iteration_{iteration}', f'color_mlp_chkpnt{iteration}.pth')
            in_channels = pipeline.in_channels + 6 + (pipeline.encoding_levels_each * 12 if pipeline.encoding_levels_each > 0 else 0) + 1 ## pl_distance
            if pipeline.shadow_map:
                in_channels += 1 + (pipeline.encoding_levels_shadow * 2 if pipeline.encoding_levels_shadow > 0 else 0)
            color_mlp = ColorMLP(in_channels, checkpoint=ckpt_path).cuda()

        depth_mlp= None
        if pipeline.depth_mlp:
            depthMLP_ckpt_path = os.path.join(dataset.model_path, 'depth_mlp', f'iteration_{iteration}', f'depth_mlp_chkpnt{iteration}.pth')
            depth_mlp = DepthMLP(in_channels=3, checkpoint=depthMLP_ckpt_path, depth_mlp_modifier=pipeline.depth_mlp_modifier)
            depth_mlp = depth_mlp.cuda()

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if render_videos:
            
            if novel_view:
                first_view = scene.getTestCameras()[0]
                cam_pos, cam_up, cam_dir = novel_view
                new_R, new_T = cam_pos_up_forward_to_Rt(cam_pos, cam_up, cam_dir)
                new_cam = Camera(
                    colmap_id=first_view.colmap_id, 
                    R=new_R, 
                    T=new_T, 
                    FoVx=first_view.FoVx, 
                    FoVy=first_view.FoVy, 
                    image=first_view.original_image,
                    gt_alpha_mask=None,
                    image_name=first_view.image_name,
                    uid=first_view.uid,
                    pl_pos=[3, 3, 3] if not collocated else cam_pos,
                    pl_intensity=1000.0,
                    full_width=first_view.full_width,
                    full_height=first_view.full_height,
                )
                print('R', new_cam.R)
                print('T', new_cam.T)
                views = [new_cam]
            else:
                views = scene.getTestCameras()[frame_idx: frame_idx+1]
                if collocated:
                    views[0].pl_pos = views[0].camera_center
            
            if render_light:
                render_video_light(dataset.model_path, scene.loaded_iter, views, gaussians, pipeline, background, color_mlp=color_mlp, depth_mlp=depth_mlp, frames=frames, y_axis=y_axis, out_name=out_light_name, dump_poses=dump_poses, bitrate=bitrate)
            if render_view:
                render_video_view(dataset.model_path, scene.loaded_iter, views, gaussians, pipeline, background, color_mlp=color_mlp, depth_mlp=depth_mlp, frames=frames, y_axis=y_axis, out_name=out_view_name, dump_poses=dump_poses, bitrate=bitrate)
            ## concate two videos together using ffmpeg
            if render_view and render_light:
                print('concatenating video... {}'.format(out_name))
                ffmepg_cmd = f'ffmpeg -y -i {os.path.join(dataset.model_path, out_light_name)} -i {os.path.join(dataset.model_path, out_view_name)} -filter_complex "[0:v][1:v]concat=n=2:v=1[outv]" -map "[outv]" -c:v h264 -b:v {bitrate} {os.path.join(dataset.model_path, out_name)} -hide_banner -loglevel error'
                print(ffmepg_cmd)
                os.system(ffmepg_cmd)
            return

        if not skip_train:
             render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, color_mlp=color_mlp, depth_mlp=depth_mlp, dump_samples=dump_samples, dump_poses=dump_poses, relight_envmap=relight_envmap, relight_spp=relight_spp, relight_intensity_scaling=relight_intensity_scaling)

        if not skip_test:
             render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, color_mlp=color_mlp, depth_mlp=depth_mlp, dump_samples=dump_samples, dump_poses=dump_poses, relight_envmap=relight_envmap, relight_spp=relight_spp, relight_intensity_scaling=relight_intensity_scaling)
        
def write_views_to_txt(views, dump_path):
    with open(dump_path, 'w') as f:
        for idx, view in enumerate(views):
            c2w = deepcopy(view.C2W)
            c2w[:3, 1:3] *= -1
            line = f'{idx} {view.pl_pos.tolist()} {c2w.reshape(-1).tolist()} {view.FoVx}\n'
            f.write(line.replace('[', '').replace(']', '').replace(',', ''))
    print(f'{len(views)} views written. -> {dump_path}')
             
if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--bitrate", type=str, default='10M')
    parser.add_argument("--light", action="store_true")
    parser.add_argument("--view", action="store_true")
    parser.add_argument("--novel_view", action="store_true")
    parser.add_argument("--collocated", action="store_true")
    parser.add_argument("--y_axis", action="store_true") ## for NRHints Real dataset, rotate along y-axis
    parser.add_argument("--dump_poses", action="store_true") ## dump all poses of rendering frames, for rendering comparisons, etc.
    parser.add_argument("--dump_samples", action="store_true")
    parser.add_argument("--frame_idx", default=0, type=int) ## if render video, which sample in the test set is the first frame?
    parser.add_argument("--frames", default=300, type=int)
    parser.add_argument("--crop_pc", default=0.0, type=float)
    parser.add_argument("--relight_envmap", type=str, default='')
    parser.add_argument("--relight_spp", default=128, type=int)
    parser.add_argument("--relight_intensity_scaling", default=1, type=float)
    args = get_combined_args(parser)
    args.load_env_ae = args.load_env_ae if hasattr(args, 'load_env_ae') else None
    args.load_envmap = args.load_envmap if hasattr(args, 'load_envmap') else None
    args.relight_weights = args.relight_weights if hasattr(args, 'relight_weights') else None
    args.relight_envmap = args.relight_envmap if hasattr(args, 'relight_envmap') else None
    args.r2w_ckpt = args.r2w_ckpt if hasattr(args, 'r2w_ckpt') else None
    print("Rendering " + args.model_path)

    if args.novel_view:
        cam_pos = np.array([2.474873734152916335402955267367, 0.0, 2.474873734152916335402955267367], dtype=np.float32),
        cam_up = np.array([0.0, 0.0, -1.0], dtype=np.float32),
        cam_dir = np.array([0.0, 0.0, 0.0], dtype=np.float32) - cam_pos,
        novel_view = (cam_pos, cam_up, cam_dir)
    else:
        novel_view = None
        
    if args.frame_idx != 0:
        args.less = 1 ## if not use the first sample in test set, then we need to load all test set samples for picking

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(
        model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test,
        y_axis=args.y_axis, crop_pc=args.crop_pc, dump_samples=args.dump_samples, 
        render_videos=args.video, bitrate=args.bitrate, frame_idx=args.frame_idx, frames=args.frames, 
        render_light=args.light, render_view=args.view, novel_view=novel_view, collocated=args.collocated, 
        dump_poses=args.dump_poses,
        relight_envmap=args.relight_envmap, relight_spp=args.relight_spp, relight_intensity_scaling=args.relight_intensity_scaling,
    )