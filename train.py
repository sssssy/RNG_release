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
import torch
from random import randint
from utils.loss_utils import *
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state, GradientScaler
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False
from matplotlib import pyplot as plt

from color_mlp import ColorMLP
from depth_mlp import DepthMLP

def training(args, dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, pc_ckpt, mlp_ckpt, d_mlp_ckpt, loss_type, debug_from, crop_pc=0.0):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    from datetime import datetime
    log_txt_file_name = f"progress-{datetime.now().strftime('%y%m%d-%H%M%S')}.txt"
    with open(os.path.join(dataset.model_path, log_txt_file_name), 'w') as f:
        for arg in vars(args):
            f.write(f"{arg}: {getattr(args, arg)}\n")
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    if pc_ckpt:
        (model_params, first_iter) = torch.load(pc_ckpt, map_location='cuda')
        gaussians.restore(model_params, opt)
        print('Loaded point cloud ({} pts) from checkpoint. Iteration: {}'.format(gaussians.get_xyz.shape[0], first_iter))
        if pipe.reset_features:
            gaussians.reset_features()
            print('Features are reset after loading point cloud.')
    gaussians.training_setup(opt)
    ## explicitly crop the point cloud after possibly loading and training setup
    gaussians.crop_pc(crop_pc)
    print(f'Total number of gaussians after training setup: {scene.gaussians.get_xyz.shape[0]}')
        
    ## initialize color mlp
    color_mlp = None
    if pipe.color_mlp:
        in_channels = pipe.in_channels + 6 + (pipe.encoding_levels_each * 12 if pipe.encoding_levels_each > 0 else 0) + 1 ## pl_distance
        if pipe.shadow_map:
            in_channels += 1 + (pipe.encoding_levels_shadow * 2 if pipe.encoding_levels_shadow > 0 else 0)
        print(f'in channels: {in_channels}')
        color_mlp = ColorMLP(in_channels=in_channels, checkpoint=mlp_ckpt).cuda()
    
    depth_mlp = None
    if pipe.depth_mlp:
        depth_mlp = DepthMLP(in_channels=3, checkpoint=d_mlp_ckpt, depth_mlp_modifier=pipe.depth_mlp_modifier).cuda()

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32).cuda()

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    loss_func = eval(f'{loss_type}_loss')
    viewpoint_stack = None
    ema_loss_for_log = 0.0
    loss_curve = []
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):        
        iter_start.record()

        # gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg, color_mlp = color_mlp, depth_mlp = depth_mlp, iteration=iteration)
        image, depth, alpha, viewspace_point_tensor, visibility_filter, radii = \
            render_pkg["render"], render_pkg['depth'], render_pkg['alpha'], \
            render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        if depth_mlp is not None:
            depth_offsets = render_pkg['depth_offset'].mean().detach().cpu().numpy()
        if pipe.gradient_scaling:
            image, alpha, depth = GradientScaler.apply(image, alpha, depth)

        # Loss
        gt_image = viewpoint_cam.original_image.cuda() ## [3, H, W]
        loss1 = loss_func(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * loss1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        loss.backward(retain_graph=True)

        if iteration == 1:
            print(image.shape)
            print(gt_image.shape)

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            if not torch.isnan(loss):
                ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                line = f"Training Loss={ema_loss_for_log:>8.5f} #G={scene.gaussians.get_xyz.shape[0]:>7d}"
                if depth_mlp is not None:
                    line += f" ΔD={depth_offsets:>7.3f}"
                progress_bar.set_description(line)
                with open(os.path.join(dataset.model_path, log_txt_file_name), 'a') as f:
                    f.write(f'{datetime.now().strftime("%H:%M:%S")} {line}\n')
                progress_bar.update(10)
                loss_curve.append(loss.item())
                plt.figure()
                plt.plot(loss_curve)
                plt.savefig(os.path.join(dataset.model_path, log_txt_file_name.replace(".txt", ".png")))
                plt.close()
            if iteration == opt.iterations:
                progress_bar.close()
                
            # Log and save
            # training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background), color_mlp)
            if (iteration in saving_iterations):
                print("[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
                if color_mlp is not None:
                    color_mlp_save_dir = os.path.join(dataset.model_path, 'color_mlp', f'iteration_{iteration}')
                    os.makedirs(color_mlp_save_dir, exist_ok=True)
                    torch.save((color_mlp.capture(), iteration), os.path.join(color_mlp_save_dir, f'color_mlp_chkpnt{iteration}.pth'))
                if depth_mlp is not None:
                    depth_mlp_save_dir = os.path.join(dataset.model_path, 'depth_mlp', f'iteration_{iteration}')
                    os.makedirs(depth_mlp_save_dir, exist_ok=True)
                    torch.save((depth_mlp.capture(), iteration), os.path.join(depth_mlp_save_dir, f'depth_mlp_chkpnt{iteration}.pth'))

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, opt.prune_min_opacity, scene.cameras_extent, size_threshold)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()
                    gaussians.crop_pc(crop_pc)
                    torch.cuda.empty_cache()
                    print("Number of Gaussians: {}".format(scene.gaussians.get_xyz.shape[0]))

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)
                if color_mlp is not None:
                    color_mlp.optimizer.step()
                    color_mlp.optimizer.zero_grad(set_to_none = True)
                if depth_mlp is not None:
                    depth_mlp.optimizer.step()
                    depth_mlp.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
                if color_mlp is not None:
                    torch.save((color_mlp.capture(), iteration), scene.model_path + "/color_mlp_chkpnt" + str(iteration) + ".pth")
                if depth_mlp is not None:
                    torch.save((depth_mlp.capture(), iteration), scene.model_path + "/depth_mlp_chkpnt" + str(iteration) + ".pth")
                    
            if iteration % 15000 == 0:
                for gidx, pgroup in enumerate(gaussians.optimizer.param_groups):
                    pgroup['lr'] *= 0.75
                    print(f"[ITER {iteration}] gaussian {gidx} lr -> {pgroup['lr']:.2e}")
                if color_mlp is not None:
                    color_mlp.optimizer.param_groups[0]['lr'] *= 0.75
                    print(f"[ITER {iteration}] color_mlp lr -> {color_mlp.optimizer.param_groups[0]['lr']:.2e}")
                if depth_mlp is not None:
                    depth_mlp.optimizer.param_groups[0]['lr'] *= 0.75
                    print(f"[ITER {iteration}] depth_mlp lr -> {depth_mlp.optimizer.param_groups[0]['lr']:.2e}")

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if False:# TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, color_mlp=None):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs, color_mlp)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    # parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    # parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--load_pc", type=str, default = None)
    parser.add_argument("--load_mlp", type=str, default = None)
    parser.add_argument("--load_d_mlp", type=str, default = None)
    parser.add_argument("--loss", type=str, default="l1", choices=['l1', 'l2', 'logl1', 'logl2'])
    parser.add_argument("--crop_pc", type=float, default=0.0)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    args.checkpoint_iterations = args.save_iterations
    
    ## save cmd-line args to txt file
    os.makedirs(args.model_path, exist_ok=True)
            
    ## recursively backup all python scripts
    os.makedirs(os.path.join(args.model_path, "codes"), exist_ok=True)
    for root, dirs, files in os.walk("/home/lab409/3dgs-pl/gaussian-splatting", topdown=True):
        for file in files:
            if file.endswith(".py"):
                os.makedirs(root.replace("/home/lab409/3dgs-pl/gaussian-splatting", os.path.join(args.model_path, "codes")), exist_ok=True)
                # os.system(f"cp {os.path.join(root, file)} {os.path.join(args.model_path, 'codes', root.split('/')[-1])}")
                os.system(f'cp {os.path.join(root, file)} {os.path.join(root.replace("/home/lab409/3dgs-pl/gaussian-splatting", os.path.join(args.model_path, "codes")), file)}')
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(args, lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.load_pc, args.load_mlp, args.load_d_mlp, args.loss, args.debug_from, args.crop_pc)

    # All done
    print("Training complete.")
