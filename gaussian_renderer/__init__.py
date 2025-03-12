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
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from diff_gaussian_rasterization_orthographic import GaussianRasterizationSettings as OrthographicGaussianRasterizationSettings, GaussianRasterizer as OrthographicGaussianRasterizer
from scene.gaussian_model import GaussianModel
from scene.cameras import Camera
from utils.sh_utils import eval_sh
from color_mlp import positional_encoding, normalize

from utils.graphics_utils import cam_pos_up_forward_to_Rt

import exr

def render(viewpoint_camera, pc: GaussianModel, pipe, bg_color: torch.Tensor, scaling_modifier = 1.0, override_color = None, color_mlp = None, depth_mlp = None, iteration = 0, dump_samples = False, relight_envmap = None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
    
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        # image_height=int(viewpoint_camera.image_height),
        # image_width=int(viewpoint_camera.image_width),
        image_height=int(viewpoint_camera.full_height),
        image_width=int(viewpoint_camera.full_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.max_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
        low_pass_filter_radius=0.3,
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs / MLP or by any other means in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
            
        ###############################################
        ## [fjh] this sector is for color mlp
        ###############################################
        elif pipe.color_mlp:
            in_feature = pc.get_features.reshape(pc.get_xyz.shape[0], -1)[:, :pipe.in_channels] ## shape: [N, (sh_degree+1) ** 2 * 3] -> [N, in_channels]
            if pipe.defer_shading or pipe.shadow_map:
                ## [fjh] NOTE: no matter defer_shading or not, as long as shadow_map is true, we need to rasterize the image for the first time to get depth_pts
                ## if defer_shading, then: rasterize -> depth_pts -> shadow_map -> shadow_hint -> color_mlp
                ## else:               rasterize -> depth_pts -> shadow_map -> shadow_hint -> color_mlp -> rasterize one more time
                ## if not shadow_map:  rasterize -> color_mlp (the simple defer_shading operation)
                colors_precomp = in_feature ## shape: [N, in_channels] --> check cuda part, config.h: NUM_CHANNELS
            else:
                pl_direction = viewpoint_camera.pl_pos.reshape(1, 3).expand(pc.get_xyz.shape[0], 3) - pc.get_xyz ## shape: [N, 3]
                pl_distance = torch.norm(pl_direction, dim=1).reshape(-1, 1)
                camera_direction = viewpoint_camera.camera_center - pc.get_xyz ## shape: [N, 3]
                mlp_input = torch.cat([
                    in_feature, 
                    positional_encoding(normalize(pl_direction), levels=pipe.encoding_levels_each),
                    positional_encoding(normalize(camera_direction), levels=pipe.encoding_levels_each),
                    1.0 / pl_distance ** 2,
                    ], dim=1) ## shape: [N, in_channels]
                colors_precomp = color_mlp(mlp_input)
                
                ## pad the colors_precomp to pipe.in_channels to fit the CUDA part
                colors_precomp = torch.cat([colors_precomp, torch.zeros(colors_precomp.shape[0], pipe.in_channels - 3).cuda()], dim=1)
        ###############################################
        ###############################################
        
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii, rendered_depth, rendered_alpha = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)
    alpha_mask = rendered_alpha > 0.0 ## [1, H, W]
    ##! correct the depth for where total alpha is not 1.0
    rendered_depth = torch.where(alpha_mask, rendered_depth / rendered_alpha, rendered_depth.max())
    
    if pipe.output_depth or pipe.output_alpha or pipe.output_feature:
        crop_offset_x, crop_offset_y = int(viewpoint_camera.crop_offset_x), int(viewpoint_camera.crop_offset_y)
        ## [NUM_CHANNELS (default: RGB=3), full_height, full_width]
        if pipe.output_depth:
            res = rendered_depth[:, crop_offset_y: crop_offset_y + viewpoint_camera.image_height, crop_offset_x: crop_offset_x + viewpoint_camera.image_width]
            res = (res - res.min()) / (res.max() - res.min())
        elif pipe.output_alpha:
            res = rendered_alpha[:, crop_offset_y: crop_offset_y + viewpoint_camera.image_height, crop_offset_x: crop_offset_x + viewpoint_camera.image_width]
        else: # pipe.output_feature:
            res = rendered_image[:, crop_offset_y: crop_offset_y + viewpoint_camera.image_height, crop_offset_x: crop_offset_x + viewpoint_camera.image_width]
        return {"render": res,
                "depth": rendered_depth,
                "alpha": rendered_alpha,
                "viewspace_points": screenspace_points,
                "visibility_filter": radii > 0,
                "radii": radii,
                "mlp_input": mlp_input if dump_samples else None,}
    
    ## early return
    if not pipe.color_mlp or (not pipe.defer_shading and not pipe.shadow_map): 
    
        ## un-pad
        rendered_image = rendered_image[:3] ## [3, full_height, full_width]
        
        crop_offset_x, crop_offset_y = int(viewpoint_camera.crop_offset_x), int(viewpoint_camera.crop_offset_y)
        ## [NUM_CHANNELS (default: RGB=3), full_height, full_width]
        rendered_image = rendered_image[:, crop_offset_y: crop_offset_y + viewpoint_camera.image_height, crop_offset_x: crop_offset_x + viewpoint_camera.image_width]
        rendered_depth = rendered_depth[:, crop_offset_y: crop_offset_y + viewpoint_camera.image_height, crop_offset_x: crop_offset_x + viewpoint_camera.image_width]
        rendered_alpha = rendered_alpha[:, crop_offset_y: crop_offset_y + viewpoint_camera.image_height, crop_offset_x: crop_offset_x + viewpoint_camera.image_width]

        # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
        # They will be excluded from value updates used in the splitting criteria.
        return {"render": rendered_image,
                "depth": rendered_depth,
                "alpha": rendered_alpha,
                "viewspace_points": screenspace_points,
                "visibility_filter": radii > 0,
                "radii": radii,
                "mlp_input": mlp_input if dump_samples else None,}

    ########################################################
    ## [fjh] if use color_mlp and (defer_shading or shadow_map)
    ########################################################
    camera_rays = viewpoint_camera.camera_rays
    camera_rays_unnorm = viewpoint_camera.camera_rays_unnorm
    _, H, W = camera_rays.shape

    ## apply the depth MLP
    depth_offset = None
    if depth_mlp:
        depth_offset = depth_mlp(viewpoint_camera.camera_center).reshape(1) ## [3] -> [1]
        depth_offset = depth_pts_modifier = depth_offset + 1
    else:
        depth_pts_modifier = pipe.depth_pts_modifier
    # rendered_depth += depth_offset * rendered_depth

    cam_center = viewpoint_camera.camera_center[:, np.newaxis, np.newaxis].repeat(1, H, W).detach()
    depth_pts = torch.where(alpha_mask, cam_center + camera_rays.cuda() * rendered_depth.detach() * depth_pts_modifier, torch.zeros_like(cam_center)) ## [fjh] NOTE: for the "unvalid" points, the depth_pts should be far enough from the camera center, otherwise the pl_distance will be buggy.

    if pipe.shadow_map:
        
        with torch.set_grad_enabled(pipe.shadow_grad):
            if iteration > pipe.enable_shadow_from: 
                if iteration % pipe.shadow_cache_rebuild == 0 or viewpoint_camera.shadow_depth_pts is None or viewpoint_camera.distance_pts_pl is None:
                    if not relight_envmap:
                        ## set up a virtual ShadowCamera and rasterize to get the shadow_depth
                        shadow_cam_pos = viewpoint_camera.pl_pos.cpu().numpy().reshape(-1) ## shape: [3,]
                                    # shadow_cam_dir = available_depth_pts.reshape(3, -1).mean(-1).cpu().numpy().reshape(-1) - shadow_cam_pos ## shape: [3,]
                        shadow_cam_dir =  - shadow_cam_pos ## shape: [3,]
                        shadow_cam_up = np.array([0., 1., 0.,]).reshape(3,)
                        crop_boundary = pc.crop_boundary if pc.crop_boundary > 0 else 1.0
                        shadow_cam_fov = np.rad2deg(np.arctan(crop_boundary / np.linalg.norm(shadow_cam_pos)) * 2.0)
                        shadow_cam_tanfov = np.tan(np.deg2rad(shadow_cam_fov / 2.0))
                        shadow_R, shadow_T = cam_pos_up_forward_to_Rt(shadow_cam_pos, shadow_cam_up, shadow_cam_dir)
                        shadow_camera = Camera(
                            colmap_id=viewpoint_camera.colmap_id, 
                            R=shadow_R, 
                            T=shadow_T, 
                            FoVx=np.deg2rad(shadow_cam_fov), 
                            FoVy=np.deg2rad(shadow_cam_fov), 
                            image=viewpoint_camera.original_image,
                            gt_alpha_mask=None,
                            image_name=viewpoint_camera.image_name,
                            uid=viewpoint_camera.uid,
                            pl_pos=None,
                            pl_intensity=None,
                            full_width=viewpoint_camera.full_width,
                            full_height=viewpoint_camera.full_height,
                        )
                        shadow_raster_settings = GaussianRasterizationSettings(
                                        image_height=int(shadow_camera.image_height),
                                        image_width=int(shadow_camera.image_width),
                                        tanfovx=shadow_cam_tanfov,
                                        tanfovy=shadow_cam_tanfov,
                                        bg=bg_color,
                                        scale_modifier=scaling_modifier,
                                        viewmatrix=shadow_camera.world_view_transform,
                                        projmatrix=shadow_camera.full_proj_transform,
                                        sh_degree=pc.max_sh_degree,
                                        campos=shadow_camera.camera_center,
                                        prefiltered=False,
                                        debug=pipe.debug,
                                        low_pass_filter_radius=0.3,
                                    )
                        shadow_rasterizer = GaussianRasterizer(raster_settings=shadow_raster_settings)
                        rendered_shadow_camera, shadow_radii, shadow_depth, shadow_alpha = shadow_rasterizer(
                                        means3D = means3D,
                                        means2D = means2D,
                                        shs = shs,
                                        colors_precomp = colors_precomp,
                                        opacities = opacity,
                                        scales = scales,
                                        rotations = rotations,
                                        cov3D_precomp = cov3D_precomp
                                    )
                                    
                        ## project all the depth_pts to shadow_cam, get the xys 
                        ## here all transform matrices are column-major, that is p' = p * full_proj_transform
                        projected_depth_pts = depth_pts.clone().reshape(3, -1).transpose(0, 1) ## [H*W, 3]
                        projected_depth_pts = torch.cat([projected_depth_pts, torch.ones(H*W, 1).cuda()], dim=1) ## [H*W, 4]
                        projected_depth_pts = torch.matmul(projected_depth_pts, shadow_camera.world_view_transform) ## [H*W, 4]
                        projected_depth_pts = projected_depth_pts.transpose(0, 1) ## [4, H*W]
                        projected_depth_pts = projected_depth_pts[:3] / projected_depth_pts[3] ## [3, H*W] (de-homogenized)
                                    
                        grid_x = projected_depth_pts[0] / projected_depth_pts[2] ## [H*W]
                        grid_y = projected_depth_pts[1] / projected_depth_pts[2]
                        
                        shadow_image_plane_half_width = shadow_cam_tanfov
                        shadow_image_plane_half_height = shadow_cam_tanfov
                        grid_x = torch.clamp(grid_x / shadow_image_plane_half_width, -1, 1)
                        grid_y = torch.clamp(grid_y / shadow_image_plane_half_height, -1, 1)
                                    
                        grid = torch.stack([grid_x, grid_y], dim=-1).reshape(1, H, W, 2)

                        shadow_depth = shadow_depth.reshape(1, 1, shadow_camera.image_height, shadow_camera.image_width).cuda()
                    else:
                        ## set up a virtual shadow camera (orthographic) for a directional light source
                        light_dir = normalize(viewpoint_camera.light_dir).float().cuda()
                        shadow_cam_pos = light_dir * 3.0
                        shadow_cam_forward = -light_dir
                        shadow_cam_right = -normalize(torch.cross(shadow_cam_forward, torch.tensor([0.0, 0.0, 1.0]).float().cuda()))
                        shadow_cam_up = torch.cross(shadow_cam_forward, shadow_cam_right)
                        shadow_raster_settings = OrthographicGaussianRasterizationSettings(
                                        image_height=int(viewpoint_camera.image_height),
                                        image_width=int(viewpoint_camera.image_width),
                                        light_dir= light_dir,
                                        cam_center=shadow_cam_pos,
                                        cam_right=shadow_cam_right,
                                        cam_up=shadow_cam_up,
                                        bg=bg_color,
                                        scale_modifier=scaling_modifier,
                                        sh_degree=pc.max_sh_degree,
                                        prefiltered=False,
                                        debug=pipe.debug,
                                        ## below args are useless
                                        low_pass_filter_radius=0.3,
                                        viewmatrix=viewpoint_camera.world_view_transform,
                                        projmatrix=viewpoint_camera.projection_matrix,
                                        campos=viewpoint_camera.camera_center,
                                    )
                        shadow_rasterizer = OrthographicGaussianRasterizer(raster_settings=shadow_raster_settings)
                        rendered_shadow_camera, shadow_radii, shadow_depth, shadow_alpha = shadow_rasterizer(
                                        means3D = means3D,
                                        means2D = means2D,
                                        shs = shs,
                                        colors_precomp = colors_precomp,
                                        opacities = opacity,
                                        scales = scales,
                                        rotations = rotations,
                                        cov3D_precomp = cov3D_precomp
                                    )
                        shadow_depth *= -1
                                    
                        ## project all the depth_pts to shadow_cam, get the xys 
                        ## here all transform matrices are column-major, that is p' = p * full_proj_transform
                        projected_depth_pts = depth_pts.clone().reshape(3, -1).transpose(0, 1) ## [H*W, 3]
                        # print('pl_pos', viewpoint_camera.pl_pos, light_dir)
                        # print('shadow cam', shadow_cam_pos, shadow_cam_forward, shadow_cam_right, shadow_cam_up)
                        
                        depth_pts_to_shadow_cam = projected_depth_pts - shadow_cam_pos
                        # print('depth_pts_to_shadow_cam', depth_pts_to_shadow_cam.shape, depth_pts_to_shadow_cam.max(), depth_pts_to_shadow_cam.max())
                        projected_depth = torch.matmul(depth_pts_to_shadow_cam, shadow_cam_forward)
                        # print('projected_depth', projected_depth.shape, projected_depth.max(), projected_depth.min())
                        world_p_proj = depth_pts_to_shadow_cam - shadow_cam_forward * projected_depth[:, np.newaxis]
                        # print('world_p_proj', world_p_proj.shape, world_p_proj.max(), world_p_proj.min())
                        grid_x = torch.matmul(world_p_proj, shadow_cam_right)
                        grid_y = torch.matmul(world_p_proj, shadow_cam_up)
                        # print('grid', grid_x.max(), grid_x.min(), grid_y.max(), grid_y.min())
                                    
                        shadow_image_plane_half_width = shadow_image_plane_half_height = 1.0
                        grid_x = torch.clamp(grid_x / shadow_image_plane_half_width, -1, 1)
                        grid_y = torch.clamp(grid_y / shadow_image_plane_half_height, -1, 1)
                                    
                        grid = torch.stack([grid_x, grid_y], dim=-1).reshape(1, H, W, 2)

                        shadow_depth = shadow_depth.reshape(1, 1, viewpoint_camera.image_height, viewpoint_camera.image_width).cuda()

                    ##! correct the depth for where total alpha is not 1.0
                    shadow_depth = torch.where(shadow_alpha > 0, shadow_depth / shadow_alpha, shadow_depth.max())

                    shadow_depth_pts = torch.nn.functional.grid_sample(shadow_depth, grid, align_corners=True, mode='bilinear').reshape(1, H, W)
                    shadow_depth_pts = torch.where(alpha_mask, shadow_depth_pts, torch.zeros_like(shadow_depth_pts)) ## [1, H, W]
                    distance_pts_pl = torch.norm(depth_pts - torch.tensor(shadow_cam_pos).cuda().reshape(3,1,1).expand_as(depth_pts), dim=0).reshape(1, H, W)
                    distance_pts_pl = torch.where(alpha_mask, distance_pts_pl, torch.zeros_like(distance_pts_pl)) ## [1, H, W]
                    
                    ## remember them for the future iterations
                    viewpoint_camera.shadow_depth_pts = shadow_depth_pts.detach().cpu()
                    viewpoint_camera.distance_pts_pl = distance_pts_pl.detach().cpu()

                else:
                    ## use cached shadow information
                    shadow_depth_pts = viewpoint_camera.shadow_depth_pts.cuda()
                    distance_pts_pl = viewpoint_camera.distance_pts_pl.cuda()

                # if iteration % 500 == 0:
                #     print("distance_pts_pl:", distance_pts_pl.shape, distance_pts_pl.min().item(), distance_pts_pl.max().item())
                #     print("shadow_depth_pts:", shadow_depth_pts.shape, shadow_depth_pts.min().item(), shadow_depth_pts.max().item())
                #     exr.write(rendered_image.detach().cpu().numpy().transpose(1, 2, 0), 'rendered_image.exr')
                #     exr.write(rendered_depth.detach().cpu().numpy().reshape(H, W), 'rendered_depth.exr')
                #     exr.write(rendered_alpha.detach().cpu().numpy().reshape(H, W), 'rendered_alpha.exr')
                #     exr.write(rendered_shadow_camera.detach().cpu().numpy().transpose(1, 2, 0), 'rendered_shadow_camera.exr')
                #     exr.write(shadow_depth.detach().cpu().numpy().reshape(H, W), 'shadow_depth.exr')
                #     exr.write(shadow_alpha.detach().cpu().numpy().reshape(H, W), 'shadow_alpha.exr')
                #     exit()
            else:
                ## not using shadow mapping for this iteration
                shadow_depth_pts = torch.ones(1, H, W).cuda()
                distance_pts_pl = torch.ones(1, H, W).cuda()

        shadow_hint = torch.cat([shadow_depth_pts, distance_pts_pl], dim=0) ## [2, H, W]
        shadow_hint = shadow_hint * rendered_alpha
        shadow_hint[0:1] = torch.clamp(shadow_hint[1:2] - shadow_hint[0:1], 0.0, 1.0) ## [1, H, W], larger means in shadow
        # shadow_hint[1:2] = torch.zeros_like(shadow_hint[1:2]) ## [1, H, W]
        # shadow_hint = torch.where(alpha_mask, shadow_hint, 0.0) ## only consider the visible points
        # shadow_hint[shadow_hint > pipe.shadow_hints_limit] = 0.0
        # shadow_hint[shadow_hint < pipe.shadow_hints_margin] = 0.0
        # shadow_hint[0:1] = torch.sign(shadow_hint[0:1] - pipe.shadow_hints_margin) ## [2, H, W]
        
        if pipe.output_shadow:
            crop_offset_x, crop_offset_y = int(viewpoint_camera.crop_offset_x), int(viewpoint_camera.crop_offset_y)
            res = shadow_hint[:, crop_offset_y: crop_offset_y + viewpoint_camera.image_height, crop_offset_x: crop_offset_x + viewpoint_camera.image_width]
            return {"render": 1 - torch.clamp(res[0:1], 0.0, 1.0),
            # return {"render": res[0:1] / res[0:1].max(), ##* vis shadow_depth_pts
            # return {"render": shadow_depth[0] / shadow_depth[0].max(), ##* vis shadow_camera (depth)
            # return {"render": depth_pts.reshape(3, H, W),
            # return {"render": projected_depth_pts.reshape(3, H, W),
            # return {"render": torch.cat([grid[0], torch.ones_like(grid[0, ..., 0:1])], dim=-1).permute(2, 0, 1),
                    "depth": rendered_depth,
                    "alpha": rendered_alpha,
                    "viewspace_points": screenspace_points,
                    "visibility_filter": radii > 0,
                    "radii": radii,
                    "mlp_input": mlp_input if dump_samples else None,}
        
        ## [fjh] if defer_shading, then this is the last operation. All features are screen-space and fed into mlp -> final rendering.
        if pipe.defer_shading:
            if relight_envmap is not None:
                light_dir = normalize(viewpoint_camera.light_dir).float().cuda()
                pl_direction = light_dir.reshape(1, 3).expand(H*W, -1) ## [H*W, 3]
                pl_distance = torch.norm(pl_direction, dim=1).reshape(-1, 1) * 100.0
            else:
                pl_direction = viewpoint_camera.pl_pos.reshape(1, 3).expand(H*W, -1) - depth_pts.permute(1, 2, 0).reshape(-1, 3) ## shape: [H*W, 3]
                pl_distance = torch.norm(pl_direction, dim=1).reshape(-1, 1)
            camera_direction = viewpoint_camera.camera_center.reshape(1, 3).expand(H*W, -1) - depth_pts.permute(1, 2, 0).reshape(-1, 3) ## shape: [H*W, 3]
            mlp_input = rendered_image.permute(1, 2, 0).reshape(-1, pipe.in_channels)
            mlp_input = torch.cat([
                mlp_input,
                positional_encoding(normalize(pl_direction), levels=pipe.encoding_levels_each),
                positional_encoding(normalize(camera_direction), levels=pipe.encoding_levels_each),
                positional_encoding(shadow_hint[0:1].reshape(1, -1).permute(1, 0), levels=pipe.encoding_levels_shadow),
                1.0 / pl_distance ** 2,
            ], dim=1)
            res = color_mlp(mlp_input).reshape(viewpoint_camera.full_height, viewpoint_camera.full_width, 3).permute(2, 0, 1)

        ## [fjh] if not defer_shading, then we need to compute colors (with shadow_hints) for all PCs and rasterize for one more time.
        else: 
            ## project the pc to view_camera and grid sample the shadow_hints to get shadow_hint for each pc
            projected_pcs = torch.cat([pc.get_xyz, torch.ones(pc.get_xyz.shape[0], 1).cuda()], dim=1) ## [N, 4]
            projected_pcs = torch.matmul(projected_pcs, viewpoint_camera.world_view_transform) ## [N, 4]
            projected_pcs = projected_pcs[:, :3] / projected_pcs[:, 3].reshape(-1, 1) ## [N, 3] (de-homogenized)
            proj_x = projected_pcs[:, 0] / projected_pcs[:, 2] ## [N]
            proj_y = projected_pcs[:, 1] / projected_pcs[:, 2]
            half_image_plane_x = tanfovx
            half_image_plane_y = tanfovy
            grid_x = proj_x / half_image_plane_x
            grid_y = proj_y / half_image_plane_y
            grid = torch.stack([grid_x, grid_y], dim=-1).reshape(1, 1, -1, 2)
            
            shadow_hint = torch.nn.functional.grid_sample(shadow_hint.reshape(1, 2, H, W), grid, align_corners=True).reshape(2, -1).permute(1, 0) ## [N, 2]
            
            if relight_envmap is not None:
                light_dir = normalize(viewpoint_camera.light_dir).float().cuda()
                pl_direction = light_dir.reshape(1, 3).expand_as(pc.get_xyz, -1) ## [H*W, 3]
                pl_distance = torch.norm(pl_direction, dim=1).reshape(-1, 1) * 100.0
            else:
                pl_direction = viewpoint_camera.pl_pos.reshape(1, 3).expand_as(pc.get_xyz) - pc.get_xyz ## shape: [N, 3]
                pl_distance = torch.norm(pl_direction, dim=1).reshape(-1, 1)
            camera_direction = viewpoint_camera.camera_center - pc.get_xyz ## shape: [N, 3]
            mlp_input = torch.cat([
                colors_precomp, ## shape: [N, NUM_CHANNELS]
                positional_encoding(normalize(pl_direction), levels=pipe.encoding_levels_each),
                positional_encoding(normalize(camera_direction), levels=pipe.encoding_levels_each),
                positional_encoding(shadow_hint[:, 0:1], levels=pipe.encoding_levels_shadow),
                1.0 / pl_distance ** 2,
            ], dim=1)
            colors_precomp = color_mlp(mlp_input) ## [N, 3]
            
            ## pad the colors_precomp to pipe.in_channels to fit the CUDA part
            colors_precomp = torch.cat([colors_precomp, torch.zeros(colors_precomp.shape[0], pipe.in_channels - 3).cuda()], dim=1)
            res, radii, _, _ = rasterizer(
                means3D = means3D,
                means2D = means2D,
                shs = shs,
                colors_precomp = colors_precomp, ## shape: [N, 3]
                opacities = opacity,
                scales = scales,
                rotations = rotations,
                cov3D_precomp = cov3D_precomp) 
            res = res[:3] ## [3, full_height, full_width]

    else: ## the simple defer_shading operation
        pl_direction = viewpoint_camera.pl_pos.reshape(1, 3).expand(H*W, -1) - depth_pts.permute(1, 2, 0).reshape(-1, 3) ## shape: [H*W, 3]
        pl_distance = torch.norm(pl_direction, dim=1).reshape(-1, 1)
        camera_direction = viewpoint_camera.camera_center.reshape(1, 3).expand(H*W, -1) - depth_pts.permute(1, 2, 0).reshape(-1, 3) ## shape: [H*W, 3]
        mlp_input = rendered_image.permute(1, 2, 0).reshape(-1, pipe.in_channels)
        mlp_input = torch.cat([
            mlp_input,
            positional_encoding(normalize(pl_direction), levels=pipe.encoding_levels_each),
            positional_encoding(normalize(camera_direction), levels=pipe.encoding_levels_each),
            1.0 / pl_distance ** 2,
        ], dim=1)
        res = color_mlp(mlp_input).reshape(viewpoint_camera.full_height, viewpoint_camera.full_width, 3).permute(2, 0, 1)

    ## clean null feature pixels
    if pipe.defer_shading:
        # res = torch.where(alpha_mask, res, 0.0)
        res = res * rendered_alpha

    crop_offset_x, crop_offset_y = int(viewpoint_camera.crop_offset_x), int(viewpoint_camera.crop_offset_y)
    ## [NUM_CHANNELS (default: RGB=3), full_height, full_width]
    res = res[:, crop_offset_y: crop_offset_y + viewpoint_camera.image_height, crop_offset_x: crop_offset_x + viewpoint_camera.image_width]
    rendered_depth = rendered_depth[:, crop_offset_y: crop_offset_y + viewpoint_camera.image_height, crop_offset_x: crop_offset_x + viewpoint_camera.image_width]
    rendered_alpha = rendered_alpha[:, crop_offset_y: crop_offset_y + viewpoint_camera.image_height, crop_offset_x: crop_offset_x + viewpoint_camera.image_width]
    
    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": res,
            "depth": rendered_depth,
            "alpha": rendered_alpha,
            "depth_offset": depth_offset,
            "viewspace_points": screenspace_points,
            "visibility_filter": radii > 0,
            "radii": radii,
            "mlp_input": mlp_input if dump_samples else None
    }