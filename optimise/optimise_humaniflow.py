import os
import cv2
import numpy as np
import torch
from tqdm import tqdm

from smplx.lbs import batch_rodrigues

from data.load_optimise_data import load_opt_initialise_data_from_pred_output
from utils.renderers.pytorch3d_textured_renderer import TexturedIUVRenderer
from utils.cam_utils import orthographic_project_torch
from utils.label_conversions import ALL_JOINTS_TO_COCO_MAP
from utils.joints2d_utils import undo_keypoint_normalisation
from utils.rigid_transform_utils import aa_rotate_translate_points_pytorch3d, so3_log
from utils.visualise_utils import render_point_est_visualisation, uncrop_point_est_visualisation


def optimise_batch_with_humaniflow_prior(humaniflow_model,
                                         humaniflow_cfg,
                                         optimise_cfg,
                                         smpl_model,
                                         pred_image_dir,
                                         pred_output_dir,
                                         opt_output_dir,
                                         device,
                                         visualise_wh=512):

    # ------------------- LOAD PREDICTION OUTPUTS AND INITIALISE OPTIMISATION VARIABLES -------------------
    data = load_opt_initialise_data_from_pred_output(pred_image_dir,
                                                     pred_output_dir)
    batch_size = data['cropped_image'].shape[0]

    # Optimisation variables
    shape = torch.from_numpy(data['shape_mode']).clone().to(device)
    pose_axis_angle = torch.from_numpy(data['pose_axisangle_point_est']).clone().to(device).view(batch_size, -1)
    glob_axis_angle = so3_log(torch.from_numpy(data['glob_rotmat']).clone().to(device).double(),
                              return_axis_angle=True).float()
    cam_wp = torch.from_numpy(data['cam_wp']).clone().to(device)

    shape.requires_grad = True
    pose_axis_angle.requires_grad = True
    glob_axis_angle.requires_grad = True
    cam_wp.requires_grad = True

    opt_variables = [pose_axis_angle, glob_axis_angle, shape, cam_wp]
    optimiser = torch.optim.SGD(opt_variables, lr=optimise_cfg.LR)
    print(f'\nOptimisation variables: '
          f'\npose_axis_angle {pose_axis_angle.shape}'
          f'\nglob_axis_angle {glob_axis_angle.shape}'
          f'\nshape {shape.shape}'
          f'\ncam_wp {cam_wp.shape}')

    input_feats = torch.from_numpy(data['input_feats']).clone().to(device)  # ResNet encoder features

    # Optimisation 2D targets
    target_joints2D = torch.from_numpy(data['cropped_joints2D']).clone().to(device)
    target_joints2D_conf = torch.from_numpy(data['hrnet_joints2D_conf']).clone().to(device)
    target_joints2D_visib = target_joints2D_conf > optimise_cfg.JOINTS2D_VISIB_THRESHOLD
    target_joints2D_visib[:, [0, 1, 2, 3, 4, 5, 6]] = True  # Only removing joints [7, 8, 9, 10, 11, 12, 13, 14, 15, 16] if occluded

    # Useful tensors that are re-used and can be pre-defined
    x_axis = torch.tensor([1., 0., 0.], device=device)
    y_axis = torch.tensor([0., 1., 0.], device=device)
    zero_trans = torch.zeros(3, device=device)
    fixed_cam_t = torch.tensor([[0., -0.2, 2.5]], device=device).expand(batch_size, -1)
    fixed_orthographic_scale = torch.tensor([[0.95, 0.95]], device=device).expand(batch_size, -1)

    # ------------------------------- OPTIMISATION -------------------------------
    humaniflow_model.eval()
    humaniflow_model.pose_so3flow_transform_modules.eval()

    for iter_num in tqdm(range(optimise_cfg.NUM_ITERS)):
        opt_smpl = smpl_model(body_pose=pose_axis_angle,
                              global_orient=glob_axis_angle,
                              betas=shape,
                              pose2rot=True)

        # Pose targets were flipped such that they are right way up in 3D space - i.e. wrong way up when projected
        # Need to flip predicted joints before projecting to get opt_joints2D
        opt_joints3D = aa_rotate_translate_points_pytorch3d(points=opt_smpl.joints[:, ALL_JOINTS_TO_COCO_MAP, :],
                                                            axes=x_axis,
                                                            angles=np.pi,
                                                            translations=zero_trans)
        opt_joints2D = undo_keypoint_normalisation(orthographic_project_torch(opt_joints3D,
                                                                              cam_wp),
                                                   humaniflow_cfg.DATA.PROXY_REP_SIZE)  # (batch_size, 17, 2)

        # Joints2D Loss
        visib_target_joints2D = target_joints2D[target_joints2D_visib, :]
        visib_opt_joints2D = opt_joints2D[target_joints2D_visib, :]
        joints2D_loss = ((visib_target_joints2D - visib_opt_joints2D) ** 2).mean()

        # Pose and Shape Prior Loss
        pose_rotmats = batch_rodrigues(pose_axis_angle.contiguous().view(-1, 3)).view(-1, 23, 3, 3)
        glob_rotmats = batch_rodrigues(glob_axis_angle.contiguous())
        dists_for_loglik = humaniflow_model(input=None,
                                            input_feats=input_feats,
                                            compute_point_est=False,
                                            num_samples=0,
                                            compute_for_loglik=True,
                                            shape_for_loglik=shape,
                                            pose_R_for_loglik=pose_rotmats,
                                            glob_R_for_loglik=glob_rotmats)

        pose_logprob = torch.tensor(0., device=device)
        for part in range(23):
            try:
                part_log_prob = dists_for_loglik['conditioned_pose_SO3flow_dists_for_loglik'][part].log_prob(pose_rotmats[:, part, :, :].double())  # (batch_size,)
                pose_logprob = pose_logprob + part_log_prob.sum()
            except AssertionError as error:
                print(error)
        pose_logprob = pose_logprob / batch_size

        shape_logprob = dists_for_loglik['shape_dist_for_loglik'].log_prob(shape).sum() / batch_size  # log prob is (batch_size, num betas)

        # Total Loss
        loss = joints2D_loss * optimise_cfg.LOSS_WEIGHTS.JOINTS2D - pose_logprob * optimise_cfg.LOSS_WEIGHTS.POSE_PRIOR - shape_logprob * optimise_cfg.LOSS_WEIGHTS.SHAPE_PRIOR

        # Caching optimisation variables in case optimisation step causes NaNs
        last_pose_axis_angle = pose_axis_angle.clone()
        last_shape = shape.clone()
        last_cam_wp = cam_wp.clone()

        # Optimisation Step
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        # Checking for NaNs in updated variables
        with torch.no_grad():
            nans_in_update = torch.any(torch.isnan(pose_axis_angle))
            if nans_in_update:
                print('\nFound NaNs in opt variables - TERMINATING opt loop.')
                pose_axis_angle = last_pose_axis_angle
                shape = last_shape
                cam_wp = last_cam_wp
                break

    # Save optimised variables
    for i, fname in enumerate(data['fnames']):
        np.savez(os.path.join(opt_output_dir, os.path.splitext(fname)[0] + '_opt.npz'),
                 pose_axisangle=pose_axis_angle[i].cpu().detach().numpy(),
                 shape=shape[i].cpu().detach().numpy(),
                 cam_wp=cam_wp[i].cpu().detach().numpy())

    # ------------------------------- VISUALISATION -------------------------------
    with torch.no_grad():
        body_vis_renderer = TexturedIUVRenderer(device=device,
                                                batch_size=batch_size,
                                                img_wh=visualise_wh,
                                                projection_type='orthographic',
                                                render_rgb=True,
                                                bin_size=None)
        lights_rgb_settings = {'location': torch.tensor([[0., -0.8, -2.0]], device=device, dtype=torch.float32),
                               'ambient_color': 0.5 * torch.ones(1, 3, device=device, dtype=torch.float32),
                               'diffuse_color': 0.3 * torch.ones(1, 3, device=device, dtype=torch.float32),
                               'specular_color': torch.zeros(1, 3, device=device, dtype=torch.float32)}

        final_vertices_all_rot = {'0': aa_rotate_translate_points_pytorch3d(points=smpl_model(body_pose=pose_axis_angle,
                                                                                              global_orient=glob_axis_angle,
                                                                                              betas=shape,
                                                                                              pose2rot=True).vertices,
                                                                            axes=x_axis,
                                                                            angles=np.pi,
                                                                            translations=zero_trans)}
        for rot in (90, 180, 270):
            final_vertices_all_rot[str(rot)] = aa_rotate_translate_points_pytorch3d(points=final_vertices_all_rot['0'],
                                                                                    axes=y_axis,
                                                                                    angles=-np.deg2rad(rot),
                                                                                    translations=zero_trans)

        cropped_rgb_for_vis = torch.nn.functional.interpolate(torch.from_numpy(data['cropped_image']).to(device),
                                                              size=(visualise_wh, visualise_wh),
                                                              mode='bilinear',
                                                              align_corners=False)
        cropped_proxy_for_vis = torch.nn.functional.interpolate(torch.from_numpy(data['proxy_rep']).to(device).sum(dim=1, keepdim=True),
                                                                size=(visualise_wh, visualise_wh),
                                                                mode='bilinear',
                                                                align_corners=False).expand(-1, 3, -1, -1)  # single-channel to RGB

        point_est_figs, point_est_mesh_renders = render_point_est_visualisation(renderer=body_vis_renderer,
                                                                                joints2D=target_joints2D,
                                                                                joints2D_confs=target_joints2D_conf,
                                                                                cropped_proxy_for_vis=cropped_proxy_for_vis,
                                                                                cropped_rgb_for_vis=cropped_rgb_for_vis,
                                                                                pred_vertices_point_est_all_rot=final_vertices_all_rot,
                                                                                pred_tpose_vertices_point_est_all_rot=None,
                                                                                vertex_colours=None,
                                                                                cam_t=torch.cat([cam_wp[:, 1:],
                                                                                                 torch.ones(batch_size, 1, device=device).float() * 2.5],
                                                                                                dim=-1),
                                                                                fixed_cam_t=fixed_cam_t,
                                                                                orthographic_scale=cam_wp[:, [0, 0]],
                                                                                fixed_orthographic_scale=fixed_orthographic_scale,
                                                                                lights_rgb_settings=lights_rgb_settings,
                                                                                visualise_wh=visualise_wh,
                                                                                proxy_orig_wh=humaniflow_cfg.DATA.PROXY_REP_SIZE)

        cropped_mesh_renders_rgb = point_est_mesh_renders['rgb_images'].permute(0, 3, 1, 2).contiguous().cpu().detach().numpy()
        cropped_mesh_renders_iuv = point_est_mesh_renders['iuv_images'].permute(0, 3, 1, 2).contiguous().cpu().detach().numpy()
        bbox_whs = np.stack([data['bbox_height'], data['bbox_width']], axis=-1).max(axis=-1)
        for i, fname in enumerate(data['fnames']):
            # uncrop_point_est_visualisation is a batched function, but we need to run this in a for loop since
            # orig_image is a list of images of potentially different sizes - i.e. different amounts of uncropping is applied.
            uncropped_point_est_fig = uncrop_point_est_visualisation(cropped_mesh_render_rgb=cropped_mesh_renders_rgb[[i]],
                                                                     cropped_mesh_render_iuv=cropped_mesh_renders_iuv[[i]],
                                                                     bbox_centres=data['bbox_centre'][[i]],
                                                                     bbox_whs=bbox_whs[[i]],
                                                                     orig_image=data['orig_image'][i][None],
                                                                     visualise_wh=visualise_wh,
                                                                     bbox_scale_factor=humaniflow_cfg.DATA.BBOX_SCALE_FACTOR)

            cv2.imwrite(os.path.join(opt_output_dir, os.path.splitext(fname)[0] + '_opt.png'),
                        point_est_figs[i][:, :, ::-1] * 255)

            cv2.imwrite(os.path.join(opt_output_dir, os.path.splitext(fname)[0] + '_opt_uncrop.png'),
                        uncropped_point_est_fig[0, :, :, ::-1])
