import os
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

from predict.predict_hrnet import predict_hrnet

from renderers.pytorch3d_textured_renderer import TexturedIUVRenderer

from utils.image_utils import batch_crop_pytorch_affine
from utils.label_conversions import convert_2Djoints_to_gaussian_heatmaps_torch
from utils.rigid_transform_utils import aa_rotate_translate_points_pytorch3d
from utils.sampling_utils import compute_vertex_variance_from_samples, joints2D_error_sorted_verts_sampling
from utils.predict_utils import save_pred_output
from utils.visualise_utils import (render_point_est_visualisation,
                                   render_samples_visualisation,
                                   uncrop_point_est_visualisation,
                                   plot_xyz_vertex_variance)


def predict_humaniflow(humaniflow_model,
                       humaniflow_cfg,
                       smpl_model,
                       hrnet_model,
                       hrnet_cfg,
                       edge_detect_model,
                       device,
                       image_dir,
                       save_dir,
                       object_detect_model=None,
                       joints2Dvisib_threshold=0.75,
                       num_pred_samples=50,
                       num_vis_samples=8,
                       visualise_wh=512,
                       visualise_uncropped=True,
                       visualise_samples=False,
                       visualise_xyz_variance=True):

    # Setting up body visualisation renderer
    body_vis_renderer = TexturedIUVRenderer(device=device,
                                            batch_size=1,
                                            img_wh=visualise_wh,
                                            projection_type='orthographic',
                                            render_rgb=True,
                                            bin_size=32)
    lights_rgb_settings = {'location': torch.tensor([[0., -0.8, -2.0]], device=device, dtype=torch.float32),
                           'ambient_color': 0.5 * torch.ones(1, 3, device=device, dtype=torch.float32),
                           'diffuse_color': 0.3 * torch.ones(1, 3, device=device, dtype=torch.float32),
                           'specular_color': torch.zeros(1, 3, device=device, dtype=torch.float32)}
    fixed_cam_t = torch.tensor([[0., -0.2, 2.5]], device=device)
    fixed_orthographic_scale = torch.tensor([[0.95, 0.95]], device=device)

    hrnet_model.eval()
    humaniflow_model.eval()
    humaniflow_model.pose_so3flow_transform_modules.eval()
    if object_detect_model is not None:
        object_detect_model.eval()
    for image_fname in tqdm(sorted([f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))])):
        with torch.no_grad():
            # ------------------------- INPUT LOADING AND PROXY REPRESENTATION GENERATION -------------------------
            image = cv2.cvtColor(cv2.imread(os.path.join(image_dir, image_fname)), cv2.COLOR_BGR2RGB)
            orig_image = image.copy()
            image = torch.from_numpy(image.transpose(2, 0, 1)).float().to(device) / 255.0
            # Predict Person Bounding Box + 2D Joints
            hrnet_output = predict_hrnet(hrnet_model=hrnet_model,
                                         hrnet_config=hrnet_cfg,
                                         object_detect_model=object_detect_model,
                                         image=image,
                                         object_detect_threshold=humaniflow_cfg.DATA.BBOX_THRESHOLD,
                                         bbox_scale_factor=humaniflow_cfg.DATA.BBOX_SCALE_FACTOR)

            # Transform predicted 2D joints and image from HRNet input size to input proxy representation size
            hrnet_input_centre = torch.tensor([[hrnet_output['cropped_image'].shape[1],
                                                hrnet_output['cropped_image'].shape[2]]],
                                              dtype=torch.float32,
                                              device=device) * 0.5
            hrnet_input_height = torch.tensor([hrnet_output['cropped_image'].shape[1]],
                                              dtype=torch.float32,
                                              device=device)
            cropped_for_proxy = batch_crop_pytorch_affine(input_wh=(hrnet_cfg.MODEL.IMAGE_SIZE[0], hrnet_cfg.MODEL.IMAGE_SIZE[1]),
                                                          output_wh=(humaniflow_cfg.DATA.PROXY_REP_SIZE, humaniflow_cfg.DATA.PROXY_REP_SIZE),
                                                          num_to_crop=1,
                                                          device=device,
                                                          joints2D=hrnet_output['joints2D'][None, :, :],
                                                          rgb=hrnet_output['cropped_image'][None, :, :, :],
                                                          bbox_centres=hrnet_input_centre,
                                                          bbox_heights=hrnet_input_height,
                                                          bbox_widths=hrnet_input_height,
                                                          orig_scale_factor=1.0)

            # Create proxy representation with 1) Edge detection and 2) 2D joints heatmaps generation
            edge_detector_output = edge_detect_model(cropped_for_proxy['rgb'])
            proxy_rep_img = edge_detector_output['thresholded_thin_edges'] if humaniflow_cfg.DATA.EDGE_NMS else edge_detector_output['thresholded_grad_magnitude']
            proxy_rep_heatmaps = convert_2Djoints_to_gaussian_heatmaps_torch(joints2D=cropped_for_proxy['joints2D'],
                                                                             img_wh=humaniflow_cfg.DATA.PROXY_REP_SIZE,
                                                                             std=humaniflow_cfg.DATA.HEATMAP_GAUSSIAN_STD)
            hrnet_joints2Dvisib = hrnet_output['joints2Dconfs'] > joints2Dvisib_threshold
            hrnet_joints2Dvisib[[0, 1, 2, 3, 4, 5, 6]] = True  # Only removing joints [7, 8, 9, 10, 11, 12, 13, 14, 15, 16] if occluded
            proxy_rep_heatmaps = proxy_rep_heatmaps * hrnet_joints2Dvisib[None, :, None, None]
            proxy_rep_input = torch.cat([proxy_rep_img, proxy_rep_heatmaps], dim=1).float()  # (1, 18, img_wh, img_wh)

            # ------------------------------- POSE AND SHAPE DISTRIBUTION PREDICTION -------------------------------
            pred = humaniflow_model(proxy_rep_input,
                                    num_samples=num_pred_samples,
                                    use_shape_mode_for_samples=True)
            # pred is a dict containing:
            # - Pose/shape samples from predicted distribution
            # - Pose/shape point estimate from predicted distribution

            # Save predicted outputs (useful for post-processing optimisation)
            save_pred_output(pred,
                             os.path.join(save_dir, os.path.splitext(image_fname)[0] + '_pred.npz'))

            # Compute SMPL point estimates and process for visualisation
            pred_smpl_output_point_est = smpl_model(body_pose=pred['pose_rotmats_point_est'],
                                                    global_orient=pred['glob_rotmat'][:, None, :, :],
                                                    betas=pred['shape_mode'],
                                                    pose2rot=False)
            pred_vertices_point_est = pred_smpl_output_point_est.vertices  # (1, 6890, 3)
            # Need to flip pred_vertices before projecting so that they project the right way up.
            pred_vertices_point_est = aa_rotate_translate_points_pytorch3d(points=pred_vertices_point_est,
                                                                           axes=torch.tensor([1., 0., 0.], device=device),
                                                                           angles=np.pi,
                                                                           translations=torch.zeros(3, device=device))
            # Rotating 90° about vertical axis for visualisation
            pred_vertices_rot90_point_est = aa_rotate_translate_points_pytorch3d(points=pred_vertices_point_est,
                                                                                 axes=torch.tensor([0., 1., 0.], device=device),
                                                                                 angles=-np.pi / 2.,
                                                                                 translations=torch.zeros(3, device=device))
            pred_vertices_rot180_point_est = aa_rotate_translate_points_pytorch3d(points=pred_vertices_rot90_point_est,
                                                                                  axes=torch.tensor([0., 1., 0.], device=device),
                                                                                  angles=-np.pi / 2.,
                                                                                  translations=torch.zeros(3, device=device))
            pred_vertices_rot270_point_est = aa_rotate_translate_points_pytorch3d(points=pred_vertices_rot180_point_est,
                                                                                  axes=torch.tensor([0., 1., 0.], device=device),
                                                                                  angles=-np.pi / 2.,
                                                                                  translations=torch.zeros(3, device=device))

            # Need to flip predicted T-pose vertices before projecting so that they project the right way up.
            pred_tpose_vertices_point_est = aa_rotate_translate_points_pytorch3d(points=smpl_model(betas=pred['shape_mode']).vertices,
                                                                                 axes=torch.tensor([1., 0., 0.], device=device),
                                                                                 angles=np.pi,
                                                                                 translations=torch.zeros(3, device=device))  # (1, 6890, 3)
            # Rotating 90° about vertical axis for visualisation
            pred_tpose_vertices_rot90_point_est = aa_rotate_translate_points_pytorch3d(points=pred_tpose_vertices_point_est,
                                                                                       axes=torch.tensor([0., 1., 0.], device=device),
                                                                                       angles=-np.pi / 2.,
                                                                                       translations=torch.zeros(3, device=device))

            # Compute SMPL samples and process for visualisation
            if visualise_samples:
                pred_smpl_output_samples = smpl_model(body_pose=pred['pose_rotmats_samples'][0, :, :, :, :],  # (num_pred_samples, 23, 3, 3)
                                                      global_orient=pred['glob_rotmat'][:, None, :, :].expand(num_pred_samples, -1, -1, -1),  # (num_pred_samples, 1, 3, 3)
                                                      betas=pred['shape_samples'][0, :, :],  # (num_pred_samples, num_betas)
                                                      pose2rot=False)
                pred_vertices_samples = pred_smpl_output_samples.vertices  # (num_pred_samples, 6890, 3)
                pred_joints_samples = pred_smpl_output_samples.joints  # (num_pred_samples, 90, 3)

                # Estimate per-vertex uncertainty - i.e. directional variance and average Euclidean distance from mean - by sampling
                # SMPL poses/shapes and computing corresponding vertex meshes
                per_vertex_avg_dist_from_mean, per_vertex_xyz_variance = compute_vertex_variance_from_samples(pred_vertices_samples)

                pred_vertices_samples = joints2D_error_sorted_verts_sampling(pred_vertices_samples=pred_vertices_samples,
                                                                             pred_joints_samples=pred_joints_samples,
                                                                             input_joints2D_heatmaps=proxy_rep_input[:, 1:, :, :],
                                                                             pred_cam_wp=pred['cam_wp'])[:num_vis_samples, :, :]  # (num_vis_samples, 6890, 3)

                # Need to flip predicted vertices samples before projecting so that they project the right way up.
                pred_vertices_samples = aa_rotate_translate_points_pytorch3d(points=pred_vertices_samples,
                                                                             axes=torch.tensor([1., 0., 0.], device=device),
                                                                             angles=np.pi,
                                                                             translations=torch.zeros(3, device=device))

                pred_vertices_rot90_samples = aa_rotate_translate_points_pytorch3d(points=pred_vertices_samples,
                                                                                   axes=torch.tensor([0., 1., 0.], device=device),
                                                                                   angles=-np.pi / 2.,
                                                                                   translations=torch.zeros(3, device=device))
                pred_vertices_samples = torch.cat([pred_vertices_point_est, pred_vertices_samples], dim=0)  # (num_vis_samples + 1, 6890, 3)
                pred_vertices_rot90_samples = torch.cat([pred_vertices_rot90_point_est, pred_vertices_rot90_samples], dim=0)  # (num_vis_samples + 1, 6890, 3)

            # --------------------------------- RENDERING AND VISUALISATION ---------------------------------
            # Predicted camera corresponding to proxy rep input
            orthographic_scale = pred['cam_wp'][:, [0, 0]]
            cam_t = torch.cat([pred['cam_wp'][:, 1:],
                               torch.ones(pred['cam_wp'].shape[0], 1, device=device).float() * 2.5],
                              dim=-1)

            cropped_for_proxy_rgb = torch.nn.functional.interpolate(cropped_for_proxy_rgb,
                                                                    size=(visualise_wh, visualise_wh),
                                                                    mode='bilinear',
                                                                    align_corners=False)

            # Generate per-vertex uncertainty colourmap
            vertex_uncertainty_norm = plt.Normalize(vmin=0.0, vmax=0.2, clip=True)
            vertex_uncertainty_colours = plt.cm.jet(vertex_uncertainty_norm(per_vertex_avg_dist_from_mean.cpu().detach().numpy()))[:, :3]
            vertex_uncertainty_colours = torch.from_numpy(vertex_uncertainty_colours[None, :, :]).to(device).float()

            # Render and save point estimate visualisation
            point_est_fig, point_est_mesh_render = render_point_est_visualisation(renderer=body_vis_renderer,
                                                                                  joints2D=cropped_for_proxy['joints2D'],
                                                                                  joints2D_confs=hrnet_output['joints2Dconfs'],
                                                                                  proxy_rep_input=proxy_rep_input,
                                                                                  cropped_for_proxy_rgb=cropped_for_proxy_rgb,
                                                                                  pred_vertices_point_est_all_rot={'0': pred_vertices_point_est,
                                                                                                                   '90': pred_vertices_rot90_point_est,
                                                                                                                   '180': pred_vertices_rot180_point_est,
                                                                                                                   '270': pred_vertices_rot270_point_est},
                                                                                  pred_tpose_vertices_point_est_all_rot={'0': pred_tpose_vertices_point_est,
                                                                                                                         '90': pred_tpose_vertices_rot90_point_est},
                                                                                  vertex_colours=vertex_uncertainty_colours,
                                                                                  cam_t=cam_t,
                                                                                  fixed_cam_t=fixed_cam_t,
                                                                                  orthographic_scale=orthographic_scale,
                                                                                  fixed_orthographic_scale=fixed_orthographic_scale,
                                                                                  lights_rgb_settings=lights_rgb_settings,
                                                                                  visualise_wh=visualise_wh)
            cv2.imwrite(os.path.join(save_dir, os.path.splitext(image_fname)[0] + '_point_est.png'),
                        point_est_fig[:, :, ::-1] * 255)

            if visualise_uncropped:
                # Render and save uncropped point estimate visualisation by projecting 3D body onto original image
                uncropped_point_est_fig = uncrop_point_est_visualisation(
                    cropped_mesh_render_rgb=point_est_mesh_render['rgb_images'].permute(0, 3, 1, 2).contiguous().cpu().detach().numpy(),
                    cropped_mesh_render_iuv=point_est_mesh_render['iuv_images'].permute(0, 3, 1, 2).contiguous().cpu().detach().numpy(),
                    bbox_centres=hrnet_output['bbox_centre'][None].cpu().detach().numpy(),
                    bbox_whs=torch.max(hrnet_output['bbox_height'], hrnet_output['bbox_width'])[None].cpu().detach().numpy(),
                    orig_image=orig_image,
                    visualise_wh=visualise_wh,
                    bbox_scale_factor=humaniflow_cfg.DATA.BBOX_SCALE_FACTOR)
                cv2.imwrite(os.path.join(save_dir, os.path.splitext(image_fname)[0] + '_point_est_uncrop.png'),
                            uncropped_point_est_fig[:, :, ::-1])

            if visualise_samples:
                # Render and save samples visualisations
                samples_fig = render_samples_visualisation(renderer=body_vis_renderer,
                                                           num_vis_samples=num_vis_samples,
                                                           samples_rows=3,
                                                           samples_cols=6,
                                                           visualise_wh=visualise_wh,
                                                           cropped_for_proxy_rgb=cropped_for_proxy_rgb,
                                                           pred_vertices_samples_all_rot={'0': pred_vertices_samples,
                                                                                          '90': pred_vertices_rot90_samples},
                                                           vertex_colours=vertex_uncertainty_colours,
                                                           cam_t=cam_t,
                                                           fixed_cam_t=fixed_cam_t,
                                                           orthographic_scale=orthographic_scale,
                                                           fixed_orthographic_scale=fixed_orthographic_scale,
                                                           lights_rgb_settings=lights_rgb_settings)
                cv2.imwrite(os.path.join(save_dir, os.path.splitext(image_fname)[0] + 'samples.png'),
                            samples_fig[:, :, ::-1] * 255)

            if visualise_xyz_variance:
                # Plot per-vertex directional (i.e. x/y/z-axis) variance (represents uncertainty)
                plot_xyz_vertex_variance(pred_vertices_point_est=pred_vertices_point_est,
                                         per_vertex_xyz_variance=per_vertex_xyz_variance.cpu().detach().numpy(),
                                         cropped_for_proxy_rgb=cropped_for_proxy_rgb.cpu().detach().numpy()[0].transpose(1, 2, 0),
                                         visualise_wh=visualise_wh,
                                         cam_wp=pred['cam_wp'],
                                         save_path=os.path.join(save_dir, os.path.splitext(image_fname)[0] + 'xyz_variance.png'))
