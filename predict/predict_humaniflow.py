import os
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

from predict.predict_hrnet import predict_hrnet

from renderers.pytorch3d_textured_renderer import TexturedIUVRenderer

from utils.image_utils import batch_add_rgb_background, batch_crop_pytorch_affine, batch_crop_opencv_affine
from utils.cam_utils import orthographic_project_torch
from utils.joints2d_utils import undo_keypoint_normalisation
from utils.label_conversions import convert_2Djoints_to_gaussian_heatmaps_torch
from utils.rigid_transform_utils import aa_rotate_translate_points_pytorch3d
from utils.sampling_utils import compute_vertex_variance_from_samples, joints2D_error_sorted_verts_sampling


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
    plain_texture = torch.ones(1, 1200, 800, 3, device=device).float() * 0.7
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
            pred_vertices_rot180_mode = aa_rotate_translate_points_pytorch3d(points=pred_vertices_rot90_point_est,
                                                                             axes=torch.tensor([0., 1., 0.], device=device),
                                                                             angles=-np.pi / 2.,
                                                                             translations=torch.zeros(3, device=device))
            pred_vertices_rot270_mode = aa_rotate_translate_points_pytorch3d(points=pred_vertices_rot180_mode,
                                                                             axes=torch.tensor([0., 1., 0.], device=device),
                                                                             angles=-np.pi / 2.,
                                                                             translations=torch.zeros(3, device=device))

            # Need to flip predicted T-pose vertices before projecting so that they project the right way up.
            pred_tpose_vertices_point_est = aa_rotate_translate_points_pytorch3d(points=smpl_model(betas=pred['shape_mode']).vertices,
                                                                                 axes=torch.tensor([1., 0., 0.], device=device),
                                                                                 angles=np.pi,
                                                                                 translations=torch.zeros(3, device=device))  # (1, 6890, 3)
            # Rotating 90° about vertical axis for visualisation
            pred_reposed_vertices_rot90_mean = aa_rotate_translate_points_pytorch3d(points=pred_tpose_vertices_point_est,
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
                per_vertex_avg_dist_from_mean, per_vertex_directional_variance = compute_vertex_variance_from_samples(pred_vertices_samples)
                # Generate per-vertex uncertainty colourmap
                vertex_var_norm = plt.Normalize(vmin=0.0, vmax=0.2, clip=True)
                vertex_colours = plt.cm.jet(vertex_var_norm(per_vertex_avg_dist_from_mean.cpu().detach().numpy()))[:, :3]
                vertex_colours = torch.from_numpy(vertex_colours[None, :, :]).to(device).float()

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

            # Render point estimate visualisation outputs
            body_vis_output = body_vis_renderer(vertices=pred_vertices_point_est,
                                                cam_t=cam_t,
                                                orthographic_scale=orthographic_scale,
                                                lights_rgb_settings=lights_rgb_settings,
                                                verts_features=vertex_colours)
            cropped_for_proxy_rgb = torch.nn.functional.interpolate(cropped_for_proxy['rgb'],
                                                                    size=(visualise_wh, visualise_wh),
                                                                    mode='bilinear',
                                                                    align_corners=False)
            body_vis_rgb = batch_add_rgb_background(backgrounds=cropped_for_proxy_rgb,
                                                    rgb=body_vis_output['rgb_images'].permute(0, 3, 1, 2).contiguous(),
                                                    seg=body_vis_output['iuv_images'][:, :, :, 0].round())
            body_vis_rgb = body_vis_rgb.cpu().detach().numpy()[0].transpose(1, 2, 0)

            body_vis_rgb_rot90 = body_vis_renderer(vertices=pred_vertices_rot90_point_est,
                                                   cam_t=fixed_cam_t,
                                                   orthographic_scale=fixed_orthographic_scale,
                                                   lights_rgb_settings=lights_rgb_settings,
                                                   verts_features=vertex_colours)['rgb_images'].cpu().detach().numpy()[0]
            body_vis_rgb_rot180 = body_vis_renderer(vertices=pred_vertices_rot180_mode,
                                                    cam_t=fixed_cam_t,
                                                    orthographic_scale=fixed_orthographic_scale,
                                                    lights_rgb_settings=lights_rgb_settings,
                                                    verts_features=vertex_colours)['rgb_images'].cpu().detach().numpy()[0]
            body_vis_rgb_rot270 = body_vis_renderer(vertices=pred_vertices_rot270_mode,
                                                    textures=plain_texture,
                                                    cam_t=fixed_cam_t,
                                                    orthographic_scale=fixed_orthographic_scale,
                                                    lights_rgb_settings=lights_rgb_settings,
                                                    verts_features=vertex_colours)['rgb_images'].cpu().detach().numpy()[0]

            # T-pose body visualisation
            tpose_body_vis_rgb = body_vis_renderer(vertices=pred_tpose_vertices_point_est,
                                                   textures=plain_texture,
                                                   cam_t=fixed_cam_t,
                                                   orthographic_scale=fixed_orthographic_scale,
                                                   lights_rgb_settings=lights_rgb_settings)['rgb_images'].cpu().detach().numpy()[0]
            tpose_body_vis_rgb_rot90 = body_vis_renderer(vertices=pred_reposed_vertices_rot90_mean,
                                                         textures=plain_texture,
                                                         cam_t=fixed_cam_t,
                                                         orthographic_scale=fixed_orthographic_scale,
                                                         lights_rgb_settings=lights_rgb_settings)['rgb_images'].cpu().detach().numpy()[0]

            # Combine all visualisations
            combined_vis_rows = 2
            combined_vis_cols = 4
            combined_vis_fig = np.zeros((combined_vis_rows * visualise_wh, combined_vis_cols * visualise_wh, 3),
                                        dtype=body_vis_rgb.dtype)
            # Cropped input image
            combined_vis_fig[:visualise_wh, :visualise_wh] = cropped_for_proxy_rgb.cpu().detach().numpy()[0].transpose(1, 2, 0)

            # Proxy representation + 2D joints scatter + 2D joints confidences
            proxy_rep_input = proxy_rep_input[0].sum(dim=0).cpu().detach().numpy()
            proxy_rep_input = np.stack([proxy_rep_input]*3, axis=-1)  # single-channel to RGB
            proxy_rep_input = cv2.resize(proxy_rep_input, (visualise_wh, visualise_wh))
            for joint_num in range(cropped_for_proxy['joints2D'].shape[1]):
                hor_coord = cropped_for_proxy['joints2D'][0, joint_num, 0].item() * visualise_wh / humaniflow_cfg.DATA.PROXY_REP_SIZE
                ver_coord = cropped_for_proxy['joints2D'][0, joint_num, 1].item() * visualise_wh / humaniflow_cfg.DATA.PROXY_REP_SIZE
                cv2.circle(proxy_rep_input,
                           (int(hor_coord), int(ver_coord)),
                           radius=3,
                           color=(255, 0, 0),
                           thickness=-1)
                cv2.putText(proxy_rep_input,
                            str(joint_num),
                            (int(hor_coord + 4), int(ver_coord + 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), lineType=2)
                cv2.putText(proxy_rep_input,
                            str(joint_num) + " {:.2f}".format(hrnet_output['joints2Dconfs'][joint_num].item()),
                            (10, 16 * (joint_num + 1)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), lineType=2)
            combined_vis_fig[visualise_wh:2*visualise_wh, :visualise_wh] = proxy_rep_input

            # Posed 3D body
            combined_vis_fig[:visualise_wh, visualise_wh:2*visualise_wh] = body_vis_rgb
            combined_vis_fig[visualise_wh:2*visualise_wh, visualise_wh:2*visualise_wh] = body_vis_rgb_rot90
            combined_vis_fig[:visualise_wh, 2*visualise_wh:3*visualise_wh] = body_vis_rgb_rot180
            combined_vis_fig[visualise_wh:2*visualise_wh, 2*visualise_wh:3*visualise_wh] = body_vis_rgb_rot270

            # T-pose 3D body
            combined_vis_fig[:visualise_wh, 3*visualise_wh:4*visualise_wh] = tpose_body_vis_rgb
            combined_vis_fig[visualise_wh:2*visualise_wh, 3*visualise_wh:4*visualise_wh] = tpose_body_vis_rgb_rot90

            cv2.imwrite(os.path.join(save_dir, os.path.splitext(image_fname)[0] + '_point_est.png'),
                        combined_vis_fig[:, :, ::-1] * 255)

            if visualise_uncropped:
                # Uncropped point estimate visualisation by projecting 3D body onto original image
                rgb_to_uncrop = body_vis_output['rgb_images'].permute(0, 3, 1, 2).contiguous().cpu().detach().numpy()
                iuv_to_uncrop = body_vis_output['iuv_images'].permute(0, 3, 1, 2).contiguous().cpu().detach().numpy()
                bbox_centres = hrnet_output['bbox_centre'][None].cpu().detach().numpy()
                bbox_whs = torch.max(hrnet_output['bbox_height'], hrnet_output['bbox_width'])[None].cpu().detach().numpy()
                bbox_whs *= humaniflow_cfg.DATA.BBOX_SCALE_FACTOR
                uncropped_for_visualise = batch_crop_opencv_affine(output_wh=(visualise_wh, visualise_wh),
                                                                   num_to_crop=1,
                                                                   rgb=rgb_to_uncrop,
                                                                   iuv=iuv_to_uncrop,
                                                                   bbox_centres=bbox_centres,
                                                                   bbox_whs=bbox_whs,
                                                                   uncrop=True,
                                                                   uncrop_wh=(orig_image.shape[1], orig_image.shape[0]))
                uncropped_rgb = uncropped_for_visualise['rgb'][0].transpose(1, 2, 0) * 255
                uncropped_seg = uncropped_for_visualise['iuv'][0, 0, :, :]
                background_pixels = uncropped_seg[:, :, None] == 0  # Body pixels are > 0
                uncropped_rgb_with_background = uncropped_rgb * (np.logical_not(background_pixels)) + \
                                                orig_image * background_pixels

                cv2.imwrite(os.path.join(save_dir, os.path.splitext(image_fname)[0] + '_point_est_uncrop.png'),
                            uncropped_rgb_with_background[:, :, ::-1])

            if visualise_samples:
                samples_rows = 3
                samples_cols = 6
                samples_fig = np.zeros((samples_rows * visualise_wh, samples_cols * visualise_wh, 3),
                                       dtype=body_vis_rgb.dtype)
                for i in range(num_vis_samples + 1):
                    body_vis_output_sample = body_vis_renderer(vertices=pred_vertices_samples[[i]],
                                                               cam_t=cam_t,
                                                               orthographic_scale=orthographic_scale,
                                                               lights_rgb_settings=lights_rgb_settings,
                                                               verts_features=vertex_colours)
                    body_vis_rgb_sample = batch_add_rgb_background(backgrounds=cropped_for_proxy_rgb,
                                                                   rgb=body_vis_output_sample['rgb_images'].permute(0, 3, 1, 2).contiguous(),
                                                                   seg=body_vis_output_sample['iuv_images'][:, :, :, 0].round())
                    body_vis_rgb_sample = body_vis_rgb_sample.cpu().detach().numpy()[0].transpose(1, 2, 0)

                    body_vis_rgb_rot90_sample = body_vis_renderer(vertices=pred_vertices_rot90_samples[[i]],
                                                                  cam_t=fixed_cam_t,
                                                                  orthographic_scale=fixed_orthographic_scale,
                                                                  lights_rgb_settings=lights_rgb_settings,
                                                                  verts_features=vertex_colours)['rgb_images'].cpu().detach().numpy()[0]

                    row = (2 * i) // samples_cols
                    col = (2 * i) % samples_cols
                    samples_fig[row*visualise_wh:(row+1)*visualise_wh, col*visualise_wh:(col+1)*visualise_wh] = body_vis_rgb_sample

                    row = (2 * i + 1) // samples_cols
                    col = (2 * i + 1) % samples_cols
                    samples_fig[row * visualise_wh:(row + 1) * visualise_wh, col * visualise_wh:(col + 1) * visualise_wh] = body_vis_rgb_rot90_sample

                    cv2.imwrite(os.path.join(save_dir, os.path.splitext(image_fname)[0] + 'samples.png'),
                                samples_fig[:, :, ::-1] * 255)

            if visualise_xyz_variance:
                pred_vertices2D_mode = orthographic_project_torch(pred_vertices_point_est, pred['cam_wp'])
                pred_vertices2D_mode = undo_keypoint_normalisation(pred_vertices2D_mode, visualise_wh).cpu().detach().numpy()[0]
                per_vertex_directional_variance = per_vertex_directional_variance.cpu().detach().numpy()

                norm = plt.Normalize(vmin=0.0, vmax=0.15, clip=True)
                scatter_s = 0.25
                img_alpha = 1.0
                plt.style.use('dark_background')
                titles = ['X-axis (Horizontal) Variance', 'Y-axis (Vertical) Variance', 'Z-axis (Depth) Variance']
                plt.figure(figsize=(14, 10))
                rows = 1
                cols = 3
                subplot_count = 1
                for i in range(3):
                    plt.subplot(rows, cols, subplot_count)
                    plt.gca().axis('off')
                    plt.title(titles[i])
                    plt.imshow(cropped_for_proxy_rgb.cpu().detach().numpy()[0].transpose(1, 2, 0), alpha=img_alpha)
                    plt.scatter(pred_vertices2D_mode[:, 0], pred_vertices2D_mode[:, 1],
                                s=scatter_s,
                                c=per_vertex_directional_variance[:, i],
                                cmap='jet',
                                norm=norm)
                    subplot_count += 1
                plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
                plt.margins(0, 0)
                plt.gca().xaxis.set_major_locator(plt.NullLocator())
                plt.gca().yaxis.set_major_locator(plt.NullLocator())
                plt.savefig(os.path.join(save_dir, os.path.splitext(image_fname)[0] + 'xyz_variance.png'), bbox_inches='tight')
                plt.close()







