import cv2
import numpy as np
import torch
import matplotlib as plt


from utils.image_utils import batch_add_rgb_background, batch_crop_opencv_affine
from utils.cam_utils import orthographic_project_torch
from utils.joints2d_utils import undo_keypoint_normalisation


def render_point_est_visualisation(renderer,
                                   joints2D,
                                   joints2D_confs,
                                   cropped_for_proxy_rgb,
                                   proxy_rep_input,
                                   pred_vertices_point_est_all_rot,
                                   pred_tpose_vertices_point_est_all_rot,
                                   vertex_colours,
                                   cam_t,
                                   fixed_cam_t,
                                   orthographic_scale,
                                   fixed_orthographic_scale,
                                   lights_rgb_settings,
                                   visualise_wh):

    body_vis_output = renderer(vertices=pred_vertices_point_est_all_rot['0'],
                               cam_t=cam_t,
                               orthographic_scale=orthographic_scale,
                               lights_rgb_settings=lights_rgb_settings,
                               verts_features=vertex_colours)

    body_vis_rgb = batch_add_rgb_background(backgrounds=cropped_for_proxy_rgb,
                                            rgb=body_vis_output['rgb_images'].permute(0, 3, 1, 2).contiguous(),
                                            seg=body_vis_output['iuv_images'][:, :, :, 0].round())
    body_vis_rgb = body_vis_rgb.cpu().detach().numpy()[0].transpose(1, 2, 0)

    body_vis_rgb_rot90 = renderer(vertices=pred_vertices_point_est_all_rot['90'],
                                  cam_t=fixed_cam_t,
                                  orthographic_scale=fixed_orthographic_scale,
                                  lights_rgb_settings=lights_rgb_settings,
                                  verts_features=vertex_colours)['rgb_images'].cpu().detach().numpy()[0]
    body_vis_rgb_rot180 = renderer(vertices=pred_vertices_point_est_all_rot['190'],
                                   cam_t=fixed_cam_t,
                                   orthographic_scale=fixed_orthographic_scale,
                                   lights_rgb_settings=lights_rgb_settings,
                                   verts_features=vertex_colours)['rgb_images'].cpu().detach().numpy()[0]
    body_vis_rgb_rot270 = renderer(vertices=pred_vertices_point_est_all_rot['270'],
                                   cam_t=fixed_cam_t,
                                   orthographic_scale=fixed_orthographic_scale,
                                   lights_rgb_settings=lights_rgb_settings,
                                   verts_features=vertex_colours)['rgb_images'].cpu().detach().numpy()[0]

    # T-pose body visualisation
    plain_texture = torch.ones(1, 1200, 800, 3, device=pred_tpose_vertices_point_est_all_rot['0'].device).float() * 0.7
    tpose_body_vis_rgb = renderer(vertices=pred_tpose_vertices_point_est_all_rot['0'],
                                  textures=plain_texture,
                                  cam_t=fixed_cam_t,
                                  orthographic_scale=fixed_orthographic_scale,
                                  lights_rgb_settings=lights_rgb_settings)['rgb_images'].cpu().detach().numpy()[0]
    tpose_body_vis_rgb_rot90 = renderer(vertices=pred_tpose_vertices_point_est_all_rot['90'],
                                        textures=plain_texture,
                                        cam_t=fixed_cam_t,
                                        orthographic_scale=fixed_orthographic_scale,
                                        lights_rgb_settings=lights_rgb_settings)['rgb_images'].cpu().detach().numpy()[0]

    # Combine all visualisations
    combined_vis_rows = 2
    combined_vis_cols = 4
    point_est_fig = np.zeros((combined_vis_rows * visualise_wh, combined_vis_cols * visualise_wh, 3),
                             dtype=body_vis_rgb.dtype)
    # Cropped input image
    point_est_fig[:visualise_wh, :visualise_wh] = cropped_for_proxy_rgb.cpu().detach().numpy()[0].transpose(1, 2, 0)

    # Proxy representation + 2D joints scatter + 2D joints confidences
    proxy_rep_input = proxy_rep_input[0].sum(dim=0).cpu().detach().numpy()
    proxy_rep_input = np.stack([proxy_rep_input] * 3, axis=-1)  # single-channel to RGB
    proxy_rep_input = cv2.resize(proxy_rep_input, (visualise_wh, visualise_wh))
    for joint_num in range(joints2D.shape[1]):
        hor_coord = joints2D[0, joint_num, 0].item() * visualise_wh / proxy_rep_input.shape[-1]
        ver_coord = joints2D[0, joint_num, 1].item() * visualise_wh / proxy_rep_input.shape[-2]
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
                    str(joint_num) + " {:.2f}".format(joints2D_confs[joint_num].item()),
                    (10, 16 * (joint_num + 1)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), lineType=2)
    point_est_fig[visualise_wh:2 * visualise_wh, :visualise_wh] = proxy_rep_input

    # Posed 3D body
    point_est_fig[:visualise_wh, visualise_wh:2 * visualise_wh] = body_vis_rgb
    point_est_fig[visualise_wh:2 * visualise_wh, visualise_wh:2 * visualise_wh] = body_vis_rgb_rot90
    point_est_fig[:visualise_wh, 2 * visualise_wh:3 * visualise_wh] = body_vis_rgb_rot180
    point_est_fig[visualise_wh:2 * visualise_wh, 2 * visualise_wh:3 * visualise_wh] = body_vis_rgb_rot270

    # T-pose 3D body
    point_est_fig[:visualise_wh, 3 * visualise_wh:4 * visualise_wh] = tpose_body_vis_rgb
    point_est_fig[visualise_wh:2 * visualise_wh, 3 * visualise_wh:4 * visualise_wh] = tpose_body_vis_rgb_rot90

    return point_est_fig, body_vis_output


def uncrop_point_est_visualisation(cropped_mesh_render_rgb,
                                   cropped_mesh_render_iuv,
                                   bbox_centres,
                                   bbox_whs,
                                   orig_image,
                                   visualise_wh,
                                   bbox_scale_factor):

    bbox_whs *= bbox_scale_factor
    uncropped_for_visualise = batch_crop_opencv_affine(output_wh=(visualise_wh, visualise_wh),
                                                       num_to_crop=1,
                                                       rgb=cropped_mesh_render_rgb,
                                                       iuv=cropped_mesh_render_iuv,
                                                       bbox_centres=bbox_centres,
                                                       bbox_whs=bbox_whs,
                                                       uncrop=True,
                                                       uncrop_wh=(orig_image.shape[1], orig_image.shape[0]))
    uncropped_rgb = uncropped_for_visualise['rgb'][0].transpose(1, 2, 0) * 255
    uncropped_seg = uncropped_for_visualise['iuv'][0, 0, :, :]
    background_pixels = uncropped_seg[:, :, None] == 0  # Body pixels are > 0
    uncropped_point_est_fig = uncropped_rgb * (np.logical_not(background_pixels)) + \
                              orig_image * background_pixels

    return uncropped_point_est_fig


def render_samples_visualisation(renderer,
                                 num_vis_samples,
                                 samples_rows,
                                 samples_cols,
                                 visualise_wh,
                                 cropped_for_proxy_rgb,
                                 pred_vertices_samples_all_rot,
                                 vertex_colours,
                                 cam_t,
                                 fixed_cam_t,
                                 orthographic_scale,
                                 fixed_orthographic_scale,
                                 lights_rgb_settings):

    samples_fig = np.zeros((samples_rows * visualise_wh, samples_cols * visualise_wh, 3),
                           dtype=torch.float32)
    for i in range(num_vis_samples + 1):
        body_vis_output_sample = renderer(vertices=pred_vertices_samples_all_rot['0'][[i]],
                                          cam_t=cam_t,
                                          orthographic_scale=orthographic_scale,
                                          lights_rgb_settings=lights_rgb_settings,
                                          verts_features=vertex_colours)
        body_vis_rgb_sample = batch_add_rgb_background(backgrounds=cropped_for_proxy_rgb,
                                                       rgb=body_vis_output_sample['rgb_images'].permute(0, 3, 1, 2).contiguous(),
                                                       seg=body_vis_output_sample['iuv_images'][:, :, :, 0].round())
        body_vis_rgb_sample = body_vis_rgb_sample.cpu().detach().numpy()[0].transpose(1, 2, 0)

        body_vis_rgb_rot90_sample = renderer(vertices=pred_vertices_samples_all_rot['90'][[i]],
                                             cam_t=fixed_cam_t,
                                             orthographic_scale=fixed_orthographic_scale,
                                             lights_rgb_settings=lights_rgb_settings,
                                             verts_features=vertex_colours)['rgb_images'].cpu().detach().numpy()[0]

        row = (2 * i) // samples_cols
        col = (2 * i) % samples_cols
        samples_fig[row * visualise_wh:(row + 1) * visualise_wh, col * visualise_wh:(col + 1) * visualise_wh] = body_vis_rgb_sample

        row = (2 * i + 1) // samples_cols
        col = (2 * i + 1) % samples_cols
        samples_fig[row * visualise_wh:(row + 1) * visualise_wh, col * visualise_wh:(col + 1) * visualise_wh] = body_vis_rgb_rot90_sample

        return samples_fig


def plot_xyz_vertex_variance(pred_vertices_point_est,
                             per_vertex_xyz_variance,
                             cropped_for_proxy_rgb,
                             visualise_wh,
                             cam_wp,
                             save_path):
    pred_vertices2D_mode = orthographic_project_torch(pred_vertices_point_est, cam_wp)
    pred_vertices2D_mode = undo_keypoint_normalisation(pred_vertices2D_mode, visualise_wh).cpu().detach().numpy()[0]

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
        plt.imshow(cropped_for_proxy_rgb, alpha=img_alpha)
        plt.scatter(pred_vertices2D_mode[:, 0], pred_vertices2D_mode[:, 1],
                    s=scatter_s,
                    c=per_vertex_xyz_variance[:, i],
                    cmap='jet',
                    norm=plt.Normalize(vmin=0.0, vmax=0.15, clip=True))
        subplot_count += 1
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
