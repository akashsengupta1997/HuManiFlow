import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt


from utils.image_utils import batch_add_rgb_background, batch_crop_opencv_affine
from utils.cam_utils import orthographic_project_torch
from utils.joints2d_utils import undo_keypoint_normalisation


def render_point_est_visualisation(renderer,
                                   joints2D,
                                   joints2D_confs,
                                   cropped_proxy_for_vis,
                                   cropped_rgb_for_vis,
                                   pred_vertices_point_est_all_rot,
                                   pred_tpose_vertices_point_est_all_rot,
                                   vertex_colours,
                                   cam_t,
                                   fixed_cam_t,
                                   orthographic_scale,
                                   fixed_orthographic_scale,
                                   lights_rgb_settings,
                                   visualise_wh,
                                   proxy_orig_wh):

    batch_size = joints2D.shape[0]

    plain_texture = torch.ones(batch_size, 1200, 800, 3, device=pred_vertices_point_est_all_rot['0'].device).float() * 0.7

    body_vis_rgb_all_rot = {}
    for rot in pred_vertices_point_est_all_rot:
        if rot == '0':
            body_vis_output = renderer(vertices=pred_vertices_point_est_all_rot[rot],
                                       cam_t=cam_t,
                                       orthographic_scale=orthographic_scale,
                                       lights_rgb_settings=lights_rgb_settings,
                                       verts_features=vertex_colours,
                                       textures=plain_texture)  # If vertex_colours is None, bodies will be rendered with plain_texture

            body_vis_rgb_all_rot[rot] = batch_add_rgb_background(backgrounds=cropped_rgb_for_vis,
                                                                 rgb=body_vis_output['rgb_images'].permute(0, 3, 1, 2).contiguous(),
                                                                 seg=body_vis_output['iuv_images'][:, :, :, 0].round()).cpu().detach().numpy()
        else:
            body_vis_rgb_all_rot[rot] = renderer(vertices=pred_vertices_point_est_all_rot[rot],
                                                 cam_t=fixed_cam_t,
                                                 orthographic_scale=fixed_orthographic_scale,
                                                 lights_rgb_settings=lights_rgb_settings,
                                                 verts_features=vertex_colours,
                                                 textures=plain_texture)['rgb_images'].cpu().detach().numpy()

    if pred_tpose_vertices_point_est_all_rot is not None:
        tpose_body_vis_rgb = renderer(vertices=pred_tpose_vertices_point_est_all_rot['0'],
                                      textures=plain_texture,
                                      cam_t=fixed_cam_t,
                                      orthographic_scale=fixed_orthographic_scale,
                                      lights_rgb_settings=lights_rgb_settings)['rgb_images'].cpu().detach().numpy()
        tpose_body_vis_rgb_rot90 = renderer(vertices=pred_tpose_vertices_point_est_all_rot['90'],
                                            textures=plain_texture,
                                            cam_t=fixed_cam_t,
                                            orthographic_scale=fixed_orthographic_scale,
                                            lights_rgb_settings=lights_rgb_settings)['rgb_images'].cpu().detach().numpy()

    cropped_rgb_for_vis = cropped_rgb_for_vis.cpu().detach().numpy().transpose(0, 2, 3, 1)
    cropped_proxy_for_vis = cropped_proxy_for_vis.cpu().detach().numpy().transpose(0, 2, 3, 1)

    point_est_figs = []
    combined_vis_rows = 2
    combined_vis_cols = 4 if pred_tpose_vertices_point_est_all_rot is not None else 3
    for i in range(batch_size):
        fig = np.zeros((combined_vis_rows * visualise_wh, combined_vis_cols * visualise_wh, 3),
                       dtype=body_vis_rgb_all_rot['0'].dtype)

        fig[:visualise_wh, :visualise_wh] = cropped_rgb_for_vis[i]

        proxy = np.ascontiguousarray(cropped_proxy_for_vis[i])
        for joint_num in range(joints2D.shape[1]):
            hor_coord = joints2D[i, joint_num, 0].item() * visualise_wh / proxy_orig_wh
            ver_coord = joints2D[i, joint_num, 1].item() * visualise_wh / proxy_orig_wh
            cv2.circle(proxy,
                       (int(hor_coord), int(ver_coord)),
                       radius=3,
                       color=(255, 0, 0),
                       thickness=-1)
            cv2.putText(proxy,
                        str(joint_num),
                        (int(hor_coord + 4), int(ver_coord + 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), lineType=2)
            cv2.putText(proxy,
                        str(joint_num) + " {:.2f}".format(joints2D_confs[i, joint_num].item()),
                        (10, 16 * (joint_num + 1)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), lineType=2)
        fig[visualise_wh:2 * visualise_wh, :visualise_wh] = proxy

        fig[:visualise_wh, visualise_wh:2 * visualise_wh] = body_vis_rgb_all_rot['0'][i].transpose(1, 2, 0)
        fig[visualise_wh:2 * visualise_wh, visualise_wh:2 * visualise_wh] = body_vis_rgb_all_rot['90'][i]
        fig[:visualise_wh, 2 * visualise_wh:3 * visualise_wh] = body_vis_rgb_all_rot['180'][i]
        fig[visualise_wh:2 * visualise_wh, 2 * visualise_wh:3 * visualise_wh] = body_vis_rgb_all_rot['270'][i]

        if pred_tpose_vertices_point_est_all_rot is not None:
            fig[:visualise_wh, 3 * visualise_wh:4 * visualise_wh] = tpose_body_vis_rgb
            fig[visualise_wh:2 * visualise_wh, 3 * visualise_wh:4 * visualise_wh] = tpose_body_vis_rgb_rot90

        point_est_figs.append(fig)

    return point_est_figs, body_vis_output


def uncrop_point_est_visualisation(cropped_mesh_render_rgb,
                                   cropped_mesh_render_iuv,
                                   bbox_centres,
                                   bbox_whs,
                                   orig_image,
                                   visualise_wh,
                                   bbox_scale_factor):

    bbox_whs *= bbox_scale_factor
    uncropped_for_visualise = batch_crop_opencv_affine(output_wh=(visualise_wh, visualise_wh),
                                                       num_to_crop=cropped_mesh_render_rgb.shape[0],
                                                       rgb=cropped_mesh_render_rgb,
                                                       iuv=cropped_mesh_render_iuv,
                                                       bbox_centres=bbox_centres,
                                                       bbox_whs=bbox_whs,
                                                       uncrop=True,
                                                       uncrop_wh=(orig_image.shape[2], orig_image.shape[1]))
    uncropped_rgb = uncropped_for_visualise['rgb'].transpose(0, 2, 3, 1) * 255
    uncropped_seg = uncropped_for_visualise['iuv'][:, 0, :, :]
    background_pixels = uncropped_seg[:, :, :, None] == 0  # Body pixels are > 0
    uncropped_point_est_figs = uncropped_rgb * (np.logical_not(background_pixels)) + \
                               orig_image * background_pixels

    return uncropped_point_est_figs


def render_samples_visualisation(renderer,
                                 num_vis_samples,
                                 samples_rows,
                                 samples_cols,
                                 visualise_wh,
                                 cropped_rgb_for_vis,
                                 pred_vertices_samples_all_rot,
                                 vertex_colours,
                                 cam_t,
                                 fixed_cam_t,
                                 orthographic_scale,
                                 fixed_orthographic_scale,
                                 lights_rgb_settings):

    samples_fig = np.zeros((samples_rows * visualise_wh, samples_cols * visualise_wh, 3),
                           dtype=np.float32)
    for i in range(num_vis_samples + 1):
        body_vis_output_sample = renderer(vertices=pred_vertices_samples_all_rot['0'][[i]],
                                          cam_t=cam_t,
                                          orthographic_scale=orthographic_scale,
                                          lights_rgb_settings=lights_rgb_settings,
                                          verts_features=vertex_colours)
        body_vis_rgb_sample = batch_add_rgb_background(backgrounds=cropped_rgb_for_vis,
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
                             cropped_rgb_for_vis,
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
        plt.imshow(cropped_rgb_for_vis, alpha=img_alpha)
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
