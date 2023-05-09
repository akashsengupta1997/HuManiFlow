import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from smplx.lbs import batch_rodrigues

from utils.renderers.pytorch3d_textured_renderer import TexturedIUVRenderer

from metrics.eval_metrics_tracker import EvalMetricsTracker

from utils.cam_utils import orthographic_project_torch
from utils.rigid_transform_utils import aa_rotate_translate_points_pytorch3d, aa_rotate_rotmats_opencv
from utils.joints2d_utils import undo_keypoint_normalisation
from utils.label_conversions import convert_multiclass_to_binary_labels, ALL_JOINTS_TO_COCO_MAP, ALL_JOINTS_TO_H36M_MAP, H36M_TO_J14


def evaluate_humaniflow(humaniflow_model,
                        humaniflow_cfg,
                        smpl_model_neutral,
                        smpl_model_male,
                        smpl_model_female,
                        edge_detect_model,
                        device,
                        eval_dataset,
                        metrics,
                        batch_size=1,
                        num_workers=4,
                        pin_memory=True,
                        num_pred_samples=10,
                        save_per_frame_metrics=True,
                        save_path=None):

    eval_dataloader = DataLoader(eval_dataset,
                                 batch_size=batch_size,
                                 shuffle=False,
                                 drop_last=False,
                                 num_workers=num_workers,
                                 pin_memory=pin_memory)

    # Instantiate metrics tracker
    metrics_tracker = EvalMetricsTracker(metrics,
                                         save_per_frame_metrics=save_per_frame_metrics,
                                         save_path=save_path,
                                         num_samples_for_prob_metrics=num_pred_samples)
    metrics_tracker.initialise_metric_sums()
    metrics_tracker.initialise_per_frame_metric_lists()

    if any('silhouette' in metric for metric in metrics):
        silhouette_renderer = TexturedIUVRenderer(device=device,
                                                  batch_size=batch_size,
                                                  img_wh=humaniflow_cfg.DATA.PROXY_REP_SIZE,
                                                  projection_type='orthographic',
                                                  render_rgb=False,
                                                  bin_size=None)

    if save_per_frame_metrics:
        fname_per_frame = []
        pose_per_frame = []
        shape_per_frame = []
        cam_per_frame = []

    humaniflow_model.eval()
    humaniflow_model.pose_so3flow_transform_modules.eval()

    for batch_num, samples_batch in enumerate(tqdm(eval_dataloader)):
        with torch.no_grad():
            # ------------------ INPUTS ------------------
            image = samples_batch['image'].to(device)
            heatmaps = samples_batch['heatmaps'].to(device)
            edge_detector_output = edge_detect_model(image)
            proxy_rep_img = edge_detector_output['thresholded_thin_edges'] if humaniflow_cfg.DATA.EDGE_NMS else edge_detector_output['thresholded_grad_magnitude']
            proxy_rep_input = torch.cat([proxy_rep_img, heatmaps], dim=1)
            num_in_batch = proxy_rep_input.shape[0]  # Not necessarily equal to batch_size, since drop_last = False

            # ------------------ Targets ------------------
            target_pose = samples_batch['pose'].to(device)
            target_shape = samples_batch['shape'].to(device)
            target_genders = np.array(samples_batch['gender'])
            target_joints2D = samples_batch['joints2D']

            # Flipping pose targets such that they are right way up in 3D space - i.e. wrong way up when projected
            target_pose_rotmats = batch_rodrigues(target_pose.contiguous().view(-1, 3)).view(num_in_batch, 24, 3, 3)
            target_glob_rotmats = target_pose_rotmats[:, 0, :, :]
            target_glob_vecs, _ = aa_rotate_rotmats_opencv(rotmats=target_glob_rotmats,
                                                           axis=[1, 0, 0],
                                                           angle=np.pi,
                                                           rot_mult_order='pre')
            target_pose[:, :3] = target_glob_vecs
            target_smpl_male = smpl_model_male(body_pose=target_pose[:, 3:],
                                               global_orient=target_pose[:, :3],
                                               betas=target_shape)
            target_vertices = target_smpl_male.vertices
            target_joints_h36mlsp = target_smpl_male.joints[:, ALL_JOINTS_TO_H36M_MAP, :][:, H36M_TO_J14, :]

            target_smpl_female = smpl_model_female(body_pose=target_pose[:, 3:],
                                                   global_orient=target_pose[:, :3],
                                                   betas=target_shape)
            target_vertices[target_genders == 'f', :, :] = target_smpl_female.vertices[target_genders == 'f', :, :]
            target_joints_h36mlsp[target_genders == 'f', :, :] = target_smpl_female.joints[:, ALL_JOINTS_TO_H36M_MAP, :][:, H36M_TO_J14, :][target_genders == 'f', :, :]

            target_tpose_smpl_male = smpl_model_male(body_pose=torch.zeros_like(target_pose[:, 3:]),
                                                     global_orient=torch.zeros_like(target_pose[:, :3]),
                                                     betas=target_shape)
            target_tpose_vertices = target_tpose_smpl_male.vertices

            target_tpose_smpl_female = smpl_model_female(body_pose=torch.zeros_like(target_pose[:, 3:]),
                                                         global_orient=torch.zeros_like(target_pose[:, :3]),
                                                         betas=target_shape)
            target_tpose_vertices[target_genders == 'f', :, :] = target_tpose_smpl_female.vertices[target_genders == 'f', :, :]

            # ------------------------------- PREDICTIONS -------------------------------
            pred = humaniflow_model(proxy_rep_input,
                                    num_samples=num_pred_samples)
            # pred is a dict containing:
            # - Pose/shape samples from predicted distribution
            # - Pose/shape point estimate from predicted distribution

            orthographic_scale = pred['cam_wp'][:, [0, 0]]
            cam_t = torch.cat([pred['cam_wp'][:, 1:],
                               torch.ones(pred['cam_wp'].shape[0], 1, device=device).float() * 2.5],
                              dim=-1)

            pred_smpl_output_point_est = smpl_model_neutral(body_pose=pred['pose_rotmats_point_est'],
                                                            global_orient=pred['glob_rotmat'][:, None, :, :],
                                                            betas=pred['shape_mode'],
                                                            pose2rot=False)
            pred_vertices_point_est = pred_smpl_output_point_est.vertices  # (bsize, 6890, 3)
            pred_joints3D_h36mlsp_point_est = pred_smpl_output_point_est.joints[:, ALL_JOINTS_TO_H36M_MAP, :][:, H36M_TO_J14, :]  # (bsize, 14, 3)

            pred_tpose_vertices_point_est = smpl_model_neutral(body_pose=torch.zeros_like(target_pose[:, 3:]),
                                                               global_orient=torch.zeros_like(target_pose[:, :3]),
                                                               betas=pred['shape_mode']).vertices  # (bsize, 6890, 3)

            if any('joints2D' in metric for metric in metrics):
                # Pose targets were flipped such that they are right way up in 3D space - i.e. wrong way up when projected
                # Need to flip pred joints before projecting to 2D for 2D metrics
                pred_joints2D_point_est = orthographic_project_torch(aa_rotate_translate_points_pytorch3d(points=pred_smpl_output_point_est.joints[:, ALL_JOINTS_TO_COCO_MAP, :],
                                                                                                          axes=torch.tensor([1., 0., 0.], device=device).float(),
                                                                                                          angles=np.pi,
                                                                                                          translations=torch.zeros(3, device=device).float()),
                                                                     pred['cam_wp'])  # (bsize, 17, 2)
                pred_joints2D_point_est = undo_keypoint_normalisation(pred_joints2D_point_est, humaniflow_cfg.DATA.PROXY_REP_SIZE)
            if any('silhouette' in metric for metric in metrics):
                pred_vertices_flipped_point_est = aa_rotate_translate_points_pytorch3d(points=pred_vertices_point_est,
                                                                                       axes=torch.tensor([1., 0., 0.], device=device),
                                                                                       angles=np.pi,
                                                                                       translations=torch.zeros(3, device=device))

            if 'silhouette-IOU' in metrics:
                # If num_in_batch != batch_size, which can happen for the LAST batch, need to make a new silhouette_renderer with appropriate batch size
                if num_in_batch != batch_size:
                    silhouette_renderer = TexturedIUVRenderer(device=device,
                                                              batch_size=num_in_batch,
                                                              img_wh=humaniflow_cfg.DATA.PROXY_REP_SIZE,
                                                              projection_type='orthographic',
                                                              render_rgb=False,
                                                              bin_size=None)

                wp_render_output = silhouette_renderer(vertices=pred_vertices_flipped_point_est,
                                                       cam_t=cam_t,
                                                       orthographic_scale=orthographic_scale)
                iuv_mode = wp_render_output['iuv_images']
                part_seg_mode = iuv_mode[:, :, :, 0].round()
                pred_silhouette_mode = convert_multiclass_to_binary_labels(part_seg_mode)

            if any('sample' in metric for metric in metrics):
                pred_smpl_output_samples = smpl_model_neutral(
                    body_pose=pred['pose_rotmats_samples'].reshape(-1, 23, 3, 3),  # (bs * num samples, 23, 3, 3)
                    global_orient=pred['glob_rotmat'][:, None, :, :].expand(-1, num_pred_samples, -1, -1).reshape(-1, 1, 3, 3),  # (bs * num samples, 1, 3, 3),
                    betas=pred['shape_samples'].reshape(-1, humaniflow_cfg.MODEL.NUM_SMPL_BETAS),  # (bs * num samples, num_smpl_betas)
                    pose2rot=False)

                pred_vertices_samples = pred_smpl_output_samples.vertices.reshape(num_in_batch,
                                                                                  num_pred_samples,
                                                                                  pred_vertices_point_est.shape[1],
                                                                                  3)  # (bs, num samples, 6890, 3)
                pred_joints3D_h36mlsp_samples = pred_smpl_output_samples.joints[:, ALL_JOINTS_TO_H36M_MAP, :][:, H36M_TO_J14, :].reshape(num_in_batch,
                                                                                                                                         num_pred_samples,
                                                                                                                                         pred_joints3D_h36mlsp_point_est.shape[1],
                                                                                                                                         3)    # (bs, num samples, 14, 3)
                pred_joints3D_coco_samples = pred_smpl_output_samples.joints[:, ALL_JOINTS_TO_COCO_MAP, :].reshape(num_in_batch,
                                                                                                                   num_pred_samples,
                                                                                                                   len(ALL_JOINTS_TO_COCO_MAP),
                                                                                                                   3)  # (bs, num samples, 17, 3)
                pred_tpose_vertices_samples = smpl_model_neutral(
                    body_pose=torch.zeros(num_in_batch * num_pred_samples, 69, device=device).float(),
                    global_orient=torch.zeros(num_in_batch * num_pred_samples, 3, device=device).float(),
                    betas=pred['shape_samples'].reshape(-1, humaniflow_cfg.MODEL.NUM_SMPL_BETAS)).vertices.reshape(num_in_batch,
                                                                                                                   num_pred_samples,
                                                                                                                   pred_tpose_vertices_point_est.shape[1],
                                                                                                                   3)  # (bs, num samples, 6890, 3)

                if 'joints2Dsamples-L2E' in metrics:
                    # Pose targets were flipped such that they are right way up in 3D space - i.e. wrong way up when projected
                    # Need to flip pred joints before projecting to 2D for 2D metrics
                    pred_joints2D_samples = orthographic_project_torch(aa_rotate_translate_points_pytorch3d(points=pred_smpl_output_samples.joints[:, ALL_JOINTS_TO_COCO_MAP, :],
                                                                                                            axes=torch.tensor([1., 0., 0.], device=device).float(),
                                                                                                            angles=np.pi,
                                                                                                            translations=torch.zeros(3, device=device).float()),
                                                                       pred['cam_wp'][:, None, :].expand(-1, num_pred_samples, -1).reshape(-1, 3))  # (bs * num samples, 17, 2)
                    pred_joints2D_samples = undo_keypoint_normalisation(pred_joints2D_samples,
                                                                        humaniflow_cfg.DATA.PROXY_REP_SIZE).reshape(num_in_batch,
                                                                                                                    num_pred_samples,
                                                                                                                    pred_joints2D_samples.shape[1],
                                                                                                                    2)  # (bs, num samples, 17, 2)

                if 'silhouettesamples-IOU' in metrics:
                    pred_vertices_samples_flipped = aa_rotate_translate_points_pytorch3d(points=pred_smpl_output_samples.vertices,
                                                                                         axes=torch.tensor([1., 0., 0.], device=device),
                                                                                         angles=np.pi,
                                                                                         translations=torch.zeros(3, device=device))
                    pred_vertices_samples_flipped = pred_vertices_samples_flipped.reshape(num_in_batch,
                                                                                          num_pred_samples,
                                                                                          pred_vertices_samples_flipped.shape[1],
                                                                                          3)  # (bs, num samples, 6890, 3)
                    pred_silhouette_samples = []
                    for i in range(num_pred_samples):
                        iuv_sample = silhouette_renderer(vertices=pred_vertices_samples_flipped[:, i, :, :],
                                                         cam_t=cam_t,
                                                         orthographic_scale=orthographic_scale)['iuv_images']
                        part_seg_sample = iuv_sample[:, :, :, 0].round()
                        pred_silhouette_samples.append(convert_multiclass_to_binary_labels(part_seg_sample))
                    pred_silhouette_samples = torch.stack(pred_silhouette_samples, dim=1)  # (bs, num samples, img wh, img wh)

            # ------------------------------- TRACKING METRICS -------------------------------
            pred_dict = {'verts3D': pred_vertices_point_est.cpu().detach().numpy(),
                         'tpose_verts3D': pred_tpose_vertices_point_est.cpu().detach().numpy(),
                         'joints3D': pred_joints3D_h36mlsp_point_est.cpu().detach().numpy()}
            target_dict = {'verts3D': target_vertices.cpu().detach().numpy(),
                           'tpose_verts3D': target_tpose_vertices.cpu().detach().numpy(),
                           'joints3D': target_joints_h36mlsp.cpu().detach().numpy()}

            if 'joints2D-L2E' in metrics:
                pred_dict['joints2D'] = pred_joints2D_point_est.cpu().detach().numpy()
                target_dict['joints2D'] = target_joints2D.numpy()
                if 'joints2D_visib' in samples_batch:
                    target_dict['joints2D_vis'] = samples_batch['joints2D_visib'].numpy()
            if 'silhouette-IOU' in metrics:
                pred_dict['silhouettes'] = pred_silhouette_mode.cpu().detach().numpy()
                target_dict['silhouettes'] = samples_batch['silhouette'].numpy()

            if any('sample' in metric for metric in metrics):
                pred_dict['verts3D_samples'] = pred_vertices_samples.cpu().detach().numpy()
                pred_dict['tpose_verts3D_samples'] = pred_tpose_vertices_samples.cpu().detach().numpy()
                pred_dict['joints3D_samples'] = pred_joints3D_h36mlsp_samples.cpu().detach().numpy()
                pred_dict['joints3D_coco_samples'] = pred_joints3D_coco_samples.cpu().detach().numpy()

                if 'joints2Dsamples-L2E' in metrics:
                    pred_dict['joints2Dsamples'] = pred_joints2D_samples.cpu().detach().numpy()
                if 'silhouettesamples-IOU' in metrics:
                    pred_dict['silhouettessamples'] = pred_silhouette_samples.cpu().detach().numpy()

            metrics_tracker.update_per_batch(pred_dict,
                                             target_dict,
                                             num_in_batch,
                                             model_input=proxy_rep_input,
                                             return_per_frame_metrics=False)

            if save_per_frame_metrics:
                fname_per_frame.append(samples_batch['fname'])
                pose_per_frame.append(np.concatenate([pred['glob_rotmat'][:, None, :, :].cpu().detach().numpy(),
                                                      pred['pose_rotmats_point_est'].cpu().detach().numpy()],
                                                     axis=1))
                shape_per_frame.append(pred['shape_mode'].cpu().detach().numpy())
                cam_per_frame.append(pred['cam_wp'].cpu().detach().numpy())

    # ------------------------------- FINAL METRICS -------------------------------
    metrics_tracker.compute_final_metrics()

    if save_per_frame_metrics:
        fname_per_frame = np.concatenate(fname_per_frame, axis=0)
        np.save(os.path.join(save_path, 'fname_per_frame.npy'), fname_per_frame)
        print('Saved filenames:', fname_per_frame.shape)

        pose_per_frame = np.concatenate(pose_per_frame, axis=0)
        np.save(os.path.join(save_path, 'pose_per_frame.npy'), pose_per_frame)
        print('Saved poses:', pose_per_frame.shape)

        shape_per_frame = np.concatenate(shape_per_frame, axis=0)
        np.save(os.path.join(save_path, 'shape_per_frame.npy'), shape_per_frame)
        print('Saved shapes:', shape_per_frame.shape)

        cam_per_frame = np.concatenate(cam_per_frame, axis=0)
        np.save(os.path.join(save_path, 'cam_per_frame.npy'), cam_per_frame)
        print('Saved cams:', cam_per_frame.shape)
