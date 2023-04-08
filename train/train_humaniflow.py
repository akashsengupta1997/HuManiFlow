import copy
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from smplx.lbs import batch_rodrigues
from tqdm import tqdm

from metrics.train_loss_and_metrics_tracker import TrainingLossesAndMetricsTracker

from utils.checkpoint_utils import load_training_info_from_checkpoint
from utils.train_utils import check_for_nans_in_output
from utils.cam_utils import perspective_project_torch, orthographic_project_torch
from utils.rigid_transform_utils import aa_rotate_translate_points_pytorch3d, aa_rotate_rotmats_pytorch3d
from utils.label_conversions import convert_2Djoints_to_gaussian_heatmaps_torch, convert_densepose_seg_to_14part_labels, \
    ALL_JOINTS_TO_H36M_MAP, ALL_JOINTS_TO_COCO_MAP, H36M_TO_J14
from utils.joints2d_utils import check_joints2d_visibility_torch, check_joints2d_occluded_torch
from utils.image_utils import batch_add_rgb_background, batch_crop_pytorch_affine

from utils.augmentation.smpl_augmentation import normal_sample_shape
from utils.augmentation.cam_augmentation import augment_cam_t
from utils.augmentation.proxy_rep_augmentation import augment_proxy_representation, random_extreme_crop
from utils.augmentation.rgb_augmentation import augment_rgb
from utils.augmentation.lighting_augmentation import augment_light


def train_humaniflow(humaniflow_model,
                     humaniflow_cfg,
                     smpl_model,
                     edge_detect_model,
                     pytorch3d_renderer,
                     device,
                     train_dataset,
                     val_dataset,
                     criterion,
                     optimiser,
                     metrics,
                     save_val_metrics,
                     model_save_dir,
                     logs_save_path,
                     checkpoint=None):
    # Set up dataloaders
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=humaniflow_cfg.TRAIN.BATCH_SIZE,
                                  shuffle=True,
                                  drop_last=True,
                                  num_workers=humaniflow_cfg.TRAIN.NUM_WORKERS,
                                  pin_memory=humaniflow_cfg.TRAIN.PIN_MEMORY)
    val_dataloader = DataLoader(val_dataset,
                                batch_size=humaniflow_cfg.TRAIN.BATCH_SIZE,
                                shuffle=True,
                                drop_last=True,
                                num_workers=humaniflow_cfg.TRAIN.NUM_WORKERS,
                                pin_memory=humaniflow_cfg.TRAIN.PIN_MEMORY)
    dataloaders = {'train': train_dataloader,
                   'val': val_dataloader}

    # Load checkpoint benchmarks if provided
    if checkpoint is not None:
        # Resuming training - note that current model and optimiser state dicts are loaded out
        # of train function (should be in run file).
        current_epoch, best_epoch, best_model_wts, best_epoch_val_metrics = load_training_info_from_checkpoint(checkpoint,
                                                                                                               save_val_metrics)
        load_logs = True
    else:
        current_epoch = 0
        best_epoch = 0
        best_epoch_val_metrics = {}
        # metrics that decide whether to save model after each epoch or not
        for metric in save_val_metrics:
            best_epoch_val_metrics[metric] = np.inf
        best_model_wts = copy.deepcopy(humaniflow_model.state_dict())
        load_logs = False

    # Instantiate metrics tracker
    metrics_tracker = TrainingLossesAndMetricsTracker(metrics_to_track=metrics,
                                                      img_wh=humaniflow_cfg.DATA.PROXY_REP_SIZE,
                                                      log_save_path=logs_save_path,
                                                      load_logs=load_logs,
                                                      current_epoch=current_epoch)

    # Useful tensors that are re-used and can be pre-defined
    x_axis = torch.tensor([1., 0., 0.],
                          device=device, dtype=torch.float32)
    delta_betas_std_vector = torch.ones(humaniflow_cfg.MODEL.NUM_SMPL_BETAS,
                                        device=device, dtype=torch.float32) * humaniflow_cfg.TRAIN.SYNTH_DATA.AUGMENT.SMPL.SHAPE_STD
    mean_shape = torch.zeros(humaniflow_cfg.MODEL.NUM_SMPL_BETAS,
                             device=device, dtype=torch.float32)
    mean_cam_t = torch.tensor(humaniflow_cfg.TRAIN.SYNTH_DATA.MEAN_CAM_T,
                              device=device, dtype=torch.float32)
    mean_cam_t = mean_cam_t[None, :].expand(humaniflow_cfg.TRAIN.BATCH_SIZE, -1)

    # Starting training loop
    for epoch in range(current_epoch, humaniflow_cfg.TRAIN.NUM_EPOCHS):
        print('\nEpoch {}/{}'.format(epoch, humaniflow_cfg.TRAIN.NUM_EPOCHS - 1))
        print('-' * 10)
        metrics_tracker.initialise_loss_metric_sums()

        for split in ['train', 'val']:
            if split == 'train':
                print('Training.')
                humaniflow_model.train()
                humaniflow_model.pose_so3flow_transform_modules.train()
            else:
                print('Validation.')
                humaniflow_model.eval()
                humaniflow_model.pose_so3flow_transform_modules.eval()

            for batch_num, samples_batch in enumerate(tqdm(dataloaders[split])):
                #############################################################
                # ---------------- SYNTHETIC DATA GENERATION ----------------
                #############################################################
                with torch.no_grad():
                    # ------------ RANDOM POSE, SHAPE, BACKGROUND, TEXTURE, CAMERA SAMPLING ------------
                    # Load target pose and random background/texture
                    target_pose = samples_batch['pose'].to(device)  # (bs, 72)
                    background = samples_batch['background'].to(device)  # (bs, 3, img_wh, img_wh)
                    texture = samples_batch['texture'].to(device)  # (bs, 1200, 800, 3)

                    # Convert target_pose from axis angle to rotmats
                    target_pose_rotmats = batch_rodrigues(target_pose.contiguous().view(-1, 3)).view(-1, 24, 3, 3)
                    target_glob_rotmats = target_pose_rotmats[:, 0, :, :]
                    target_pose_rotmats = target_pose_rotmats[:, 1:, :, :]
                    # Flipping pose targets such that they are right way up in 3D space - i.e. wrong way up when projected
                    # Then pose predictions will also be right way up in 3D space - network doesn't need to learn to flip.
                    _, target_glob_rotmats = aa_rotate_rotmats_pytorch3d(rotmats=target_glob_rotmats,
                                                                         angles=np.pi,
                                                                         axes=x_axis,
                                                                         rot_mult_order='post')
                    # Random sample body shape
                    target_shape = normal_sample_shape(batch_size=humaniflow_cfg.TRAIN.BATCH_SIZE,
                                                       mean_shape=mean_shape,
                                                       std_vector=delta_betas_std_vector)
                    # Random sample camera translation
                    target_cam_t = augment_cam_t(mean_cam_t,
                                                 xy_std=humaniflow_cfg.TRAIN.SYNTH_DATA.AUGMENT.CAM.XY_STD,
                                                 delta_z_range=humaniflow_cfg.TRAIN.SYNTH_DATA.AUGMENT.CAM.DELTA_Z_RANGE)

                    # Compute target vertices and joints
                    target_smpl_output = smpl_model(body_pose=target_pose_rotmats,
                                                    global_orient=target_glob_rotmats.unsqueeze(1),
                                                    betas=target_shape,
                                                    pose2rot=False)
                    target_vertices = target_smpl_output.vertices
                    target_joints_all = target_smpl_output.joints
                    target_joints_h36mlsp = target_joints_all[:, ALL_JOINTS_TO_H36M_MAP, :][:, H36M_TO_J14, :]

                    target_tpose_vertices = smpl_model(body_pose=torch.zeros_like(target_pose)[:, 3:],
                                                         global_orient=torch.zeros_like(target_pose)[:, :3],
                                                         betas=target_shape).vertices

                    # ------------ INPUT PROXY REPRESENTATION GENERATION + 2D TARGET JOINTS ------------
                    # Pose targets were flipped such that they are right way up in 3D space - i.e. wrong way up when projected
                    # Need to flip target_vertices_for_rendering 180° about x-axis so they are right way up when projected
                    # Need to flip target_joints_coco 180° about x-axis so they are right way up when projected
                    target_vertices_for_rendering = aa_rotate_translate_points_pytorch3d(points=target_vertices,
                                                                                         axes=x_axis,
                                                                                         angles=np.pi,
                                                                                         translations=torch.zeros(3, device=device).float())
                    target_joints_coco = aa_rotate_translate_points_pytorch3d(points=target_joints_all[:, ALL_JOINTS_TO_COCO_MAP, :],
                                                                              axes=x_axis,
                                                                              angles=np.pi,
                                                                              translations=torch.zeros(3, device=device).float())
                    target_joints2d_coco = perspective_project_torch(target_joints_coco,
                                                                     None,
                                                                     target_cam_t,
                                                                     focal_length=humaniflow_cfg.TRAIN.SYNTH_DATA.FOCAL_LENGTH,
                                                                     img_wh=humaniflow_cfg.DATA.PROXY_REP_SIZE)

                    # Check if joints within image dimensions before cropping + recentering.
                    target_joints2d_visib_coco = check_joints2d_visibility_torch(target_joints2d_coco,
                                                                                 humaniflow_cfg.DATA.PROXY_REP_SIZE)  # (batch_size, 17)

                    # Render RGB/IUV image
                    lights_rgb_settings = augment_light(batch_size=1,
                                                        device=device,
                                                        rgb_augment_config=humaniflow_cfg.TRAIN.SYNTH_DATA.AUGMENT.RGB)
                    renderer_output = pytorch3d_renderer(vertices=target_vertices_for_rendering,
                                                         textures=texture,
                                                         cam_t=target_cam_t,
                                                         lights_rgb_settings=lights_rgb_settings)
                    iuv_in = renderer_output['iuv_images'].permute(0, 3, 1, 2).contiguous()  # (bs, 3, img_wh, img_wh)
                    iuv_in[:, 1:, :, :] = iuv_in[:, 1:, :, :] * 255
                    iuv_in = iuv_in.round()
                    rgb_in = renderer_output['rgb_images'].permute(0, 3, 1, 2).contiguous()  # (bs, 3, img_wh, img_wh)

                    # Prepare seg for extreme crop augmentation
                    seg_extreme_crop = random_extreme_crop(seg=iuv_in[:, 0, :, :],
                                                           extreme_crop_probability=humaniflow_cfg.TRAIN.SYNTH_DATA.AUGMENT.PROXY_REP.EXTREME_CROP_PROB)

                    # Crop to person bounding box after bbox scale and centre augmentation
                    crop_outputs = batch_crop_pytorch_affine(input_wh=(humaniflow_cfg.DATA.PROXY_REP_SIZE, humaniflow_cfg.DATA.PROXY_REP_SIZE),
                                                             output_wh=(humaniflow_cfg.DATA.PROXY_REP_SIZE, humaniflow_cfg.DATA.PROXY_REP_SIZE),
                                                             num_to_crop=humaniflow_cfg.TRAIN.BATCH_SIZE,
                                                             device=device,
                                                             rgb=rgb_in,
                                                             iuv=iuv_in,
                                                             joints2D=target_joints2d_coco,
                                                             bbox_determiner=seg_extreme_crop,
                                                             orig_scale_factor=humaniflow_cfg.DATA.BBOX_SCALE_FACTOR,
                                                             delta_scale_range=humaniflow_cfg.TRAIN.SYNTH_DATA.AUGMENT.BBOX.DELTA_SCALE_RANGE,
                                                             delta_centre_range=humaniflow_cfg.TRAIN.SYNTH_DATA.AUGMENT.BBOX.DELTA_CENTRE_RANGE,
                                                             out_of_frame_pad_val=-1)
                    iuv_in = crop_outputs['iuv']
                    target_joints2d_coco = crop_outputs['joints2D']
                    rgb_in = crop_outputs['rgb']

                    # Check if joints within image dimensions after cropping + recentering.
                    target_joints2d_visib_coco = check_joints2d_visibility_torch(target_joints2d_coco,
                                                                                 humaniflow_cfg.DATA.PROXY_REP_SIZE,
                                                                                 visibility=target_joints2d_visib_coco)  # (bs, 17)
                    # Check if joints are occluded by the body.
                    seg_14_part_occlusion_check = convert_densepose_seg_to_14part_labels(iuv_in[:, 0, :, :])
                    target_joints2d_visib_coco = check_joints2d_occluded_torch(seg_14_part_occlusion_check,
                                                                               target_joints2d_visib_coco,
                                                                               pixel_count_threshold=50)  # (bs, 17)

                    # Apply segmentation/IUV-based render augmentations + 2D joints augmentations
                    seg_aug, target_joints2d_coco_input, target_joints2d_visib_coco = augment_proxy_representation(
                        seg=iuv_in[:, 0, :, :],  # Note: out of frame pixels marked with -1
                        joints2D=target_joints2d_coco,
                        joints2D_visib=target_joints2d_visib_coco,
                        proxy_rep_augment_config=humaniflow_cfg.TRAIN.SYNTH_DATA.AUGMENT.PROXY_REP)

                    # Add background rgb
                    rgb_in = batch_add_rgb_background(backgrounds=background,
                                                      rgb=rgb_in,
                                                      seg=seg_aug)
                    # Apply RGB-based render augmentations + 2D joints augmentations
                    rgb_in, target_joints2d_coco_input, target_joints2d_visib_coco = augment_rgb(rgb=rgb_in,
                                                                                                 joints2D=target_joints2d_coco_input,
                                                                                                 joints2D_visib=target_joints2d_visib_coco,
                                                                                                 rgb_augment_config=humaniflow_cfg.TRAIN.SYNTH_DATA.AUGMENT.RGB)
                    # Compute edge-images edges
                    edge_detector_output = edge_detect_model(rgb_in)
                    edge_in = edge_detector_output['thresholded_thin_edges'] if humaniflow_cfg.DATA.EDGE_NMS else edge_detector_output['thresholded_grad_magnitude']

                    # Compute 2D joint heatmaps
                    j2d_heatmaps = convert_2Djoints_to_gaussian_heatmaps_torch(target_joints2d_coco_input,
                                                                               humaniflow_cfg.DATA.PROXY_REP_SIZE,
                                                                               std=humaniflow_cfg.DATA.HEATMAP_GAUSSIAN_STD)
                    j2d_heatmaps = j2d_heatmaps * target_joints2d_visib_coco[:, :, None, None]

                    # Concatenate edge-image and 2D joint heatmaps to create input proxy representation
                    proxy_rep_input = torch.cat([edge_in, j2d_heatmaps], dim=1).float()  # (batch_size, C, img_wh, img_wh)

                with torch.set_grad_enabled(split == 'train'):
                    #############################################################
                    # ---------------------- FORWARD PASS -----------------------
                    #############################################################
                    pred = humaniflow_model(proxy_rep_input,
                                            compute_point_est=True,
                                            num_samples=humaniflow_cfg.LOSS.NUM_J2D_SAMPLES,
                                            compute_for_loglik=True,
                                            shape_for_loglik=target_shape,
                                            pose_R_for_loglik=target_pose_rotmats,
                                            glob_R_for_loglik=target_glob_rotmats)
                    # pred is a dict containing:
                    # - Pose/shape point estimates for metrics and J2D reprojection loss,
                    # - Pose/shape samples for J2D sample reprojection loss (and metrics)
                    # - Gaussian distribution over shape for log-likelihood loss
                    # - List of ancestor-conditioned SO(3) flow (pyro) distributions over per-part poses for log-likelihood loss

                    if split == 'train':
                        # Check for NaNs in model outputs
                        # Happens very occasionally after SGD step - some numerical instability in pyro norm flows?
                        nans_in_output = check_for_nans_in_output(pred)
                        if nans_in_output:
                            print('Reloading model and optimiser state dict from before last SGD step.')
                            humaniflow_model.load_state_dict(last_batch_model_state_dict)
                            optimiser.load_state_dict(last_batch_opt_state_dict)
                            print('Re-doing current batch forward pass.')
                            pred = humaniflow_model(proxy_rep_input,
                                                    num_samples=humaniflow_cfg.LOSS.NUM_J2D_SAMPLES,
                                                    compute_for_loglik=True,
                                                    shape_for_loglik=target_shape,
                                                    pose_R_for_loglik=target_pose_rotmats,
                                                    glob_R_for_loglik=target_glob_rotmats)

                    # Compute SMPL point estimate vertices/joints
                    pred_smpl_output_point_est = smpl_model(body_pose=pred['pose_rotmats_point_est'],
                                                            global_orient=pred['glob_rotmat'][:, None, :, :],
                                                            betas=pred['shape_mode'],
                                                            pose2rot=False)
                    pred_vertices_point_est = pred_smpl_output_point_est.vertices  # (batch_size, 6890, 3)
                    pred_joints_all_point_est = pred_smpl_output_point_est.joints
                    pred_joints_h36mlsp_point_est = pred_joints_all_point_est[:, ALL_JOINTS_TO_H36M_MAP, :][:, H36M_TO_J14, :]  # (batch_size, 14, 3)
                    # Pose targets were flipped such that they are right way up in 3D space - i.e. wrong way up when projected
                    # Need to flip pred_joints_coco 180° about x-axis so they are right way up when projected
                    pred_joints_coco_point_est = aa_rotate_translate_points_pytorch3d(points=pred_joints_all_point_est[:, ALL_JOINTS_TO_COCO_MAP, :],
                                                                                      axes=x_axis,
                                                                                      angles=np.pi,
                                                                                      translations=torch.zeros(3, device=device).float())
                    pred_joints2d_coco_point_est = orthographic_project_torch(pred_joints_coco_point_est,
                                                                              pred['cam_wp'])  # (batch_size, 17, 2)

                    with torch.no_grad():
                        pred_tpose_vertices_point_est = smpl_model(body_pose=torch.zeros_like(target_pose)[:, 3:],
                                                                   global_orient=torch.zeros_like(target_pose)[:, :3],
                                                                   betas=pred['shape_mode']).vertices  # (batch_size, 6890, 3)

                    if 'samples' in criterion.loss_cfg.J2D_LOSS_ON:
                        pred_joints_coco_samples = smpl_model(
                            body_pose=pred['pose_rotmats_samples'].reshape(-1, 23, 3, 3),  # (bs * num samples, 23, 3, 3)
                            global_orient=pred['glob_rotmat'][:, None, :, :].expand(-1, humaniflow_cfg.LOSS.NUM_J2D_SAMPLES, -1, -1).reshape(-1, 1, 3, 3),  # (bs * num samples, 1, 3, 3),
                            betas=pred['shape_samples'].reshape(-1, humaniflow_cfg.MODEL.NUM_SMPL_BETAS),  # (bs * num samples, num_smpl_betas)
                            pose2rot=False).joints[:, ALL_JOINTS_TO_COCO_MAP, :]  # (bs * num samples, 17, 3)

                        # Pose targets were flipped such that they are right way up in 3D space - i.e. wrong way up when projected
                        # Need to flip pred_joints_coco_samples 180° about x-axis so they are right way up when projected
                        pred_joints_coco_samples = aa_rotate_translate_points_pytorch3d(points=pred_joints_coco_samples,
                                                                                        axes=x_axis,
                                                                                        angles=np.pi,
                                                                                        translations=torch.zeros(3, device=device).float())

                        pred_joints2d_coco_samples = orthographic_project_torch(pred_joints_coco_samples,
                                                                                pred['cam_wp'][:, None, :].expand(-1, humaniflow_cfg.LOSS.NUM_J2D_SAMPLES, -1).reshape(-1, 3))
                        pred_joints2d_coco_samples = pred_joints2d_coco_samples.reshape(-1,
                                                                                        humaniflow_cfg.LOSS.NUM_J2D_SAMPLES,
                                                                                        pred_joints_coco_samples.shape[1], 2)  # (bs, num samples, 17, 2)

                        if criterion.loss_cfg.J2D_LOSS_ON == 'point_est+samples':
                            pred_joints2d_coco_samples = torch.cat([pred_joints2d_coco_point_est[:, None, :, :],
                                                                    pred_joints2d_coco_samples], dim=1)  # (bs, num samples + 1, 17, 2)
                    else:
                        pred_joints2d_coco_samples = pred_joints2d_coco_point_est[:, None, :, :]  # (batch_size, 1, 17, 2)

                    #############################################################
                    # ----------------- LOSS AND BACKWARD PASS ------------------
                    #############################################################
                    pred_dict_for_loss = {'pose_dist': pred['conditioned_pose_SO3flow_dists_for_loglik'],
                                          'shape_dist': pred['shape_dist_for_loglik'],
                                          'verts3D': pred_vertices_point_est,
                                          'joints3D': pred_joints_h36mlsp_point_est,
                                          'joints2D': pred_joints2d_coco_samples,
                                          'glob_rotmats': pred['glob_rotmat']}

                    target_dict_for_loss = {'pose_params_rotmats': target_pose_rotmats,
                                            'shape_params': target_shape,
                                            'verts3D': target_vertices,
                                            'joints3D': target_joints_h36mlsp,
                                            'joints2D': target_joints2d_coco,
                                            'joints2D_vis': target_joints2d_visib_coco,
                                            'glob_rotmats': target_glob_rotmats}

                    optimiser.zero_grad()
                    loss = criterion(target_dict_for_loss, pred_dict_for_loss)
                    if split == 'train':
                        last_batch_model_state_dict = copy.deepcopy(humaniflow_model.state_dict())
                        last_batch_opt_state_dict = copy.deepcopy(optimiser.state_dict())
                        loss.backward()
                        optimiser.step()
                    for dist in humaniflow_model.pose_SO3flow_dists:
                        dist.clear_cache()

                #############################################################
                # --------------------- TRACK METRICS ----------------------
                #############################################################
                pred_dict_for_loss['joints2D'] = pred_joints2d_coco_point_est
                if criterion.loss_cfg.J2D_LOSS_ON == 'samples':
                    pred_dict_for_loss['joints2Dsamples'] = pred_joints2d_coco_samples
                elif criterion.loss_cfg.J2D_LOSS_ON == 'point_est+samples':
                    pred_dict_for_loss['joints2Dsamples'] = pred_joints2d_coco_samples[:, 1:, :, :]
                del pred_dict_for_loss['pose_dist']
                del pred_dict_for_loss['shape_dist']
                metrics_tracker.update_per_batch(split=split,
                                                 loss=loss,
                                                 pred_dict=pred_dict_for_loss,
                                                 target_dict=target_dict_for_loss,
                                                 batch_size=humaniflow_cfg.TRAIN.BATCH_SIZE,
                                                 pred_tpose_vertices=pred_tpose_vertices_point_est,
                                                 target_tpose_vertices=target_tpose_vertices)

        #############################################################
        # ------------- UPDATE METRICS HISTORY and SAVE -------------
        #############################################################
        metrics_tracker.update_per_epoch()

        save_model_weights_this_epoch = metrics_tracker.determine_save_model_weights_this_epoch(save_val_metrics,
                                                                                                best_epoch_val_metrics)

        if save_model_weights_this_epoch:
            for metric in save_val_metrics:
                best_epoch_val_metrics[metric] = metrics_tracker.epochs_history['val_' + metric][-1]
            print("Best epoch val metrics updated to ", best_epoch_val_metrics)
            best_model_wts = copy.deepcopy(humaniflow_model.state_dict())
            best_epoch = epoch
            print("Best model weights updated!")

        if epoch % humaniflow_cfg.TRAIN.EPOCHS_PER_SAVE == 0:
            save_dict = {'epoch': epoch,
                         'best_epoch': best_epoch,
                         'best_epoch_val_metrics': best_epoch_val_metrics,
                         'model_state_dict': humaniflow_model.state_dict(),
                         'best_model_state_dict': best_model_wts,
                         'optimiser_state_dict': optimiser.state_dict()}
            torch.save(save_dict,
                       os.path.join(model_save_dir, 'epoch_{}'.format(str(epoch).zfill(3)) + '.tar'))
            print('Model saved! Best Val Metrics:\n',
                  best_epoch_val_metrics,
                  '\nin epoch {}'.format(best_epoch))

    print('Training Completed. Best Val Metrics:\n',
          best_epoch_val_metrics)

    humaniflow_model.load_state_dict(best_model_wts)
    return humaniflow_model
