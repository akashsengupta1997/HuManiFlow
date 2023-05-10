import numpy as np
import os

from utils.eval_utils import procrustes_analysis_batch, scale_and_translation_transform_batch
from utils.label_conversions import convert_heatmaps_to_2Djoints_coordinates_torch


class EvalMetricsTracker:
    """
    Tracks metrics during evaluation.
    Point-estimate metrics:
        - PVE (aka MPVPE):
            - Per vertex 3D position error (L2 norm in millimetres) (should really rename this to MPVPE)
        - PVE-T (aka MPVPE-T):
            - Per vertex 3D position error in the T-pose/neutral pose i.e. body shape error (L2 norm in millimetres)
        - MPJPE: 
            - Per joint 3D position error (L2 norm in millimetres)
        - Joints2D L2E: 
            - Per joint 2D position error after projection to image plane (L2 norm in pixels)
            - If the visibility of target 2D joint is provided, this metric will be computed using only visible targets
        - Silhouette IOU:
            - Silhouette intersection over union
    Sample metrics:
        - Distribution Accuracy:
            - PVE/PVE-T/MPJPE Samples Minimum:
                - Minimum 3D position error out of N samples obtained from shape/pose distribution.
        - Sample-Input Consistency
            - Joints2D Samples L2E:
                - Mean per joint 2D position error over N samples obtained from shape/pose distribution.
                    - All 2D samples from predicted 3D distribution should match 2D target joints.
            - Silhouette samples IOU
                - Mean silhouette-IOU over N samples obtained from shape/pose distribution.
                    - All 2D samples from predicted 3D distribution should match 2D target silhouette.
        - Sample Diversity
            - Visible / Invisible Joints3D Diversity:
                - Average Euclidean distance from mean of COCO 3D Joints over N samples.
    """
    def __init__(self,
                 metrics_to_track,
                 save_path=None,
                 save_per_frame_metrics=False,
                 num_samples_for_prob_metrics=None):

        self.metrics_to_track = metrics_to_track
        self.num_samples_for_prob_metrics = num_samples_for_prob_metrics

        self.metric_sums = None
        self.num_total_test_data = 0
        self.save_per_frame_metrics = save_per_frame_metrics
        self.save_path = save_path
        print('\nInitialised metrics tracker.')

    def initialise_metric_sums(self):
        self.metric_sums = {}
        for metric_type in self.metrics_to_track:
            if metric_type == 'silhouette-IOU':
                self.metric_sums['num_true_positives'] = 0.
                self.metric_sums['num_false_positives'] = 0.
                self.metric_sums['num_true_negatives'] = 0.
                self.metric_sums['num_false_negatives'] = 0.
            elif metric_type == 'silhouettesamples-IOU':
                self.metric_sums['num_samples_true_positives'] = 0.
                self.metric_sums['num_samples_false_positives'] = 0.
                self.metric_sums['num_samples_true_negatives'] = 0.
                self.metric_sums['num_samples_false_negatives'] = 0.

            elif metric_type == 'joints2D-L2E':
                self.metric_sums['num_vis_joints2D'] = 0.
                self.metric_sums[metric_type] = 0.
            elif metric_type == 'joints2Dsamples-L2E':
                self.metric_sums['num_vis_joints2Dsamples'] = 0.
                self.metric_sums[metric_type] = 0.
            elif metric_type == 'input_joints2D-L2E':
                self.metric_sums['num_vis_input_joints2D'] = 0.
                self.metric_sums[metric_type] = 0.
            elif metric_type == 'input_joints2Dsamples-L2E':
                self.metric_sums['num_vis_input_joints2Dsamples'] = 0.
                self.metric_sums[metric_type] = 0.

            elif metric_type == 'joints3D_invis_sample_diversity':
                self.metric_sums['num_invis_joints3Dsamples'] = 0
                self.metric_sums[metric_type] = 0.
            elif metric_type == 'joints3D_vis_sample_diversity':
                self.metric_sums['num_vis_joints3Dsamples'] = 0
                self.metric_sums[metric_type] = 0.

            else:
                self.metric_sums[metric_type] = 0.

    def initialise_per_frame_metric_lists(self):
        self.per_frame_metrics = {}
        for metric_type in self.metrics_to_track:
            self.per_frame_metrics[metric_type] = []

    def update_per_batch(self,
                         pred_dict,
                         target_dict,
                         batch_size,
                         model_input=None,
                         return_per_frame_metrics=False):
        self.num_total_test_data += batch_size

        if return_per_frame_metrics:
            per_frame_metrics_return_dict = {}
        else:
            per_frame_metrics_return_dict = None

        if model_input is not None:
            input_joints2D_coco, input_joints2D_vis_coco = convert_heatmaps_to_2Djoints_coordinates_torch(joints2D_heatmaps=model_input[:, 1:, :, :],
                                                                                                          eps=1e-6,
                                                                                                          gaussian_heatmaps=True)
            input_joints2D_coco = input_joints2D_coco.cpu().detach().numpy()  # (bsize, 17, 2)
            input_joints2D_vis_coco = input_joints2D_vis_coco.cpu().detach().numpy()  # (bsize, 17)

        ##################################################################################################################
        # --------------------------------------------- Update metrics sums ---------------------------------------------
        ##################################################################################################################

        # --------------------------------- 3D Point Estimate Metrics ---------------------------------
        if 'PVE' in self.metrics_to_track:
            pve_batch = np.linalg.norm(pred_dict['verts3D'] - target_dict['verts3D'], axis=-1)  # (bsize, 6890)
            self.metric_sums['PVE'] += np.sum(pve_batch)  # scalar
            self.per_frame_metrics['PVE'].append(np.mean(pve_batch, axis=-1))    # (bsize,)
            if return_per_frame_metrics:
                per_frame_metrics_return_dict['PVE'] = np.mean(pve_batch, axis=-1)

        # Scale and translation correction
        if 'PVE-SC' in self.metrics_to_track:
            pred_vertices = pred_dict['verts3D']  # (bsize, 6890, 3)
            target_vertices = target_dict['verts3D']  # (bsize, 6890, 3)
            pred_vertices_sc = scale_and_translation_transform_batch(pred_vertices, target_vertices)
            pve_sc_batch = np.linalg.norm(pred_vertices_sc - target_vertices, axis=-1)    # (bsize, 6890)
            self.metric_sums['PVE-SC'] += np.sum(pve_sc_batch)  # scalar
            self.per_frame_metrics['PVE-SC'].append(np.mean(pve_sc_batch, axis=-1))    # (bsize,)
            if return_per_frame_metrics:
                per_frame_metrics_return_dict['PVE-SC'] = np.mean(pve_sc_batch, axis=-1)

        # Procrustes analysis
        if 'PVE-PA' in self.metrics_to_track:
            pred_vertices = pred_dict['verts3D']  # (bsize, 6890, 3)
            target_vertices = target_dict['verts3D']  # (bsize, 6890, 3)
            pred_vertices_pa = procrustes_analysis_batch(pred_vertices, target_vertices)
            pve_pa_batch = np.linalg.norm(pred_vertices_pa - target_vertices, axis=-1)  # (bsize, 6890)
            self.metric_sums['PVE-PA'] += np.sum(pve_pa_batch)  # scalar
            self.per_frame_metrics['PVE-PA'].append(np.mean(pve_pa_batch, axis=-1))    # (bsize,)
            if return_per_frame_metrics:
                per_frame_metrics_return_dict['PVE-PA'] = np.mean(pve_pa_batch, axis=-1)

        # T-Pose
        if 'PVE-T' in self.metrics_to_track:
            pvet_batch = np.linalg.norm(pred_dict['tpose_verts3D'] - target_dict['tpose_verts3D'], axis=-1)  # (bsize, 6890)
            self.metric_sums['PVE-T'] += np.sum(pvet_batch)  # scalar
            self.per_frame_metrics['PVE-T'].append(np.mean(pvet_batch, axis=-1))    # (bsize,)
            if return_per_frame_metrics:
                per_frame_metrics_return_dict['PVE-T'] = np.mean(pvet_batch, axis=-1)

        # T-Pose + Scale and translation correction
        if 'PVE-T-SC' in self.metrics_to_track:
            pred_tpose_vertices = pred_dict['tpose_verts3D']  # (bsize, 6890, 3)
            target_tpose_vertices = target_dict['tpose_verts3D']  # (bsize, 6890, 3)
            pred_tpose_vertices_sc = scale_and_translation_transform_batch(pred_tpose_vertices,
                                                                             target_tpose_vertices)
            pvet_sc_batch = np.linalg.norm(pred_tpose_vertices_sc - target_tpose_vertices, axis=-1)    # (bsize, 6890)
            self.metric_sums['PVE-T-SC'] += np.sum(pvet_sc_batch)  # scalar
            self.per_frame_metrics['PVE-T-SC'].append(np.mean(pvet_sc_batch, axis=-1))    # (bsize,)
            if return_per_frame_metrics:
                per_frame_metrics_return_dict['PVE-T-SC'] = np.mean(pvet_sc_batch, axis=-1)

        if 'MPJPE' in self.metrics_to_track:
            mpjpe_batch = np.linalg.norm(pred_dict['joints3D'] - target_dict['joints3D'], axis=-1)  # (bsize, 14)
            self.metric_sums['MPJPE'] += np.sum(mpjpe_batch)  # scalar
            self.per_frame_metrics['MPJPE'].append(np.mean(mpjpe_batch, axis=-1))    # (bsize,)
            if return_per_frame_metrics:
                per_frame_metrics_return_dict['MPJPE'] = np.mean(mpjpe_batch, axis=-1)

        # Scale and translation correction
        if 'MPJPE-SC' in self.metrics_to_track:
            pred_joints3D_h36mlsp = pred_dict['joints3D']  # (bsize, 14, 3)
            target_joints3D_h36mlsp = target_dict['joints3D']  # (bsize, 14, 3)
            pred_joints3D_h36mlsp_sc = scale_and_translation_transform_batch(pred_joints3D_h36mlsp,
                                                                             target_joints3D_h36mlsp)
            mpjpe_sc_batch = np.linalg.norm(pred_joints3D_h36mlsp_sc - target_joints3D_h36mlsp, axis=-1)  # (bsize, 14)
            self.metric_sums['MPJPE-SC'] += np.sum(mpjpe_sc_batch)  # scalar
            self.per_frame_metrics['MPJPE-SC'].append(np.mean(mpjpe_sc_batch, axis=-1))    # (bsize,)
            if return_per_frame_metrics:
                per_frame_metrics_return_dict['MPJPE-SC'] = np.mean(mpjpe_sc_batch, axis=-1)

        # Procrustes analysis
        if 'MPJPE-PA' in self.metrics_to_track:
            pred_joints3D_h36mlsp = pred_dict['joints3D']  # (bsize, 14, 3)
            target_joints3D_h36mlsp = target_dict['joints3D']  # (bsize, 14, 3)
            pred_joints3D_h36mlsp_pa = procrustes_analysis_batch(pred_joints3D_h36mlsp,
                                                                 target_joints3D_h36mlsp)
            mpjpe_pa_batch = np.linalg.norm(pred_joints3D_h36mlsp_pa - target_joints3D_h36mlsp, axis=-1)  # (bsize, 14)
            self.metric_sums['MPJPE-PA'] += np.sum(mpjpe_pa_batch)  # scalar
            self.per_frame_metrics['MPJPE-PA'].append(np.mean(mpjpe_pa_batch, axis=-1))    # (bsize,)
            if return_per_frame_metrics:
                per_frame_metrics_return_dict['MPJPE-PA'] = np.mean(mpjpe_pa_batch, axis=-1)

        # --------------------------------- 3D Minimum Sample Metrics (Distribution Accuracy) ---------------------------------
        if 'PVE_samples_min' in self.metrics_to_track:
            pve_per_sample = np.linalg.norm(pred_dict['verts3D_samples'] - target_dict['verts3D'][:, None, :, :], axis=-1)  # (bsize, num samples, 6890)
            min_pve_sample = np.argmin(np.mean(pve_per_sample, axis=-1), axis=-1)  # (bsize,)
            pve_samples_min_batch = pve_per_sample[np.arange(pve_per_sample.shape[0]), min_pve_sample, :]  # (bsize, 6890)
            self.metric_sums['PVE_samples_min'] += np.sum(pve_samples_min_batch)  # scalar
            self.per_frame_metrics['PVE_samples_min'].append(np.mean(pve_samples_min_batch, axis=-1))  # (bsize,)

        # Scale and translation correction
        if 'PVE-SC_samples_min' in self.metrics_to_track:
            pred_vertices_samples = pred_dict['verts3D_samples']  # (bsize, num samples, 6890, 3)
            target_vertices = np.tile(target_dict['verts3D'][:, None, :, :], (1, pred_vertices_samples.shape[1], 1, 1))  # (bsize, num samples, 6890, 3)
            pred_vertices_samples_sc = scale_and_translation_transform_batch(pred_vertices_samples, target_vertices)
            pve_sc_per_sample = np.linalg.norm(pred_vertices_samples_sc - target_vertices, axis=-1)  # (bsize, num samples, 6890)
            min_pve_sc_sample = np.argmin(np.mean(pve_sc_per_sample, axis=-1), axis=-1)  # (bsize,)
            pve_sc_samples_min_batch = pve_sc_per_sample[np.arange(pve_sc_per_sample.shape[0]), min_pve_sc_sample, :]  # (bsize, 6890)
            self.metric_sums['PVE-SC_samples_min'] += np.sum(pve_sc_samples_min_batch)  # scalar
            self.per_frame_metrics['PVE-SC_samples_min'].append(np.mean(pve_sc_samples_min_batch, axis=-1))  # (bsize,)

        # Procrustes analysis
        if 'PVE-PA_samples_min' in self.metrics_to_track:
            pred_vertices_samples = pred_dict['verts3D_samples']  # (bsize, num samples, 6890, 3)
            target_vertices = np.tile(target_dict['verts3D'][:, None, :, :], (1, pred_vertices_samples.shape[1], 1, 1))  # (bsize, num samples, 6890, 3)
            pred_vertices_samples_pa = procrustes_analysis_batch(pred_vertices_samples.reshape(-1, *pred_vertices_samples.shape[2:]),
                                                                 target_vertices.reshape(-1, *target_vertices.shape[2:])).reshape(*pred_vertices_samples.shape)
            pve_pa_per_sample = np.linalg.norm(pred_vertices_samples_pa - target_vertices, axis=-1)  # (bsize, num samples, 6890)
            min_pve_pa_sample = np.argmin(np.mean(pve_pa_per_sample, axis=-1), axis=-1)  # (bsize,)
            pve_pa_samples_min_batch = pve_pa_per_sample[np.arange(pve_pa_per_sample.shape[0]), min_pve_pa_sample, :]  # (bsize, 6890)
            self.metric_sums['PVE-PA_samples_min'] += np.sum(pve_pa_samples_min_batch)  # scalar
            self.per_frame_metrics['PVE-PA_samples_min'].append(np.mean(pve_pa_samples_min_batch, axis=-1))  # (bsize)

        # T-Pose
        if 'PVE-T_samples_min' in self.metrics_to_track:
            pvet_per_sample = np.linalg.norm(pred_dict['tpose_verts3D_samples'] - target_dict['tpose_verts3D'][:, None, :, :], axis=-1)  # (bsize, num samples, 6890)
            min_pvet_sample = np.argmin(np.mean(pvet_per_sample, axis=-1), axis=-1)  # (bsize,)
            pvet_samples_min_batch = pvet_per_sample[np.arange(pvet_per_sample.shape[0]), min_pvet_sample, :]  # (bsize, 6890)
            self.metric_sums['PVE-T_samples_min'] += np.sum(pvet_samples_min_batch)  # scalar
            self.per_frame_metrics['PVE-T_samples_min'].append(np.mean(pvet_samples_min_batch, axis=-1))  # (bsize)

        # T-pose + Scale and translation correction
        if 'PVE-T-SC_samples_min' in self.metrics_to_track:
            pred_tpose_vertices_samples = pred_dict['tpose_verts3D_samples']  # (bsize, num samples, 6890, 3)
            target_tpose_vertices = np.tile(target_dict['tpose_verts3D'][:, None, :, :], (1, pred_tpose_vertices_samples.shape[1], 1, 1))  # (bsize, num samples, 6890, 3)
            pred_tpose_vertices_samples_sc = scale_and_translation_transform_batch(pred_tpose_vertices_samples,
                                                                                     target_tpose_vertices)
            pvet_sc_per_sample = np.linalg.norm(pred_tpose_vertices_samples_sc - target_tpose_vertices, axis=-1)  # (bsize, num samples, 6890)
            min_pvet_sc_sample = np.argmin(np.mean(pvet_sc_per_sample, axis=-1), axis=-1)  # (bsize,)
            pvet_sc_samples_min_batch = pvet_sc_per_sample[np.arange(pvet_sc_per_sample.shape[0]), min_pvet_sc_sample, :]  # (bsize, 6890)
            self.metric_sums['PVE-T-SC_samples_min'] += np.sum(pvet_sc_samples_min_batch)  # scalar
            self.per_frame_metrics['PVE-T-SC_samples_min'].append(np.mean(pvet_sc_samples_min_batch, axis=-1))  # (bsize)

        if 'MPJPE_samples_min' in self.metrics_to_track:
            mpjpe_per_sample = np.linalg.norm(pred_dict['joints3D_samples'] - target_dict['joints3D'][:, None, :, :], axis=-1)  # (num samples, 14)
            min_mpjpe_sample = np.argmin(np.mean(mpjpe_per_sample, axis=-1), axis=-1)  # (bsize,)
            mpjpe_samples_min_batch = mpjpe_per_sample[np.arange(min_mpjpe_sample.shape[0]), min_mpjpe_sample, :]  # (bsize, 14)
            self.metric_sums['MPJPE_samples_min'] += np.sum(mpjpe_samples_min_batch)  # scalar
            self.per_frame_metrics['MPJPE_samples_min'].append(np.mean(mpjpe_samples_min_batch, axis=-1))  # (bsize)

        # Scale and translation correction
        if 'MPJPE-SC_samples_min' in self.metrics_to_track:
            pred_joints3D_h36mlsp_samples = pred_dict['joints3D_samples']  # (bsize, num samples, 14, 3)
            target_joints3D_h36mlsp = np.tile(target_dict['joints3D'][:, None, :, :], (1, pred_joints3D_h36mlsp_samples.shape[1], 1, 1))  # (bsize, num samples, 14, 3)
            pred_joints3D_h36mlsp_sc = scale_and_translation_transform_batch(pred_joints3D_h36mlsp_samples,
                                                                             target_joints3D_h36mlsp)
            mpjpe_sc_per_sample = np.linalg.norm(pred_joints3D_h36mlsp_sc - target_joints3D_h36mlsp, axis=-1)  # (bsize, num samples, 14)
            min_mpjpe_sc_sample = np.argmin(np.mean(mpjpe_sc_per_sample, axis=-1), axis=-1)  # (bsize,)
            mpjpe_sc_samples_min_batch = mpjpe_sc_per_sample[np.arange(mpjpe_sc_per_sample.shape[0]), min_mpjpe_sc_sample, :]  # (bsize, 14)
            self.metric_sums['MPJPE-SC_samples_min'] += np.sum(mpjpe_sc_samples_min_batch)  # scalar
            self.per_frame_metrics['MPJPE-SC_samples_min'].append(np.mean(mpjpe_sc_samples_min_batch, axis=-1))  # (bsize)

        # Procrustes analysis
        if 'MPJPE-PA_samples_min' in self.metrics_to_track:
            pred_joints3D_h36mlsp_samples = pred_dict['joints3D_samples']  # (bsize, num samples, 14, 3)
            target_joints3D_h36mlsp = np.tile(target_dict['joints3D'][:, None, :, :], (1, pred_joints3D_h36mlsp_samples.shape[1], 1, 1))  # (bsize, num samples, 14, 3)
            pred_joints3D_h36mlsp_pa = procrustes_analysis_batch(pred_joints3D_h36mlsp_samples.reshape(-1, *pred_joints3D_h36mlsp_samples.shape[2:]),
                                                                 target_joints3D_h36mlsp.reshape(-1, *target_joints3D_h36mlsp.shape[2:])).reshape(*pred_joints3D_h36mlsp_samples.shape)
            mpjpe_pa_per_sample = np.linalg.norm(pred_joints3D_h36mlsp_pa - target_joints3D_h36mlsp, axis=-1)  # (bsize, num samples, 14)
            min_mpjpe_pa_sample = np.argmin(np.mean(mpjpe_pa_per_sample, axis=-1), axis=-1)  # (bsize,)
            mpjpe_pa_samples_min_batch = mpjpe_pa_per_sample[np.arange(mpjpe_pa_per_sample.shape[0]), min_mpjpe_pa_sample, :]  # (bsize, 14)
            self.metric_sums['MPJPE-PA_samples_min'] += np.sum(mpjpe_pa_samples_min_batch)  # scalar
            self.per_frame_metrics['MPJPE-PA_samples_min'].append(np.mean(mpjpe_pa_samples_min_batch, axis=-1))  # (bsize)

        # --------------------------------- 2D Point-Estimate Metrics ---------------------------------

        if 'joints2D-L2E' in self.metrics_to_track:
            pred_joints2D_coco = pred_dict['joints2D']  # (bsize, 17, 2)
            target_joints2D_coco = target_dict['joints2D']  # (bsize, 17, 2)
            joints2D_l2e_batch = np.linalg.norm(pred_joints2D_coco - target_joints2D_coco, axis=-1)  # (bsize, 17)

            if 'joints2D_vis' in target_dict.keys():
                target_joints2D_vis_coco = target_dict['joints2D_vis']  # (bsize, 17)
                joints2D_l2e_batch = joints2D_l2e_batch * target_joints2D_vis_coco  # (bsize, 17)  masking out invisible joints2D
                self.metric_sums['num_vis_joints2D'] += target_joints2D_vis_coco.sum()
                per_frame_joints2D_l2e = np.sum(joints2D_l2e_batch, axis=-1) / target_joints2D_vis_coco.sum(axis=-1)  # (bsize,)
            else:
                self.metric_sums['num_vis_joints2D'] += joints2D_l2e_batch.size  # Adding bsize * 17 to num visible joints2D
                per_frame_joints2D_l2e = np.mean(joints2D_l2e_batch, axis=-1)  # (bsize,)

            self.metric_sums['joints2D-L2E'] += np.sum(joints2D_l2e_batch)  # scalar
            self.per_frame_metrics['joints2D-L2E'].append(per_frame_joints2D_l2e)
            if return_per_frame_metrics:
                per_frame_metrics_return_dict['joints2D-L2E'] = per_frame_joints2D_l2e

        # Using input 2D joints (from HRNet) as target, rather than GT labels
        if 'input_joints2D-L2E' in self.metrics_to_track:
            assert model_input is not None
            pred_joints2D_coco = pred_dict['joints2D']  # (bsize, 17, 2)

            input_joints2D_l2e_batch = np.linalg.norm(pred_joints2D_coco - input_joints2D_coco, axis=-1)  # (bsize, 17)
            input_joints2D_l2e_batch = input_joints2D_l2e_batch * input_joints2D_vis_coco  # (bsize, 17)  masking out invisible joints2D
            per_frame_input_joints2D_l2e = np.sum(input_joints2D_l2e_batch, axis=-1) / input_joints2D_vis_coco.sum(axis=-1)  # (bsize,)

            self.metric_sums['input_joints2D-L2E'] += np.sum(input_joints2D_l2e_batch)  # scalar
            self.metric_sums['num_vis_input_joints2D'] += input_joints2D_vis_coco.sum()
            self.per_frame_metrics['input_joints2D-L2E'].append(per_frame_input_joints2D_l2e)  # (bsize,)
            if return_per_frame_metrics:
                per_frame_metrics_return_dict['input_joints2D-L2E'] = per_frame_input_joints2D_l2e

        if 'silhouette-IOU' in self.metrics_to_track:
            pred_silhouettes = pred_dict['silhouettes']  # (bsize, img_wh, img_wh)
            target_silhouettes = target_dict['silhouettes']  # (bsize, img_wh, img_wh)
            true_positive = np.logical_and(pred_silhouettes, target_silhouettes)
            false_positive = np.logical_and(pred_silhouettes, np.logical_not(target_silhouettes))
            true_negative = np.logical_and(np.logical_not(pred_silhouettes), np.logical_not(target_silhouettes))
            false_negative = np.logical_and(np.logical_not(pred_silhouettes), target_silhouettes)
            num_tp = np.sum(true_positive, axis=(1, 2))  # (bsize,)
            num_fp = np.sum(false_positive, axis=(1, 2))
            num_tn = np.sum(true_negative, axis=(1, 2))
            num_fn = np.sum(false_negative, axis=(1, 2))
            self.metric_sums['num_true_positives'] += np.sum(num_tp)  # scalar
            self.metric_sums['num_false_positives'] += np.sum(num_fp)
            self.metric_sums['num_true_negatives'] += np.sum(num_tn)
            self.metric_sums['num_false_negatives'] += np.sum(num_fn)
            iou_per_frame = num_tp / (num_tp + num_fp + num_fn)  # (bsize,)
            self.per_frame_metrics['silhouette-IOU'].append(iou_per_frame)
            if return_per_frame_metrics:
                per_frame_metrics_return_dict['silhouette-IOU'] = iou_per_frame

        # ---------------------------- 2D Sample Average Metrics (Sample-Input Consistency) ----------------------------

        if 'joints2Dsamples-L2E' in self.metrics_to_track:
            pred_joints2D_coco_samples = pred_dict['joints2Dsamples']  # (bsize, num_samples, 17, 2)
            target_joints2D_coco = np.tile(target_dict['joints2D'][:, None, :, :], (1, pred_joints2D_coco_samples.shape[1], 1, 1))  # (bsize, num_samples, 17, 2)
            num_samples = pred_joints2D_coco_samples.shape[1]

            joints2Dsamples_l2e_batch = np.linalg.norm(pred_joints2D_coco_samples - target_joints2D_coco, axis=-1)    # (bsize, num_samples, 17)

            if 'joints2D_vis' in target_dict.keys():
                target_joints2D_vis_coco = target_dict['joints2D_vis']    # (bsize, 17)
                joints2Dsamples_l2e_batch = joints2Dsamples_l2e_batch * target_joints2D_vis_coco[:, None, :]    # (bsize, num_samples, 17) masking out invisible joints2D
                self.metric_sums['num_vis_joints2Dsamples'] += target_joints2D_vis_coco.sum() * num_samples
                per_frame_joints2Dsamples_l2e = np.sum(joints2Dsamples_l2e_batch, axis=(1, 2)) / (target_joints2D_vis_coco.sum(axis=-1) * num_samples)    # (bsize,)
            else:
                self.metric_sums['num_vis_joints2Dsamples'] += joints2Dsamples_l2e_batch.size  # Adding bsize * num_samples * 17 to num visible joints2D samples
                per_frame_joints2Dsamples_l2e = np.mean(joints2Dsamples_l2e_batch, axis=(1, 2))    # (bsize,)

            self.metric_sums['joints2Dsamples-L2E'] += np.sum(joints2Dsamples_l2e_batch)  # scalar
            self.per_frame_metrics['joints2Dsamples-L2E'].append(per_frame_joints2Dsamples_l2e)
            if return_per_frame_metrics:
                per_frame_metrics_return_dict['joints2Dsamples-L2E'] = per_frame_joints2Dsamples_l2e

        # Using input 2D joints (from HRNet) as target, rather than GT labels
        if 'input_joints2Dsamples-L2E' in self.metrics_to_track:
            assert model_input is not None
            pred_joints2D_coco_samples = pred_dict['joints2Dsamples']    # (bsize, num_samples, 17, 2)
            num_samples = pred_joints2D_coco_samples.shape[1]

            input_joints2Dsamples_l2e_batch = np.linalg.norm(pred_joints2D_coco_samples - input_joints2D_coco[:, None, :, :], axis=-1)    # (bsize, num_samples, 17)
            input_joints2Dsamples_l2e_batch = input_joints2Dsamples_l2e_batch * input_joints2D_vis_coco[:, None, :]    # (bsize, num_samples, 17)  masking out invisible joints2D
            per_frame_input_joints2Dsamples_l2e = np.sum(input_joints2Dsamples_l2e_batch, axis=(1, 2)) / (input_joints2D_vis_coco.sum(axis=-1) * num_samples)    # (bsize,)

            self.metric_sums['num_vis_input_joints2Dsamples'] += input_joints2D_vis_coco.sum() * num_samples
            self.metric_sums['input_joints2Dsamples-L2E'] += np.sum(input_joints2Dsamples_l2e_batch)  # scalar
            self.per_frame_metrics['input_joints2Dsamples-L2E'].append(per_frame_input_joints2Dsamples_l2e)
            if return_per_frame_metrics:
                per_frame_metrics_return_dict['input_joints2Dsamples-L2E'] = per_frame_input_joints2Dsamples_l2e

        if 'silhouettesamples-IOU' in self.metrics_to_track:
            pred_silhouettes_samples = pred_dict['silhouettessamples']  # (bsize, num_samples, img_wh, img_wh)
            target_silhouettes = np.tile(target_dict['silhouettes'][:, None, :, :], (1, pred_silhouettes_samples.shape[1], 1, 1))  # (bsize, num_samples, img_wh, img_wh)
            true_positive = np.logical_and(pred_silhouettes_samples, target_silhouettes)
            false_positive = np.logical_and(pred_silhouettes_samples, np.logical_not(target_silhouettes))
            true_negative = np.logical_and(np.logical_not(pred_silhouettes_samples), np.logical_not(target_silhouettes))
            false_negative = np.logical_and(np.logical_not(pred_silhouettes_samples), target_silhouettes)
            num_tp = np.sum(true_positive, axis=(1, 2, 3))  # (bsize,)
            num_fp = np.sum(false_positive, axis=(1, 2, 3))
            num_tn = np.sum(true_negative, axis=(1, 2, 3))
            num_fn = np.sum(false_negative, axis=(1, 2, 3))
            self.metric_sums['num_samples_true_positives'] += np.sum(num_tp)  # scalar
            self.metric_sums['num_samples_false_positives'] += np.sum(num_fp)
            self.metric_sums['num_samples_true_negatives'] += np.sum(num_tn)
            self.metric_sums['num_samples_false_negatives'] += np.sum(num_fn)
            sample_iou_per_frame = num_tp / (num_tp + num_fp + num_fn)
            self.per_frame_metrics['silhouettesamples-IOU'].append(sample_iou_per_frame)  # (bsize,)
            if return_per_frame_metrics:
                per_frame_metrics_return_dict['silhouettesamples-IOU'] = sample_iou_per_frame

        # --------------------------------- 3D Sample Diversity Metrics ---------------------------------
        if 'verts3D_sample_diversity' in self.metrics_to_track:
            assert self.num_samples_for_prob_metrics is not None, "Need to specify number of samples for diversity metrics"
            verts_samples_mean = pred_dict['verts3D_samples'].mean(axis=1)  # (bsize, 6890, 3)
            verts3D_sample_diversity = np.linalg.norm(pred_dict['verts3D_samples'] - verts_samples_mean[:, None, :, :], axis=-1)  # (bsize, num samples, 6890)
            self.metric_sums['verts3D_sample_diversity'] += verts3D_sample_diversity.sum()
            self.per_frame_metrics['verts3D_sample_diversity'].append(verts3D_sample_diversity.mean(axis=(1, 2)))  # (bsize,)

        if 'joints3D_sample_diversity' in self.metrics_to_track:
            assert self.num_samples_for_prob_metrics is not None, "Need to specify number of samples for diversity metrics"
            joints3D_samples_mean = pred_dict['joints3D_coco_samples'].mean(axis=1)  # (bsize, 17, 3)
            joints3D_samples_dist_from_mean = np.linalg.norm(pred_dict['joints3D_coco_samples'] - joints3D_samples_mean[:, None, :, :], axis=-1)  # (bsize, num samples, 17)
            self.metric_sums['joints3D_sample_diversity'] += joints3D_samples_dist_from_mean.sum()  # scalar
            self.per_frame_metrics['joints3D_sample_diversity'].append(joints3D_samples_dist_from_mean.mean(axis=(1, 2)))  # (bsize,)

        if 'joints3D_invis_sample_diversity' in self.metrics_to_track:
            # (In)visibility of specific joints determined by input joint heatmaps obtained using HRNet
            assert model_input is not None, "Need to pass model input"
            assert self.num_samples_for_prob_metrics is not None, "Need to specify number of samples for diversity metrics"
            assert 'joints3D_sample_diversity' in self.metrics_to_track

            input_joints2D_invis_coco = np.logical_not(input_joints2D_vis_coco)  # (bsize, 17)
            joints3D_invis_samples_dist_from_mean = joints3D_samples_dist_from_mean * input_joints2D_invis_coco[:, None, :]   # (bsize, num samples, 17)
            self.metric_sums['joints3D_invis_sample_diversity'] += joints3D_invis_samples_dist_from_mean.sum()  # scalar
            self.metric_sums['num_invis_joints3Dsamples'] += input_joints2D_invis_coco.sum() * self.num_samples_for_prob_metrics
            self.per_frame_metrics['joints3D_invis_sample_diversity'].append(joints3D_invis_samples_dist_from_mean.mean(axis=(1, 2)))  # (bsize,)

        if 'joints3D_vis_sample_diversity' in self.metrics_to_track:
            # Visibility of specific joints determined by model input joint heatmaps
            assert model_input is not None, "Need to pass model input"
            assert self.num_samples_for_prob_metrics is not None, "Need to specify number of samples for diversity metrics"
            assert 'joints3D_sample_diversity' in self.metrics_to_track

            joints3D_vis_samples_dist_from_mean = joints3D_samples_dist_from_mean * input_joints2D_vis_coco[:, None, :]   # (bsize, num samples, 17)
            self.metric_sums['joints3D_vis_sample_diversity'] += joints3D_vis_samples_dist_from_mean.sum()  # scalar
            self.metric_sums['num_vis_joints3Dsamples'] += input_joints2D_vis_coco.sum() * self.num_samples_for_prob_metrics
            self.per_frame_metrics['joints3D_vis_sample_diversity'].append(joints3D_vis_samples_dist_from_mean.mean(axis=(1, 2)))  # (bsize,)

        return per_frame_metrics_return_dict

    def compute_final_metrics(self):
        final_metrics = {}
        for metric_type in self.metrics_to_track:
            mult = 1.
            if metric_type == 'silhouette-IOU':
                iou = self.metric_sums['num_true_positives'] / \
                      (self.metric_sums['num_true_positives'] +
                       self.metric_sums['num_false_negatives'] +
                       self.metric_sums['num_false_positives'])
                final_metrics['silhouette-IOU'] = iou
            elif metric_type == 'silhouettesamples-IOU':
                iou = self.metric_sums['num_samples_true_positives'] / \
                      (self.metric_sums['num_samples_true_positives'] +
                       self.metric_sums['num_samples_false_negatives'] +
                       self.metric_sums['num_samples_false_positives'])
                final_metrics['silhouettesamples-IOU'] = iou

            elif metric_type == 'joints2D-L2E':
                joints2D_l2e = self.metric_sums['joints2D-L2E'] / self.metric_sums['num_vis_joints2D']
                final_metrics[metric_type] = joints2D_l2e
            elif metric_type == 'joints2Dsamples-L2E':
                joints2Dsamples_l2e = self.metric_sums['joints2Dsamples-L2E'] / self.metric_sums['num_vis_joints2Dsamples']
                final_metrics[metric_type] = joints2Dsamples_l2e
            elif metric_type == 'input_joints2D-L2E':
                joints2D_l2e = self.metric_sums['input_joints2D-L2E'] / self.metric_sums['num_vis_input_joints2D']
                final_metrics[metric_type] = joints2D_l2e
            elif metric_type == 'input_joints2Dsamples-L2E':
                joints2Dsamples_l2e = self.metric_sums['input_joints2Dsamples-L2E'] / self.metric_sums['num_vis_input_joints2Dsamples']
                final_metrics[metric_type] = joints2Dsamples_l2e

            elif metric_type == 'verts3D_sample_diversity':
                mult = 1000.  # mult used to convert 3D metrics from metres to millimetres
                final_metrics[metric_type] = self.metric_sums[metric_type] / (self.num_total_test_data * self.num_samples_for_prob_metrics * 6890)
            elif metric_type == 'joints3D_sample_diversity':
                mult = 1000.  # mult used to convert 3D metrics from metres to millimetres
                final_metrics[metric_type] = self.metric_sums[metric_type] / (self.num_total_test_data * self.num_samples_for_prob_metrics * 17)
            elif metric_type == 'joints3D_invis_sample_diversity':
                mult = 1000.  # mult used to convert 3D metrics from metres to millimetres
                if self.metric_sums['num_invis_joints3Dsamples'] > 0:
                    final_metrics[metric_type] = self.metric_sums[metric_type] / self.metric_sums['num_invis_joints3Dsamples']
                else:
                    print('No invisible 3D COCO joints!')
                    final_metrics[metric_type] = 0.
            elif metric_type == 'joints3D_vis_sample_diversity':
                mult = 1000.  # mult used to convert 3D metrics from metres to millimetres
                final_metrics[metric_type] = self.metric_sums[metric_type] / self.metric_sums['num_vis_joints3Dsamples']

            else:
                if 'PVE' in metric_type:
                    num_per_sample = 6890
                    mult = 1000.  # mult used to convert 3D metrics from metres to millimetres
                elif 'MPJPE' in metric_type:
                    num_per_sample = 14
                    mult = 1000.
                elif 'joints2D' in metric_type:
                    print('NEVER GETS HERE')
                    num_per_sample = 17
                final_metrics[metric_type] = self.metric_sums[metric_type] / (self.num_total_test_data * num_per_sample)

            print(metric_type, '{:.2f}'.format(final_metrics[metric_type] * mult))

        if self.save_per_frame_metrics:
            for metric_type in self.metrics_to_track:
                per_frame = np.concatenate(self.per_frame_metrics[metric_type], axis=0)
                np.save(os.path.join(self.save_path, metric_type+'_per_frame.npy'), per_frame)
                # print(f'Saved {metric_type}: {per_frame.shape}')