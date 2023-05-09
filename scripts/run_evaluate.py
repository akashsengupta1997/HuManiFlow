import os
import numpy as np
import torch
import argparse
import sys
sys.path.append('.')

from configs.humaniflow_config import get_humaniflow_cfg_defaults
from configs import paths

from data.pw3d_eval_dataset import PW3DEvalDataset
from data.ssp3d_eval_dataset import SSP3DEvalDataset

from models.humaniflow_model import HumaniflowModel
from models.smpl import SMPL
from models.canny_edge_detector import CannyEdgeDetector

from evaluate.evaluate_humaniflow import evaluate_humaniflow


def run_evaluate(device,
                 dataset_name,
                 humaniflow_weights_path,
                 humaniflow_cfg_path=None,
                 save_path=None,
                 batch_size=1,
                 num_samples_for_metrics=10):

    # ------------------ Models ------------------
    # Config
    humaniflow_cfg = get_humaniflow_cfg_defaults()
    if humaniflow_cfg_path is not None:
        humaniflow_cfg.merge_from_file(humaniflow_cfg_path)
        print('\nLoaded Distribution Predictor config from', humaniflow_cfg_path)
    else:
        print('\nUsing default Distribution Predictor config.')

    # Edge detector
    edge_detect_model = CannyEdgeDetector(non_max_suppression=humaniflow_cfg.DATA.EDGE_NMS,
                                          gaussian_filter_std=humaniflow_cfg.DATA.EDGE_GAUSSIAN_STD,
                                          gaussian_filter_size=humaniflow_cfg.DATA.EDGE_GAUSSIAN_SIZE,
                                          threshold=humaniflow_cfg.DATA.EDGE_THRESHOLD).to(device)

    # SMPL neutral/male/female models
    smpl_model_neutral = SMPL(paths.SMPL,
                              batch_size=batch_size,
                              num_betas=humaniflow_cfg.MODEL.NUM_SMPL_BETAS,
                              create_transl=False).to(device)
    smpl_immediate_parents = smpl_model_neutral.parents.tolist()
    smpl_model_male = SMPL(paths.SMPL,
                           batch_size=batch_size,
                           gender='male',
                           create_transl=False).to(device)
    smpl_model_female = SMPL(paths.SMPL,
                             batch_size=batch_size,
                             gender='female',
                             create_transl=False).to(device)

    # 3D shape and pose distribution predictor
    humaniflow_model = HumaniflowModel(device=device,
                                       model_cfg=humaniflow_cfg.MODEL,
                                       smpl_parents=smpl_model_neutral.parents.tolist()).to(device)
    checkpoint = torch.load(humaniflow_weights_path, map_location=device)
    humaniflow_model.load_state_dict(checkpoint['best_model_state_dict'], strict=True)
    print('\nLoaded HuManiFlow weights from', humaniflow_weights_path)

    # ------------------ Dataset + Metrics ------------------
    if dataset_name == '3dpw':
        # 3D point estimate metrics
        metrics = ['PVE', 'PVE-SC', 'PVE-PA', 'PVE-T-SC', 'MPJPE', 'MPJPE-SC', 'MPJPE-PA']
        # Distribution accuracy - 3D point estimate metrics with minimum sample
        metrics += [metric + '_samples_min' for metric in metrics]
        # Sample-input consistency - 2D sample (and point estimate) reprojection error
        metrics += ['joints2D-L2E']
        metrics += ['joints2Dsamples-L2E']
        # Sample diversity - 3D sample average distance from mean
        metrics += ['verts3D_sample_diversity', 'joints3D_sample_diversity',
                    'joints3D_invis_sample_diversity', 'joints3D_vis_sample_diversity']

        if save_path is None:
            save_path = f'./evaluations/3dpw_eval_{num_samples_for_metrics}_samples'
        eval_dataset = PW3DEvalDataset(pw3d_dir_path=paths.PW3D_PATH,
                                       config=humaniflow_cfg,
                                       visible_joints_threshold=0.6)

    elif dataset_name == 'ssp3d':
        # 3D point estimate metrics
        metrics = ['PVE-SC', 'PVE-PA', 'PVE-T-SC']
        # Sample-input consistency - 2D sample (and point estimate) reprojection error
        metrics += ['joints2D-L2E', 'silhouette-IOU']
        metrics += ['joints2Dsamples-L2E', 'silhouettesamples-IOU']
        # Sample diversity - 3D sample average distance from mean
        metrics += ['verts3D_sample_diversity', 'joints3D_sample_diversity',
                    'joints3D_invis_sample_diversity', 'joints3D_vis_sample_diversity']

        if save_path is None:
            save_path = f'./evaluations/ssp3d_eval_{num_samples_for_metrics}_samples'
        eval_dataset = SSP3DEvalDataset(ssp3d_dir_path=paths.SSP3D_PATH,
                                        config=humaniflow_cfg)

    print('\nEvaluating on {} with {} eval examples.'.format(dataset_name, str(len(eval_dataset))))
    print('Metrics:', metrics)
    print(f'Using {num_samples_for_metrics} samples for sample metrics.')
    print('Saving to:', save_path)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # ------------------ Evaluate ------------------
    torch.manual_seed(0)
    np.random.seed(0)
    evaluate_humaniflow(humaniflow_model=humaniflow_model,
                        humaniflow_cfg=humaniflow_cfg,
                        smpl_model_neutral=smpl_model_neutral,
                        smpl_model_male=smpl_model_male,
                        smpl_model_female=smpl_model_female,
                        edge_detect_model=edge_detect_model,
                        device=device,
                        eval_dataset=eval_dataset,
                        metrics=metrics,
                        batch_size=batch_size,
                        num_pred_samples=num_samples_for_metrics,
                        save_per_frame_metrics=True,
                        save_path=save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', '-D', type=str, choices=['3dpw', 'ssp3d'])

    parser.add_argument('--humaniflow_weights', '-W3D', type=str, default='./model_files/humaniflow_weights.tar')
    parser.add_argument('--humaniflow_cfg', type=str, default=None)
    parser.add_argument('--save_path', '-S', type=str, default=None)

    parser.add_argument('--batch_size', '-B', type=int, default=32)
    parser.add_argument('--num_samples', '-N', type=int, default=10, help='Number of samples to use for sample-based evaluation metrics.')
    parser.add_argument('--gpu', type=int, default=0)

    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('\nDevice: {}'.format(device))

    run_evaluate(device=device,
                 dataset_name=args.dataset,
                 humaniflow_weights_path=args.humaniflow_weights,
                 humaniflow_cfg_path=args.humaniflow_cfg,
                 save_path=args.save_path,
                 batch_size=args.batch_size,
                 num_samples_for_metrics=args.num_samples)



