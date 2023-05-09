import os
import torch
import torchvision
import numpy as np
import argparse
import sys
sys.path.append('.')

from models.humaniflow_model import HumaniflowModel
from models.smpl import SMPL
from models.pose2D_hrnet import PoseHighResolutionNet
from models.canny_edge_detector import CannyEdgeDetector

from configs.humaniflow_config import get_humaniflow_cfg_defaults
from configs.pose2D_hrnet_config import get_pose2D_hrnet_cfg_defaults
from configs import paths

from predict.predict_humaniflow import predict_humaniflow


def run_predict(device,
                image_dir,
                save_dir,
                humaniflow_weights_path,
                pose2D_hrnet_weights_path,
                humaniflow_cfg_path=None,
                already_cropped_images=False,
                joints2Dvisib_threshold=0.75,
                visualise_uncropped=True,
                visualise_xyz_variance=True,
                visualise_samples=False,
                num_pred_samples=50,
                num_vis_samples=8,
                gender='neutral'):

    # ------------------------- Models -------------------------
    # Configs
    pose2D_hrnet_cfg = get_pose2D_hrnet_cfg_defaults()
    humaniflow_cfg = get_humaniflow_cfg_defaults()
    if humaniflow_cfg_path is not None:
        humaniflow_cfg.merge_from_file(humaniflow_cfg_path)
        print('\nLoaded HuManiFlow config from', humaniflow_cfg_path)
    else:
        print('\nUsing default HuManiFlow config.')

    # Bounding box / Object detection model
    if not already_cropped_images:
        object_detect_model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True).to(device)
    else:
        object_detect_model = None

    # HRNet model for 2D joint detection
    hrnet_model = PoseHighResolutionNet(pose2D_hrnet_cfg).to(device)
    hrnet_checkpoint = torch.load(pose2D_hrnet_weights_path, map_location=device)
    hrnet_model.load_state_dict(hrnet_checkpoint, strict=False)
    print('\nLoaded HRNet weights from', pose2D_hrnet_weights_path)

    # Edge detector
    edge_detect_model = CannyEdgeDetector(non_max_suppression=humaniflow_cfg.DATA.EDGE_NMS,
                                          gaussian_filter_std=humaniflow_cfg.DATA.EDGE_GAUSSIAN_STD,
                                          gaussian_filter_size=humaniflow_cfg.DATA.EDGE_GAUSSIAN_SIZE,
                                          threshold=humaniflow_cfg.DATA.EDGE_THRESHOLD).to(device)

    # SMPL model
    print('\nUsing {} SMPL model with {} shape parameters.'.format(gender, str(humaniflow_cfg.MODEL.NUM_SMPL_BETAS)))
    smpl_model = SMPL(paths.SMPL,
                      batch_size=1,
                      gender=gender,
                      num_betas=humaniflow_cfg.MODEL.NUM_SMPL_BETAS).to(device)

    # HuManiFlow - 3D shape and pose distribution predictor
    humaniflow_model = HumaniflowModel(device=device,
                                       model_cfg=humaniflow_cfg.MODEL,
                                       smpl_parents=smpl_model.parents.tolist()).to(device)
    checkpoint = torch.load(humaniflow_weights_path, map_location=device)
    humaniflow_model.load_state_dict(checkpoint['best_model_state_dict'], strict=True)
    print('\nLoaded HuManiFlow weights from', humaniflow_weights_path)

    # ------------------------- Predict -------------------------
    torch.manual_seed(0)
    np.random.seed(0)
    predict_humaniflow(humaniflow_model=humaniflow_model,
                       humaniflow_cfg=humaniflow_cfg,
                       smpl_model=smpl_model,
                       hrnet_model=hrnet_model,
                       hrnet_cfg=pose2D_hrnet_cfg,
                       edge_detect_model=edge_detect_model,
                       device=device,
                       image_dir=image_dir,
                       save_dir=save_dir,
                       object_detect_model=object_detect_model,
                       joints2Dvisib_threshold=joints2Dvisib_threshold,
                       num_pred_samples=num_pred_samples,
                       num_vis_samples=num_vis_samples,
                       visualise_uncropped=visualise_uncropped,
                       visualise_samples=visualise_samples,
                       visualise_xyz_variance=visualise_xyz_variance)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', '-I', type=str, help='Path to directory of test images.')
    parser.add_argument('--save_dir', '-S', type=str, help='Path to directory where test outputs will be saved.')

    parser.add_argument('--humaniflow_weights', '-W3D', type=str, default='./model_files/humaniflow_weights.tar')
    parser.add_argument('--humaniflow_cfg', type=str, default=None)
    parser.add_argument('--pose2D_hrnet_weights', '-W2D', type=str, default='./model_files/pose_hrnet_w48_384x288.pth')

    parser.add_argument('--cropped_images', '-C', action='store_true', help='Images already cropped and centred.')
    parser.add_argument('--gender', '-G', type=str, default='neutral', choices=['neutral', 'male', 'female'], help='Gendered SMPL models may be used.')
    parser.add_argument('--joints2Dvisib_threshold', '-T', type=float, default=0.75)
    parser.add_argument('--num_pred_samples', '-NP', type=int, default=50)
    parser.add_argument('--num_vis_samples', '-NV', type=int, default=8)

    parser.add_argument('--visualise_samples', '-VS', action='store_true')
    parser.add_argument('--visualise_uncropped', '-VU', action='store_true')
    parser.add_argument('--visualise_xyz_variance', '-VXYZ', action='store_true')

    parser.add_argument('--gpu', type=int, default=0)

    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('\nDevice: {}'.format(device))

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    if args.gender != 'neutral':
        raise NotImplementedError

    run_predict(device=device,
                image_dir=args.image_dir,
                save_dir=args.save_dir,
                humaniflow_weights_path=args.humaniflow_weights,
                pose2D_hrnet_weights_path=args.pose2D_hrnet_weights,
                humaniflow_cfg_path=args.humaniflow_cfg,
                already_cropped_images=args.cropped_images,
                joints2Dvisib_threshold=args.joints2Dvisib_threshold,
                visualise_uncropped=args.visualise_uncropped,
                visualise_xyz_variance=args.visualise_xyz_variance,
                visualise_samples=args.visualise_samples,
                num_pred_samples=args.num_pred_samples,
                num_vis_samples=args.num_vis_samples,
                gender=args.gender)



