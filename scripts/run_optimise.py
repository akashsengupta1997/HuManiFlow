import os
import torch
import numpy as np
import argparse
import sys
sys.path.append('.')

from models.humaniflow_model import HumaniflowModel
from models.smpl import SMPL

from configs.humaniflow_config import get_humaniflow_cfg_defaults
from configs.optimise_config import get_optimise_cfg_defaults
from configs import paths

from optimise.optimise_humaniflow import optimise_batch_with_humaniflow_prior


def run_optimise(device,
                 pred_image_dir,
                 pred_output_dir,
                 save_dir,
                 humaniflow_weights_path,
                 humaniflow_cfg_path=None,
                 optimise_cfg_path=None):

    # ------------------------- Models -------------------------
    # Configs
    humaniflow_cfg = get_humaniflow_cfg_defaults()
    if humaniflow_cfg_path is not None:
        humaniflow_cfg.merge_from_file(humaniflow_cfg_path)
        print('\nLoaded HuManiFlow config from', humaniflow_cfg_path)
    else:
        print('\nUsing default HuManiFlow config.')

    optimise_cfg = get_optimise_cfg_defaults()
    if optimise_cfg_path is not None:
        optimise_cfg.merge_from_file(optimise_cfg_path)
        print('\nLoaded optimisation config from', optimise_cfg_path)
    else:
        print('\nUsing default optimisation config.')

    # SMPL model
    smpl_model = SMPL(paths.SMPL,
                      batch_size=1,
                      gender='neutral',
                      num_betas=humaniflow_cfg.MODEL.NUM_SMPL_BETAS).to(device)

    # HuManiFlow - 3D shape and pose distribution predictor - to be used as a 3D prior during optimisation
    humaniflow_model = HumaniflowModel(device=device,
                                       model_cfg=humaniflow_cfg.MODEL,
                                       smpl_parents=smpl_model.parents.tolist()).to(device)
    checkpoint = torch.load(humaniflow_weights_path, map_location=device)
    humaniflow_model.load_state_dict(checkpoint['best_model_state_dict'], strict=True)
    print('\nLoaded HuManiFlow weights from', humaniflow_weights_path)

    # ------------------------- Predict -------------------------
    torch.manual_seed(0)
    np.random.seed(0)
    optimise_batch_with_humaniflow_prior(humaniflow_model=humaniflow_model,
                                         humaniflow_cfg=humaniflow_cfg,
                                         optimise_cfg=optimise_cfg,
                                         smpl_model=smpl_model,
                                         pred_image_dir=pred_image_dir,
                                         pred_output_dir=pred_output_dir,
                                         opt_output_dir=save_dir,
                                         device=device,
                                         visualise_wh=512)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_image_dir', '-I', type=str, help='Path to directory of test images.')
    parser.add_argument('--pred_output_dir', '-O', type=str, help='Path to directory of predicted outputs corresponding to test images.')
    parser.add_argument('--save_dir', '-S', type=str, help='Path to directory where optimisation outputs will be saved.')

    parser.add_argument('--humaniflow_weights', '-W3D', type=str, default='./model_files/humaniflow_weights.tar')
    parser.add_argument('--humaniflow_cfg', type=str, default=None)
    parser.add_argument('--optimise_cfg', type=str, default=None)

    parser.add_argument('--gpu', type=int, default=0)

    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('\nDevice: {}'.format(device))

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    run_optimise(device=device,
                 pred_image_dir=args.pred_image_dir,
                 pred_output_dir=args.pred_output_dir,
                 save_dir=args.save_dir,
                 humaniflow_weights_path=args.humaniflow_weights,
                 humaniflow_cfg_path=args.humaniflow_cfg,
                 optimise_cfg_path=args.optimise_cfg)



