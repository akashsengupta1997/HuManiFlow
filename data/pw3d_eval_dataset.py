import cv2
import numpy as np
import os
import torch
from torch.utils.data import Dataset

from utils.label_conversions import convert_2Djoints_to_gaussian_heatmaps


class PW3DEvalDataset(Dataset):
    def __init__(self,
                 pw3d_dir_path,
                 config,
                 extreme_crop_scale=None,
                 visible_joints_threshold=None,
                 threshold_hip_joints=False):
        super(PW3DEvalDataset, self).__init__()

        # Input images and 2D keypoints (obtained using HRNet)
        if extreme_crop_scale is None:
            self.cropped_frames_dir = os.path.join(pw3d_dir_path, 'cropped_frames')
            self.keypoints = np.load(os.path.join(pw3d_dir_path, 'hrnet_results_centred.npy'))
        else:
            self.cropped_frames_dir = os.path.join(pw3d_dir_path, f'extreme_cropped_{extreme_crop_scale}_frames')
            self.keypoints = np.load(os.path.join(pw3d_dir_path,  f'extreme_cropped_{extreme_crop_scale}_hrnet_results_centred.npy'))

        # Ground Truth Targets
        data = np.load(os.path.join(pw3d_dir_path, '3dpw_test.npz'))
        self.frame_fnames = data['imgname']
        self.pose = data['pose']
        self.shape = data['shape']
        self.gender = data['gender']
        if extreme_crop_scale is None:
            self.joints2D = data['joints2D_coco']
        else:
            self.joints2D = np.load(os.path.join(pw3d_dir_path,  f'extreme_cropped_{extreme_crop_scale}_joints2D.npy'))

        self.img_wh = config.DATA.PROXY_REP_SIZE
        self.hmaps_gaussian_std = config.DATA.HEATMAP_GAUSSIAN_STD
        self.visible_joints_threshold = visible_joints_threshold
        self.threshold_hip_joints = threshold_hip_joints

    def __len__(self):
        return len(self.frame_fnames)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        # ---------------------- Inputs ----------------------
        fname = self.frame_fnames[index]
        cropped_frame_path = os.path.join(self.cropped_frames_dir, fname)

        image = cv2.cvtColor(cv2.imread(cropped_frame_path), cv2.COLOR_BGR2RGB)
        orig_height, orig_width = image.shape[:2]
        assert (orig_height == orig_width), "Resizing non-square image to square will cause unwanted stretching/squeezing!"
        image = cv2.resize(image, (self.img_wh, self.img_wh), interpolation=cv2.INTER_LINEAR)
        image = np.transpose(image, [2, 0, 1]) / 255.0

        keypoints = self.keypoints[index]   # (17, 3) Predicted 2D joints locations and confidences, from HRNet
        keypoints_confidence = keypoints[:, 2]  # (17,)
        keypoints = keypoints[:, :2]
        keypoints = keypoints * np.array([self.img_wh / float(orig_width),
                                          self.img_wh / float(orig_height)])
        heatmaps = convert_2Djoints_to_gaussian_heatmaps(keypoints.round().astype(np.int16),
                                                         self.img_wh,
                                                         std=self.hmaps_gaussian_std)
        if self.visible_joints_threshold is not None:
            keypoints_visib_flag = keypoints_confidence > self.visible_joints_threshold
            if not self.threshold_hip_joints:
                keypoints_visib_flag[[0, 1, 2, 3, 4, 5, 6, 11, 12]] = True  # Only removing joints [7, 8, 9, 10, 13, 14, 15, 16] if occluded
            else:
                keypoints_visib_flag[[0, 1, 2, 3, 4, 5, 6]] = True  # Only removing joints [7, 8, 9, 10, 11, 12, 13, 14, 15, 16] if occluded
            heatmaps = heatmaps * keypoints_visib_flag[None, None, :]
        heatmaps = np.transpose(heatmaps, [2, 0, 1])

        # ---------------------- Targets ----------------------
        pose = self.pose[index]
        shape = self.shape[index]
        gender = self.gender[index]

        joints2D = self.joints2D[index]  # (17, 3) Ground Truth 2D joints locations and confidences
        joints2D_conf = joints2D[:, 2]
        joints2D = joints2D[:, :2] * np.array([self.img_wh / float(orig_width),
                                               self.img_wh / float(orig_height)])
        joints2D_visib_flag = joints2D_conf > self.visible_joints_threshold
        joints2D_visib_flag[[1, 2, 3, 4]] = joints2D_conf[[1, 2, 3, 4]] > 0.1  # Different threshold for face because confidences are generally very low for GT 2D face keypoints.

        image = torch.from_numpy(image).float()
        heatmaps = torch.from_numpy(heatmaps).float()
        pose = torch.from_numpy(pose).float()
        shape = torch.from_numpy(shape).float()
        joints2D = torch.from_numpy(joints2D).float()
        joints2D_visib_flag = torch.from_numpy(joints2D_visib_flag).bool()

        return {'image': image,
                'heatmaps': heatmaps,
                'pose': pose,
                'shape': shape,
                'fname': fname,
                'joints2D': joints2D,
                'joints2D_visib': joints2D_visib_flag,
                'gender': gender}
