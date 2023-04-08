import torch
import numpy as np

from utils.rigid_transform_utils import quat_to_rotmat, aa_rotate_translate_points_pytorch3d
from utils.cam_utils import orthographic_project_torch
from utils.joints2d_utils import undo_keypoint_normalisation
from utils.label_conversions import convert_heatmaps_to_2Djoints_coordinates_torch, ALL_JOINTS_TO_COCO_MAP


def so3_uniform_sampling_torch(num_matrices):
    """
    Sampling uniformly-distributed (w.r.t Haar measure) random rotation matrices from SO(3).
    This is done by sampling random unit quaternions on S^3, which is a double cover of SO(3),
    then converting quaternions to rotmats.
    :param num_matrices: int, number of samples
    """
    random_quat = torch.randn(num_matrices, 4)
    random_quat = random_quat / torch.norm(random_quat, p=2, dim=-1, keepdim=True)
    return quat_to_rotmat(random_quat)


def compute_vertex_variance_from_samples(vertices_samples):
    """

    :param vertices_samples: (N, 6890, 3)
    :return:
    """
    mean_vertices = torch.mean(vertices_samples, dim=0)
    diff_from_mean = vertices_samples - mean_vertices
    directional_vertex_variances = torch.sqrt(torch.mean(diff_from_mean ** 2, dim=0))  # (6890, 3)
    avg_vertex_l2_distance_from_mean = torch.norm(diff_from_mean, dim=-1).mean(dim=0)  # (6890,)

    return avg_vertex_l2_distance_from_mean, directional_vertex_variances


def joints2D_error_sorted_verts_sampling(pred_vertices_samples,
                                         pred_joints_samples,
                                         input_joints2D_heatmaps,
                                         pred_cam_wp):
    """
    Sort 3D vertex mesh samples according to consistency (error) between projected 2D joint samples
    and input 2D joints.
    :param pred_vertices_samples: (N, 6890, 3) tensor of candidate vertex mesh samples.
    :param pred_joints_samples: (N, 90, 3) tensor of candidate J3D samples.
    :param input_joints2D_heatmaps: (1, 17, img_wh, img_wh) tensor of 2D joint locations and confidences.
    :param pred_cam_wp: (1, 3) array with predicted weak-perspective camera.
    :return: pred_vertices_samples_error_sorted: (N, 6890, 3) tensor of J2D-error-sorted vertex mesh samples.
    """
    # Project 3D joint samples to 2D (using COCO joints)
    pred_joints_coco_samples = pred_joints_samples[:, ALL_JOINTS_TO_COCO_MAP, :]
    pred_joints_coco_samples = aa_rotate_translate_points_pytorch3d(points=pred_joints_coco_samples,
                                                                    axes=torch.tensor([1., 0., 0.], device=pred_vertices_samples.device).float(),
                                                                    angles=np.pi,
                                                                    translations=torch.zeros(3, device=pred_vertices_samples.device).float())
    pred_joints2D_coco_samples = orthographic_project_torch(pred_joints_coco_samples, pred_cam_wp)
    pred_joints2D_coco_samples = undo_keypoint_normalisation(pred_joints2D_coco_samples,
                                                             input_joints2D_heatmaps.shape[-1])

    # Convert input 2D joint heatmaps into coordinates
    input_joints2D_coco, input_joints2D_coco_vis = convert_heatmaps_to_2Djoints_coordinates_torch(joints2D_heatmaps=input_joints2D_heatmaps,
                                                                                                  eps=1e-6)  # (1, 17, 2) and (1, 17)

    # Gather visible 2D joint samples and input
    pred_visible_joints2D_coco_samples = pred_joints2D_coco_samples[:, input_joints2D_coco_vis[0], :]  # (N, num vis joints, 2)
    input_visible_joints2D_coco = input_joints2D_coco[:, input_joints2D_coco_vis[0], :]  # (1, num vis joints, 2)

    # Compare 2D joint samples and input using Euclidean distance on image plane.
    j2d_l2es = torch.norm(pred_visible_joints2D_coco_samples - input_visible_joints2D_coco, dim=-1)  # (N, num vis joints)
    j2d_l2e_max, _ = torch.max(j2d_l2es, dim=-1)  # (N,)  # Max joint L2 error for each sample
    _, error_sort_idx = torch.sort(j2d_l2e_max, descending=False)

    pred_vertices_samples_error_sorted = pred_vertices_samples[error_sort_idx]

    return pred_vertices_samples_error_sorted
