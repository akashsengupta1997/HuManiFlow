import math
import cv2
import torch
import numpy as np
from itertools import product
from torch.nn import functional as F
try:
    from pytorch3d.transforms.so3 import so3_log_map as so3_log_pytorch3d, so3_exp_map as so3_exp_pytorch3d
except ImportError:
    print('Failed to import pytorch3d in rigid_transform_utils.py')

from utils.lin_alg_utils import batch_trace

# TODO use so3_exp and so3_log defined here instead of pytorch3d


def aa_rotate_rotmats_pytorch3d(rotmats, axes, angles, rot_mult_order='post'):
    """
    Batched rotation of rotation matrices about given axes and angles.

    :param rotmats: (B, 3, 3), batch of rotation matrices
    :param axes: (B, 3) or (3,), rotation axes (may be batched)
    :param angles: (B, 1) or scalar, rotation angles in radians (may be batched)
    :return: rotated_axisangle (B, 3) and rotated_rotmats (B, 3, 3)
    """
    assert rot_mult_order in ['pre', 'post']
    r = axes * angles
    if r.dim() < 2:
        r = r[None, :].expand(rotmats.shape[0], -1)
    R = so3_exp_pytorch3d(log_rot=r)  # (B, 3, 3)
    if rot_mult_order == 'post':
        rotated_rotmats = torch.matmul(rotmats, R)
    elif rot_mult_order == 'pre':
        rotated_rotmats = torch.matmul(R, rotmats)
    rotated_axisangle = so3_log_pytorch3d(R=rotated_rotmats)

    return rotated_axisangle, rotated_rotmats


def aa_rotate_rotmats_opencv(axis, angle, rotmats, rot_mult_order='post'):
    """
    This does the same thing as aa_rotate_rotmats_pytorch3d, except using openCV instead of pytorch3d.
    This is preferred when computing rotated rotation vectors (SO(3) log map) because pytorch3d's
    SO(3) log map is broken for R = I.
    However pytorch3d function is batched and only requires torch, so should be faster - use when
    rotation vectors are not needed (e.g. during training).

    :param rotmats: (B, 3, 3), batch of rotation matrices
    :param axis: (3, ) numpy array, axis of rotation
    :param angle: scalar, angle of rotation
    :return: rotated_vecs (B, 3) and rotated_rotmats (B, 3, 3)
    """
    assert rot_mult_order in ['pre', 'post']
    R = cv2.Rodrigues(np.array(axis)*angle)[0]
    rotmats = rotmats.cpu().detach().numpy()
    if rot_mult_order == 'post':
        rotated_rotmats = np.matmul(rotmats, R)
    elif rot_mult_order == 'pre':
        rotated_rotmats = np.matmul(R, rotmats)
    rotated_vecs = []
    for i in range(rotated_rotmats.shape[0]):
        rotated_vecs.append(cv2.Rodrigues(rotated_rotmats[i, :, :])[0].squeeze())
    rotated_vecs = torch.from_numpy(np.stack(rotated_vecs, axis=0)).float()
    return rotated_vecs, torch.from_numpy(rotated_rotmats).float()


def aa_rotate_translate_points_pytorch3d(points, axes, angles, translations):
    """
    Rotates and translates batch of points from a mesh about given axes and angles.
    :param points: B, N, 3, batch of meshes with N points each
    :param axes: (B,3) or (3,), rotation axes
    :param angles: (B,1) or scalar, rotation angles in radians
    :param translations: (B,3) or (3,), translation vectors
    :return:
    """
    r = axes * angles
    if r.dim() < 2:
        r = r[None, :].expand(points.shape[0], -1)
    R = so3_exp_pytorch3d(log_rot=r)  # (B, 3, 3)
    points = torch.einsum('bij,bkj->bki', R, points)
    points = points + translations

    return points


def rot6d_to_rotmat(x):
    """Convert 6D rotation representation to 3x3 rotation matrix.
    Based on Zhou et al., "On the Continuity of Rotation Representations in Neural Networks", CVPR 2019
    Input:
        (B,6) or (B, 24*6) Batch of 6-D rotation representations
    Output:
        (B,3,3) Batch of corresponding rotation matrices
    """
    x = x.view(-1,3,2)
    a1 = x[:, :, 0]
    a2 = x[:, :, 1]
    b1 = F.normalize(a1)  # Ensuring columns are unit vectors
    b2 = F.normalize(a2 - torch.einsum('bi,bi->b', b1, a2).unsqueeze(-1) * b1)  # Ensuring column 1 and column 2 are orthogonal with Gram-Schmidt orthogonalisation
    b3 = torch.cross(b1, b2)
    return torch.stack((b1, b2, b3), dim=-1)


def rotmat_to_rot6d(R, stack_columns=False):
    """
    :param R: (B, 3, 3)
    :param stack_columns:
        if True, 6D pose representation is [1st col of R, 2nd col of R]^T = [R11, R21, R31, R12, R22, R32]^T
        if False, 6D pose representation is [R11, R12, R21, R22, R31, R32]^T
        Set to False if doing inverse of rot6d_to_rotmat
    :return: rot6d: (B, 6)
    """
    if stack_columns:
        rot6d = torch.cat([R[:, :, 0], R[:, :, 1]], dim=1)
    else:
        rot6d = R[:, :, :2].contiguous().view(-1, 6)
    return rot6d


def quat_to_rotmat(quat):
    """Convert quaternion coefficients to rotation matrix.
    Args:
        quat: size = [B, 4] 4 <===>(w, x, y, z)
    Returns:
        Rotation matrix corresponding to the quaternion -- size = [B, 3, 3]
    """
    norm_quat = quat
    norm_quat = norm_quat/norm_quat.norm(p=2, dim=1, keepdim=True)
    w, x, y, z = norm_quat[:,0], norm_quat[:,1], norm_quat[:,2], norm_quat[:,3]

    B = quat.size(0)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z

    rotmat = torch.stack([w2 + x2 - y2 - z2, 2*xy - 2*wz, 2*wy + 2*xz,
                          2*wz + 2*xy, w2 - x2 + y2 - z2, 2*yz - 2*wx,
                          2*xz - 2*wy, 2*wx + 2*yz, w2 - x2 - y2 + z2], dim=1).view(B, 3, 3)
    return rotmat


def so3_hat(v):
    """
    Code from https://github.com/pimdh/relie.
    Isomorphism from R^3 to Lie algebra so(3),
    i.e. 3D vectors to skew-symmetric matrices.
    Inverse of so3_vee.

    :param  v: (*, 3) batch of 3D vectors
    :return v_hat: (*, 3, 3) batch of skew-symmetric matrices
    """
    assert v.shape[-1] == 3

    e_x = v.new_tensor([[0.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])

    e_y = v.new_tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])

    e_z = v.new_tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])

    v_hat = (
            e_x * v[..., 0, None, None]
            + e_y * v[..., 1, None, None]
            + e_z * v[..., 2, None, None]
    )
    return v_hat


def so3_vee(v_hat):
    """
    Code from https://github.com/pimdh/relie.
    Isomorphism from Lie algebra so(3) to R^3,
    i.e. skew-symmetric matrices to 3D vectors.
    Inverse of so3_hat.

    :param  v_hat: (*, 3, 3) batch of skew-symmetric matrices
    :return v: (*, 3) batch of 3D vectors
    """
    assert v_hat.shape[-2:] == (3, 3)
    return torch.stack((-v_hat[..., 1, 2], v_hat[..., 0, 2], -v_hat[..., 0, 1]), -1)


def so3_exp(v):
    """
    Code from https://github.com/pimdh/relie.
    Exponential map of SO(3) with Rordigues formula.
    :param v: algebra vector of shape (..., 3)
    :return: group element of shape (..., 3, 3)
    """
    assert v.dtype == torch.double
    theta = v.norm(p=2, dim=-1)

    mask = theta > 1e-10
    theta = torch.where(mask, theta, torch.ones_like(theta))

    # sin(x)/x -> 1-x^2/6 as x->0
    alpha = torch.where(mask, torch.sin(theta) / theta, 1 - theta ** 2 / 6)
    # (1-cos(x))/x^2 -> 0.5-x^2/24 as x->0
    beta = torch.where(mask, (1 - torch.cos(theta)) / theta ** 2, 0.5 - theta ** 2 / 24)
    eye = torch.eye(3, device=v.device, dtype=v.dtype)
    x = so3_hat(v)
    return eye + alpha[..., None, None] * x + beta[..., None, None] * x @ x


def so3_log(r,
            return_axis_angle=False):
    """
    Code from https://github.com/pimdh/relie.
    Logarithm map of SO(3).
    :param r: group element of shape (..., 3, 3)
    :param return_axis_angle: bool, convert skew-symetric matrix output to 3D vector if True
    :return: Algebra element in matrix basis of shape (..., 3, 3)
             Converted to 3D vector if return_axis_angle is True.

    Uses https://en.wikipedia.org/wiki/Rotation_group_SO(3)#Logarithm_map
    """
    assert r.dtype == torch.double
    anti_sym = 0.5 * (r - r.transpose(-1, -2))
    cos_theta = 0.5 * (batch_trace(r)[..., None, None] - 1)
    cos_theta = cos_theta.clamp(-1, 1)  # Ensure we get a correct angle
    theta = torch.acos(cos_theta)
    ratio = theta / torch.sin(theta)

    # x/sin(x) -> 1 + x^2/6 as x->0
    mask = (theta[..., 0, 0] < 1e-20).nonzero()
    ratio[mask] = 1 + theta[mask] ** 2 / 6

    log = ratio * anti_sym

    # Separately handle theta close to pi
    mask = ((math.pi - theta[..., 0, 0]).abs() < 1e-2).nonzero()
    if mask.numel():
        log[mask[:, 0]] = so3_log_pi(r[mask[:, 0]], theta[mask[:, 0]])

    if return_axis_angle:
        log = so3_vee(log)

    return log


def so3_log_pi(r, theta):
    """
    Code from https://github.com/pimdh/relie.
    Logarithm map of SO(3) for cases with theta close to pi.
    Note: inaccurate for theta around 0.
    :param r: group element of shape (..., 3, 3)
    :param theta: rotation angle
    :return: Algebra element in matrix basis of shape (..., 3, 3)
    """
    sym = 0.5 * (r + r.transpose(-1, -2))
    eye = torch.eye(3, device=r.device, dtype=r.dtype).expand_as(sym)
    z = theta ** 2 / (1 - torch.cos(theta)) * (sym - eye)

    q_1 = z[..., 0, 0]
    q_2 = z[..., 1, 1]
    q_3 = z[..., 2, 2]
    x_1 = torch.sqrt(torch.clamp(q_1 - q_2 - q_3, min=1e-8) / 2)
    x_2 = torch.sqrt(torch.clamp(-q_1 + q_2 - q_3, min=1e-8) / 2)
    x_3 = torch.sqrt(torch.clamp(-q_1 - q_2 + q_3, min=1e-8) / 2)
    x = torch.stack([x_1, x_2, x_3], -1)

    # Flatten batch dim
    batch_shape = x.shape[:-1]
    x = x.view(-1, 3)
    r = r.view(-1, 3, 3)

    # We know components up to a sign, search for correct one
    signs = torch.tensor(list(product([0, 1], repeat=3)), dtype=x.dtype, device=x.device) * 2 - 1
    # signs = zero_one_outer_product(3, dtype=x.dtype, device=x.device) * 2 - 1
    x_stack = signs.view(8, 1, 3) * x[None]
    with torch.no_grad():
        r_stack = so3_exp(x_stack)
        diff = (r[None] - r_stack).pow(2).sum(-1).sum(-1)
        selector = torch.argmin(diff, dim=0)
    x = x_stack[selector, torch.arange(len(selector))]

    # Restore shape
    x = x.view(*batch_shape, 3)

    return so3_hat(x)


def so3_xset(x, k_max):
    """
    Code from https://github.com/pimdh/relie.
    Return set of x's that have same image as exp(x) excluding x itself.
    :param x: Tensor of shape (..., 3) of algebra elements.
    :param k_max: int. Number of 2pi shifts in either direction
    :return: Tensor of shape (2 * k_max+1, ..., 3)
    """
    x = x[None]
    x_norm = x.norm(dim=-1, keepdim=True)
    shape = [-1, *[1] * (x.dim() - 1)]
    k_range = torch.arange(1, k_max + 1, dtype=x.dtype, device=x.device)
    k_range = torch.cat([-k_range, k_range]).view(shape)
    return x / x_norm * (x_norm + 2 * math.pi * k_range)


def so3_log_abs_det_jacobian(x):
    """
    Code from https://github.com/pimdh/relie.
    Return element wise log abs det jacobian of exponential map
    :param x: Algebra tensor of shape (..., 3)
    :return: Tensor of shape (..., 3)

    Removable pole: (2-2 cos x)/x^2 -> 1-x^2/12 as x->0
    """
    x_norm = x.double().norm(dim=-1)
    mask = x_norm > 1e-10
    x_norm = torch.where(mask, x_norm, torch.ones_like(x_norm))

    ratio = torch.where(
        mask, (2 - 2 * torch.cos(x_norm)) / x_norm ** 2, 1 - x_norm ** 2 / 12
    )
    return torch.log(ratio).to(x.dtype)


def so3_exp_opencv(rotvecs):
    """
    Simple for loop over a batch of axis-angle vectors, converting each to
    rotation matrix using OpenCV.
    :param rotvecs: (B, 3) batch of axis-angle rotation vectors
    :return R: (B, 3, 3) batch of rotation matrices
    """
    B = rotvecs.shape[0]
    R = np.zeros((B, 3, 3))
    for i in range(B):
        R[i] = cv2.Rodrigues(rotvecs[i])[0]
    return R


def so3_log_opencv(R):
    """
    Simple for loop over a batch of rotation matrices, converting each to
    axis-angle representation using OpenCV.
    :param R: (B, 3, 3) batch of rotation matrices
    :return rotvecs: (B, 3) batch of axis-angle rotation vectors
    """
    B = R.shape[0]
    rotvecs = np.zeros((B, 3))
    for i in range(B):
        rotvecs[i] = np.squeeze(cv2.Rodrigues(R[i])[0])
    return rotvecs


# # Code snippet showing that pytorch3d so3_log_map is broken near 180° rotations.
# # Random target R to recover
# rot_angle = np.deg2rad(180)
# rot_axis = np.random.randn(3)
# rot_axis = rot_axis/np.linalg.norm(rot_axis)
# rotvec = rot_angle * rot_axis
# R1 = cv2.Rodrigues(rotvec)[0]
# # Fixed target R to recover, right rotvec should be [0, pi, 0]
# R2 = np.array([[-1.,  0.,  0.],
#                [ 0.,  1.,  0.],
#                [ 0., 0.,  -1.]])
# R = np.stack([R1, R2])
#
# # Convert to axis-angle vector
# print('TARGET R\n', R)
# rotcv2 = so3_log_opencv(R)
# rotp3d = so3_log_pytorch3d(torch.from_numpy(R))
# rotrelie = so3_log(torch.from_numpy(R).double(),
#                    return_axis_angle=True)
#
# # Convert back to rotation matrix - compare with target R
# Rcv2 = so3_exp_opencv(rotcv2)
# Rp3d = so3_exp_pytorch3d(rotp3d)
# Rrelie = so3_exp(rotrelie)
# print('\nOPENCV')
# print(Rcv2)
# print('\nPYTORCH3D')
# print(Rp3d)
# print('\nRELIE')
# print(Rrelie)