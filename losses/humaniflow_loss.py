import torch
import torch.nn as nn


class HumaniflowLoss(nn.Module):
    """
    """
    def __init__(self,
                 loss_cfg,
                 img_wh):
        super(HumaniflowLoss, self).__init__()

        self.loss_cfg = loss_cfg
        self.img_wh = img_wh

        self.joints2D_loss = nn.MSELoss(reduction=self.loss_cfg.REDUCTION)
        self.glob_rotmats_loss = nn.MSELoss(reduction=self.loss_cfg.REDUCTION)

        if self.loss_cfg.APPLY_POINT_EST_LOSS:
            self.verts3D_loss = nn.MSELoss(reduction=self.loss_cfg.REDUCTION)
            self.joints3D_loss = nn.MSELoss(reduction=self.loss_cfg.REDUCTION)

    def forward(self, target_dict, pred_dict):

        # ----------------- Pose NLL -----------------
        pred_pose_dists = pred_dict['pose_dist']  # List of per-bodypart ancestor-conditioned SO(3) flow distributions
        target_pose_rotmats = target_dict['pose_params_rotmats'].double()
        batch_size, num_bodyparts = target_pose_rotmats.shape[:2]
        pose_nll = 0
        for bodypart in range(len(pred_pose_dists)):
            pose_nll -= pred_pose_dists[bodypart].log_prob(target_pose_rotmats[:, bodypart, :, :]).sum(dim=0)  # scalar
            # pred_pose_dists[bodypart].log_prob(target_pose_rotmats[:, bodypart, :, :]) has shape (batch_size,)

        if self.loss_cfg.REDUCTION == 'mean':
            pose_nll = pose_nll / (batch_size * num_bodyparts)
            # Technically should only be dividing by batch size, since per-bodypart distributions (i.e. log conditionals) should be
            # summed to get log joint distribution over pose. In practice, doesn't really matter as the extra division just acts as a down-weighting,
            # which ends up getting subsumed by the loss weight hyperparameters anyway.

        # ----------------- Shape NLL -----------------
        shape_nll = -(pred_dict['shape_dist'].log_prob(target_dict['shape_params']).sum(dim=1))  # (batch_size,)
        # shape_dist is num_betas independent univariate normals (i.e. "multivariate" with diagonal covariance matrix), hence the sum over dim=1
        if self.loss_cfg.REDUCTION == 'mean':
            shape_nll = torch.mean(shape_nll)
        elif self.loss_cfg.REDUCTION == 'sum':
            shape_nll = torch.sum(shape_nll)

        # ----------------- Visible 2D Joints Reprojection MSE -----------------
        target_joints2D = target_dict['joints2D']          # (batch_size, 17, 2)
        target_joints2D_vis = target_dict['joints2D_vis']  # (batch_size, 17)
        pred_joints2D = pred_dict['joints2D']              # (batch_size, num_samples, 17, 2)

        # Selecting visible 2D joint targets and predictions
        target_joints2D = target_joints2D[:, None, :, :].expand_as(pred_joints2D)
        target_joints2D_vis = target_joints2D_vis[:, None, :].expand(-1, pred_joints2D.shape[1], -1)
        pred_joints2D = pred_joints2D[target_joints2D_vis, :]
        target_joints2D = target_joints2D[target_joints2D_vis, :]

        target_joints2D = (2.0 * target_joints2D) / self.img_wh - 1.0  # normalising 2D joints to [-1, 1] x [-1, 1] plane.
        joints2D_loss = self.joints2D_loss(pred_joints2D, target_joints2D)

        # Glob Rotmats MSE
        glob_rotmats_loss = self.glob_rotmats_loss(pred_dict['glob_rotmats'], target_dict['glob_rotmats'])

        total_loss = pose_nll * self.loss_cfg.WEIGHTS.POSE \
                     + shape_nll * self.loss_cfg.WEIGHTS.SHAPE \
                     + joints2D_loss * self.loss_cfg.WEIGHTS.JOINTS2D \
                     + glob_rotmats_loss * self.loss_cfg.WEIGHTS.GLOB_ROTMATS

        # ----------------- Optional 3D point estimate losses -----------------
        # Very slightly improve 3D point estimate metrics, but often degrade sample-input consistency and sample diversity
        if self.loss_cfg.APPLY_POINT_EST_LOSS:
            verts3D_loss = self.verts3D_loss(pred_dict['verts'], target_dict['verts'])
            total_loss += verts3D_loss * self.loss_cfg.WEIGHTS.VERTS3D

            joints3D_loss = self.joints3D_loss(pred_dict['joints3D'], target_dict['joints3D'])
            total_loss += joints3D_loss * self.loss_cfg.WEIGHTS.JOINTS3D

        return total_loss

