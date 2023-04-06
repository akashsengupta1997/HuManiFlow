from collections import defaultdict
import torch
from torch import nn as nn
from torch.distributions import Normal

from smplx.lbs import batch_rodrigues

from models.resnet import resnet18, resnet50
from models.norm_flows import ConditionalLocalDiffeoTransformedDistribution
from models.norm_flows.pyro_conditional_norm_flow import create_conditional_norm_flow, forward_trans_conditional_norm_flow
from models.norm_flows.transforms import ToTransform, SO3ExpCompactTransform

from utils.rigid_transform_utils import rotmat_to_rot6d, rot6d_to_rotmat


def immediate_parent_to_all_ancestors(immediate_parents):
    """

    :param immediate_parents: list with len = num joints, contains index of each joint's parent.
            - includes root joint, but its parent index is -1.
    :return: ancestors_dict: dict of lists, dict[joint] is ordered list of parent joints.
            - DOES NOT INCLUDE ROOT JOINT! Joint 0 here is actually joint 1 in SMPL.
    """
    ancestors_dict = defaultdict(list)
    for i in range(1, len(immediate_parents)):  # Excluding root joint
        joint = i - 1
        immediate_parent = immediate_parents[i] - 1
        if immediate_parent >= 0:
            ancestors_dict[joint] += [immediate_parent] + ancestors_dict[immediate_parent]
    return ancestors_dict


class HumaniflowModel(nn.Module):
    def __init__(self,
                 device,
                 model_cfg,
                 smpl_parents):
        """
        """
        super(HumaniflowModel, self).__init__()

        # Num pose parameters + Kinematic tree pre-processing
        self.parents = smpl_parents
        self.ancestors_dict = immediate_parent_to_all_ancestors(smpl_parents)
        self.num_bodyparts = len(self.ancestors_dict)

        # Number of shape, glob and cam parameters
        self.num_shape_params = model_cfg.NUM_SMPL_BETAS

        self.num_glob_params = 6  # 6D rotation representation for glob
        init_glob = rotmat_to_rot6d(torch.eye(3)[None, :].float())
        self.register_buffer('init_glob', init_glob)

        self.num_cam_params = 3
        init_cam = torch.tensor([0.9, 0.0, 0.0]).float()  # Initialise orthographic camera scale at 0.9
        self.register_buffer('init_cam', init_cam)

        # ResNet Image Encoder
        if model_cfg.NUM_RESNET_LAYERS == 18:
            self.image_encoder = resnet18(in_channels=model_cfg.NUM_IN_CHANNELS,
                                          pretrained=False)
            input_feats_dim = 512
            fc1_dim = 512
        elif model_cfg.NUM_RESNET_LAYERS == 50:
            self.image_encoder = resnet50(in_channels=model_cfg.NUM_IN_CHANNELS,
                                          pretrained=False)
            input_feats_dim = 2048
            fc1_dim = 1024

        # FC Shape/Glob/Cam networks
        self.activation = nn.ELU()
        self.fc1 = nn.Linear(input_feats_dim, fc1_dim)
        self.fc_shape = nn.Linear(fc1_dim, self.num_shape_params * 2)  # Means and variances for SMPL betas and/or measurements
        self.fc_glob = nn.Linear(fc1_dim, self.num_glob_params)
        self.fc_cam = nn.Linear(fc1_dim, self.num_cam_params)

        # Pose Normalising Flow networks for each bodypart
        self.fc_input_shape_glob_cam_feats = nn.Linear(input_feats_dim + self.num_shape_params + 9 + self.num_cam_params,
                                                       model_cfg.INPUT_SHAPE_GLOB_CAM_FEATS_DIM)
        self.fc_flow_context = nn.ModuleList()
        self.pose_so3flow_transform_modules = nn.ModuleList()
        self.pose_so3flow_transforms = []
        self.pose_so3flow_dists = []
        self.pose_SO3flow_dists = []

        for bodypart in range(self.num_bodyparts):
            num_ancestors = len(self.ancestors_dict[bodypart])
            # Input to fc_flow_context is input/shape/glob/cam features and 3x3 rotmats for all ancestors
            self.fc_flow_context.append(nn.Linear(model_cfg.INPUT_SHAPE_GLOB_CAM_FEATS_DIM + num_ancestors * 9,
                                                  model_cfg.FLOW_CONTEXT_DIM))
            # Set up normalising flow distribution on Lie algebra so(3) for each bodypart.
            so3flow_dist, so3flow_transform_modules, so3flow_transforms = create_conditional_norm_flow(
                device=device,
                event_dim=3,
                context_dim=model_cfg.FLOW_CONTEXT_DIM,
                num_transforms=model_cfg.NORM_FLOW.NUM_TRANSFORMS,
                transform_type=model_cfg.NORM_FLOW.TRANSFORM_TYPE,
                transform_hidden_dims=model_cfg.NORM_FLOW.TRANSFORM_NN_HIDDEN_DIMS,
                permute_type=model_cfg.NORM_FLOW.PERMUTE_TYPE,
                permute_hidden_dims=model_cfg.NORM_FLOW.PERMUTE_HIDDEN_DIMS,
                bound=model_cfg.NORM_FLOW.COMPACT_SUPPORT_RADIUS,
                count_bins=model_cfg.NORM_FLOW.NUM_SPLINE_SEGMENTS,
                radial_tanh_radius=model_cfg.NORM_FLOW.COMPACT_SUPPORT_RADIUS,
                base_dist_std=model_cfg.NORM_FLOW.BASE_DIST_STD)

            # Pushforward distribution on Lie group SO(3) for each bodypart.
            SO3flow_dist = ConditionalLocalDiffeoTransformedDistribution(base_dist=so3flow_dist,
                                                                         transforms=[ToTransform(dict(dtype=torch.float32), dict(dtype=torch.float64)),
                                                                                     SO3ExpCompactTransform(support_radius=model_cfg.NORM_FLOW.COMPACT_SUPPORT_RADIUS)])

            self.pose_so3flow_transform_modules.extend(so3flow_transform_modules)
            self.pose_so3flow_transforms.append(so3flow_transforms)
            self.pose_so3flow_dists.append(so3flow_dist)
            self.pose_SO3flow_dists.append(SO3flow_dist)

    def compute_input_shape_glob_cam_feats(self,
                                           input_feats,
                                           shape,
                                           glob_R,
                                           cam):
        """
        Compute intermediate features given input features, SMPL shape, global rotation and camera.
        Shape could be point estimate ( i.e. (B, num_smpl_betas) tensor) or samples (i.e. (B, N, num_smpl_betas)).
        Other tensors are broadcasted if needed to match shape tensor.

        :param input_feats: (B, N, input_feats_dim) or (B, input_feats_dim) features from ResNet encoder.
        :param shape: (B, N, num_smpl_betas) or (B, num_smpl_betas) SMPL shape parameter samples or point estimate.
        :param glob_R: (B, N, 3, 3) or (B, 3, 3) global rotation.
        :param cam: (B, N, 3) or (B, 3) camera parameters.
        :return: feats: (B, N, input_shape_glob_cam_feats_dim) or (B, input_shape_glob_cam_feats_dim) intermediate features computed from
                        input features, SMPL shape, global rotation and camera..
        """
        if shape.dim() == 3:
            bsize, num_samples = shape.shape[:2]
            infeats_shape_glob_cam = torch.cat([input_feats[:, None, :].expand(-1, num_samples, -1),  # (bsize, num_samples, num_image_features)
                                                shape,  # (bsize, num_samples, num_shape_params)
                                                glob_R.view(bsize, 1, -1).expand(-1, num_samples, -1),  # (bsize, num_samples, 9)
                                                cam[:, None, :].expand(-1, num_samples, -1)],  # (bsize, num_samples, 3)
                                               dim=-1)
        else:
            bsize = shape.shape[0]
            infeats_shape_glob_cam = torch.cat([input_feats,  # (bsize, num_image_features)
                                                shape,  # (bsize, num_shape_params)
                                                glob_R.view(bsize, -1),  # (bsize, 9)
                                                cam],  # (bsize, 3)
                                               dim=-1)

        feats = self.activation(self.fc_input_shape_glob_cam_feats(infeats_shape_glob_cam))  # (*, input_shape_glob_cam_feats_dim)

        return feats

    def compute_flow_context(self,
                             bodypart_idx,
                             ancestors_idx,
                             input_shape_glob_cam_feats,
                             pose_SO3):
        """
        Compute normalising flow input context (or condition) for body-part specific by bodypart_idx.
        Context is computed using ancestor rotations and input/shape/glob/cam features given in input_shape_glob_cam_feats.

        :param bodypart_idx: int, index of body-part of interest
        :param ancestors_idx: list of ints, indices of kinematic ancestors of body-part of interest
        :param input_shape_glob_cam_feats: (B, N, input_shape_glob_cam_feats_dim) or (B, input_shape_glob_cam_feats_dim)
        :param pose_SO3: (B, N, num bodyparts, 3, 3) or (B, num bodyparts, 3, 3) per-part rotations samples or
                          point estimates. Only ancestor rotations will be used (should have been computed already in sequence)
        :return: so3_flow_context: (B, N, flow_context_dim) or (B, flow_context_dim)
        """
        if len(ancestors_idx) > 0:
            if pose_SO3.dim() == 5:
                bsize, num_samples = pose_SO3.shape[:2]
                ancestors_SO3 = pose_SO3[:, :, ancestors_idx, :, :]  # (bsize, num_samples, num ancestors, 3, 3)
                so3flow_context = torch.cat([input_shape_glob_cam_feats,
                                             ancestors_SO3.view(bsize, num_samples, -1)],
                                            dim=-1)  # (bsize, num_samples, input_shape_glob_cam_feats_dim + num_ancestors * 9)
            else:
                bsize = pose_SO3.shape[0]
                ancestors_SO3_zero = pose_SO3[:, ancestors_idx, :, :]  # (bsize, num ancestors, 3, 3)
                so3flow_context = torch.cat([input_shape_glob_cam_feats,
                                             ancestors_SO3_zero.view(bsize, -1)],
                                            dim=-1)  # (bsize, input_shape_glob_cam_feats_dim + num_ancestors * 9)
        else:
            so3flow_context = input_shape_glob_cam_feats  # (*, input_shape_glob_cam_feats_dim)

        so3flow_context = self.activation(self.fc_flow_context[bodypart_idx](so3flow_context))  # (*, flow_context_dim)

        return so3flow_context

    def forward(self,
                input,
                compute_point_est=True,
                num_samples=0,
                use_shape_mode_for_samples=False,
                compute_for_loglik=False,
                shape_for_loglik=None,
                pose_R_for_loglik=None,
                glob_R_for_loglik=None,
                input_feats=None,
                grad_for_pose_point_est=False,
                return_input_feats=False,
                return_input_feats_only=False):
        """
        :param input: (batch_size, num_channels, D, D) tensor.
        :param num_samples: int, number of hierarchical samples to draw from predicted shape and pose distribution.
        :param use_shape_mode_for_samples: bool, only use the shape distribution mode for hierarchical sampling.
        :param compute_for_loglik: bool, trying to compute log-likelihood of given target shape and pose rotmats.
        :param shape_for_loglik: (B, num shape params) tensor of target shapes.
        :param pose_R_for_loglik: (B, num joints, 3, 3) tensor of target pose rotation matrices.
        :param glob_R_for_loglik: (B, 3, 3) tensor of target global body rotation matrices.

        Need to input shape_for_loglik, pose_R_for_loglik and glob_R_for_loglik to compute the log-likelihood of
        target shape and pose parameters w.r.t predicted distributions.
        Since this is an auto-regressive model, predicted distribution parameters depend on samples from up the kinematic tree.
        Need to set target shape and pose as "samples" to compute distribution parameters down the kinematic tree.
        """
        if input_feats is None:
            input_feats = self.image_encoder(input)  # (bsize, num_image_features)

        if return_input_feats_only:
            return {'input_feats': input_feats}

        batch_size = input_feats.shape[0]
        device = input_feats.device

        if compute_for_loglik:
            assert pose_R_for_loglik is not None
            assert pose_R_for_loglik.shape[0] == batch_size
            assert shape_for_loglik is not None
            assert shape_for_loglik.shape[0] == batch_size
            assert glob_R_for_loglik is not None
            assert glob_R_for_loglik.shape[0] == batch_size

        x = self.activation(self.fc1(input_feats))

        ######################################################################################################
        # -------------------------------------- Weak perspective camera -------------------------------------
        ######################################################################################################
        delta_cam = self.fc_cam(x)
        cam = delta_cam + self.init_cam  # (bsize, 3)

        #######################################################################################################
        # ------------------------------------------ Global rotation ------------------------------------------
        #######################################################################################################
        delta_glob = self.fc_glob(x)
        glob = delta_glob + self.init_glob  # (bsize, 6)
        glob_R = rot6d_to_rotmat(glob)  # (bsize, 3, 3)

        #######################################################################################################
        # ----------------------------------------------- Shape ----------------------------------------------
        #######################################################################################################
        shape_params = self.fc_shape(x)  # (bsize, num_shape_params * 2)
        shape_mode = shape_params[:, :self.num_shape_params]
        shape_log_std = shape_params[:, self.num_shape_params:]
        shape_dist = Normal(loc=shape_mode, scale=torch.exp(shape_log_std))
        if num_samples > 0:
            if use_shape_mode_for_samples:
                shape_samples = shape_mode[:, None, :].expand(-1, num_samples, -1)  # (bsize, num_samples, num_shape_params)
            else:
                shape_samples = shape_dist.rsample([num_samples]).transpose(0, 1)  # (bsize, num_samples, num_shape_params)

        #######################################################################################################
        # ------------------------------------------------ Pose -----------------------------------------------
        #######################################################################################################
        if compute_point_est:
            base_zero = torch.zeros(batch_size, 3, device=device)
            pose_so3_point_est = torch.zeros(batch_size, self.num_bodyparts, 3, device=device)  # (bsize, num_bodyparts, 3)
            pose_SO3_point_est = torch.zeros(batch_size, self.num_bodyparts, 3, 3, device=device)  # (bsize, num_bodyparts, 3, 3)
            input_shape_glob_cam_feats_point_est = self.compute_input_shape_glob_cam_feats(input_feats,
                                                                                           shape_mode,
                                                                                           glob_R,
                                                                                           cam)
        if num_samples > 0:
            pose_SO3_samples = torch.zeros(batch_size, num_samples, self.num_bodyparts, 3, 3, device=device)  # (bsize, num_samples, num_bodyparts, 3, 3)
            input_shape_glob_cam_feats_samples = self.compute_input_shape_glob_cam_feats(input_feats,
                                                                                         shape_samples,
                                                                                         glob_R,
                                                                                         cam)
        if compute_for_loglik:
            conditioned_pose_so3flow_dists_for_loglik = []
            conditioned_pose_SO3flow_dists_for_loglik = []
            input_shape_glob_cam_feats_loglik = self.compute_input_shape_glob_cam_feats(input_feats,
                                                                                        shape_for_loglik,
                                                                                        glob_R_for_loglik,
                                                                                        cam)

        # ----------------- Loop over joints and sequentially obtain SO(3) flow distributions -----------------
        for bodypart in range(self.num_bodyparts):  # Note that bodypart 0 here is actually bodypart 1 in SMPL (since we exlcude root bodypart)
            ancestors = self.ancestors_dict[bodypart]

            # ----------------- Compute point estimate -----------------
            if compute_point_est:
                with torch.set_grad_enabled(grad_for_pose_point_est):
                    so3flow_context_point_est = self.compute_flow_context(bodypart_idx=bodypart,
                                                                          ancestors_idx=ancestors,
                                                                          input_shape_glob_cam_feats=input_shape_glob_cam_feats_point_est,
                                                                          pose_SO3=pose_SO3_point_est)
                    so3_zero_sample = forward_trans_conditional_norm_flow(transforms=self.pose_so3flow_transforms[bodypart],
                                                                          base_sample=base_zero,
                                                                          context=so3flow_context_point_est)  # (bsize, 3)
                    SO3_zero_sample = batch_rodrigues(so3_zero_sample)  # (bsize, 3, 3)
                    pose_so3_point_est[:, bodypart, :] = so3_zero_sample
                    pose_SO3_point_est[:, bodypart, :, :] = SO3_zero_sample

            # ------------ Randomly sample rotations for bodypart ------------
            if num_samples > 0:
                so3flow_context_samples = self.compute_flow_context(bodypart_idx=bodypart,
                                                                    ancestors_idx=ancestors,
                                                                    input_shape_glob_cam_feats=input_shape_glob_cam_feats_samples,
                                                                    pose_SO3=pose_SO3_samples)
                SO3_samples = self.pose_SO3flow_dists[bodypart].condition(so3flow_context_samples).rsample([batch_size, num_samples])  # (bsize, num_samples, 3, 3)
                # Note SO3_samples will have dtype = float64
                pose_SO3_samples[:, :, bodypart, :, :] = SO3_samples

            # ------------ Per-part rotation distributions conditioned on GT parent rotations for log-likelihood (teacher forcing) ------------
            if compute_for_loglik:
                so3flow_context_for_loglik = self.compute_flow_context(bodypart_idx=bodypart,
                                                                       ancestors_idx=ancestors,
                                                                       input_shape_glob_cam_feats=input_shape_glob_cam_feats_loglik,
                                                                       pose_SO3=pose_R_for_loglik)
                conditioned_pose_so3flow_dists_for_loglik.append(self.pose_so3flow_dists[bodypart].condition(so3flow_context_for_loglik))
                conditioned_pose_SO3flow_dists_for_loglik.append(self.pose_SO3flow_dists[bodypart].condition(so3flow_context_for_loglik))

        return_dict = {'cam_wp': cam,
                       'glob_rotmat': glob_R}
        if compute_point_est:
            return_dict['pose_rotvecs_point_est'] = pose_so3_point_est
            return_dict['pose_rotmats_point_est'] = pose_SO3_point_est
            return_dict['shape_mode'] = shape_mode
        if num_samples > 0:
            return_dict['pose_rotmats_samples'] = pose_SO3_samples
            return_dict['shape_samples'] = shape_samples
        if compute_for_loglik:
            return_dict['conditioned_pose_so3flow_dists_for_loglik'] = conditioned_pose_so3flow_dists_for_loglik
            return_dict['conditioned_pose_SO3flow_dists_for_loglik'] = conditioned_pose_SO3flow_dists_for_loglik
            return_dict['shape_dist_for_loglik'] = shape_dist

        if return_input_feats:
            return_dict['input_feats'] = input_feats

        return return_dict
