import numpy as np


def save_pred_output(hrnet_output,
                     cropped_for_proxy,
                     proxy_rep,
                     humaniflow_output,
                     save_path):

    np.savez(save_path,
             bbox_centre=hrnet_output['bbox_centre'].cpu().detach().numpy(),
             bbox_height=hrnet_output['bbox_height'].cpu().detach().numpy(),
             bbox_width=hrnet_output['bbox_width'].cpu().detach().numpy(),
             hrnet_joints2D=hrnet_output['joints2D'].cpu().detach().numpy(),
             hrnet_joints2D_conf=hrnet_output['joints2Dconfs'].cpu().detach().numpy(),
             cropped_image=cropped_for_proxy['rgb'].cpu().detach().numpy()[0],
             cropped_joints2D=cropped_for_proxy['joints2D'].cpu().detach().numpy()[0],
             proxy_rep=proxy_rep.cpu().detach().numpy()[0],
             pose_axisangle_point_est=humaniflow_output['pose_axisangle_point_est'].cpu().detach().numpy()[0],
             pose_rotmats_point_est=humaniflow_output['pose_rotmats_point_est'].cpu().detach().numpy()[0],
             shape_mode=humaniflow_output['shape_mode'].cpu().detach().numpy()[0],
             glob_rotmat=humaniflow_output['glob_rotmat'].cpu().detach().numpy()[0],
             cam_wp=humaniflow_output['cam_wp'].cpu().detach().numpy()[0],
             input_feats=humaniflow_output['input_feats'].cpu().detach().numpy()[0])

