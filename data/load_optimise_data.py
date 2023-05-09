import os
import cv2
import numpy as np
from collections import defaultdict


def load_opt_initialise_data_from_pred_output(pred_image_dir,
                                              pred_output_dir):

    opt_init_data = defaultdict(list)
    opt_init_data['fnames'] = sorted([f for f in os.listdir(pred_image_dir) if f.endswith(('.jpg', '.jpeg', '.png'))])

    for image_fname in opt_init_data['fnames']:
        image = cv2.cvtColor(cv2.imread(os.path.join(pred_image_dir, image_fname)), cv2.COLOR_BGR2RGB)
        opt_init_data['orig_image'].append(image)

        pred_outputs = np.load(os.path.join(pred_output_dir, os.path.splitext(image_fname)[0] + '_pred.npz'))
        for key in pred_outputs:
            opt_init_data[key].append(pred_outputs[key])

    print('\nLoaded optimisation initialisation data:')
    for key in opt_init_data:
        if key not in ('orig_image', 'fnames'):
            opt_init_data[key] = np.stack(opt_init_data[key], axis=0)
            print(key, opt_init_data[key].shape)

    return opt_init_data
