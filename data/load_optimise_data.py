import os
import cv2
import numpy as np
from collections import defaultdict


def load_optimise_data_from_pred_output(pred_image_dir,
                                        pred_output_dir):

    optimise_data = defaultdict(list)
    optimise_data['fnames'] = sorted([f for f in os.listdir(pred_image_dir) if f.endswith(('.jpg', '.jpeg', '.png'))])

    for image_fname in optimise_data['fnames']:
        image = cv2.cvtColor(cv2.imread(os.path.join(pred_image_dir, image_fname)), cv2.COLOR_BGR2RGB)
        optimise_data['orig_image'].append(image)

        pred_outputs = np.load(os.path.join(pred_output_dir, os.path.splitext(image_fname)[0] + '_pred.npz'))
        for key in pred_outputs:
            optimise_data[key].append(pred_outputs[key])

    print('\nLoaded optimisation data:')
    for key in optimise_data:
        if key not in ('orig_image', 'fnames'):
            optimise_data[key] = np.stack(optimise_data[key], axis=0)
            print(key, optimise_data[key].shape)

    return optimise_data
