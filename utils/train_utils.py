import torch


def check_for_nans_in_output(return_dict):
    found_nans = False
    for key, val in return_dict.items():
        if isinstance(val, torch.Tensor):
            num_nans = val.isnan().sum()
            if num_nans != 0:
                print(f'Found {num_nans} NaNs in {key} with shape {val.shape}')
                found_nans = True
    return found_nans
