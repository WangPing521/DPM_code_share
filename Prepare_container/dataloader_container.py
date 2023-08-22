from datasets.UDA.BraTS20 import flairInterface, t1ceInterface, t2Interface
from datasets.UDA.mmwhs_new import CTNewInterface, MRNewInterface
from general_utils.fixSeed_fn_tool import fix_all_seed


def data_load(config_box):
    fix_all_seed(config_box['seed'])

    if config_box['Data']['name'] == 'BraTS20':
        handler_flair = flairInterface()
        handler_t1ce = t1ceInterface()
        handler_t2 = t2Interface()
        handler_flair.compile_dataloader_params(**config_box["DataLoader"])
        handler_t1ce.compile_dataloader_params(**config_box["DataLoader"])
        handler_t2.compile_dataloader_params(**config_box["DataLoader"])

        train_flair_loader, val_flair_loader, test_flair_loader = handler_flair.DataLoaders()
        train_t1ce_loader, val_t1ce_loader, test_t1ce_loader = handler_t1ce.DataLoaders()
        train_t2_loader, val_t2_loader, test_t2_loader = handler_t2.DataLoaders()

        Loaders_container = {
            "flair": [train_flair_loader, val_flair_loader, test_flair_loader],
            "t1ce": [train_t1ce_loader, val_t1ce_loader, test_t1ce_loader],
            "t2": [train_t2_loader, val_t2_loader, test_t2_loader]
        }

    elif config_box['Data']['name'] == 'mmwhs':
        handler_ct = CTNewInterface()
        handler_mr = MRNewInterface()
        handler_ct.compile_dataloader_params(**config_box["DataLoader"])
        handler_mr.compile_dataloader_params(**config_box["DataLoader"])
        train_ct_loader, val_ct_loader, test_ct_loader = handler_ct.DataLoaders()
        train_mr_loader, val_mr_loader, test_mr_loader = handler_mr.DataLoaders()
        Loaders_container = {
            "CT": [train_ct_loader, val_ct_loader, test_ct_loader],
            "MR": [train_mr_loader, val_mr_loader, test_mr_loader]
        }

    else:
        raise NotImplementedError(config_box['Data']['name'])

    return Loaders_container
