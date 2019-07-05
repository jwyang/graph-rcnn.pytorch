from .vg_hdf5 import vg_hdf5

def build_data_loader(cfg, split="train", num_im=0):
    if cfg.DATASET.NAME == "vg_bm":
        return vg_hdf5(cfg, split=split, num_im=num_im)
    else:
        raise NotImplementedError("Unsupported dataset {}.".format(dataset))
#         cfg.data_dir = "data/vg"
