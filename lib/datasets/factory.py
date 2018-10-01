from datasets.vg_hdf5 import vg_hdf5

def get_imdb(roidb_name, imdb_name, rpndb_name, split=-1, num_im=-1):
    return vg_hdf5('%s.h5'%roidb_name, '%s-dicts.json'%roidb_name, imdb_name, rpndb_name, split=split, num_im=num_im)
