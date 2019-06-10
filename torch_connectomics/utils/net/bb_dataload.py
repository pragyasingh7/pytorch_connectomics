import os,sys
import numpy as np
import h5py

import torch
import torch.nn as nn
import torch.utils.data
import torchvision.utils as vutils

from torch_connectomics.utils.net import *
from torch_connectomics.model.model_zoo import *

TASK_MAP = {0: 'neuron segmentation',
            1: 'synapse detection',
            11: 'synapse polarity detection',
            2: 'mitochondria segmentation',
            22:'mitochondira segmentation with skeleton transform'}
 

def get_output(args, model_io_size, model):
    """Prepare dataloader for training and inference.
    """

    pad_size = model_io_size // 2

    image_file = '/n/coxfs01/donglai/jwr/20um_20180720/bfly_v2-2.json'
    DimB = json.load(open(jn))

    bb_file = '/n/coxfs01/donglai/data/JWR/toufiq_synapse/v2/syn_bbox.txt'
    bb = open(bb_file, 'r')
    bb_lines = bb.readlines()
    bb_syn = np.array([[int(j) for j in i.split(' ')] for i in bb_lines])

    inp_file = '/n/coxfs01/donglai/data/JWR/toufiq_synapse/eval/cell26_t_0_all6.txt'
    inp = open(inp_file, 'r')
    inp_syn = inp.readlines()
    inp_syn = np.array([int(i) for i in inp_syn])
    '''
    target_file = '/n/coxfs01/donglai/data/JWR/toufiq_synapse/eval/cell26_t_0_all6_proofread.txt'
    tar = open(target_file, 'r')
    tar_syn = tar.readlines()
    tar_syn = np.array([int(i) for i in tar_syn])
    '''
    inp_bb = bb_syn[inp_syn]
    x0 = inp_bb[:,4]; x1 = inp_bb[:,5]
    y0 = inp_bb[:,2]; y1 = inp_bb[:,3]
    z0 = inp_bb[:,0]; z1 = inp_bb[:,1]

    for i in range(len(x0)):
        print('loading data')
        output = bfly(DimB, x0[i], x1[i], y0[i], y1[i], z0[i], z1[i], model)
        print('saving output')
        hf = h5py.File('cell26/inp_index_'+str(i)+'.h5', 'w')
        hf.create_dataset('main', data = output, compression='gzip')
        hf.close()



def bfly(bfly_db, x0, x1, y0, y1, z0, z1, model, tile_sz=2560, dt=np.uint8,st=1, tile_ratio=1, resize_order=1):
    # x: column
    # y: row
    # no padding at the boundary
    # st: starting index 0 or 1
    result = np.zeros((8, 256, 256)).astype(np.uint8)
    c0 = x0 // tile_sz # floor
    c1 = (x1 + tile_sz-1) // tile_sz # ceil
    r0 = y0 // tile_sz
    r1 = (y1 + tile_sz-1) // tile_sz
    for z in range(8):
        pattern = bfly_db["sections"][z0+z]
        pattern = '/mnt'+pattern[2:]
        for row in range(r0, r1):
            for column in range(c0, c1):
                path = pattern.format(row=row+st, column=column+st)
                if not os.path.exists(path): 
                    #return None
                    patch = 128*np.ones((tile_sz,tile_sz),dtype=np.uint8)/255.0
                else:
                    if path[-3:]=='tif':
                        import tifffile
                        patch = tifffile.imread(path)/255.0
                    else:
                        import scipy.misc
                        patch = scipy.misc.imread(path, 'L')/255.0
                if tile_ratio != 1:
                    # scipy.misc.imresize: only do uint8
                    from scipy.ndimage import zoom
                    patch = zoom(patch, tile_ratio, order=resize_order)

                xp0 = column * tile_sz
                xp1 = (column+1) * tile_sz
                yp0 = row * tile_sz
                yp1 = (row + 1) * tile_sz
                if patch is not None:
                    x_ = (x0+x1)//2
                    y_ = (y0+y1)//2
                    x0a = max(x_-128, xp0)
                    x1a = min(x_+128, xp1)
                    y0a = max(y_-128, yp0)
                    y1a = min(y_+128, yp1)
                    #print(x0a, x1a, y0a, y1a)
                    dx = (x1a-x0a)
                    dy = (y1a-y0a)
                    margin_x = (256-dx)//2
                    margin_y = (256-dy)//2
                    #print(margin_x, margin_y, dx, dy)
                    result[z, margin_y:margin_y+dy, margin_x:margin_x+dx] = patch[y0a-yp0:y1a-yp0, x0a-xp0:x1a-xp0]
    result = np.pad(result,((pad_size[0],pad_size[0]), 
                            (pad_size[1],pad_size[1]), 
                            (pad_size[2],pad_size[2])), 'reflect') 
    result = np.expand_dims(result, axis=0)
    result = torch.tensor(result)
    print('getting model output')
    output = model(result)

    return output


def main():
    args = get_args(mode='test')
    model_io_size, device = init(args)
    model = fpn(in_channel=1, out_channel=3)
    model = DataParallelWithCallback(model, device_ids=range(args.num_gpu))
    model = model.to(device)
    print('loading model')
    if bool(args.load_model):
        print('Load pretrained model:')
        print(args.pre_model)
        if exact:
            model.load_state_dict(torch.load(args.pre_model))
        else:
            pretrained_dict = torch.load(args.pre_model)
            model_dict = model.state_dict()
            # 1. filter out unnecessary keys
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            # 2. overwrite entries in the existing state dict 
            if size_match:
                model_dict.update(pretrained_dict) 
            else:
                for param_tensor in pretrained_dict:
                    if model_dict[param_tensor].size() == pretrained_dict[param_tensor].size():
                        model_dict[param_tensor] = pretrained_dict[param_tensor]       
            # 3. load the new state dict
            model.load_state_dict(model_dict)
    print('beginning test...')
    get_output(args, model_io_size, model)
    print('finishing.')


