# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import pickle
import shutil
import tempfile

import mmcv
import torch
import torch.distributed as dist
from mmcv.runner import get_dist_info
import cv2
import numpy as np
from mmpose.core.utils.trilateration import keypoints_from_heatmaps_tril
from mpl_toolkits.mplot3d import Axes3D
import json

def trans_heatmap(hm):
    prob = []
    H, W = hm.shape
    for i in range(H):
        for j in range(W):
            prob.append(hm[i][j])
    return prob

def topK_sort(dis, topK=4):
    # import pdb
    # pdb.set_trace()
    
    dis_new=[]
    length=len(dis)-1
    sorted_nums = sorted(enumerate(dis), key=lambda x: x[1])
    for i in range(topK):
        dis_new.append( sorted_nums[length-i][1] )
    
    return dis_new

def topK(heatmaps, topK=[0,5] ):
    N, K, H, W = heatmaps.shape
    heatmaps_trans = np.reshape(heatmaps, (N,K,-1))
    # heatmaps_trans_map = np.argsort(heatmaps_trans) # zhengxu
    heatmaps_trans_map = np.argsort(-heatmaps_trans) # fanxu
    
    indices = np.indices((N,K,H*W))
    indices[2,:,:,:] = heatmaps_trans_map
    heatmaps_trans_sort = heatmaps_trans[indices[0], indices[1], indices[2]]
    heatmaps_trans_sort=heatmaps_trans_sort[:,:,topK[0]:topK[1]]
    return heatmaps_trans_sort

def gauss_plot(dis_map, save_path=None):
    import matplotlib.pyplot as plt
    from scipy.stats import multivariate_normal
    from mpl_toolkits.mplot3d import Axes3D
    
    # import pdb
    # pdb.set_trace()
    
    M=64
    x, y = np.meshgrid(np.linspace(0,256,M), np.linspace(0,256,M))
    z = dis_map
    
    
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap='rainbow',alpha = 1.0)
    
    if save_path is not None:
        plt.savefig(save_path)
    # plt.show()
    


def single_gpu_test(model, data_loader):
    """Test model with a single gpu.

    This method tests model with a single gpu and displays test progress bar.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.


    Returns:
        list: The prediction results.
    """

    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    for data in data_loader:
        with torch.no_grad():
            result = model(return_loss=False, return_heatmap=True, **data)
        results.append(result)

        # use the first key as main key to calculate the batch size
        batch_size = len(next(iter(data.values())))
        for _ in range(batch_size):
            prog_bar.update()

    return results


def multi_gpu_test(model, data_loader, tmpdir=None, gpu_collect=False):
    """Test model with multiple gpus.

    This method tests model with multiple gpus and collects the results
    under two different modes: gpu and cpu modes. By setting 'gpu_collect=True'
    it encodes results to gpu tensors and use gpu communication for results
    collection. On cpu mode it saves the results on different gpus to 'tmpdir'
    and collects them by the rank 0 worker.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.
        tmpdir (str): Path of directory to save the temporary results from
            different gpus under cpu mode.
        gpu_collect (bool): Option to use either gpu or cpu to collect results.

    Returns:
        list: The prediction results.
    """
    model.eval()
    results = []
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))
    for data in data_loader:
        with torch.no_grad():
            result = model(return_loss=False, **data)
        results.append(result)

        if rank == 0:
            # use the first key as main key to calculate the batch size
            batch_size = len(next(iter(data.values())))
            for _ in range(batch_size * world_size):
                prog_bar.update()

    # collect results from all ranks
    if gpu_collect:
        results = collect_results_gpu(results, len(dataset))
    else:
        results = collect_results_cpu(results, len(dataset), tmpdir)
    return results


def collect_results_cpu(result_part, size, tmpdir=None):
    """Collect results in cpu mode.

    It saves the results on different gpus to 'tmpdir' and collects
    them by the rank 0 worker.

    Args:
        result_part (list): Results to be collected
        size (int): Result size.
        tmpdir (str): Path of directory to save the temporary results from
            different gpus under cpu mode. Default: None

    Returns:
        list: Ordered results.
    """
    rank, world_size = get_dist_info()
    # create a tmp dir if it is not specified
    if tmpdir is None:
        MAX_LEN = 512
        # 32 is whitespace
        dir_tensor = torch.full((MAX_LEN, ),
                                32,
                                dtype=torch.uint8,
                                device='cuda')
        if rank == 0:
            mmcv.mkdir_or_exist('.dist_test')
            tmpdir = tempfile.mkdtemp(dir='.dist_test')
            tmpdir = torch.tensor(
                bytearray(tmpdir.encode()), dtype=torch.uint8, device='cuda')
            dir_tensor[:len(tmpdir)] = tmpdir
        dist.broadcast(dir_tensor, 0)
        tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
    else:
        mmcv.mkdir_or_exist(tmpdir)
    # synchronizes all processes to make sure tmpdir exist
    dist.barrier()
    # dump the part result to the dir
    mmcv.dump(result_part, osp.join(tmpdir, f'part_{rank}.pkl'))
    # synchronizes all processes for loading pickle file
    dist.barrier()
    # collect all parts
    if rank != 0:
        return None

    # load results of all parts from tmp dir
    part_list = []
    for i in range(world_size):
        part_file = osp.join(tmpdir, f'part_{i}.pkl')
        part_list.append(mmcv.load(part_file))
    # sort the results
    ordered_results = []
    for res in zip(*part_list):
        ordered_results.extend(list(res))
    # the dataloader may pad some samples
    ordered_results = ordered_results[:size]
    # remove tmp dir
    shutil.rmtree(tmpdir)
    return ordered_results


def collect_results_gpu(result_part, size):
    """Collect results in gpu mode.

    It encodes results to gpu tensors and use gpu communication for results
    collection.

    Args:
        result_part (list): Results to be collected
        size (int): Result size.

    Returns:
        list: Ordered results.
    """

    rank, world_size = get_dist_info()
    # dump result part to tensor with pickle
    part_tensor = torch.tensor(
        bytearray(pickle.dumps(result_part)), dtype=torch.uint8, device='cuda')
    # gather all result part tensor shape
    shape_tensor = torch.tensor(part_tensor.shape, device='cuda')
    shape_list = [shape_tensor.clone() for _ in range(world_size)]
    dist.all_gather(shape_list, shape_tensor)
    # padding result part tensor to max length
    shape_max = torch.tensor(shape_list).max()
    part_send = torch.zeros(shape_max, dtype=torch.uint8, device='cuda')
    part_send[:shape_tensor[0]] = part_tensor
    part_recv_list = [
        part_tensor.new_zeros(shape_max) for _ in range(world_size)
    ]
    # gather all result part
    dist.all_gather(part_recv_list, part_send)

    if rank == 0:
        part_list = []
        for recv, shape in zip(part_recv_list, shape_list):
            part_list.append(
                pickle.loads(recv[:shape[0]].cpu().numpy().tobytes()))
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        return ordered_results
    return None
