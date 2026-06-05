import argparse
import builtins
import logging
import math
import os
import random
import shutil
import time
import warnings
from tqdm import tqdm
import numpy as np
import faiss
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.models as models
from datetime import timedelta
import torch.distributed as dist
import math
import os, sys
import numpy as np
import torch


import hcsc.loader
from hcsc.logger import EasyLogger
from utils.utils import init_distributed_mode
import hcsc

from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10
from datetime import datetime
from PIL import Image
# ===================== 命令行参数配置 =====================
parser = argparse.ArgumentParser(description='Train HCSC on CIFAR-10')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50', help='model architecture')
parser.add_argument('--lr', '--learning-rate', default=0.03, type=float, metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--epochs', default=300, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('--schedule', default=[120, 160], nargs='*', type=int, help='learning rate schedule')
parser.add_argument('--cos', type=int, default=1, help='use cosine lr schedule')
parser.add_argument('-b', '--batch-size', default=64, type=int, metavar='N')
parser.add_argument('--wd', default=5e-4, type=float, metavar='W', help='weight decay')
parser.add_argument('--dataset', default='cifar10', type=str, help='dataset name')
# 需要添加：
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight_decay', default=1e-4, type=float, help='weight decay')
parser.add_argument('--warmup_epoch', default=0, type=int, help='warmup epochs')
parser.add_argument('--print_freq', default=10, type=int, help='print frequency')
parser.add_argument('--exp_dir', default='experiment', type=str, help='experiment directory')
parser.add_argument('--num-cluster', default='3000,2000,1000', type=str, 
                        help='number of clusters')
parser.add_argument('--aug-plus', type=int, default=1,
                        help='use moco-v2/SimCLR data augmentation')
parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                        help='number of data loading workers (default: 32)')
parser.add_argument('--dist-url', default='tcp://localhost:10034', type=str,
                        help='url used to set up distributed training')
# 模型核心参数
parser.add_argument('--dim', default=128, type=int, help='feature dimension')
parser.add_argument('--queue_length', default=16384, type=int, help='queue size; number of negative pairs')
parser.add_argument('--m', default=0.999, type=float, help='moco momentum of updating key encoder')
parser.add_argument('--T', default=0.2, type=float, help='temperature')
parser.add_argument('--mlp', type=int, default=1, help='use mlp head')
parser.add_argument('--multi_crop', action='store_true', default=False, help='Whether to enable multi-crop transformation')
parser.add_argument("--selection_on_local", action="store_true", default=False, help="whether enable mining on local views")
parser.add_argument("--instance_selection", type=int, default=1, help="Whether enable instance selection")
parser.add_argument("--proto_selection", type=int, default=1, help="Whether enable prototype selection")

# knn monitor
parser.add_argument('--knn-k', default=20, type=int, help='k in kNN monitor')
parser.add_argument('--knn-t', default=0.1, type=float, help='softmax temperature in kNN monitor')

# 工具参数
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint')
parser.add_argument('--results-dir', default='', type=str, metavar='PATH', help='path to cache')

# 运行参数（默认使用空字符串，适配IDE运行）
args = parser.parse_args('')
# 立即添加device属性
args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 补充参数配置
args.epochs = 300
args.cos = True
args.schedule = []  # cos调度时禁用step调度
args.symmetric = False
if args.results_dir == '':
    args.results_dir = "F:\\contrast-11\\HCSC\\run\\cache-" + datetime.now().strftime("%Y-%m-%d-%H-%M-%S-moco")
hcsc_args =args
# 创建结果目录
if not os.path.exists(args.results_dir):
    os.makedirs(args.results_dir)

# # ===================== 数据加载 =====================
# class CIFAR10Pair(CIFAR10):
#     """CIFAR10 Dataset（返回一对增强样本）"""
#     def __getitem__(self, index):
#         img = self.data[index]
#         img = Image.fromarray(img)
#         if self.transform is not None:
#             im_1 = self.transform(img)
#             im_2 = self.transform(img)
#         return im_1, im_2

# # 数据增强
# train_transform = transforms.Compose([
#     transforms.RandomResizedCrop(32),
#     transforms.RandomHorizontalFlip(p=0.5),
#     transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
#     transforms.RandomGrayscale(p=0.2),
#     transforms.ToTensor(),
#     transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
# ])

# test_transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
# ])

# # 加载数据集
# train_data = CIFAR10Pair(root="F:\\contrast-11\\HCSC\\data_DYV", train=True, transform=train_transform, download=False)
# hcsc_train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)

# memory_data = CIFAR10(root="F:\\contrast-11\\HCSC\\data_DYV", train=True, transform=test_transform, download=False)
# memory_loader = DataLoader(memory_data, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)

# test_data = CIFAR10(root="F:\\contrast-11\\HCSC\\data_DYV", train=False, transform=test_transform, download=False)
# test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)



import torch
import torch.nn as nn
import numpy as np

class HCSC(nn.Module):
    """
    Our proposed HCSC framework with instance selection
    and prototype selection.

    Args:
        base_encoder (nn.Module class): query encoder model class(use ResNet50 by default)
        dim (int): feature dimension (default: 128)
        queue_length: queue size; number of negative samples/prototypes (default: 16384)
        m: momentum for updating key encoder (default: 0.999)
        T: temperature 
        mlp: whether to use mlp projection
        multi_crop: (bool) whether using multi crops augmentation
        instance_selection: (bool) whether enable instance selection
        proto_selection: (bool) whether enable prototype selection
        selection_on_local: (bool) whether apply mining strategy on local views.
        logger: (obj) a logger used to store some mediate variables during training.
    """
    def __init__(self, 
                 base_encoder, 
                 dim=128, 
                 queue_length=16384, 
                 m=0.999, 
                 T=0.2, 
                 mlp=True,
                 multi_crop=False,
                 instance_selection=True,
                 proto_selection=True,
                 selection_on_local=True,
                 logger=None,
                 **kwargs):
        super().__init__()

        self.queue_length = queue_length
        self.m = m
        self.T = T
        self.multi_crop = multi_crop
        self.selection_on_local = selection_on_local
        self.logger = logger
        self.instance_selection = instance_selection
        self.proto_selection = proto_selection
        # create the encoders

        self.encoder_q = base_encoder(num_classes=dim)
        self.encoder_k = base_encoder(num_classes=dim)

        if mlp:
            dim_mlp = self.encoder_q.fc.weight.shape[1]
            self.encoder_q.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_q.fc)
            self.encoder_k.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_k.fc)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(dim, queue_length))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        self.register_buffer("queue_index", torch.arange(0, queue_length))
        self.buffer_dict = dict()
        self.mined_index = list()
    
    def set_epoch(self, epoch):
        self.epoch = epoch

    def momentum_update_key_encoder(self):
        self._momentum_update_key_encoder()

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, index=None):
        # gather keys before updating queue
        keys = concat_all_gather(keys)
        if index is not None:
            index = concat_all_gather(index)
        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.queue_length % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        if index is not None:
            self.queue_index[ptr: ptr + batch_size] = index
        ptr = (ptr + batch_size) % self.queue_length  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

    @torch.no_grad()
    def sample_neg_instance(self, im2cluster, centroids, density, index):
        """
        mining based on the clustering results
        """
        queue_p_samples = []
        for layer in range(len(im2cluster)):
            proto_logit = torch.mm(self.queue.clone().detach().permute(1, 0), centroids[layer].permute(1, 0)) 
            density[layer] = density[layer].clamp(min=1e-3)
            proto_logit /= density[layer]
            label = im2cluster[layer][index] 
            logit = proto_logit.clone().detach().softmax(-1) 
            p_sample = 1 - logit[:, label].t() 
            queue_p_samples.append(p_sample) 

        self.selected_masks = []
        avg_sample_ratios = []
        for p_sample in queue_p_samples:
            neg_sampler = torch.distributions.bernoulli.Bernoulli(p_sample.clamp(0.0, 0.999))
            selected_mask = neg_sampler.sample() # [N_q, N_queue]
            try:
                self.selected_masks.append(selected_mask)
                avg_sample_ratios.append(p_sample.mean())
            except:
                # when no samples are selected
                selected_mask = torch.ones([index.shape[0], self.queue.shape[1]]).cuda()
                self.selected_masks.append(selected_mask)
        return self.selected_masks, avg_sample_ratios   

    @torch.no_grad()
    def extract_key_feat(self, im_k):
        self._momentum_update_key_encoder()  # update the key encoder
        # shuffle for making use of BN
        im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)

        k = self.encoder_k(im_k)  # keys: NxC
        k = nn.functional.normalize(k, dim=1)

        # undo shuffle
        k = self._batch_unshuffle_ddp(k, idx_unshuffle)
        return k

    def extract_feat(self, images, is_eval=False):
        # global views
        if is_eval:
            k = self.encoder_k(images)  
            k = nn.functional.normalize(k, dim=1)            
            return k
        im_q, im_k = images[0], images[1]
        
        q = self.encoder_q(im_q)  # queries: NxC
        q = nn.functional.normalize(q, dim=1)        
        # compute key features
        if self.multi_crop:
            k = self.extract_key_feat(im_k)
            local_views = list()
            for n, im_local in enumerate(images[2:]):
                local_q = self.encoder_q(im_local)
                local_q = nn.functional.normalize(local_q, dim=1)
                local_views.append(local_q)

            return q, k, local_views
        else:
            k = self.extract_key_feat(im_k)
            # compute query features
            return q, k, None

    def forward(self, images, is_eval=False, cluster_result=None, index=None):
        """
        Input:
            images: a list of images, where
                images[0] as im_q and
                images[1] as im_k 
                others are local views, which are also treated 
                as keys
            is_eval: return momentum embeddings (used for clustering)
            cluster_result: cluster assignments, centroids, and density
            index: indices for training samples
        Output:
            logits, targets, proto_logits, proto_targets
        """
        if is_eval:
            return self.extract_feat(images, is_eval)
        else:
            q, k, local_views = self.extract_feat(images, is_eval)
        
        proto_logits, proto_labels, proto_selected, temp_protos = self.get_protos(q, index, cluster_result)

        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # negative logits: Nxr
        if proto_labels is not None and self.instance_selection:
            try:
                self.selected_masks, sample_ratios = self.sample_neg_instance(cluster_result['im2cluster'], proto_selected, temp_protos, index)
                self.buffer_dict['avg_sample_ratios'] = sum(sample_ratios) / len(sample_ratios)
                l_neg = list()
                for selected_mask in self.selected_masks:
                    logit = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])
                    mask = selected_mask.clone().float()
                    l_neg.append(logit * mask)
            except:
                l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])
        else:
            l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        # logits: Nx(1+queue_length) or list(Nx(1+queue_length))
        if isinstance(l_neg, list):
            logits = [torch.cat([l_pos, l_n], dim=1)/self.T for l_n in l_neg]
            labels = [torch.zeros(logit.shape[0], dtype=torch.long).cuda() for logit in logits]
        else:
            logits = torch.cat([l_pos, l_neg], dim=1)
            logits /= self.T
            labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        # dequeue and enqueue
        local_proto_logits, local_proto_targets = None, None
        local_logits, local_labels = None, None
        # computing local logits when enabling multi-crop
        if self.multi_crop:
            local_logits, local_labels = self.compute_local_logits(q, k, local_views, index)
            local_proto_logits, local_proto_targets = self.compute_local_proto_logits(local_views, proto_selected, cluster_result, index)

            # print(self.inst_temp[index])
        self._dequeue_and_enqueue(k, index)
        return logits, labels, proto_logits, proto_labels, local_logits, local_labels, local_proto_logits, local_proto_targets

    def compute_local_proto_logits(self, local_views, proto_selected, cluster_result, index):
        """
        Compute prototype logits for local views. 
        """
        if cluster_result is not None:
            local_proto_logits = list()
            local_proto_targets = list()
            for local_view in local_views:
                # reuse the get_protos() with q replaced by local_view
                proto_logits, proto_labels, proto_selected, temp_protos = self.get_protos(local_view, index, cluster_result)
                local_proto_logits.append(proto_logits)
                local_proto_targets.append(proto_labels)
            return local_proto_logits, local_proto_targets
        else:
            return None, None

    def compute_local_logits(self, q, k, local_views, index):
        """
        Args:
            q: (torch.Tensor([N, D]))
            k: (torch.Tensor([N, D]))
            local_views: (list[torch.Tensor([N, D])]) 
                features of local views that could be additional keys or
                queries

        Returns:
            local_logits: (list[torch.Tensor([N, queue_length+1])])
            local_labels: (list[torch.Tensor([N])])
        """
            # mining
        if self.selection_on_local and hasattr(self, "selected_masks"):
            l_pos_list = list()
            l_neg_list = list()
            for selected_mask in self.selected_masks:
                l_pos_list.extend([torch.einsum('nc,nc->n', [local_view, k]).unsqueeze(-1) for local_view in local_views])
                for local_view in local_views:
                    logit = torch.einsum('nc,ck->nk', [local_view, self.queue.clone().detach()])
                    mask = selected_mask.clone().float()
                    l_neg_list.append(logit * mask)
        else:
            l_pos_list = [torch.einsum('nc,nc->n', [local_view, k]).unsqueeze(-1) for local_view in local_views]
            l_neg_list = [torch.einsum('nc,ck->nk', [local_view, self.queue.clone().detach()]) for local_view in local_views]
        

        local_logits = [torch.cat([l_pos, l_neg], dim=1)/self.T for (l_pos, l_neg) in zip(l_pos_list, l_neg_list)]
        local_labels = [torch.zeros(logit.shape[0], dtype=torch.long).cuda() for logit in local_logits]
        
        return local_logits, local_labels

    def get_protos(self, q, index, cluster_result):
        # prototypical contrast
        if cluster_result is not None:  
            proto_labels = []
            proto_logits = []
            proto_selecteds = []
            temp_protos = []
            for n, (im2cluster,prototypes,density) in enumerate(zip(cluster_result['im2cluster'],cluster_result['centroids'],cluster_result['density'])):

                pos_proto_id = im2cluster[index]
                pos_prototypes = prototypes[pos_proto_id] 
                proto_selecteds.append(prototypes)
                temp_protos.append(density)

                # sample negative prototypes
                all_proto_id = [i for i in range(im2cluster.max())]

                neg_proto_id = set(all_proto_id)-set(pos_proto_id.tolist())
                if self.proto_selection:
                    if n==(len(cluster_result['im2cluster']) - 1):
                        neg_proto_id = list(neg_proto_id)
                        neg_proto_id = torch.LongTensor(neg_proto_id).to(pos_proto_id.device)
                        neg_prototypes = prototypes[neg_proto_id] # [N_neg, D]
                        logits_proto = torch.cat([torch.einsum('nc,nc->n',[q, pos_prototypes]).unsqueeze(-1),
                                                    torch.mm(q, neg_prototypes.t())], dim=1) # [N_q, 1+N_neg]
                        temp_map = torch.cat([density[pos_proto_id].unsqueeze(-1), 
                                          density[neg_proto_id].unsqueeze(0).repeat([q.shape[0], 1])], dim=1)
                        logits_proto = logits_proto / temp_map 
                    else:
                        cluster2cluster = cluster_result['cluster2cluster'][n]
                        prot_logits = cluster_result['logits'][n]
                        neg_proto_id = list(neg_proto_id)
                        neg_proto_id = torch.LongTensor(neg_proto_id).to(pos_proto_id.device)
                        neg_prototypes = prototypes[neg_proto_id] # [N_neg, D]
                        neg_mask = self.sample_neg_protos(im2cluster, cluster2cluster, pos_proto_id, prot_logits, n, cluster_result) # [N, N_neg]
                        neg_logit_mask = neg_mask.clone().float() # [N_q, N_neg]
                        neg_logits = torch.mm(q, neg_prototypes.t()) #[N_q, N_neg] ~ range([-1, 1])
                        neg_logits *= neg_logit_mask
                        logits_proto = torch.cat([torch.einsum('nc,nc->n',[q, pos_prototypes]).unsqueeze(-1),
                                                 neg_logits], dim=1)
                        temp_map = torch.cat([density[pos_proto_id].unsqueeze(-1), 
                                          density[neg_proto_id].unsqueeze(0).repeat([q.shape[0], 1])], dim=1)
                        logits_proto = logits_proto / temp_map
                else:
                    neg_proto_id = list(neg_proto_id)
                    neg_proto_id = torch.LongTensor(neg_proto_id).to(pos_proto_id.device)
                    neg_prototypes = prototypes[neg_proto_id] # [N_neg, D]
                    # [N, 1] + [N, N_neg] => [N, 1 + N_neg]
                    logits_proto = torch.cat([torch.einsum('nc,nc->n',[q, pos_prototypes]).unsqueeze(-1),
                                                torch.mm(q, neg_prototypes.t())], dim=1)
                    temp_map = torch.cat([density[pos_proto_id].unsqueeze(-1), 
                                          density[neg_proto_id].unsqueeze(0).repeat([q.shape[0], 1])], dim=1)

                    logits_proto = logits_proto / temp_map
       
                
                labels_proto = torch.zeros(q.shape[0], dtype=torch.long).cuda()
               
                proto_labels.append(labels_proto)
                proto_logits.append(logits_proto)

            return proto_logits, proto_labels, proto_selecteds, temp_protos
        else:
            return None, None, None, None

    def sample_neg_protos(self, im2cluster, cluster2cluster, pos_proto_id, prot_logits, n, cluster_results):
        """
        Sampling negative prototypes given pos_proto_id and layer

        Args:
            im2cluster: [N_bs]
            pos_proto_id: [N_bs] actually im2cluster[index]
            proto_dist_mat: [N_bs, N_l] used for sampling strategy.
            prot_logits: [N_l, N_{l+1}] proto logits of cucrrent layer
        """
        all_proto_id = [i for i in range(im2cluster.max())] 
        neg_proto_id = set(all_proto_id)-set(pos_proto_id.tolist())
        neg_proto_id = torch.LongTensor(list(neg_proto_id)).to(pos_proto_id.device)
        upper_pos_proto_id = cluster2cluster[pos_proto_id] # [N_q]
        densities = cluster_results['density'][n+1] / cluster_results['density'][n+1].mean() * self.T
        sampling_prob = 1 - (prot_logits / densities).softmax(-1)[neg_proto_id, :][:, upper_pos_proto_id].t()
        neg_sampler = torch.distributions.bernoulli.Bernoulli(sampling_prob.clamp(0.0001, 0.999))
        selected_mask = neg_sampler.sample() #[N_q, N_neg]
        return selected_mask

# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output




def build_model(args, logger):
    backbone = models.__dict__[args.arch]
    model = HCSC(
        backbone,
        args.dim,
        args.queue_length,
        args.m,
        args.T,
        args.mlp,
        args.multi_crop,
        args.instance_selection,
        args.proto_selection,
        args.selection_on_local,
        logger)
    return model

hcsc_model=build_model

def build_dataloaders(args):
    return getattr(hcsc.loader, args.dataset)(args)
hcsc_train_loader=build_dataloaders

def build_optimizer(args, model):
    total_batch_size = args.batch_size * dist.get_world_size()
    ## scale up the batch size
    args.lr = args.lr * total_batch_size / 256
    print("total batch size is {}, lr is scaled up to {}".format(total_batch_size, args.lr))

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                            momentum=args.momentum,
                            weight_decay=args.weight_decay)
    return optimizer
hcsc_optimizer=build_optimizer
criterion = nn.CrossEntropyLoss().to(args.device)

def main():

    
    init_distributed_mode(args)
    

    args.num_cluster = args.num_cluster.split(',')
    if not os.path.exists(args.exp_dir):
        os.makedirs(args.exp_dir, exist_ok=True)
    
    logger = EasyLogger(args.exp_dir, 0, args.rank)
    # create dataset
    train_loader, eval_loader, train_dataset, eval_dataset, train_sampler = build_dataloaders(args)
    dist.barrier()
    args.dataset_size = len(train_dataset)
    # create model
    if args.rank == 0:
        print("=> creating model '{}'".format(args.arch))
    model = build_model(args, logger)
    model = model.to(args.device)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().to(args.device)

    optimizer = build_optimizer(args, model)
    scheduler = adjust_learning_rate
    
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location=args.device)

            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'], strict=False)
            if "optimizer" in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer'])
            else:
                print("No optimizer state!")
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True
    
    for epoch in range(args.start_epoch, args.epochs):
        logger.set_epoch(epoch)
        if hasattr(model.module, "set_epoch"):
            model.module.set_epoch(epoch)
        cluster_result = None
        if epoch >= args.warmup_epoch:
            # compute momentum features for center-cropped images
            features = compute_features(eval_loader, model, args)         
            # placeholder for clustering result
            cluster_result = {'im2cluster':[],'centroids':[],'density':[], 'cluster2cluster': [], 'logits': []}
            for i, num_cluster in enumerate(args.num_cluster):
                cluster_result['im2cluster'].append(torch.zeros(len(eval_dataset),dtype=torch.long).cuda())
                cluster_result['centroids'].append(torch.zeros(int(num_cluster),args.dim).cuda())
                cluster_result['density'].append(torch.zeros(int(num_cluster)).cuda())
                if i < (len(args.num_cluster) - 1):
                    cluster_result['cluster2cluster'].append(torch.zeros(int(num_cluster), dtype=torch.long).cuda())
                    cluster_result['logits'].append(torch.zeros([int(num_cluster), int(args.num_cluster[i+1])]).cuda())
            if dist.get_rank() == 0:
                features[torch.norm(features,dim=1)>1.5] /= 2 
                features = features.numpy()
                cluster_result = run_hkmeans(features,args)  
                # save the clustering result
                try:
                    torch.save(cluster_result,os.path.join(args.exp_dir, 'clusters_%d'%epoch))  
                except:
                    pass
            dist.barrier()  
            # broadcast clustering result
            for k, data_list in cluster_result.items():
                for data_tensor in data_list:                
                    dist.broadcast(data_tensor, 0, async_op=False)   
            
        train_sampler.set_epoch(epoch)
        scheduler(optimizer, epoch, args)

        # train for one epoch
        hcsc_train(train_loader, model, criterion, optimizer, epoch, args, cluster_result)

        if (epoch+1)%5==0 and dist.get_rank()==0:
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
            }, is_best=False, filename='{}/checkpoint_{:04d}.pth.tar'.format(args.exp_dir,epoch))

def hcsc_train(train_loader, model, criterion, optimizer, epoch, args, cluster_result=None):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = dict()
    acc_inst = dict()
    losses['InsLoss_sum'] = AverageMeter('InsLoss_sum', ':.4e')
    acc_inst['Acc@Inst_avg'] = AverageMeter('Acc@Inst_avg', ':6.2f')
    acc_proto = AverageMeter('Acc@Proto', ':6.2f')
    buffer_meter = dict()
    
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, acc_inst, acc_proto, buffer_meter],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()
    end = time.time()

    for i, (images, index) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        images = [image.to(args.device) for image in images]
                
        # compute output
        output, target, output_proto, target_proto, local_logits, local_labels, local_proto_logits, local_proto_targets = model(images, 
                                                                                cluster_result=cluster_result, 
                                                                                index=index)

        # InfoNCE loss
        loss = 0.
        if isinstance(target, list):
            loss_total = 0.
            for k, (out, tar) in enumerate(zip(output, target)):
                loss = criterion(out, tar)
                loss_total += loss
                if f'InsLoss_{k}' not in buffer_meter:
                    buffer_meter[f'InsLoss_{k}'] = AverageMeter(f'InsLoss_{k}', ':.4e')
                buffer_meter[f'InsLoss_{k}'].update(loss.item(), images[0].size(0))
                acc = accuracy(out, tar)[0] 
                if f'Acc@Inst{k}' not in buffer_meter:
                    buffer_meter[f'Acc@Inst{k}'] = AverageMeter(f'Acc@Inst{k}', ":6.2f")
                buffer_meter[f'Acc@Inst{k}'].update(acc[0], images[0].size(0))
                losses['InsLoss_sum'].update(loss.item(), images[0].size(0))
                acc_inst['Acc@Inst_avg'].update(acc[0], images[0].size(0))
                loss = loss_total
        else:
            loss = criterion(output, target)  
            losses['InsLoss_sum'].update(loss.item(), images[0].size(0))
            acc = accuracy(output, target)[0] 
            acc_inst['Acc@Inst_avg'].update(acc[0], images[0].size(0))
        
        # InfoNCE Loss on local views with multi-crop
        if local_logits is not None:
            # print("local nce")
            loss_local = 0
            for vid, (local_logit, local_target) in enumerate(zip(local_logits, local_labels)):
                loss_local += criterion(local_logit, local_target)
                acc_local = accuracy(local_logit, local_target)[0]
                if f"acc_local{vid}" not in buffer_meter:
                    buffer_meter[f"acc_local{vid}"] = AverageMeter(f"acc_local{vid}", ":6.4f")
                
                buffer_meter[f"acc_local{vid}"].update(acc_local[0], images[0].size(0))
            loss += loss_local

        # HProtoNCE loss
        if output_proto is not None:
            loss_proto = 0
            for proto_out,proto_target in zip(output_proto, target_proto):
                loss_proto += criterion(proto_out, proto_target)  
                accp = accuracy(proto_out, proto_target)[0] 
                acc_proto.update(accp[0], images[0].size(0))
                
            # average loss across all sets of prototypes
            loss_proto /= len(args.num_cluster) 
            loss += loss_proto

        # HProtoNCE Loss on local views
        if local_proto_logits is not None:
            loss_local_proto = 0
            for vid, (proto_logits, proto_targets) in enumerate(zip(local_proto_logits, local_proto_targets)):
                for logit, target in zip(proto_logits, proto_targets):
                    loss_local_proto += criterion(logit, target)
                    accp = accuracy(logit, target)[0]
                    if f"proto_acc_local{vid}" not in buffer_meter:
                        buffer_meter[f"proto_acc_local{vid}"] = AverageMeter(f"proto_acc_local{vid}", ":6.4f")
                    buffer_meter[f"proto_acc_local{vid}"].update(accp[0], images[0].size(0))
            loss_local_proto /= len(args.num_cluster)
            loss += loss_local_proto
        
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            if dist.get_rank() == 0:
                progress.display(i)

def compute_features(eval_loader, model, args):
    print('Computing features...')
    model.eval()
    features = torch.zeros(len(eval_loader.dataset),args.dim).cuda()
    for i, (images, index) in enumerate(tqdm(eval_loader)):
        with torch.no_grad():
            images = images.cuda(non_blocking=True)
            feat = model(images, is_eval=True)
            features[index] = feat
    dist.barrier()        
    dist.all_reduce(features, op=dist.ReduceOp.SUM)
    return features.cpu()

def run_hkmeans(x, args):
    """
    This function is a hierarchical 
    k-means: the centroids of current hierarchy is used
    to perform k-means in next step
    """
    
    print('performing kmeans clustering')
    results = {'im2cluster':[],'centroids':[],'density':[], 'cluster2cluster':[], 'logits':[]}
    
    for seed, num_cluster in enumerate(args.num_cluster):
        # intialize faiss clustering parameters
        d = x.shape[1]
        k = int(num_cluster)
        clus = faiss.Clustering(d, k)
        clus.verbose = True
        clus.niter = 20
        clus.nredo = 5
        clus.seed = seed
        clus.max_points_per_centroid = 1000
        clus.min_points_per_centroid = 10

        res = faiss.StandardGpuResources()
        cfg = faiss.GpuIndexFlatConfig()
        cfg.useFloat16 = False
        cfg.device = args.local_rank  
        index = faiss.GpuIndexFlatL2(res, d, cfg)  
        if seed==0: # the first hierarchy from instance directly
            clus.train(x, index)   
            D, I = index.search(x, 1) # for each sample, find cluster distance and assignments
        else:
            # the input of higher hierarchy is the centorid of lower one
            clus.train(results['centroids'][seed - 1].cpu().numpy(), index)
            D, I = index.search(results['centroids'][seed - 1].cpu().numpy(), 1)
        
        im2cluster = [int(n[0]) for n in I]
        # sample-to-centroid distances for each cluster 
        ## centroid in lower level to higher level
        Dcluster = [[] for c in range(k)]          
        for im,i in enumerate(im2cluster):
            Dcluster[i].append(D[im][0])

       # get cluster centroids
        centroids = faiss.vector_to_array(clus.centroids).reshape(k,d)

        if seed>0: # the im2cluster of higher hierarchy is the index of previous hierachy
            im2cluster = np.array(im2cluster) # enable batch indexing
            results['cluster2cluster'].append(torch.LongTensor(im2cluster).cuda())
            im2cluster = im2cluster[results['im2cluster'][seed - 1].cpu().numpy()]
            im2cluster = list(im2cluster)
    
        if len(set(im2cluster))==1:
            print("Warning! All samples are assigned to one cluster")

        # concentration estimation (phi)        
        density = np.zeros(k)
        for i,dist in enumerate(Dcluster):
            if len(dist)>1:
                d = (np.asarray(dist)**0.5).mean()/np.log(len(dist)+10)            
                density[i] = d     
                
        #if cluster only has one point, use the max to estimate its concentration        
        dmax = density.max()
        for i,dist in enumerate(Dcluster):
            if len(dist)<=1:
                density[i] = dmax 

        density = density.clip(np.percentile(density,10),np.percentile(density,90)) 
        density = args.T*density/density.mean() 
        
        # convert to cuda Tensors for broadcast
        centroids = torch.Tensor(centroids).cuda()
        centroids = nn.functional.normalize(centroids, p=2, dim=1)    
        if seed > 0: #maintain a logits from lower prototypes to higher
            proto_logits = torch.mm(results['centroids'][-1], centroids.t())
            results['logits'].append(proto_logits.cuda())


        density = torch.Tensor(density).cuda()
        im2cluster = torch.LongTensor(im2cluster).cuda()    
        results['centroids'].append(centroids)
        results['density'].append(density)
        results['im2cluster'].append(im2cluster)    
        
    return results

    
def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        for meter in self.meters:
            if isinstance(meter, AverageMeter):
                entries += [str(meter)]
            elif isinstance(meter, dict):
                entries += [str(v) for (k, v) in meter.items()]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    if args.cos:  # cosine lr schedule
        lr = args.lr_final + 0.5 * (1. + math.cos(math.pi * epoch / args.epochs)) * (args.lr - args.lr_final)
    else:  # stepwise lr schedule
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()
