# from datetime import datetime
# from functools import partial
# import json
# from turtle import pd
# from PIL import Image
# from torch.utils.data import DataLoader
# from torch.version import cuda
# from torchvision import transforms
# from torchvision.datasets import CIFAR10
# from torchvision.models import resnet
# from tqdm import tqdm
# import argparse


# import os
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import argparse
# from datetime import datetime
# from functools import partial
# import math
# import os
# from tqdm import tqdm
# import numpy as np
# import faiss
# import torch
# import torch.nn as nn
# import torch.backends.cudnn as cudnn
# import torch.distributed as dist
# import torch.optim
# import torchvision.models as models
# from torchvision import transforms
# from torchvision.datasets import CIFAR10
# from PIL import Image
# from torch.utils.data import DataLoader
# from torchvision.models import resnet

# import os
# from pathlib import Path
# # 在 HCSC/main.py 文件的最开头添加以下代码
# import sys
# import os
# from pathlib import Path

# # 获取当前文件的绝对路径
# current_file = Path(__file__).resolve()
# print(f"当前文件: {current_file}")

# # 获取HCSC目录的路径
# hcsc_dir = current_file.parent
# print(f"HCSC目录: {hcsc_dir}")

# # 获取项目根目录（contrast-11）的路径
# project_root = hcsc_dir.parent
# print(f"项目根目录: {project_root}")

# # 清除可能存在的重复路径
# sys.path = [p for p in sys.path if str(project_root) not in p]

# # 添加项目根目录到Python路径（最优先）
# if str(project_root) not in sys.path:
#     sys.path.insert(0, str(project_root))
#     print(f"✅ 已添加项目根目录到Python路径: {project_root}")

# # 添加HCSC目录到Python路径
# if str(hcsc_dir) not in sys.path:
#     sys.path.insert(0, str(hcsc_dir))
#     print(f"✅ 已添加HCSC目录到Python路径: {hcsc_dir}")

# # 打印调试信息
# print("\n=== 当前Python路径（前5个）===")
# for i, path in enumerate(sys.path[:5]):
#     print(f"{i}: {path}")

# # 现在尝试导入HCSC模块
# try:
#     from HCSC.hcsc.hcsc import HCSC
#     print("✅ HCSC模块导入成功")
# except ImportError as e:
#     print(f"❌ HCSC模块导入失败: {e}")
#     # 提供详细的错误信息
#     import traceback
#     traceback.print_exc()
#     sys.exit(1)  # 如果导入失败，退出程序
# import numpy as np
# import torch
# import torch.nn.functional as F

# import hcsc
# from HCSC.hcsc.hcsc import HCSC

# from hcsc.logger import EasyLogger
# from utils.options import parse_args_main
# from utils.utils import init_distributed_mode

# from sklearn.cluster import MiniBatchKMeans


# # from .utils.utils import init_distributed_mode

# # 设置参数
# parser = argparse.ArgumentParser(description='Train HCSC on CIFAR-10')
# parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50', help='model architecture')

# parser.add_argument('--lr', '--learning-rate', default=0.03, type=float, metavar='LR', help='initial learning rate', dest='lr')
# parser.add_argument('--epochs', default=300, type=int, metavar='N',help='number of total epochs to run')
# parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
#                         help='manual epoch number (useful on restarts)')
# parser.add_argument('--schedule', default=[120, 160], nargs='*', type=int, help='learning rate schedule (when to drop lr by 10x)')
# parser.add_argument('--cos', type=int, default=1, help='use cosine lr schedule')

# parser.add_argument('-b', '--batch-size', default=64, type=int, metavar='N')
# parser.add_argument('--wd', default=5e-4, type=float, metavar='W', help='weight decay')

# # 🔽 添加以下层级聚类参数 🔽
# parser.add_argument('--cluster-levels', default=100, type=int,
#                    help='number of hierarchical clustering levels')
# parser.add_argument('--cluster-nums', default=[1000,500], nargs='*', type=int,
#                    help='number of clusters at each level')
# # 🔼 添加结束 🔼

# parser.add_argument('--dim', default=128, type=int,help='feature dimension')
# parser.add_argument('--queue_length', default=16384, type=int,help='queue size; number of negative pairs')
# parser.add_argument('--m', default=0.999, type=float, help='moco momentum of updating key encoder (default: 0.999)')
# parser.add_argument('--T', default=0.2, type=float, help='temperature')
# parser.add_argument('--mlp', type=int, default=1, help='use mlp head')
# parser.add_argument('--multi_crop', action='store_true',default=False,help='Whether to enable multi-crop transformation')
# parser.add_argument("--selection_on_local", action="store_true", default=False, help="whether enable mining on local views")

# parser.add_argument("--instance_selection", type=int, default=1, help="Whether enable instance selection")
# parser.add_argument("--proto_selection", type=int, default=1,help="Whether enable prototype selection")

# # knn monitor
# parser.add_argument('--knn-k', default=20, type=int, help='k in kNN monitor')
# parser.add_argument('--knn-t', default=0.1, type=float, help='softmax temperature in kNN monitor; could be different with moco-t')


# # utils
# parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
# parser.add_argument('--results-dir', default='', type=str, metavar='PATH', help='path to cache (default: none)')


# '''
# args = parser.parse_args()  # running in command line
# '''
# args = parser.parse_args('')  # running in ipynb

# # set command line arguments here when running in ipynb
# args.epochs = 300               # 修改处
# args.cos = True
# args.schedule = []  # cos in use
# args.symmetric = False
# if args.results_dir == '':
#     args.results_dir = "F:\\contrast-11\\HCSC\\run\\cache-" + datetime.now().strftime("%Y-%m-%d-%H-%M-%S-moco")

# hcsc_args =args


# class SplitBatchNorm(nn.BatchNorm2d):
#     def __init__(self, num_features, num_splits, **kw):
#         super().__init__(num_features, **kw)
#         self.num_splits = num_splits

#     def forward(self, input):
#         N, C, H, W = input.shape
#         if self.training or not self.track_running_stats:
#             running_mean_split = self.running_mean.repeat(self.num_splits)
#             running_var_split = self.running_var.repeat(self.num_splits)
#             outcome = nn.functional.batch_norm(
#                 input.view(-1, C * self.num_splits, H, W), running_mean_split, running_var_split,
#                 self.weight.repeat(self.num_splits), self.bias.repeat(self.num_splits),
#                 True, self.momentum, self.eps).view(N, C, H, W)
#             self.running_mean.data.copy_(running_mean_split.view(self.num_splits, C).mean(dim=0))
#             self.running_var.data.copy_(running_var_split.view(self.num_splits, C).mean(dim=0))
#             return outcome
#         else:
#             return nn.functional.batch_norm(
#                 input, self.running_mean, self.running_var,
#                 self.weight, self.bias, False, self.momentum, self.eps)

# class ModelBase(nn.Module):
#     """
#     Common CIFAR ResNet recipe.
#     Comparing with ImageNet ResNet recipe, it:
#     (i) replaces conv1 with kernel=3, str=1
#     (ii) removes pool1
#     """

#     def __init__(self, feature_dim=128, arch=None, bn_splits=16):
#         super(ModelBase, self).__init__()

#         # use split batchnorm
#         norm_layer = partial(SplitBatchNorm, num_splits=bn_splits) if bn_splits > 1 else nn.BatchNorm2d
#         resnet_arch = getattr(resnet, arch)
#         net = resnet_arch(num_classes=feature_dim, norm_layer=norm_layer)

#         self.net = []
#         for name, module in net.named_children():
#             if name == 'conv1':
#                 module = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
#             if isinstance(module, nn.MaxPool2d):
#                 continue
#             if isinstance(module, nn.Linear):
#                 self.net.append(nn.Flatten(1))
#             self.net.append(module)

#         self.net = nn.Sequential(*self.net)

#     def forward(self, x):
#         x = self.net(x)
#         # note: not normalized here
#         return x

# class CIFAR10Pair(CIFAR10):
#     """CIFAR10 Dataset.
#     """
#     def __getitem__(self, index):
#         img = self.data[index]
#         img = Image.fromarray(img)

#         if self.transform is not None:
#             im_1 = self.transform(img)
#             im_2 = self.transform(img)

#         return im_1, im_2

# train_transform = transforms.Compose([
#     transforms.RandomResizedCrop(32),
#     transforms.RandomHorizontalFlip(p=0.5),
#     transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
#     transforms.RandomGrayscale(p=0.2),
#     transforms.ToTensor(),
#     transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])

# test_transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])

# # data_processing prepare
# train_data = CIFAR10Pair(root="F:\\contrast-11\\HCSC\\data", train=True, transform=train_transform, download=False)
# hcsc_train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)

# memory_data = CIFAR10(root="F:\\contrast-11\\HCSC\\data", train=True, transform=test_transform, download=False)
# memory_loader = DataLoader(memory_data, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)

# test_data = CIFAR10(root="F:\\contrast-11\\HCSC\\data", train=False, transform=test_transform, download=False)
# test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)


# def create_hcsc_model(args):
#     """创建HCSC模型实例"""
#     import torchvision.models as models

#     # 根据arch参数选择基础编码器
#     if args.arch == 'resnet18':
#         base_encoder = models.resnet18
#     elif args.arch == 'resnet50':
#         base_encoder = models.resnet50
#     else:
#         base_encoder = models.resnet50  # 默认使用ResNet50


#     hcsc_model = HCSC(
#         base_encoder=base_encoder,
#         dim=args.dim,
#         queue_length=args.queue_length,
#         m=args.m,
#         T=args.T,
#         mlp=args.mlp,
#         multi_crop=args.multi_crop,
#         instance_selection=args.instance_selection,
#         proto_selection=args.proto_selection,
#         selection_on_local=args.selection_on_local
#     ).cuda()

#     return hcsc_model

# # utils
# @torch.no_grad()
# def concat_all_gather(tensor):
#     """
#     Performs all_gather operation on the provided tensors.
#     *** Warning ***: torch.distributed.all_gather has no gradient.
#     """
#     tensors_gather = [torch.ones_like(tensor)
#         for _ in range(torch.distributed.get_world_size())]
#     torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

#     output = torch.cat(tensors_gather, dim=0)
#     return output

# class HCSC(nn.Module):
#     """
#     Our proposed HCSC framework with instance selection
#     and prototype selection.

#     Args:
#         base_encoder (nn.Module class): query encoder model class(use ResNet50 by default)
#         dim (int): feature dimension (default: 128)
#         queue_length: queue size; number of negative samples/prototypes (default: 16384)
#         m: momentum for updating key encoder (default: 0.999)
#         T: temperature
#         mlp: whether to use mlp projection
#         multi_crop: (bool) whether using multi crops augmentation
#         instance_selection: (bool) whether enable instance selection
#         proto_selection: (bool) whether enable prototype selection
#         selection_on_local: (bool) whether apply mining strategy on local views.
#         logger: (obj) a logger used to store some mediate variables during training.
#     """
#     def __init__(self,  base_encoder, dim=128, queue_length=16384, m=0.999, T=0.2, mlp=True,multi_crop=False, instance_selection=True, proto_selection=True, selection_on_local=True, logger=None,
#                  **kwargs):
#         super().__init__()

#         self.queue_length = queue_length
#         self.m = m
#         self.T = T
#         self.multi_crop = multi_crop
#         self.selection_on_local = selection_on_local
#         self.logger = logger
#         self.instance_selection = instance_selection
#         self.proto_selection = proto_selection
#         # create the encoders

#         self.encoder_q = base_encoder(num_classes=dim)
#         self.encoder_k = base_encoder(num_classes=dim)

#         if mlp:
#             dim_mlp = self.encoder_q.fc.weight.shape[1]
#             self.encoder_q.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_q.fc)
#             self.encoder_k.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_k.fc)

#         for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
#             param_k.data.copy_(param_q.data)  # initialize
#             param_k.requires_grad = False  # not update by gradient

#         # create the queue
#         self.register_buffer("queue", torch.randn(dim, queue_length))
#         self.queue = nn.functional.normalize(self.queue, dim=0)

#         self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
#         self.register_buffer("queue_index", torch.arange(0, queue_length))
#         self.buffer_dict = dict()
#         self.mined_index = list()

#    #动量更新，缓慢更新键编码器参数
#     @torch.no_grad()
#     def _momentum_update_key_encoder(self):
#         """
#         Momentum update of the key encoder
#         """
#         for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
#             param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)


#     #队列管理，维护负样本队列
#     @torch.no_grad()
#     def _dequeue_and_enqueue(self, keys, index=None):
#         # gather keys before updating queue
#         keys = concat_all_gather(keys)
#         if index is not None:
#             index = concat_all_gather(index)
#         batch_size = keys.shape[0]

#         ptr = int(self.queue_ptr)
#         assert self.queue_length % batch_size == 0  # for simplicity

#         # replace the keys at ptr (dequeue and enqueue)
#         self.queue[:, ptr:ptr + batch_size] = keys.T
#         if index is not None:
#             self.queue_index[ptr: ptr + batch_size] = index
#         ptr = (ptr + batch_size) % self.queue_length  # move pointer

#         self.queue_ptr[0] = ptr
#     #批处理打乱，改善BatchNorm效果
#     @torch.no_grad()
#     def _batch_shuffle_ddp(self, x):
#         """
#         Batch shuffle, for making use of BatchNorm.
#         *** Only support DistributedDataParallel (DDP) model. ***
#         """
#         # gather from all gpus
#         batch_size_this = x.shape[0]
#         x_gather = concat_all_gather(x)
#         batch_size_all = x_gather.shape[0]

#         num_gpus = batch_size_all // batch_size_this

#         # random shuffle index
#         idx_shuffle = torch.randperm(batch_size_all).cuda()

#         # broadcast to all gpus
#         torch.distributed.broadcast(idx_shuffle, src=0)

#         # index for restoring
#         idx_unshuffle = torch.argsort(idx_shuffle)

#         # shuffled index for this gpu
#         gpu_idx = torch.distributed.get_rank()
#         idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

#         return x_gather[idx_this], idx_unshuffle

#     @torch.no_grad()
#     def _batch_unshuffle_ddp(self, x, idx_unshuffle):
#         """
#         Undo batch shuffle.
#         *** Only support DistributedDataParallel (DDP) model. ***
#         """
#         # gather from all gpus
#         batch_size_this = x.shape[0]
#         x_gather = concat_all_gather(x)
#         batch_size_all = x_gather.shape[0]

#         num_gpus = batch_size_all // batch_size_this

#         # restored index for this gpu
#         gpu_idx = torch.distributed.get_rank()
#         idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

#         return x_gather[idx_this]

#     @torch.no_grad()
#     def sample_neg_instance(self, im2cluster, centroids, density, index):
#         """
#         mining based on the clustering results
#         """
#         queue_p_samples = []
#         for layer in range(len(im2cluster)):
#             proto_logit = torch.mm(self.queue.clone().detach().permute(1, 0), centroids[layer].permute(1, 0))
#             density[layer] = density[layer].clamp(min=1e-3)
#             proto_logit /= density[layer]
#             label = im2cluster[layer][index]
#             logit = proto_logit.clone().detach().softmax(-1)
#             p_sample = 1 - logit[:, label].t()
#             queue_p_samples.append(p_sample)

#         self.selected_masks = []
#         avg_sample_ratios = []
#         for p_sample in queue_p_samples:
#             neg_sampler = torch.distributions.bernoulli.Bernoulli(p_sample.clamp(0.0, 0.999))
#             selected_mask = neg_sampler.sample() # [N_q, N_queue]
#             try:
#                 self.selected_masks.append(selected_mask)
#                 avg_sample_ratios.append(p_sample.mean())
#             except:
#                 # when no samples are selected
#                 selected_mask = torch.ones([index.shape[0], self.queue.shape[1]]).cuda()
#                 self.selected_masks.append(selected_mask)
#         return self.selected_masks, avg_sample_ratios
#     #特征提取，获取查询和键特征
#     @torch.no_grad()
#     def extract_key_feat(self, im_k):
#         self._momentum_update_key_encoder()  # update the key encoder
#         # shuffle for making use of BN
#         im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)

#         k = self.encoder_k(im_k)  # keys: NxC
#         k = nn.functional.normalize(k, dim=1)

#         # undo shuffle
#         k = self._batch_unshuffle_ddp(k, idx_unshuffle)
#         return k

#     def extract_feat(self, images, is_eval=False):
#         # global views
#         if is_eval:
#             k = self.encoder_k(images)
#             k = nn.functional.normalize(k, dim=1)
#             return k
#         im_q, im_k = images[0], images[1]

#         q = self.encoder_q(im_q)  # queries: NxC
#         q = nn.functional.normalize(q, dim=1)
#         # compute key features
#         if self.multi_crop:
#             k = self.extract_key_feat(im_k)
#             local_views = list()
#             for n, im_local in enumerate(images[2:]):
#                 local_q = self.encoder_q(im_local)
#                 local_q = nn.functional.normalize(local_q, dim=1)
#                 local_views.append(local_q)

#             return q, k, local_views
#         else:
#             k = self.extract_key_feat(im_k)
#             # compute query features
#             return q, k, None


#    #实例对比损失
#     def forward(self, images, is_eval=False, cluster_result=None, index=None):
#         """
#         Input:
#             images: a list of images, where
#                 images[0] as im_q and
#                 images[1] as im_k
#                 others are local views, which are also treated
#                 as keys
#             is_eval: return momentum embeddings (used for clustering)
#             cluster_result: cluster assignments, centroids, and density
#             index: indices for training samples
#         Output:
#             logits, targets, proto_logits, proto_targets
#         """
#         #提取特征
#         if is_eval:
#             return self.extract_feat(images, is_eval)
#         else:
#             q, k, local_views = self.extract_feat(images, is_eval)
#              #正样本对比损失
#         proto_logits, proto_labels, proto_selected, temp_protos = self.get_protos(q, index, cluster_result)
#          # 负样本对比损失
#         l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
#        #负样本相似度计算，带随机性机制
#         if proto_labels is not None and self.instance_selection:
#             try:
#                 self.selected_masks, sample_ratios = self.sample_neg_instance(cluster_result['im2cluster'], proto_selected, temp_protos, index)
#                 self.buffer_dict['avg_sample_ratios'] = sum(sample_ratios) / len(sample_ratios)
#                 l_neg = list()
#                 for selected_mask in self.selected_masks:
#                     logit = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])
#                     mask = selected_mask.clone().float()
#                     l_neg.append(logit * mask)
#             except:
#                 l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])
#         else:
#             l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])#负样本相似度计算

#         # logits: Nx(1+queue_length) or list(Nx(1+queue_length))合并logits
#         if isinstance(l_neg, list):
#             logits = [torch.cat([l_pos, l_n], dim=1)/self.T for l_n in l_neg]
#             labels = [torch.zeros(logit.shape[0], dtype=torch.long).cuda() for logit in logits]
#         else:
#             logits = torch.cat([l_pos, l_neg], dim=1)#合并logits
#             logits /= self.T#温度缩放
#             labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda() #标签，样本索引为0

#         # # dequeue and enqueue
#         # local_proto_logits, local_proto_targets = None, None
#         # local_logits, local_labels = None, None
#         # # computing local logits when enabling multi-crop
#         # if self.multi_crop:
#         #     local_logits, local_labels = self.compute_local_logits(q, k, local_views, index)
#         #     local_proto_logits, local_proto_targets = self.compute_local_proto_logits(local_views, proto_selected, cluster_result, index)

#         #     # print(self.inst_temp[index])
#         # self._dequeue_and_enqueue(k, index)

#         #计算实例损失
#         instance_loss= nn.CrossEntropyLoss()(logits, labels)#最大化正样本相似度，最小化负样本相似度
#         # return logits, labels, proto_logits, proto_labels, local_logits, local_labels, local_proto_logits, local_proto_targets
#         # 原型损失
#         proto_loss = 0
#         if proto_logits is not None and self.proto_selection:
#             for i, (plogits, plabels) in enumerate(zip(proto_logits, proto_labels)):
#                 proto_loss += nn.CrossEntropyLoss()(plogits, plabels)
#             proto_loss /= len(proto_logits)

#         # 总损失
#         total_loss = instance_loss + 0.5 * proto_loss

#         # 更新队列
#         self._dequeue_and_enqueue(k, index)

#         return total_loss


#     def compute_local_proto_logits(self, local_views, proto_selected, cluster_result, index):
#         """
#         Compute prototype logits for local views.
#         """
#         if cluster_result is not None:
#             local_proto_logits = list()
#             local_proto_targets = list()
#             for local_view in local_views:
#                 # reuse the get_protos() with q replaced by local_view
#                 proto_logits, proto_labels, proto_selected, temp_protos = self.get_protos(local_view, index, cluster_result)
#                 local_proto_logits.append(proto_logits)
#                 local_proto_targets.append(proto_labels)
#             return local_proto_logits, local_proto_targets
#         else:
#             return None, None

#     def compute_local_logits(self, q, k, local_views, index):
#         """
#         Args:
#             q: (torch.Tensor([N, D]))
#             k: (torch.Tensor([N, D]))
#             local_views: (list[torch.Tensor([N, D])])
#                 features of local views that could be additional keys or
#                 queries

#         Returns:
#             local_logits: (list[torch.Tensor([N, queue_length+1])])
#             local_labels: (list[torch.Tensor([N])])
#         """
#             # mining
#         if self.selection_on_local and hasattr(self, "selected_masks"):
#             l_pos_list = list()
#             l_neg_list = list()
#             for selected_mask in self.selected_masks:
#                 l_pos_list.extend([torch.einsum('nc,nc->n', [local_view, k]).unsqueeze(-1) for local_view in local_views])
#                 for local_view in local_views:
#                     logit = torch.einsum('nc,ck->nk', [local_view, self.queue.clone().detach()])
#                     mask = selected_mask.clone().float()
#                     l_neg_list.append(logit * mask)
#         else:
#             l_pos_list = [torch.einsum('nc,nc->n', [local_view, k]).unsqueeze(-1) for local_view in local_views]
#             l_neg_list = [torch.einsum('nc,ck->nk', [local_view, self.queue.clone().detach()]) for local_view in local_views]


#         local_logits = [torch.cat([l_pos, l_neg], dim=1)/self.T for (l_pos, l_neg) in zip(l_pos_list, l_neg_list)]
#         local_labels = [torch.zeros(logit.shape[0], dtype=torch.long).cuda() for logit in local_logits]

#         return local_logits, local_labels


#     #原型对比损失
#     def get_protos(self, q, index, cluster_result):
#         # prototypical contrast
#         if cluster_result is not None:
#             proto_labels = []
#             proto_logits = []
#             proto_selecteds = []
#             temp_protos = []
#             #遍历每个层级
#             for n, (im2cluster,prototypes,density) in enumerate(zip(cluster_result['im2cluster'],cluster_result['centroids'],cluster_result['density'])):
#                 #获取正原型
#                 pos_proto_id = im2cluster[index]
#                 pos_prototypes = prototypes[pos_proto_id]
#                 proto_selecteds.append(prototypes)
#                 temp_protos.append(density)

#                 # sample negative prototypes获取负原型
#                 all_proto_id = [i for i in range(im2cluster.max())]

#                 neg_proto_id = set(all_proto_id)-set(pos_proto_id.tolist())
#                 if self.proto_selection:
#                     if n==(len(cluster_result['im2cluster']) - 1):
#                         neg_proto_id = list(neg_proto_id)
#                         neg_proto_id = torch.LongTensor(neg_proto_id).to(pos_proto_id.device)
#                         neg_prototypes = prototypes[neg_proto_id] # [N_neg, D]
#                         logits_proto = torch.cat([torch.einsum('nc,nc->n',[q, pos_prototypes]).unsqueeze(-1),
#                                                     torch.mm(q, neg_prototypes.t())], dim=1) # [N_q, 1+N_neg]计算原型logits
#                         temp_map = torch.cat([density[pos_proto_id].unsqueeze(-1),
#                                           density[neg_proto_id].unsqueeze(0).repeat([q.shape[0], 1])], dim=1)#密度加权
#                         logits_proto = logits_proto / temp_map
#                     else:
#                         cluster2cluster = cluster_result['cluster2cluster'][n]
#                         prot_logits = cluster_result['logits'][n]
#                         neg_proto_id = list(neg_proto_id)
#                         neg_proto_id = torch.LongTensor(neg_proto_id).to(pos_proto_id.device)
#                         neg_prototypes = prototypes[neg_proto_id] # [N_neg, D]
#                         neg_mask = self.sample_neg_protos(im2cluster, cluster2cluster, pos_proto_id, prot_logits, n, cluster_result) # [N, N_neg]
#                         neg_logit_mask = neg_mask.clone().float() # [N_q, N_neg]
#                         neg_logits = torch.mm(q, neg_prototypes.t()) #[N_q, N_neg] ~ range([-1, 1])
#                         neg_logits *= neg_logit_mask
#                         logits_proto = torch.cat([torch.einsum('nc,nc->n',[q, pos_prototypes]).unsqueeze(-1),
#                                                  neg_logits], dim=1)
#                         temp_map = torch.cat([density[pos_proto_id].unsqueeze(-1),
#                                           density[neg_proto_id].unsqueeze(0).repeat([q.shape[0], 1])], dim=1)
#                         logits_proto = logits_proto / temp_map
#                 else:
#                     neg_proto_id = list(neg_proto_id)
#                     neg_proto_id = torch.LongTensor(neg_proto_id).to(pos_proto_id.device)
#                     neg_prototypes = prototypes[neg_proto_id] # [N_neg, D]
#                     # [N, 1] + [N, N_neg] => [N, 1 + N_neg]
#                     logits_proto = torch.cat([torch.einsum('nc,nc->n',[q, pos_prototypes]).unsqueeze(-1),
#                                                 torch.mm(q, neg_prototypes.t())], dim=1)
#                     temp_map = torch.cat([density[pos_proto_id].unsqueeze(-1),
#                                           density[neg_proto_id].unsqueeze(0).repeat([q.shape[0], 1])], dim=1)

#                     logits_proto = logits_proto / temp_map


#                 labels_proto = torch.zeros(q.shape[0], dtype=torch.long).cuda()

#                 proto_labels.append(labels_proto)
#                 proto_logits.append(logits_proto)

#             return proto_logits, proto_labels, proto_selecteds, temp_protos
#         else:
#             return None, None, None, None


#     def sample_neg_protos(self, im2cluster, cluster2cluster, pos_proto_id, prot_logits, n, cluster_results):
#         """
#         Sampling negative prototypes given pos_proto_id and layer

#         Args:
#             im2cluster: [N_bs]
#             pos_proto_id: [N_bs] actually im2cluster[index]
#             proto_dist_mat: [N_bs, N_l] used for sampling strategy.
#             prot_logits: [N_l, N_{l+1}] proto logits of cucrrent layer
#         """
#         all_proto_id = [i for i in range(im2cluster.max())]
#         neg_proto_id = set(all_proto_id)-set(pos_proto_id.tolist())
#         neg_proto_id = torch.LongTensor(list(neg_proto_id)).to(pos_proto_id.device)
#         upper_pos_proto_id = cluster2cluster[pos_proto_id] # [N_q]
#         densities = cluster_results['density'][n+1] / cluster_results['density'][n+1].mean() * self.T
#         sampling_prob = 1 - (prot_logits / densities).softmax(-1)[neg_proto_id, :][:, upper_pos_proto_id].t()
#         neg_sampler = torch.distributions.bernoulli.Bernoulli(sampling_prob.clamp(0.0001, 0.999))
#         selected_mask = neg_sampler.sample() #[N_q, N_neg]
#         return selected_mask


# # def forward(self,loss,proto_logits, proto_labels,temp_protos):
# #     if proto_logits is not None and self.proto_selection:
# #        proto_loss = 0
# #        for i, (plogits, plabels) in enumerate(zip(proto_logits, proto_labels)):
# #         proto_loss += nn.CrossEntropyLoss()(plogits / temp_protos[i], plabels)
# #        proto_loss /= len(proto_logits)
# #        total_loss = loss + proto_loss  # 结合实例损失和原型损失
# #     else:
# #         total_loss = loss
# #         return total_loss

# # hcsc_model = HCSC (
# #     dim=args.hcsc_dim,
# #     queue_length= args.hcsc_q,
# #     m= args.hcsc_m,
# #     T =args.hcsc_T,
# #     mlp=args.mlp,
# #     multi_crop=args.multi_crop,
# #     instance_selection=args.instance_selection,
# #     proto_selection=args.proto_selection,
# #     selection_on_local=args.selection_on_local,
# #     logger=args.logger,
# # ).cuda()


# """
#     CIFAR10 Dataset.


# """

# def optimized_hierarchical_kmeans(features, levels=2, clusters=[1000, 500]):
#     """优化版的层级聚类，减少计算负担和内存占用"""
#     cluster_result = {
#         'im2cluster': [],
#         'centroids': [],
#         'density': [],
#         'cluster2cluster': [],
#         'logits': []
#     }

#     # 检查特征有效性
#     if torch.isnan(features).any() or torch.isinf(features).any():
#         print("警告: 特征包含NaN或Inf值，进行清理")
#         features = torch.nan_to_num(features)

#     current_features = features.numpy()

#     # 限制特征数量，避免内存溢出
#     max_samples = 10000  # 最大样本数
#     if current_features.shape[0] > max_samples:
#         print(f"采样特征: {current_features.shape[0]} -> {max_samples}")
#         indices = np.random.choice(current_features.shape[0], max_samples, replace=False)
#         current_features = current_features[indices]

#     for level in range(levels):
#         try:
#             n_clusters = clusters[level] if level < len(clusters) else clusters[-1]

#             # 确保聚类数不超过样本数
#             n_clusters = min(n_clusters, current_features.shape[0] - 1)
#             if n_clusters <= 1:
#                 print(f"层级 {level}: 样本数不足，跳过聚类")
#                 # 创建虚拟聚类结果
#                 labels = np.zeros(current_features.shape[0], dtype=int)
#                 centers = np.mean(current_features, axis=0, keepdims=True)
#                 cluster_result['im2cluster'].append(torch.from_numpy(labels))
#                 cluster_result['centroids'].append(torch.from_numpy(centers).float())
#                 cluster_result['density'].append(torch.ones(1))
#                 break

#             print(f"层级 {level} 聚类: {current_features.shape} -> {n_clusters} 个类别")

#             # 使用更高效的聚类算法
#             from sklearn.cluster import KMeans
#             kmeans = KMeans(
#                 n_clusters=n_clusters,
#                 random_state=42,
#                 n_init=3,          # 减少初始化次数
#                 max_iter=50,        # 减少迭代次数
#                 algorithm='elkan'   # 更快的算法
#             )

#             labels = kmeans.fit_predict(current_features)
#             centers = kmeans.cluster_centers_

#             # 计算密度估计
#             from scipy.spatial.distance import cdist
#             distances = cdist(current_features, centers)
#             min_distances = np.min(distances, axis=1)
#             density = 1.0 / (np.mean(min_distances) + 1e-8)

#             # 存储结果
#             cluster_result['im2cluster'].append(torch.from_numpy(labels))
#             cluster_result['centroids'].append(torch.from_numpy(centers).float())
#             cluster_result['density'].append(torch.tensor(density))

#             # 为下一层准备特征
#             if level < levels - 1:
#                 current_features = centers

#         except Exception as e:
#             print(f"层级 {level} 聚类失败: {e}")
#             # 创建降级方案
#             labels = np.zeros(current_features.shape[0], dtype=int)
#             centers = np.mean(current_features, axis=0, keepdims=True)
#             cluster_result['im2cluster'].append(torch.from_numpy(labels))
#             cluster_result['centroids'].append(torch.from_numpy(centers).float())
#             cluster_result['density'].append(torch.ones(1))
#             break

#     return cluster_result
# # 层级聚类函数
# import time
# import signal

# class TimeoutError(Exception):
#     pass

# def timeout_handler(signum, frame):
#     raise TimeoutError("操作超时")

# def safe_update_clustering(model, data_loader, timeout=300):
#     """带超时机制的聚类更新"""
#     # 设置超时处理
#     signal.signal(signal.SIGALRM, timeout_handler)
#     signal.alarm(timeout)

#     try:
#         start_time = time.time()
#         cluster_result = update_hierarchical_clustering(model, data_loader)
#         elapsed_time = time.time() - start_time
#         print(f"聚类完成，耗时: {elapsed_time:.2f}秒")
#         signal.alarm(0)  # 取消超时
#         return cluster_result
#     except TimeoutError:
#         print(f"聚类超时（>{timeout}秒），使用简化聚类")
#         signal.alarm(0)
#         return create_simplified_clustering()
#     except Exception as e:
#         print(f"聚类错误: {e}")
#         signal.alarm(0)
#         return create_simplified_clustering()

# def create_simplified_clustering():
#     """创建简化的聚类结果作为降级方案"""
#     return {
#         'im2cluster': [torch.zeros(1000, dtype=torch.long)],  # 虚拟标签
#         'centroids': [torch.randn(1, 128)],  # 虚拟中心
#         'density': [torch.tensor(1.0)],
#         'cluster2cluster': [],
#         'logits': []
#     }

# def update_hierarchical_clustering(model, data_loader):
#     """
#     更新层级聚类结果，添加详细监控
#     """
#     model.eval()
#     features = []

#     print("开始提取特征用于聚类...")

#     # 限制批次数量以避免内存问题
#     max_batches = 100
#     batch_count = 0

#     with torch.no_grad():
#         for im_1, im_2 in tqdm(data_loader, desc='提取特征'):
#             if batch_count >= max_batches:
#                 break

#             try:
#                 im = im_1.cuda(non_blocking=True)
#                 feat = model.encoder_q(im)
#                 feat = torch.nn.functional.normalize(feat, dim=1)
#                 features.append(feat.cpu())
#                 batch_count += 1

#                 # 定期检查内存
#                 if batch_count % 20 == 0:
#                     if torch.cuda.is_available():
#                         allocated = torch.cuda.memory_allocated() / 1024**3
#                         print(f"已处理 {batch_count} 批次, GPU内存: {allocated:.2f}GB")

#             except Exception as e:
#                 print(f"批次 {batch_count} 特征提取失败: {e}")
#                 continue

#     if not features:
#         raise ValueError("未能提取任何特征")

#     features = torch.cat(features, dim=0)
#     print(f"特征形状: {features.shape}")

#     # 检查特征有效性
#     if torch.isnan(features).any() or torch.isinf(features).any():
#         print("警告: 特征包含NaN或Inf值，进行清理")
#         features = torch.nan_to_num(features)

#     # 执行层级聚类
#     cluster_result = optimized_hierarchical_kmeans(features)

#     return cluster_result

# # 学习率调整
# def adjust_learning_rate(optimizer, epoch, args):
#     """调整学习率"""
#     lr = args.lr
#     if args.cos:  # cosine lr schedule
#         lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
#     else:  # stepwise lr schedule
#         for milestone in args.schedule:
#             lr *= 0.1 if epoch >= milestone else 1.
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = lr

# # HCSC训练函数
# # def hcsc_train(model, data_loader, optimizer, epoch, args):
# #     """
# #     HCSC训练函数
# #     """
# #     model.train()
# #     adjust_learning_rate(optimizer, epoch, args)

# #     total_loss, total_num = 0.0, 0
# #     train_bar = tqdm(data_loader, desc=f'Training Epoch {epoch}')

# #     print("🔄 检查聚类更新条件...")

# #     # 定期更新层级聚类
# #     if epoch == 1 or epoch % 10 == 0:
# #         print("🔄 更新层级聚类...")
# #         cluster_result = update_hierarchical_clustering(model, data_loader)
# #         # 缓存聚类结果
# #         torch.save(cluster_result, os.path.join(args.results_dir, f'cluster_result_epoch_{epoch}.pth'))
# #     else:
# #         # 使用最近缓存的聚类结果
# #         try:
# #             cluster_result = torch.load(os.path.join(args.results_dir, f'cluster_result_epoch_{epoch-1}.pth'))
# #             print("✅ 使用缓存的聚类结果")
# #         except:
# #             cluster_result = torch.load(os.path.join(args.results_dir, 'cluster_result_epoch_1.pth'))
# #             print("⚠️ 使用初始聚类结果")
# #     print("🔄 开始批次训练...")
# #     for batch_idx, (im_1, im_2) in enumerate(train_bar):
# #         im_1, im_2 = im_1.cuda(non_blocking=True), im_2.cuda(non_blocking=True)

# #         # 准备输入和索引
# #         images = [im_1, im_2]
# #         batch_size = im_1.size(0)
# #         index = torch.arange(batch_idx * batch_size, (batch_idx + 1) * batch_size).cuda()

# #         try:
# #             # HCSC前向传播
# #             loss = model(images, is_eval=False, cluster_result=cluster_result, index=index)

# #             # 反向传播
# #             optimizer.zero_grad()
# #             loss.backward()
# #             optimizer.step()

# #             total_num += batch_size
# #             total_loss += loss.item() * batch_size

# #             train_bar.set_description(
# #                 f'Epoch {epoch}: Loss: {total_loss/total_num:.4f}')

# #         except Exception as e:
# #             print(f"Error in batch {batch_idx}: {e}")
# #             continue

# #     return total_loss / total_num
# def hcsc_train(model, data_loader, optimizer, epoch, args):
#     """
#     HCSC训练函数
#     """
#     print(f"🎯 进入hcsc_train函数，第{epoch}轮")

#     try:
#         model.train()
#         adjust_learning_rate(optimizer, epoch, args)

#         total_loss, total_num = 0.0, 0
#         train_bar = tqdm(data_loader, desc=f'Training Epoch {epoch}')

#         print("🔄 检查聚类更新条件...")

#         # 定期更新层级聚类
#         cluster_result = None
#         try:
#             if epoch == 1 or epoch % 10 == 0:
#                 print("🔄 更新层级聚类...")
#                 cluster_result = update_hierarchical_clustering(model, data_loader)
#                 # 缓存聚类结果
#                 torch.save(cluster_result, os.path.join(args.results_dir, f'cluster_result_epoch_{epoch}.pth'))
#                 print("✅ 聚类更新完成")
#             else:
#                 # 使用最近缓存的聚类结果
#                 try:
#                     cluster_result = torch.load(os.path.join(args.results_dir, f'cluster_result_epoch_{epoch-1}.pth'))
#                     print("✅ 使用缓存的聚类结果")
#                 except:
#                     cluster_result = torch.load(os.path.join(args.results_dir, 'cluster_result_epoch_1.pth'))
#                     print("⚠️ 使用初始聚类结果")
#         except Exception as e:
#             print(f"❌ 聚类更新失败: {e}")
#             # 创建简单的聚类结果继续训练
#             cluster_result = {
#                 'im2cluster': [torch.zeros(1000, dtype=torch.long)],
#                 'centroids': [torch.randn(1, 128)],
#                 'density': [torch.tensor(1.0)],
#                 'cluster2cluster': [],
#                 'logits': []
#             }

#         print("🔄 开始批次训练...")
#         batch_count = 0

#         for batch_idx, (im_1, im_2) in enumerate(train_bar):
#             try:
#                 im_1, im_2 = im_1.cuda(non_blocking=True), im_2.cuda(non_blocking=True)

#                 # 准备输入和索引
#                 images = [im_1, im_2]
#                 batch_size = im_1.size(0)
#                 index = torch.arange(batch_idx * batch_size, (batch_idx + 1) * batch_size).cuda()

#                 # HCSC前向传播
#                 loss = model(images, is_eval=False, cluster_result=cluster_result, index=index)

#                 # 反向传播
#                 optimizer.zero_grad()
#                 loss.backward()
#                 optimizer.step()

#                 total_num += batch_size
#                 total_loss += loss.item() * batch_size
#                 batch_count += 1

#                 train_bar.set_description(f'Epoch {epoch}: Loss: {total_loss/total_num:.4f}')

#                 # 每10个批次打印一次进度
#                 if batch_count % 10 == 0:
#                     print(f"✅ 已处理 {batch_count} 个批次，当前损失: {loss.item():.4f}")

#             except Exception as e:
#                 print(f"❌ 批次 {batch_idx} 训练失败: {e}")
#                 continue

#         if total_num == 0:
#             print("⚠️ 警告：没有成功处理任何批次")
#             return 0.0

#         avg_loss = total_loss / total_num
#         print(f"✅ 第 {epoch} 轮训练完成，平均损失: {avg_loss:.4f}")
#         return avg_loss

#     except Exception as e:
#         print(f"❌ hcsc_train函数执行失败: {e}")
#         import traceback
#         traceback.print_exc()
#         return 0.0


# def test(net, memory_data_loader, test_data_loader, epoch, args):
#     net.eval()
#     classes = len(memory_data_loader.dataset.classes)
#     total_top1, total_top5, total_num, feature_bank = 0.0, 0.0, 0, []
#     with torch.no_grad():
#         # generate feature bank
#         for data, target in tqdm(memory_data_loader, desc='Feature extracting'):
#             feature = net(data.cuda(non_blocking=True))
#             feature = F.normalize(feature, dim=1)
#             feature_bank.append(feature)
#         # [D, N]
#         feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
#         # [N]
#         feature_labels = torch.tensor(memory_data_loader.dataset.targets, device=feature_bank.device)
#         # loop test data_processing to predict the label by weighted knn search
#         test_bar = tqdm(test_data_loader)
#         for data, target in test_bar:
#             data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
#             feature = net(data)
#             feature = F.normalize(feature, dim=1)

#             pred_labels = knn_predict(feature, feature_bank, feature_labels, classes, args.knn_k, args.knn_t)

#             total_num += data.size(0)
#             total_top1 += (pred_labels[:, 0] == target).float().sum().item()
#             test_bar.set_description(
#                 'Test Epoch: [{}/{}] Acc@1:{:.2f}%'.format(epoch, args.epochs, total_top1 / total_num * 100))

#     return total_top1 / total_num * 100

# # KNN评估函数
# def knn_predict(feature, feature_bank, feature_labels, classes, knn_k, knn_t):
#     """KNN预测"""
#     sim_matrix = torch.mm(feature, feature_bank)
#     sim_weight, sim_indices = sim_matrix.topk(k=knn_k, dim=-1)
#     sim_labels = torch.gather(feature_labels.expand(feature.size(0), -1), dim=-1, index=sim_indices)
#     sim_weight = (sim_weight / knn_t).exp()

#     one_hot_label = torch.zeros(feature.size(0) * knn_k, classes, device=sim_labels.device)
#     one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0)
#     # weighted score ---> [B, C]
#     pred_scores = torch.sum(one_hot_label.view(feature.size(0), -1, classes) * sim_weight.unsqueeze(dim=-1), dim=1)

#     pred_labels = pred_scores.argsort(dim=-1, descending=True)
#     return pred_labels

# # 1. 导入

# # 2. 模型创建函数
# def create_hcsc_model(args):
#     """创建HCSC模型实例，处理Namespace和字典两种参数格式"""
#     import torchvision.models as models

#     # 正确处理参数
#     if hasattr(args, '__dict__'):
#         # 如果是Namespace对象，转换为字典或直接使用属性
#         params = vars(args) if hasattr(args, '__dict__') else {}
#     else:
#         params = args if isinstance(args, dict) else {}

#     # 获取参数值，提供默认值
#     arch = params.get('arch', 'resnet50')
#     dim = params.get('dim', 128)
#     queue_length = params.get('queue_length', 16384)
#     m = params.get('m', 0.999)
#     T = params.get('T', 0.2)
#     mlp = params.get('mlp', 1)

#     # 根据arch参数选择基础编码器
#     if arch == 'resnet18':
#         base_encoder = models.resnet18
#     elif arch == 'resnet50':
#         base_encoder = models.resnet50
#     else:
#         base_encoder = models.resnet50

#     # 创建模型实例
#     model = HCSC(
#         base_encoder=base_encoder,
#         dim=dim,
#         queue_length=queue_length,
#         m=m,
#         T=T,
#         mlp=mlp,
#         multi_crop=params.get('multi_crop', False),
#         instance_selection=params.get('instance_selection', True),
#         proto_selection=params.get('proto_selection', True),
#         selection_on_local=params.get('selection_on_local', True)
#     ).cuda()

#     return model

# # 3. 创建模型实例（重要：需要调用函数）
# print("创建HCSC模型实例...")
# hcsc_model = create_hcsc_model(args)  # 注意：这里调用了函数


# # 4. 验证模型
# print(f"模型类型: {type(hcsc_model)}")
# print(f"参数数量: {sum(p.numel() for p in hcsc_model.parameters()):,}")

# # 5. 创建优化器（现在应该可以正常工作）
# hcsc_optimizer = torch.optim.SGD(
#     hcsc_model.parameters(),  # 现在 hcsc_model 是模型实例
#     lr=args.lr,
#     weight_decay=args.wd,
#     momentum=0.9
# )
# print("优化器创建成功！")

# # load model if resume
# epoch_start = 1
# if args.resume != '' :         # 加载预模型
#     checkpoint = torch.load(args.resume)
#     hcsc_model.load_state_dict(checkpoint['state_dict'])
#     hcsc_optimizer.load_state_dict(checkpoint['optimizer'])
#     epoch_start = checkpoint['epoch'] + 1
#     print('Loaded from: {}'.format(args.resume))

# # logging
# results = {'train_loss': [], 'test_acc@1': []}
# if not os.path.exists(args.results_dir):
#     os.mkdir(args.results_dir)
# # dump args
# with open(args.results_dir + '/args.json', 'w') as fid:
#     json.dump(args.__dict__, fid, indent=2)

# #training loop

# min_acc = 0
# for epoch in range(epoch_start, args.epochs + 1):
#    train_loss = hcsc_train(hcsc_model, hcsc_train_loader, hcsc_optimizer, epoch, args)
#    results['train_loss'].append(train_loss)
#    test_acc_1 = test(hcsc_model.encoder_q, memory_loader, test_loader, epoch, args)
#    results['test_acc@1'].append(test_acc_1)

# #save statistics
#    data_frame = pd.DataFrame(data=results, index=range(epoch_start, epoch + 1))
#    data_frame.to_csv(args.results_dir + '/log.csv', index_label='epoch')

#    # save model
#    torch.save({'epoch': epoch, 'state_dict': hcsc_model.state_dict(), 'optimizer': hcsc_optimizer.state_dict(), },
#               args.results_dir + '/model_last.pth')

#     #save model
#    if test_acc_1 > min_acc:
#        min_acc = test_acc_1
#        torch.save({'epoch': epoch, 'state_dict': hcsc_model.state_dict(), 'optimizer': hcsc_optimizer.state_dict(), },
#            args.results_dir + '/model_best.pth')
#    if epoch == args.epochs:
#       torch.save({'epoch': epoch, 'state_dict': hcsc_model.state_dict(), 'optimizer': hcsc_optimizer.state_dict(), },
#                    args.results_dir + '/model_last.pth')

import argparse
import math
import os
from datetime import datetime

import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torchvision.models import resnet
from tqdm import tqdm

# 设置环境变量（单GPU环境）
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
cudnn.benchmark = True

# ===================== 路径配置与Python环境 =====================
import sys
from pathlib import Path

# 获取当前文件路径
current_file = Path(__file__).resolve()
print(f"当前文件: {current_file}")

# 获取目录路径
hcsc_dir = current_file.parent
project_root = hcsc_dir.parent
print(f"HCSC目录: {hcsc_dir}")
print(f"项目根目录: {project_root}")

# 清理重复路径并添加
sys.path = [p for p in sys.path if str(project_root) not in p]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
    print(f"✅ 已添加项目根目录到Python路径: {project_root}")
if str(hcsc_dir) not in sys.path:
    sys.path.insert(0, str(hcsc_dir))
    print(f"✅ 已添加HCSC目录到Python路径: {hcsc_dir}")

# 打印路径调试信息
print("\n=== 当前Python路径（前5个）===")
for i, path in enumerate(sys.path[:5]):
    print(f"{i}: {path}")

# ===================== 命令行参数配置 =====================
parser = argparse.ArgumentParser(description="Train HCSC on CIFAR-10")
parser.add_argument("-a", "--arch", metavar="ARCH", default="resnet50", help="model architecture")
parser.add_argument(
    "--lr", "--learning-rate", default=0.03, type=float, metavar="LR", help="initial learning rate", dest="lr"
)
parser.add_argument("--epochs", default=300, type=int, metavar="N", help="number of total epochs to run")
parser.add_argument("--start-epoch", default=0, type=int, metavar="N", help="manual epoch number (useful on restarts)")
parser.add_argument("--schedule", default=[120, 160], nargs="*", type=int, help="learning rate schedule")
parser.add_argument("--cos", type=int, default=1, help="use cosine lr schedule")
parser.add_argument("-b", "--batch-size", default=64, type=int, metavar="N")
parser.add_argument("--wd", default=5e-4, type=float, metavar="W", help="weight decay")

# 层级聚类参数
parser.add_argument("--cluster-levels", default=2, type=int, help="number of hierarchical clustering levels")
parser.add_argument("--cluster-nums", default=[1000, 500], nargs="*", type=int, help="number of clusters at each level")

# 模型核心参数
parser.add_argument("--dim", default=128, type=int, help="feature dimension")
parser.add_argument("--queue_length", default=16384, type=int, help="queue size; number of negative pairs")
parser.add_argument("--m", default=0.999, type=float, help="moco momentum of updating key encoder")
parser.add_argument("--T", default=0.2, type=float, help="temperature")
parser.add_argument("--mlp", type=int, default=1, help="use mlp head")
parser.add_argument(
    "--multi_crop", action="store_true", default=False, help="Whether to enable multi-crop transformation"
)
parser.add_argument(
    "--selection_on_local", action="store_true", default=False, help="whether enable mining on local views"
)
parser.add_argument("--instance_selection", type=int, default=1, help="Whether enable instance selection")
parser.add_argument("--proto_selection", type=int, default=1, help="Whether enable prototype selection")

# knn monitor
parser.add_argument("--knn-k", default=20, type=int, help="k in kNN monitor")
parser.add_argument("--knn-t", default=0.1, type=float, help="softmax temperature in kNN monitor")

# 工具参数
parser.add_argument("--resume", default="", type=str, metavar="PATH", help="path to latest checkpoint")
parser.add_argument("--results-dir", default="", type=str, metavar="PATH", help="path to cache")

# 运行参数（默认使用空字符串，适配IDE运行）
args = parser.parse_args("")

# 补充参数配置
args.epochs = 300
args.cos = True
args.schedule = []  # cos调度时禁用step调度
args.symmetric = False
if args.results_dir == "":
    args.results_dir = "F:\\contrast-11\\HCSC\\run\\cache-" + datetime.now().strftime("%Y-%m-%d-%H-%M-%S-moco")
hcsc_args = args
# 创建结果目录
if not os.path.exists(args.results_dir):
    os.makedirs(args.results_dir)


# ===================== 数据加载 =====================
class CIFAR10Pair(CIFAR10):
    """CIFAR10 Dataset（返回一对增强样本）."""

    def __getitem__(self, index):
        img = self.data[index]
        img = Image.fromarray(img)
        if self.transform is not None:
            im_1 = self.transform(img)
            im_2 = self.transform(img)
        return im_1, im_2


# 数据增强
train_transform = transforms.Compose(
    [
        transforms.RandomResizedCrop(32),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
    ]
)

test_transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])]
)

# 加载数据集
train_data = CIFAR10Pair(
    root="F:\\contrast-11\\HCSC\\data_visdrone2019", train=True, transform=train_transform, download=False
)
hcsc_train_loader = DataLoader(
    train_data, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True, drop_last=True
)

memory_data = CIFAR10(
    root="F:\\contrast-11\\HCSC\\data_visdrone2019", train=True, transform=test_transform, download=False
)
memory_loader = DataLoader(memory_data, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)

test_data = CIFAR10(
    root="F:\\contrast-11\\HCSC\\data_visdrone2019", train=False, transform=test_transform, download=False
)
test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)


# ===================== 辅助模块 =====================
class SplitBatchNorm(nn.BatchNorm2d):
    """分割BatchNorm（适配多GPU，单GPU也可使用）."""

    def __init__(self, num_features, num_splits, **kw):
        super().__init__(num_features, **kw)
        self.num_splits = num_splits

    def forward(self, input):
        N, C, H, W = input.shape
        if self.training or not self.track_running_stats:
            running_mean_split = self.running_mean.repeat(self.num_splits)
            running_var_split = self.running_var.repeat(self.num_splits)
            outcome = nn.functional.batch_norm(
                input.view(-1, C * self.num_splits, H, W),
                running_mean_split,
                running_var_split,
                self.weight.repeat(self.num_splits),
                self.bias.repeat(self.num_splits),
                True,
                self.momentum,
                self.eps,
            ).view(N, C, H, W)
            self.running_mean.data.copy_(running_mean_split.view(self.num_splits, C).mean(dim=0))
            self.running_var.data.copy_(running_var_split.view(self.num_splits, C).mean(dim=0))
            return outcome
        else:
            return nn.functional.batch_norm(
                input, self.running_mean, self.running_var, self.weight, self.bias, False, self.momentum, self.eps
            )


# ===================== 核心模型定义 =====================
class HCSC(nn.Module):
    """HCSC框架（修复单GPU适配问题）."""

    def __init__(
        self,
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
        **kwargs,
    ):
        super().__init__()
        self.queue_length = queue_length
        self.m = m
        self.T = T
        self.multi_crop = multi_crop
        self.selection_on_local = selection_on_local
        self.logger = logger
        self.instance_selection = instance_selection
        self.proto_selection = proto_selection

        # 创建编码器
        self.encoder_q = base_encoder(num_classes=dim)
        self.encoder_k = base_encoder(num_classes=dim)

        # MLP头
        if mlp:
            dim_mlp = self.encoder_q.fc.weight.shape[1]
            self.encoder_q.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_q.fc)
            self.encoder_k.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_k.fc)

        # 初始化key编码器
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        # 创建队列
        self.register_buffer("queue", torch.randn(dim, queue_length))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        self.register_buffer("queue_index", torch.arange(0, queue_length))
        self.buffer_dict = {}
        self.mined_index = []

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """动量更新key编码器."""
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, index=None):
        """队列管理（单GPU适配）."""
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        assert self.queue_length % batch_size == 0

        # 入队出队
        self.queue[:, ptr : ptr + batch_size] = keys.T
        if index is not None:
            self.queue_index[ptr : ptr + batch_size] = index
        ptr = (ptr + batch_size) % self.queue_length
        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle(self, x):
        """单GPU版本Batch Shuffle（替代DDP版本）."""
        batch_size = x.shape[0]
        idx_shuffle = torch.randperm(batch_size).cuda()
        idx_unshuffle = torch.argsort(idx_shuffle)
        return x[idx_shuffle], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle(self, x, idx_unshuffle):
        """单GPU版本Batch Unshuffle."""
        return x[idx_unshuffle]

    @torch.no_grad()
    def sample_neg_instance(self, im2cluster, centroids, density, index):
        """负实例采样（修复维度匹配）."""
        queue_p_samples = []
        for layer in range(len(im2cluster)):
            # 确保centroids是cuda张量
            centroid = centroids[layer].cuda() if not centroids[layer].is_cuda else centroids[layer]
            proto_logit = torch.mm(self.queue.clone().detach().permute(1, 0), centroid.permute(1, 0))

            # 密度处理
            density_layer = density[layer].cuda() if not density[layer].is_cuda else density[layer]
            density_layer = density_layer.clamp(min=1e-3)
            proto_logit /= density_layer

            # 标签处理（确保索引不越界）
            label = im2cluster[layer][index].clamp(0, centroid.shape[0] - 1)
            logit = proto_logit.clone().detach().softmax(-1)
            p_sample = 1 - logit[:, label].t()
            queue_p_samples.append(p_sample)

        self.selected_masks = []
        avg_sample_ratios = []
        for p_sample in queue_p_samples:
            try:
                neg_sampler = torch.distributions.bernoulli.Bernoulli(p_sample.clamp(0.0, 0.999))
                selected_mask = neg_sampler.sample()
                self.selected_masks.append(selected_mask)
                avg_sample_ratios.append(p_sample.mean().item())
            except:
                selected_mask = torch.ones([index.shape[0], self.queue.shape[1]]).cuda()
                self.selected_masks.append(selected_mask)
        return self.selected_masks, avg_sample_ratios

    @torch.no_grad()
    def extract_key_feat(self, im_k):
        """提取key特征（单GPU适配）."""
        self._momentum_update_key_encoder()
        im_k_shuffled, idx_unshuffle = self._batch_shuffle(im_k)
        k = self.encoder_k(im_k_shuffled)
        k = nn.functional.normalize(k, dim=1)
        k = self._batch_unshuffle(k, idx_unshuffle)
        return k

    def extract_feat(self, images, is_eval=False):
        """提取特征（统一接口）."""
        if is_eval:
            k = self.encoder_k(images)
            k = nn.functional.normalize(k, dim=1)
            return k
        im_q, im_k = images[0], images[1]

        q = self.encoder_q(im_q)
        q = nn.functional.normalize(q, dim=1)

        if self.multi_crop:
            k = self.extract_key_feat(im_k)
            local_views = [nn.functional.normalize(self.encoder_q(im_local), dim=1) for im_local in images[2:]]
            return q, k, local_views
        else:
            k = self.extract_key_feat(im_k)
            return q, k, None

    def get_protos(self, q, index, cluster_result):
        """获取原型（修复cluster2cluster缺失问题）."""
        if cluster_result is None:
            return None, None, None, None

        proto_labels = []
        proto_logits = []
        proto_selecteds = []
        temp_protos = []

        # 补全缺失的cluster2cluster和logits
        cluster2cluster = cluster_result.get("cluster2cluster", [[] for _ in cluster_result["im2cluster"]])
        logits = cluster_result.get(
            "logits", [torch.randn(c.shape[0], c.shape[0]).cuda() for c in cluster_result["centroids"]]
        )

        for n, (im2cluster, prototypes, density) in enumerate(
            zip(cluster_result["im2cluster"], cluster_result["centroids"], cluster_result["density"])
        ):
            # 转换为cuda
            im2cluster = im2cluster.cuda() if not im2cluster.is_cuda else im2cluster
            prototypes = prototypes.cuda() if not prototypes.is_cuda else prototypes
            density = density.cuda() if not density.is_cuda else density

            # 正原型
            pos_proto_id = im2cluster[index].clamp(0, prototypes.shape[0] - 1)  # 防止索引越界
            pos_prototypes = prototypes[pos_proto_id]
            proto_selecteds.append(prototypes)
            temp_protos.append(density)

            # 负原型ID
            all_proto_id = torch.arange(prototypes.shape[0], device=pos_proto_id.device)
            neg_proto_mask = ~torch.isin(all_proto_id, pos_proto_id.unique())
            neg_proto_id = all_proto_id[neg_proto_mask]

            if len(neg_proto_id) == 0:  # 避免无负样本
                neg_proto_id = all_proto_id[1:] if all_proto_id.numel() > 1 else all_proto_id

            # 原型对比逻辑
            if self.proto_selection:
                if n == len(cluster_result["im2cluster"]) - 1:  # 最后一层
                    neg_prototypes = prototypes[neg_proto_id]
                    logits_proto = torch.cat(
                        [torch.einsum("nc,nc->n", [q, pos_prototypes]).unsqueeze(-1), torch.mm(q, neg_prototypes.t())],
                        dim=1,
                    )
                    # 密度加权
                    temp_map = torch.cat(
                        [density[pos_proto_id].unsqueeze(-1), density[neg_proto_id].unsqueeze(0).repeat(q.shape[0], 1)],
                        dim=1,
                    )
                    logits_proto = logits_proto / temp_map
                else:
                    # 补全cluster2cluster（层级映射）
                    if len(cluster2cluster[n]) == 0:
                        cluster2cluster[n] = torch.randint(
                            0, logits[n].shape[1], (prototypes.shape[0],), device=q.device
                        )
                    upper_pos_proto_id = cluster2cluster[n][pos_proto_id]

                    # 采样概率计算
                    densities = density.clamp(min=1e-3)
                    sampling_prob = (
                        1
                        - (logits[n][neg_proto_id] / densities[neg_proto_id].unsqueeze(1))
                        .softmax(-1)[:, upper_pos_proto_id]
                        .t()
                    )

                    # 负原型采样
                    neg_sampler = torch.distributions.bernoulli.Bernoulli(sampling_prob.clamp(0.001, 0.999))
                    selected_mask = neg_sampler.sample()

                    # 负原型logits
                    neg_prototypes = prototypes[neg_proto_id]
                    neg_logits = torch.mm(q, neg_prototypes.t()) * selected_mask
                    logits_proto = torch.cat(
                        [torch.einsum("nc,nc->n", [q, pos_prototypes]).unsqueeze(-1), neg_logits], dim=1
                    )
                    temp_map = torch.cat(
                        [density[pos_proto_id].unsqueeze(-1), density[neg_proto_id].unsqueeze(0).repeat(q.shape[0], 1)],
                        dim=1,
                    )
                    logits_proto = logits_proto / temp_map
            else:
                neg_prototypes = prototypes[neg_proto_id]
                logits_proto = torch.cat(
                    [torch.einsum("nc,nc->n", [q, pos_prototypes]).unsqueeze(-1), torch.mm(q, neg_prototypes.t())],
                    dim=1,
                )
                temp_map = torch.cat(
                    [density[pos_proto_id].unsqueeze(-1), density[neg_proto_id].unsqueeze(0).repeat(q.shape[0], 1)],
                    dim=1,
                )
                logits_proto = logits_proto / temp_map

            labels_proto = torch.zeros(q.shape[0], dtype=torch.long, device=q.device)
            proto_labels.append(labels_proto)
            proto_logits.append(logits_proto)

        return proto_logits, proto_labels, proto_selecteds, temp_protos

    def forward(self, images, is_eval=False, cluster_result=None, index=None):
        """前向传播（修复损失计算逻辑）."""
        if is_eval:
            return self.extract_feat(images, is_eval)

        q, k, _local_views = self.extract_feat(images, is_eval)
        proto_logits, proto_labels, _proto_selected, _temp_protos = self.get_protos(q, index, cluster_result)

        # 实例对比损失
        l_pos = torch.einsum("nc,nc->n", [q, k]).unsqueeze(-1)

        if proto_labels is not None and self.instance_selection and hasattr(self, "selected_masks"):
            l_neg = []
            for mask in self.selected_masks:
                logit = torch.einsum("nc,ck->nk", [q, self.queue.clone().detach()])
                l_neg.append(logit * mask)
            logits = [torch.cat([l_pos, ln], dim=1) / self.T for ln in l_neg]
            labels = [torch.zeros(logit.shape[0], dtype=torch.long, device=q.device) for logit in logits]
            instance_loss = sum([F.cross_entropy(logit, label) for logit, label in zip(logits, labels)]) / len(logits)
        else:
            l_neg = torch.einsum("nc,ck->nk", [q, self.queue.clone().detach()])
            logits = torch.cat([l_pos, l_neg], dim=1) / self.T
            labels = torch.zeros(logits.shape[0], dtype=torch.long, device=q.device)
            instance_loss = F.cross_entropy(logits, labels)

        # 原型对比损失
        proto_loss = 0.0
        if proto_logits is not None and self.proto_selection:
            for plogits, plabels in zip(proto_logits, proto_labels):
                proto_loss += F.cross_entropy(plogits / self.T, plabels)
            proto_loss /= len(proto_logits)

        # 总损失
        # total_loss = instance_loss + 0.5 * proto_loss
        total_loss = instance_loss

        # 更新队列
        self._dequeue_and_enqueue(k, index)
        return total_loss


# ===================== 聚类功能修复 =====================
# def optimized_hierarchical_kmeans(features, levels=2, clusters=[1000, 500]):
#     """优化层级聚类（确保结构完整）"""
#     cluster_result = {
#         'im2cluster': [],
#         'centroids': [],
#         'density': [],
#         'cluster2cluster': [],  # 补全层级映射
#         'logits': []  # 补全logits
#     }

#     # 特征清理
#     features = torch.nan_to_num(features)
#     current_features = features.numpy()

#     # 限制最大样本数（防止内存溢出）
#     max_samples = min(10000, current_features.shape[0])
#     if current_features.shape[0] > max_samples:
#         indices = np.random.choice(current_features.shape[0], max_samples, replace=False)
#         current_features = current_features[indices]

#     for level in range(levels):
#         try:
#             n_clusters = clusters[level] if level < len(clusters) else clusters[-1]
#             n_clusters = min(n_clusters, current_features.shape[0] - 1)
#             if n_clusters <= 1:
#                 n_clusters = 2  # 至少2个聚类

#             print(f"层级 {level} 聚类: {current_features.shape} -> {n_clusters} 个类别")
#             kmeans = KMeans(
#                 n_clusters=n_clusters,
#                 random_state=42,
#                 n_init=3,
#                 max_iter=50,
#                 algorithm='elkan'
#             )
#             labels = kmeans.fit_predict(current_features)
#             centers = kmeans.cluster_centers_

#             # 计算密度
#             distances = cdist(current_features, centers)
#             min_distances = np.min(distances, axis=1)
#             density = 1.0 / (np.mean(min_distances) + 1e-8)

#             # 存储结果
#             cluster_result['im2cluster'].append(torch.from_numpy(labels))
#             cluster_result['centroids'].append(torch.from_numpy(centers).float())
#             cluster_result['density'].append(torch.tensor(density))

#             # 补全cluster2cluster（层级映射）
#             if level < levels - 1:
#                 next_clusters = clusters[level+1] if (level+1) < len(clusters) else clusters[-1]
#                 next_clusters = min(next_clusters, centers.shape[0] - 1)
#                 next_kmeans = KMeans(n_clusters=next_clusters, n_init=2, max_iter=30)
#                 next_labels = next_kmeans.fit_predict(centers)
#                 cluster_result['cluster2cluster'].append(torch.from_numpy(next_labels))

#                 # 补全logits（原型相似度）
#                 logits = np.dot(centers, next_kmeans.cluster_centers_.T)
#                 cluster_result['logits'].append(torch.from_numpy(logits).float())
#             else:
#                 cluster_result['cluster2cluster'].append(torch.zeros(n_clusters, dtype=torch.long))
#                 cluster_result['logits'].append(torch.zeros(n_clusters, n_clusters).float())

#             # 下一层特征
#             current_features = centers

#         except Exception as e:
#             print(f"层级 {level} 聚类失败: {e}，使用降级方案")
#             labels = np.zeros(current_features.shape[0], dtype=int)
#             centers = np.mean(current_features, axis=0, keepdims=True)
#             cluster_result['im2cluster'].append(torch.from_numpy(labels))
#             cluster_result['centroids'].append(torch.from_numpy(centers).float())
#             cluster_result['density'].append(torch.tensor(1.0))
#             cluster_result['cluster2cluster'].append(torch.zeros(1, dtype=torch.long))
#             cluster_result['logits'].append(torch.zeros(1, 1).float())
#             break

#     return cluster_result


# def update_hierarchical_clustering(model, data_loader):
#     """更新聚类（添加完整监控）"""
#     model.eval()
#     features = []
#     print("开始提取特征用于聚类...")

#     with torch.no_grad():
#         for batch_idx, (im_1, im_2) in enumerate(tqdm(data_loader, desc='提取特征')):
#             if batch_idx >= 100:  # 限制批次，加速聚类
#                 break
#             try:
#                 im = im_1.cuda(non_blocking=True)
#                 feat = model.encoder_q(im)
#                 feat = F.normalize(feat, dim=1)
#                 features.append(feat.cpu())

#                 # 内存监控
#                 if batch_idx % 20 == 0 and torch.cuda.is_available():
#                     allocated = torch.cuda.memory_allocated() / 1024**3
#                     print(f"已处理 {batch_idx+1} 批次, GPU内存: {allocated:.2f}GB")
#             except Exception as e:
#                 print(f"批次 {batch_idx} 提取失败: {e}")
#                 continue

#     if not features:
#         raise ValueError("未提取到有效特征")

#     features = torch.cat(features, dim=0)
#     print(f"特征形状: {features.shape}")
#     return optimized_hierarchical_kmeans(features, levels=args.cluster_levels, clusters=args.cluster_nums)


# ===================== 训练与评估函数 =====================
def adjust_learning_rate(optimizer, epoch, args):
    """学习率调整."""
    if args.cos:
        lr = args.lr * 0.5 * (1.0 + math.cos(math.pi * epoch / args.epochs))
    else:
        lr = args.lr
        for milestone in args.schedule:
            if epoch >= milestone:
                lr *= 0.1
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def hcsc_train(model, data_loader, optimizer, epoch, args):
    """训练函数（添加错误捕获）."""
    print(f"🎯 进入hcsc_train函数，第{epoch}轮")
    model.train()
    adjust_learning_rate(optimizer, epoch, args)

    total_loss, total_num = 0.0, 0
    train_bar = tqdm(data_loader, desc=f"Training Epoch {epoch}")
    cluster_result = None

    for batch_idx, (im_1, im_2) in enumerate(train_bar):
        try:
            im_1, im_2 = im_1.cuda(non_blocking=True), im_2.cuda(non_blocking=True)
            images = [im_1, im_2]
            batch_size = im_1.size(0)
            index = torch.arange(batch_idx * batch_size, (batch_idx + 1) * batch_size, device=im_1.device)

            loss = model(images, is_eval=False, cluster_result=cluster_result, index=index)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_num += batch_size
            total_loss += loss.item() * batch_size
            train_bar.set_description(f"Epoch {epoch}: Loss: {total_loss / total_num:.4f}")

        except Exception as e:
            print(f"❌ 批次 {batch_idx} 失败: {e}")
            continue

    return total_loss / total_num if total_num > 0 else 0.0


def knn_predict(feature, feature_bank, feature_labels, classes, knn_k, knn_t):
    """KNN预测."""
    sim_matrix = torch.mm(feature, feature_bank)
    sim_weight, sim_indices = sim_matrix.topk(k=knn_k, dim=-1)
    sim_labels = torch.gather(feature_labels.expand(feature.size(0), -1), dim=-1, index=sim_indices)
    sim_weight = (sim_weight / knn_t).exp()

    one_hot_label = torch.zeros(feature.size(0) * knn_k, classes, device=sim_labels.device)
    one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0)
    pred_scores = torch.sum(one_hot_label.view(feature.size(0), -1, classes) * sim_weight.unsqueeze(dim=-1), dim=1)
    return pred_scores.argsort(dim=-1, descending=True)


def test(net, memory_data_loader, test_data_loader, epoch, args):
    """测试函数."""
    net.eval()
    classes = len(memory_data_loader.dataset.classes)
    total_top1, total_num = 0.0, 0
    feature_bank, feature_labels = [], []

    with torch.no_grad():
        # 生成特征库
        for data, target in tqdm(memory_data_loader, desc="特征提取"):
            feature = net(data.cuda(non_blocking=True))
            feature = F.normalize(feature, dim=1)
            feature_bank.append(feature)
            feature_labels.append(target)
        feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
        feature_labels = torch.cat(feature_labels, dim=0).cuda()

        # KNN预测
        test_bar = tqdm(test_data_loader)
        for data, target in test_bar:
            data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
            feature = net(data)
            feature = F.normalize(feature, dim=1)

            pred_labels = knn_predict(feature, feature_bank, feature_labels, classes, args.knn_k, args.knn_t)
            total_num += data.size(0)
            total_top1 += (pred_labels[:, 0] == target).float().sum().item()
            test_bar.set_description(f"Test Epoch {epoch}: Acc@1: {total_top1 / total_num * 100:.2f}%")

    return total_top1 / total_num * 100


# ===================== 主函数 =====================
def create_hcsc_model(args):
    """创建模型实例."""
    if args.arch == "resnet18":
        base_encoder = resnet.resnet18
    else:
        base_encoder = resnet.resnet50

    return HCSC(
        base_encoder=base_encoder,
        dim=args.dim,
        queue_length=args.queue_length,
        m=args.m,
        T=args.T,
        mlp=args.mlp,
        multi_crop=args.multi_crop,
        instance_selection=args.instance_selection,
        proto_selection=args.proto_selection,
        selection_on_local=args.selection_on_local,
    ).cuda()


# 初始化模型和优化器
print("创建HCSC模型实例...")
hcsc_model = create_hcsc_model(args)
print(f"模型类型: {type(hcsc_model)}")
print(f"参数数量: {sum(p.numel() for p in hcsc_model.parameters()):,}")

hcsc_optimizer = torch.optim.SGD(hcsc_model.parameters(), lr=args.lr, weight_decay=args.wd, momentum=0.9)

# # 加载断点
# epoch_start = 1
# if args.resume and os.path.exists(args.resume):
#     checkpoint = torch.load(args.resume)
#     hcsc_model.load_state_dict(checkpoint['state_dict'])
#     hcsc_optimizer.load_state_dict(checkpoint['optimizer'])
#     epoch_start = checkpoint['epoch'] + 1
#     print(f'加载断点: {args.resume}')

# # 训练日志
# results = {'train_loss': [], 'test_acc@1': []}
# with open(os.path.join(args.results_dir, 'args.json'), 'w') as f:
#     json.dump(args.__dict__, f, indent=2)

# # 主训练循环
# min_acc = 0.0
# for epoch in range(epoch_start, args.epochs + 1):
#     train_loss = hcsc_train(hcsc_model, hcsc_train_loader, hcsc_optimizer, epoch, args)
#     results['train_loss'].append(train_loss)

#     # 测试
#     test_acc = test(hcsc_model.encoder_q, memory_loader, test_loader, epoch, args)
#     results['test_acc@1'].append(test_acc)

#     # 保存日志
#     pd.DataFrame(results, index=range(epoch_start, epoch+1)).to_csv(
#         os.path.join(args.results_dir, 'log.csv'), index_label='epoch')

#     # 保存模型
#     torch.save({
#         'epoch': epoch,
#         'state_dict': hcsc_model.state_dict(),
#         'optimizer': hcsc_optimizer.state_dict(),
#     }, os.path.join(args.results_dir, 'model_last.pth'))

#     if test_acc > min_acc:
#         min_acc = test_acc
#         torch.save({
#             'epoch': epoch,
#             'state_dict': hcsc_model.state_dict(),
#             'optimizer': hcsc_optimizer.state_dict(),
#         }, os.path.join(args.results_dir, 'model_best.pth'))
#         print(f"💾 保存最佳模型 (Acc: {min_acc:.2f}%)")

# print("训练完成!")
