import time
import torch.distributed as dist
from datetime import datetime
from functools import partial
import json
import pandas as pd
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torchvision.models import resnet
from tqdm import tqdm
import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import faiss
import torch.backends.cudnn as cudnn
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

from hcsc.logger import EasyLogger
import torchvision.models as models

import hcsc.loader
import shutil


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
parser = argparse.ArgumentParser(description='Train HCSC on CIFAR-10')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50', help='model architecture')
parser.add_argument('--lr', '--learning-rate', default=0.03, type=float, metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--epochs', default=300, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('--schedule', default=[120, 160], nargs='*', type=int, help='learning rate schedule')
parser.add_argument('--cos', type=int, default=1, help='use cosine lr schedule')
parser.add_argument('-b', '--batch-size', default=64, type=int, metavar='N')
parser.add_argument('--wd', default=5e-4, type=float, metavar='W', help='weight decay')

# 层级聚类参数
parser.add_argument('--cluster-levels', default=3, type=int, help='number of hierarchical clustering levels')
parser.add_argument('--cluster-nums', default=[3000,2000, 1000], nargs='*', type=int, help='number of clusters at each level')
# 在参数解析部分添加
parser.add_argument('--warmup-epoch', default=10, type=int, 
                    help='number of warm-up epochs before clustering')
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

# ===================== 数据加载 =====================
class CIFAR10Pair(CIFAR10):
    """CIFAR10 Dataset（返回一对增强样本）"""
    def __getitem__(self, index):
        img = self.data[index]
        img = Image.fromarray(img)
        if self.transform is not None:
            im_1 = self.transform(img)
            im_2 = self.transform(img)
        return [im_1, im_2], index

# 数据增强
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(32),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
])

# 加载数据集
train_data = CIFAR10Pair(root="F:\\contrast-11\\HCSC\\data_visdrone2019", train=True, transform=train_transform, download=False)
hcsc_train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)

memory_data = CIFAR10(root="F:\\contrast-11\\HCSC\\data_visdrone2019", train=True, transform=test_transform, download=False)
memory_loader = DataLoader(memory_data, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)

test_data = CIFAR10(root="F:\\contrast-11\\HCSC\\data_visdrone2019", train=False, transform=test_transform, download=False)
test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)

# ===================== 辅助模块 =====================
class SplitBatchNorm(nn.BatchNorm2d):
    """分割BatchNorm（适配多GPU，单GPU也可使用）"""
    def __init__(self, num_features, num_splits, **kw):
        super().__init__(num_features, **kw)
        self.num_splits = num_splits

    def forward(self, input):
        N, C, H, W = input.shape
        if self.training or not self.track_running_stats:
            running_mean_split = self.running_mean.repeat(self.num_splits)
            running_var_split = self.running_var.repeat(self.num_splits)
            outcome = nn.functional.batch_norm(
                input.view(-1, C * self.num_splits, H, W), running_mean_split, running_var_split,
                self.weight.repeat(self.num_splits), self.bias.repeat(self.num_splits),
                True, self.momentum, self.eps).view(N, C, H, W)
            self.running_mean.data.copy_(running_mean_split.view(self.num_splits, C).mean(dim=0))
            self.running_var.data.copy_(running_var_split.view(self.num_splits, C).mean(dim=0))
            return outcome
        else:
            return nn.functional.batch_norm(
                input, self.running_mean, self.running_var,
                self.weight, self.bias, False, self.momentum, self.eps)

# ===================== 核心模型定义 =====================
class HCSC(nn.Module):
    """HCSC框架（修复单GPU适配问题）"""
    def __init__(self, base_encoder, dim=128, queue_length=16384, m=0.999, T=0.2, mlp=True,
                 multi_crop=False, instance_selection=True, proto_selection=True, selection_on_local=True, logger=None,** kwargs):
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
        self.buffer_dict = dict()
        self.mined_index = list()

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """动量更新key编码器"""
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, index=None):
        """队列管理（单GPU适配）"""
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        assert self.queue_length % batch_size == 0

        # 入队出队
        self.queue[:, ptr:ptr + batch_size] = keys.T
        if index is not None:
            self.queue_index[ptr: ptr + batch_size] = index
        ptr = (ptr + batch_size) % self.queue_length
        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle(self, x):
        """单GPU版本Batch Shuffle（替代DDP版本）"""
        batch_size = x.shape[0]
        idx_shuffle = torch.randperm(batch_size).cuda()
        idx_unshuffle = torch.argsort(idx_shuffle)
        return x[idx_shuffle], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle(self, x, idx_unshuffle):
        """单GPU版本Batch Unshuffle"""
        return x[idx_unshuffle]

    @torch.no_grad()
    def sample_neg_instance(self, im2cluster, centroids, density, index):
        """负实例采样（修复维度匹配）"""
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
            label = im2cluster[layer][index].clamp(0, centroid.shape[0]-1)
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
        """提取key特征（单GPU适配）"""
        self._momentum_update_key_encoder()
        im_k_shuffled, idx_unshuffle = self._batch_shuffle(im_k)
        k = self.encoder_k(im_k_shuffled)
        k = nn.functional.normalize(k, dim=1)
        k = self._batch_unshuffle(k, idx_unshuffle)
        return k

    def extract_feat(self, images, is_eval=False):
        """提取特征（统一接口）"""
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
    # def extract_feat(self, images, is_eval=False):
    #     """提取特征（统一接口）"""
    #     if is_eval:
    #         if isinstance(images, list):
    #             if len(images) > 0:
    #                 images = images[0]
    #             else:
    #                 raise ValueError("评估模式下传入的图像列表为空")
        
    #         if not isinstance(images, torch.Tensor):
    #             if isinstance(images, tuple) and len(images) > 0:
    #                 images = images[0]  # 假设第一个元素是图像
    #             else:
    #                 raise TypeError(f"不支持的输入类型: {type(images)}")
        
    #         k = self.encoder_k(images)
    #         k = nn.functional.normalize(k, dim=1)
    #         return k
    
    #     im_q, im_k = images[0], images[1]
    #     q = self.encoder_q(im_q)
    #     q = nn.functional.normalize(q, dim=1)
    #     if self.multi_crop:
    #         k = self.extract_key_feat(im_k)
    #         local_views = [nn.functional.normalize(self.encoder_q(im_local), dim=1) for im_local in images[2:]]
    #         return q, k, local_views
    #     else:
    #         k = self.extract_key_feat(im_k)
    #         return q, k, None
    

    def get_protos(self, q, index, cluster_result):
        """获取原型（修复cluster2cluster缺失问题）"""
        if cluster_result is None:
            return None, None, None, None
        
        proto_labels = []
        proto_logits = []
        proto_selecteds = []
        temp_protos = []
        
        # 补全缺失的cluster2cluster和logits
        cluster2cluster = cluster_result.get('cluster2cluster', [[] for _ in cluster_result['im2cluster']])
        logits = cluster_result.get('logits', [torch.randn(c.shape[0], c.shape[0]).cuda() for c in cluster_result['centroids']])
        
        for n, (im2cluster, prototypes, density) in enumerate(zip(
                cluster_result['im2cluster'], cluster_result['centroids'], cluster_result['density'])):
            # 转换为cuda
            im2cluster = im2cluster.cuda() if not im2cluster.is_cuda else im2cluster
            prototypes = prototypes.cuda() if not prototypes.is_cuda else prototypes
            density = density.cuda() if not density.is_cuda else density
            
            # 正原型
            pos_proto_id = im2cluster[index].clamp(0, prototypes.shape[0]-1)  # 防止索引越界
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
                if n == len(cluster_result['im2cluster']) - 1:  # 最后一层
                    neg_prototypes = prototypes[neg_proto_id]
                    logits_proto = torch.cat([
                        torch.einsum('nc,nc->n', [q, pos_prototypes]).unsqueeze(-1),
                        torch.mm(q, neg_prototypes.t())
                    ], dim=1)
                    # 密度加权
                    temp_map = torch.cat([
                        density[pos_proto_id].unsqueeze(-1),
                        density[neg_proto_id].unsqueeze(0).repeat(q.shape[0], 1)
                    ], dim=1)
                    logits_proto = logits_proto / temp_map
                else:
                    # 补全cluster2cluster（层级映射）
                    if len(cluster2cluster[n]) == 0:
                        cluster2cluster[n] = torch.randint(0, logits[n].shape[1], (prototypes.shape[0],), device=q.device)
                    upper_pos_proto_id = cluster2cluster[n][pos_proto_id]
                    
                    # 采样概率计算
                    densities = density.clamp(min=1e-3)
                    sampling_prob = 1 - (logits[n][neg_proto_id] / densities[neg_proto_id].unsqueeze(1)).softmax(-1)[:, upper_pos_proto_id].t()
                    
                    # 负原型采样
                    neg_sampler = torch.distributions.bernoulli.Bernoulli(sampling_prob.clamp(0.001, 0.999))
                    selected_mask = neg_sampler.sample()
                    
                    # 负原型logits
                    neg_prototypes = prototypes[neg_proto_id]
                    neg_logits = torch.mm(q, neg_prototypes.t()) * selected_mask
                    logits_proto = torch.cat([
                        torch.einsum('nc,nc->n', [q, pos_prototypes]).unsqueeze(-1),
                        neg_logits
                    ], dim=1)
                    temp_map = torch.cat([
                        density[pos_proto_id].unsqueeze(-1),
                        density[neg_proto_id].unsqueeze(0).repeat(q.shape[0], 1)
                    ], dim=1)
                    logits_proto = logits_proto / temp_map
            else:
                neg_prototypes = prototypes[neg_proto_id]
                logits_proto = torch.cat([
                    torch.einsum('nc,nc->n', [q, pos_prototypes]).unsqueeze(-1),
                    torch.mm(q, neg_prototypes.t())
                ], dim=1)
                temp_map = torch.cat([
                    density[pos_proto_id].unsqueeze(-1),
                    density[neg_proto_id].unsqueeze(0).repeat(q.shape[0], 1)
                ], dim=1)
                logits_proto = logits_proto / temp_map

            labels_proto = torch.zeros(q.shape[0], dtype=torch.long, device=q.device)
            proto_labels.append(labels_proto)
            proto_logits.append(logits_proto)

        return proto_logits, proto_labels, proto_selecteds, temp_protos

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
        total_loss: 总损失
       """
        if is_eval:
            return self.extract_feat(images, is_eval)
        else:
            q, k, local_views = self.extract_feat(images, is_eval)
            
        proto_logits, proto_labels, proto_selected, temp_protos = self.get_protos(q, index, cluster_result)
    
        # 正样本相似度计算
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
    
         # 初始化损失
        loss1 = 0.0
        loss2 = 0.0
        loss3 = 0.0
        loss4 = 0.0
    
         # 实例对比损失
        if proto_labels is not None and self.instance_selection:
            try:
                self.selected_masks, sample_ratios = self.sample_neg_instance(cluster_result['im2cluster'], proto_selected, temp_protos, index)
                self.buffer_dict['avg_sample_ratios'] = sum(sample_ratios) / len(sample_ratios)
                l_neg = []
                for selected_mask in self.selected_masks:
                    logit = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])
                    mask = selected_mask.clone().float()
                    l_neg.append(logit * mask)
            
                # 计算多级实例对比损失
                instance_losses = []
                for l_n in l_neg:
                    logits = torch.cat([l_pos, l_n], dim=1) / self.T
                    labels = torch.zeros(logits.shape[0], dtype=torch.long, device=q.device)
                    instance_losses.append(nn.CrossEntropyLoss()(logits, labels))
            
                 # 平均多个实例损失
                instance_loss = sum(instance_losses) / len(instance_losses)
            except:
                l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])
                logits = torch.cat([l_pos, l_neg], dim=1) / self.T
                labels = torch.zeros(logits.shape[0], dtype=torch.long, device=q.device)
                instance_loss = nn.CrossEntropyLoss()(logits, labels)
        else:
            l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])
            logits = torch.cat([l_pos, l_neg], dim=1) / self.T
            labels = torch.zeros(logits.shape[0], dtype=torch.long, device=q.device)
            instance_loss = nn.CrossEntropyLoss()(logits, labels)
            
        loss1 += instance_loss
    
    # 原型对比损失
        if proto_logits is not None and self.proto_selection:
            proto_losses = []
            for plogits, plabels in zip(proto_logits, proto_labels):
                if plogits is not None and plabels is not None:
                # 添加温度缩放
                    proto_loss = nn.CrossEntropyLoss()(plogits / self.T, plabels)
                    proto_losses.append(proto_loss)
        
            if proto_losses:
            # 平均多个原型损失
                total_proto_loss = sum(proto_losses) / len(proto_losses)
            # 加权原型损失（可以根据需要调整权重）
                # proto_weight = 0.5
                # loss2 += proto_weight * total_proto_loss
                loss2 += total_proto_loss
    
    # 局部视图的实例和原型对比损失（如果启用了多crop）
        if self.multi_crop and local_views is not None:
        # 局部实例对比损失
            if hasattr(self, 'compute_local_logits'):
                local_logits, local_labels = self.compute_local_logits(q, k, local_views, index)
                if local_logits is not None and local_labels is not None:
                    local_instance_loss = 0.0
                    for logit, label in zip(local_logits, local_labels):
                        if logit is not None and label is not None:
                            local_instance_loss += nn.CrossEntropyLoss()(logit, label)
                
                    if local_instance_loss > 0:
                        loss3 += local_instance_loss / len(local_logits)
        
        # 局部原型对比损失
            if hasattr(self, 'compute_local_proto_logits'):
                local_proto_logits, local_proto_targets = self.compute_local_proto_logits(
                    local_views, proto_selected, cluster_result, index
                 )
                if local_proto_logits is not None and local_proto_targets is not None:
                    local_proto_loss = 0.0
                    for proto_logits_list, proto_targets_list in zip(local_proto_logits, local_proto_targets):
                        for plogits, ptargets in zip(proto_logits_list, proto_targets_list):
                            if plogits is not None and ptargets is not None:
                                local_proto_loss += nn.CrossEntropyLoss()(plogits, ptargets)
                
                    if local_proto_loss > 0:
                        loss4 += local_proto_loss / (len(local_proto_logits) * len(local_proto_logits[0]))
    
        self._dequeue_and_enqueue(k, index)
        total_loss = 0*loss1 + loss2 + loss3 + loss4
    
        return total_loss

# ===================== 聚类功能修复 =====================
def run_hkmeans(x, args):
    """
    This function is a hierarchical k-means
    修改：添加GPU检测和回退机制
    """
    
    print('performing kmeans clustering')
    results = {'im2cluster':[], 'centroids':[], 'density':[], 'cluster2cluster':[], 'logits':[]}
    
    # 检查GPU可用性
    gpu_available = False
    try:
        # 尝试导入GPU资源
        if hasattr(faiss, 'StandardGpuResources'):
            gpu_available = True
            print(f"✅ GPU版本faiss可用")
        else:
            print(f"⚠️  GPU版本faiss不可用，将使用CPU版本")
    except:
        print(f"⚠️  无法检测GPU支持，将使用CPU版本")
    
    for seed, num_cluster in enumerate(args.num_cluster):
        # 初始化faiss聚类参数
        d = x.shape[1]
        k = int(num_cluster)
        clus = faiss.Clustering(d, k)
        clus.verbose = True
        clus.niter = 20
        clus.nredo = 5
        clus.seed = seed
        clus.max_points_per_centroid = 1000
        clus.min_points_per_centroid = 10
        
        # 根据GPU可用性选择索引类型
        if gpu_available and hasattr(faiss, 'StandardGpuResources'):
            try:
                res = faiss.StandardGpuResources()
                cfg = faiss.GpuIndexFlatConfig()
                cfg.useFloat16 = False
                cfg.device = args.local_rank if hasattr(args, 'local_rank') else 0
                index = faiss.GpuIndexFlatL2(res, d, cfg)
                print(f"  使用GPU进行聚类")
            except Exception as e:
                print(f"  GPU初始化失败: {e}，回退到CPU")
                gpu_available = False
                index = faiss.IndexFlatL2(d)
        else:
            index = faiss.IndexFlatL2(d)
            print(f"  使用CPU进行聚类")
        
        if seed == 0:  # 第一层直接在实例特征上聚类
            clus.train(x, index)   
            D, I = index.search(x, 1)  # 为每个样本找到聚类距离和分配
        else:
            # 更高层在前一层的聚类中心上聚类
            clus.train(results['centroids'][seed - 1].cpu().numpy(), index)
            D, I = index.search(results['centroids'][seed - 1].cpu().numpy(), 1)
        
        im2cluster = [int(n[0]) for n in I]
        # 样本到聚类中心的距离
        Dcluster = [[] for c in range(k)]          
        for im, i in enumerate(im2cluster):
            Dcluster[i].append(D[im][0])

        # 获取聚类中心
        centroids = faiss.vector_to_array(clus.centroids).reshape(k, d)

        if seed > 0:  # 高层级的聚类分配是前一层的索引
            im2cluster = np.array(im2cluster)  # 启用批索引
            results['cluster2cluster'].append(torch.LongTensor(im2cluster).cuda())
            im2cluster = im2cluster[results['im2cluster'][seed - 1].cpu().numpy()]
            im2cluster = list(im2cluster)
    
        if len(set(im2cluster)) == 1:
            print("Warning! All samples are assigned to one cluster")

        # 浓度估计
        density = np.zeros(k)
        for i, dist in enumerate(Dcluster):
            if len(dist) > 1:
                d_val = (np.asarray(dist)**0.5).mean() / np.log(len(dist) + 10)            
                density[i] = d_val     
                
        # 如果聚类只有一个点，使用最大值估计其浓度
        dmax = density.max()
        for i, dist in enumerate(Dcluster):
            if len(dist) <= 1:
                density[i] = dmax 

        density = density.clip(np.percentile(density, 10), np.percentile(density, 90)) 
        density = args.T * density / density.mean() 
        
        # 转换为cuda张量
        centroids = torch.Tensor(centroids).cuda()
        centroids = nn.functional.normalize(centroids, p=2, dim=1)    
        if seed > 0:  # 维护从低层原型到高层的logits
            proto_logits = torch.mm(results['centroids'][-1], centroids.t())
            results['logits'].append(proto_logits.cuda())

        density = torch.Tensor(density).cuda()
        im2cluster = torch.LongTensor(im2cluster).cuda()    
        results['centroids'].append(centroids)
        results['density'].append(density)
        results['im2cluster'].append(im2cluster)    
        
    return results

# ===================== 训练与评估函数 =====================
def adjust_learning_rate(optimizer, epoch, args):
    """学习率调整"""
    if args.cos:
        lr = args.lr * 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    else:
        lr = args.lr
        for milestone in args.schedule:
            if epoch >= milestone:
                lr *= 0.1
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

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
def build_optimizer(args, model):
    # 安全地获取world_size
    if args.distributed and dist.is_initialized():
        total_batch_size = args.batch_size * dist.get_world_size()
    else:
        total_batch_size = args.batch_size  # 单GPU模式
    
    # scale up the batch size
    args.lr = args.lr * total_batch_size / 256
    print("total batch size is {}, lr is scaled up to {}".format(total_batch_size, args.lr))

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                            momentum=args.momentum,
                            weight_decay=args.weight_decay)
    return optimizer

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

def compute_features_single_gpu(eval_loader, model, args):
    """单GPU版本的特征计算"""
    print('计算特征（单GPU）...')
    model.eval()
    
    all_features = []
    all_indices = []
    
    with torch.no_grad():
        for images, index in tqdm(eval_loader, desc='提取特征'):
            # 处理输入格式
            if isinstance(images, list):
                eval_image = images[0].to(args.device)  # 取第一个视图进行评估
            elif isinstance(images, torch.Tensor):
                eval_image = images.to(args.device)
            else:
                continue
            
            # 提取特征
            feat = model.extract_feat(eval_image, is_eval=True)
            
            if feat is not None:
                all_features.append(feat)
                all_indices.append(index)
    
    # 合并所有特征
    if all_features:
        features = torch.cat(all_features, dim=0)
        indices = torch.cat(all_indices, dim=0)
        
        # 确保特征顺序正确
        sorted_features = torch.zeros(len(eval_loader.dataset), args.dim, device=args.device)
        sorted_features[indices] = features
        
        print(f"特征计算完成: {sorted_features.shape}")
        return sorted_features
    else:
        print("❌ 无法提取特征")
        return torch.zeros(0, args.dim, device=args.device)
       
def hcsc_train(model, train_loader, optimizer, epoch, args, cluster_result=None):
    """简化的训练函数，适配当前数据格式"""
    # 确保有 device 属性
    if not hasattr(args, 'device'):
        args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 添加聚类状态信息
    if cluster_result is not None:
        print(f"   使用聚类结果: {len(cluster_result['im2cluster'])} 层")
    else:
        print(f"   未使用聚类结果")
    
    model.train()
    total_loss = 0
    num_batches = 0
    
    # 创建进度条
    train_bar = tqdm(train_loader, desc=f'训练 Epoch {epoch}', total=len(train_loader), leave=True)
    
    for batch_idx, (images, index) in enumerate(train_bar):
        # 移动数据到设备
        images = [image.to(args.device) for image in images]
        index = index.to(args.device)
        
        # 前向传播
        loss = model(images, cluster_result=cluster_result, index=index)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 记录损失
        loss_value = loss.item()
        total_loss += loss_value
        num_batches += 1
        
        # 更新进度条
        avg_loss = total_loss / num_batches
        train_bar.set_postfix({
            'Loss': f'{loss_value:.4f}',
            'AvgLoss': f'{avg_loss:.4f}',
            'Batch': f'{batch_idx+1}/{len(train_loader)}'
        })
    
    # 计算平均损失
    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    print(f"   Epoch {epoch} 完成 - 平均损失: {avg_loss:.4f}")
    
    return avg_loss


def knn_predict(feature, feature_bank, feature_labels, classes, knn_k, knn_t):
    """KNN预测"""
    sim_matrix = torch.mm(feature, feature_bank)
    sim_weight, sim_indices = sim_matrix.topk(k=knn_k, dim=-1)
    sim_labels = torch.gather(feature_labels.expand(feature.size(0), -1), dim=-1, index=sim_indices)
    sim_weight = (sim_weight / knn_t).exp()
    
    one_hot_label = torch.zeros(feature.size(0) * knn_k, classes, device=sim_labels.device)
    one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0)
    pred_scores = torch.sum(one_hot_label.view(feature.size(0), -1, classes) * sim_weight.unsqueeze(dim=-1), dim=1)
    return pred_scores.argsort(dim=-1, descending=True)


def test(net, memory_data_loader, test_data_loader, epoch, args):
    """测试函数"""
    net.eval()
    classes = len(memory_data_loader.dataset.classes)
    total_top1, total_num = 0.0, 0
    feature_bank, feature_labels = [], []
    
    with torch.no_grad():
        # 生成特征库
        for data, target in tqdm(memory_data_loader, desc='特征提取'):
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
            test_bar.set_description(f'Test Epoch {epoch}: Acc@1: {total_top1/total_num*100:.2f}%')
    
    return total_top1 / total_num * 100


# ===================== 主函数 =====================
def create_hcsc_model(args):
    """创建模型实例"""
    if args.arch == 'resnet18':
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
        selection_on_local=args.selection_on_local
    ).cuda()

def build_dataloaders(args):
    return getattr(hcsc.loader, args.dataset)(args)

# 初始化模型和优化器
print("创建HCSC模型实例...")
hcsc_model = create_hcsc_model(args)
print(f"模型类型: {type(hcsc_model)}")
print(f"参数数量: {sum(p.numel() for p in hcsc_model.parameters()):,}")

hcsc_optimizer = torch.optim.SGD(
    hcsc_model.parameters(),
    lr=args.lr,
    weight_decay=args.wd,
    momentum=0.9
)
criterion = nn.CrossEntropyLoss()
# 修改主函数调用
if __name__ == "__main__":
    # 检查必要的参数
    if not hasattr(args, 'device'):
        args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"✅ 设置 device: {args.device}")
    
    if not hasattr(args, 'num_cluster'):
        args.num_cluster = [1000, 500]
        print(f"✅ 设置聚类层级: {args.num_cluster}")
    
    if not hasattr(args, 'warmup_epoch'):
        args.warmup_epoch = 10
        print(f"✅ 设置热身epoch: {args.warmup_epoch}")
    
    # 创建结果目录
    if not os.path.exists(args.results_dir):
        os.makedirs(args.results_dir, exist_ok=True)
        print(f"✅ 创建结果目录: {args.results_dir}")
    
    # 测试聚类功能
    print("\n" + "="*60)
    print("测试聚类功能")
    print("="*60)
    
    # 创建模型
    hcsc_model = create_hcsc_model(args)
    hcsc_model = hcsc_model.to(args.device)
    # create dataset
    train_loader, eval_loader, train_dataset, eval_dataset, train_sampler = build_dataloaders(args)
    # 测试特征提取
    with torch.no_grad():
        test_batch, _ = next(iter(hcsc_train_loader))
        if isinstance(test_batch, tuple) and len(test_batch) == 2:
            images_list, _ = test_batch
            eval_image = images_list[0].to(args.device)
        else:
            eval_image = test_batch[0].to(args.device) if isinstance(test_batch, list) else test_batch.to(args.device)
        
        test_features = hcsc_model.extract_feat(eval_image, is_eval=True)
        print(f"✅ 特征提取测试成功: {test_features.shape}")
    
    # 创建优化器
    hcsc_optimizer = torch.optim.SGD(
        hcsc_model.parameters(),
        lr=args.lr,
        weight_decay=args.wd,
        momentum=0.9
    )
    
    print("\n" + "="*60)
    print("开始训练")
    print("="*60)
    
    # 主训练循环
    for epoch in range(args.start_epoch, args.epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{args.epochs}")
        print(f"{'='*60}")
        
        # 设置模型epoch
        if hasattr(hcsc_model, 'set_epoch'):
            hcsc_model.set_epoch(epoch)
        
        cluster_result = None
        
        # 层级聚类
        if epoch >= args.warmup_epoch:
            print(f"🚀 进行层级聚类...")
            
            # 收集所有数据的特征
            all_features = []
            hcsc_model.eval()
            
            with torch.no_grad():
                for images, _ in tqdm(eval_loader, desc="提取特征", leave=False):
                    if isinstance(images, tuple) and len(images) == 2:
                        images_list, _ = images
                        eval_image = images_list[0].to(args.device)
                    else:
                        eval_image = images[0].to(args.device) if isinstance(images, list) else images.to(args.device)
                    
                    features = hcsc_model.extract_feat(eval_image, is_eval=True)
                    all_features.append(features)
            
            if all_features:
                all_features = torch.cat(all_features, dim=0)
                features_np = all_features.cpu().numpy()
                
                # 运行层级聚类
                try:
                    cluster_result = run_hkmeans(features_np, args)
                    print(f"✅ 聚类成功: 层级数={len(cluster_result['im2cluster'])}, "
                          f"聚类数={[c.shape[0] for c in cluster_result['centroids']]}")
                except Exception as e:
                    print(f"❌ 聚类失败: {e}")
                    cluster_result = None
            else:
                print("❌ 没有提取到特征")
        else:
            print(f"⏳ 热身阶段 (epoch {epoch}/{args.warmup_epoch})，跳过聚类")
        
        # 调整学习率
        adjust_learning_rate(hcsc_optimizer, epoch, args)
        
        # 训练一个epoch
        print(f"\n🎯 开始训练...")
        hcsc_model.train()
        train_loss = hcsc_train(hcsc_model, hcsc_train_loader, hcsc_optimizer, epoch, args, cluster_result)
        
        # 保存检查点
        if (epoch + 1) % 5 == 0:
            checkpoint_path = os.path.join(args.results_dir, f'checkpoint_epoch{epoch+1}.pth')
            torch.save({
                'epoch': epoch + 1,
                'state_dict': hcsc_model.state_dict(),
                'optimizer': hcsc_optimizer.state_dict(),
            }, checkpoint_path)
            print(f"💾 检查点已保存: {checkpoint_path}")
        
        # 测试
        if (epoch + 1) % 20 == 0:
            test_acc = test(hcsc_model.encoder_q, memory_loader, test_loader, epoch, args)
            print(f"🧪 测试准确率: {test_acc:.2f}%")
    
    print("\n" + "="*60)
    print("训练完成!")
    print("="*60)

# def main():

#     args.num_cluster = args.num_cluster.split(',')
#     if not os.path.exists(args.exp_dir):
#         os.makedirs(args.exp_dir, exist_ok=True)
    
#     logger = EasyLogger(args.exp_dir, 0, args.rank)
#     # create dataset
#     train_loader, eval_loader, train_dataset, eval_dataset, train_sampler = build_dataloaders(args)
#     # dist.barrier()
#     # args.dataset_size = len(train_dataset)
#     # 只在分布式模式下调用barrier
#     if args.distributed:
#         dist.barrier()
    
#     args.dataset_size = len(train_dataset)
#     # create model
#     if args.rank == 0:
#         print("=> creating model '{}'".format(args.arch))
#     model = hcsc_model(args, logger)
#     model = model.to(args.device)
#     # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank)
#     # 只在分布式模式下使用DistributedDataParallel
#     if args.distributed:
#         model = torch.nn.parallel.DistributedDataParallel(
#             model, 
#             device_ids=[args.local_rank], 
#             output_device=args.local_rank
#         )
#     else:
#         # 单GPU模式，使用DataParallel或直接使用模型
#         if torch.cuda.device_count() > 1:
#             model = torch.nn.DataParallel(model)

#     # define loss function (criterion) and optimizer
#     criterion = nn.CrossEntropyLoss().to(args.device)

#     optimizer = build_optimizer(args, model)
#     scheduler = adjust_learning_rate
    
#     # optionally resume from a checkpoint
#     if args.resume:
#         if os.path.isfile(args.resume):
#             print("=> loading checkpoint '{}'".format(args.resume))
#             checkpoint = torch.load(args.resume, map_location=args.device)

#             args.start_epoch = checkpoint['epoch']
#             model.load_state_dict(checkpoint['state_dict'], strict=False)
#             if "optimizer" in checkpoint:
#                 optimizer.load_state_dict(checkpoint['optimizer'])
#             else:
#                 print("No optimizer state!")
#             print("=> loaded checkpoint '{}' (epoch {})"
#                   .format(args.resume, checkpoint['epoch']))
#         else:
#             print("=> no checkpoint found at '{}'".format(args.resume))

#     cudnn.benchmark = True



    
#     # 主训练循环
#     for epoch in range(args.start_epoch, args.epochs):
#         print(f"\n{'='*60}")
#         print(f"Epoch {epoch}/{args.epochs}")
#         print(f"{'='*60}")
        
#         # 设置模型epoch
#         if hasattr(model, 'module') and hasattr(model.module, "set_epoch"):
#             model.module.set_epoch(epoch)
#         elif hasattr(model, "set_epoch"):
#             model.set_epoch(epoch)
        
#         cluster_result = None
        
#         # 层级聚类
#         if epoch >= args.warmup_epoch:
#             print(f"🚀 进行层级聚类 (warmup_epoch={args.warmup_epoch})...")
            
#             # 1. 计算所有数据的特征
#             print("   1. 计算特征...")
#             features = compute_features_single_gpu(eval_loader, model, args)
            
#             if features.shape[0] > 0:
#                 # 2. 特征归一化处理
#                 features_np = features.cpu().numpy()
#                 norm_mask = np.linalg.norm(features_np, axis=1) > 1.5
#                 features_np[norm_mask] /= 2
                
#                 # 3. 检查数据量
#                 n_samples = features_np.shape[0]
#                 print(f"   样本数: {n_samples}")
                
#                 # 4. 调整聚类数（确保不超过样本数）
#                 if not hasattr(args, 'num_cluster'):
#                     args.num_cluster = [1000, 500]  # 默认聚类层级
                
#                 # 调整聚类数
#                 adjusted_clusters = []
#                 for num_cluster in args.num_cluster:
#                     if n_samples >= num_cluster * 2:  # 每个聚类至少2个样本
#                         adjusted_clusters.append(num_cluster)
#                     else:
#                         adjusted = max(2, n_samples // 2)  # 调整为样本数的一半
#                         print(f"   ⚠️ 聚类数 {num_cluster} 调整为 {adjusted}")
#                         adjusted_clusters.append(adjusted)
                
#                 args.num_cluster = adjusted_clusters
                
#                 # 5. 运行层级聚类
#                 print(f"   2. 运行层级聚类: {args.num_cluster}")
#                 try:
#                     cluster_result = run_hkmeans(features_np, args)
#                     print(f"   ✅ 聚类成功: 层级数={len(cluster_result['im2cluster'])}, "
#                           f"聚类数={[c.shape[0] for c in cluster_result['centroids']]}")
                    
#                     # 6. 保存聚类结果
#                     try:
#                         torch.save(cluster_result, os.path.join(args.results_dir, f'clusters_epoch{epoch}.pth'))
#                         print(f"   ✅ 聚类结果已保存")
#                     except Exception as e:
#                         print(f"   ❌ 保存聚类结果失败: {e}")
                        
#                 except Exception as e:
#                     print(f"   ❌ 聚类失败: {e}")
#                     import traceback
#                     traceback.print_exc()
#                     cluster_result = None
#             else:
#                 print("   ❌ 没有提取到特征")
#         else:
#             print(f"⏳ 热身阶段 (epoch {epoch}/{args.warmup_epoch})，跳过聚类")
        
#         # 设置采样器epoch（分布式训练）
#         if args.distributed and train_sampler is not None:
#             train_sampler.set_epoch(epoch)
        
#         # 调整学习率
#         adjust_learning_rate(optimizer, epoch, args)
        
#         # 训练一个epoch
#         print(f"\n🎯 开始训练...")
#         train_loss = hcsc_train(train_loader, model, criterion, optimizer, epoch, args, cluster_result)
        
#         # 保存检查点
#         if (epoch + 1) % 5 == 0 and (not args.distributed or args.rank == 0):
#             checkpoint_path = os.path.join(args.results_dir, f'checkpoint_{epoch+1:04d}.pth.tar')
#             save_checkpoint({
#                 'epoch': epoch + 1,
#                 'arch': args.arch,
#                 'state_dict': model.state_dict(),
#                 'optimizer': optimizer.state_dict(),
#             }, is_best=False, filename=checkpoint_path)
#             print(f"💾 检查点已保存: {checkpoint_path}")
    
#     print("\n" + "="*60)
#     print("训练完成!")
#     print("="*60)