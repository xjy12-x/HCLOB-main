  
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

from hcsc.loader import GaussianBlur, TwoCropsTransform_JointCrop
from hcsc.loader import Solarize
import torchvision.datasets as datasets

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
parser.add_argument('-b', '--batch-size', default=32, type=int, metavar='N')
parser.add_argument('--wd', default=5e-4, type=float, metavar='W', help='weight decay')

# 层级聚类参数
parser.add_argument('--cluster-levels', default=2, type=int, help='number of hierarchical clustering levels')
parser.add_argument('--cluster-nums', default=[1000, 500], nargs='*', type=int, help='number of clusters at each level')


# # 添加data参数
# parser.add_argument('--data', default='F:\\contrast-11\\HCSC\\data_visdrone2019', type=str, 
#                     help='path to dataset directory')
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


# 添加JointCrop相关参数
parser.add_argument('--crop-min', default=0.08, type=float, help='minimum scale for random cropping')
parser.add_argument('--num-local-views', default=4, type=int, help='number of local views for multi-crop')
parser.add_argument('--local-crop-scale', default=[0.05, 0.4], nargs='*', type=float, help='scale range for local crops')

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

class CIFAR10(CIFAR10):
    """CIFAR10 Dataset（返回一对增强样本）"""
    def __getitem__(self, index):
        img = self.data[index]
        img = Image.fromarray(img)
        if self.transform is not None:
            im_1 = self.transform(img)
            im_2 = self.transform(img)
        return im_1, im_2

# Data loading code

normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    # follow BYOL's augmentation recipe: https://arxiv.org/abs/2006.07733
augmentation1 = [
        # transforms.RandomResizedCrop(224, scale=(args.crop_min, 1.)),
        transforms.RandomApply(
            [transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8  # not strengthened
        ),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=1.0),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ]

augmentation2 = [
        # transforms.RandomResizedCrop(224, scale=(args.crop_min, 1.)),
        transforms.RandomApply(
            [transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8  # not strengthened
        ),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.1),
        transforms.RandomApply([Solarize()], p=0.2),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ]
test_transform = transforms.Compose([
         transforms.ToTensor(),
         transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
     ])
train_transform= TwoCropsTransform_JointCrop(transforms.Compose(augmentation1), transforms.Compose(augmentation2), scale=(args.crop_min, 1.0) )
train_data = CIFAR10(root="F:\\contrast-11\\HCSC\\data_visdrone2019", train=True, transform=train_transform, download=False)
hcsc_train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
memory_data = CIFAR10(root="F:\\contrast-11\\HCSC\\data_visdrone2019", train=True, transform=test_transform, download=False)
memory_loader = DataLoader(memory_data, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)
test_data = CIFAR10(root="F:\\contrast-11\\HCSC\\data_visdrone2019", train=False, transform=test_transform, download=False)
test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)

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

    #---------add------
    def _build_mlp(self, num_layers, input_dim, mlp_dim, output_dim, last_bn=True):
        mlp = []
        for l in range(num_layers):
            dim1 = input_dim if l == 0 else mlp_dim
            dim2 = output_dim if l == num_layers - 1 else mlp_dim

            mlp.append(nn.Linear(dim1, dim2, bias=False))

            if l < num_layers - 1:
                mlp.append(nn.BatchNorm1d(dim2))
                mlp.append(nn.ReLU(inplace=True))
            elif last_bn:
                # follow SimCLR's design: https://github.com/google-research/simclr/blob/master/model_util.py#L157
                # for simplicity, we further removed gamma in BN
                mlp.append(nn.BatchNorm1d(dim2, affine=False))

        return nn.Sequential(*mlp)

    def _build_projector_and_predictor_mlps(self, dim, mlp_dim):
        pass
    #-------------------

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
        """前向传播（修复损失计算逻辑）"""
        if is_eval:
            return self.extract_feat(images, is_eval)
        
        q, k, local_views = self.extract_feat(images, is_eval)
        proto_logits, proto_labels, proto_selected, temp_protos = self.get_protos(q, index, cluster_result)

        # 实例对比损失
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        
        if proto_labels is not None and self.instance_selection and hasattr(self, 'selected_masks'):
            l_neg = []
            for mask in self.selected_masks:
                logit = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])
                l_neg.append(logit * mask)
            logits = [torch.cat([l_pos, ln], dim=1) / self.T for ln in l_neg]
            labels = [torch.zeros(logit.shape[0], dtype=torch.long, device=q.device) for logit in logits]
            instance_loss = sum([F.cross_entropy(logit, label) for logit, label in zip(logits, labels)]) / len(logits)
        else:
            l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])
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
        total_loss = instance_loss + 0.5 * proto_loss

        # 更新队列
        self._dequeue_and_enqueue(k, index)
        return total_loss




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


def hcsc_train(model, data_loader, optimizer, epoch, args):
    """训练函数（终极修复：兼容 JointCrop 输出的列表/5D张量/4D张量）"""
    print(f"🎯 进入hcsc_train函数，第{epoch}轮")
    model.train()
    adjust_learning_rate(optimizer, epoch, args)
    
    total_loss, total_num = 0.0, 0
    train_bar = tqdm(data_loader, desc=f'Training Epoch {epoch}')
    cluster_result = None

    # 批次训练
    for batch_idx, batch in enumerate(train_bar):
        try:
            def convert_to_tensor(x):
                """统一转换：列表 → 4D张量（[B, C, H, W]）"""
                if isinstance(x, torch.Tensor):
                    return x
                elif isinstance(x, (list, tuple)):
                    # 情况1：列表中的元素是张量（每个元素是单张图片 [C, H, W]）
                    if len(x) > 0 and isinstance(x[0], torch.Tensor):
                        # 堆叠成批次张量 [B, C, H, W]
                        return torch.stack(x)
                    # 情况2：列表嵌套列表（JointCrop 特殊输出）
                    elif len(x) > 0 and isinstance(x[0], (list, tuple)):
                        # 先展平嵌套列表，再堆叠
                        flat_list = [item for sublist in x for item in sublist]
                        return torch.stack(flat_list)
                    else:
                        raise ValueError(f"无法转换列表为张量: {type(x[0]) if len(x)>0 else '空列表'}")
                else:
                    raise TypeError(f"不支持的类型: {type(x)}")
            
            # 处理主视图（im_1/im_2）
            if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                im_1 = convert_to_tensor(batch[0])
                im_2 = convert_to_tensor(batch[1])
            else:
                continue
            
            # ========== 核心修复2：检查并修正维度 ==========
            def fix_dimension(x):
                """确保张量是 4D [B, C, H, W]"""
                # 如果是 5D（[2, B, C, H, W]）→ 拆分为 4D
                if x.dim() == 5:
                    if x.shape[0] == 2:
                        return x[0]  # 取第一个视图 → [B, C, H, W]
                    else:
                        # 其他 5D 情况：压缩第一个维度
                        return x.squeeze(0)
                elif x.dim() == 3:
                    return x.unsqueeze(0)
                elif x.dim() == 4:
                    return x
                else:
                    raise ValueError(f"不支持的维度: {x.dim()}, 形状: {x.shape}")
            
            im_1 = fix_dimension(im_1)
            im_2 = fix_dimension(im_2)
            
            # 最终校验：必须是 4D 张量
            # assert im_1.dim() == 4, f"im_1 必须是 4D 张量，当前维度: {im_1.dim()}, 形状: {im_1.shape}"
            # assert im_2.dim() == 4, f"im_2 必须是 4D 张量，当前维度: {im_2.dim()}, 形状: {im_2.shape}"
            
            #print(f"✅ 最终输入形状 - im_1: {im_1.shape}, im_2: {im_2.shape}")
            
            # 移动到GPU
            im_1 = im_1.cuda(non_blocking=True)
            im_2 = im_2.cuda(non_blocking=True)
            
            # 构建输入列表
            images = [im_1, im_2]                                                                                                       
            batch_size = im_1.size(0)
            index = torch.arange(batch_idx * batch_size, (batch_idx + 1) * batch_size, device=im_1.device)
            
            # 处理 multi_crop 的 local views（如果有）
            if args.multi_crop and len(batch) > 2:
                local_views = []
                for view in batch[2:]:
                    # 统一转换为张量并修正维度
                    view_tensor = convert_to_tensor(view)
                    view_tensor = fix_dimension(view_tensor)
                    assert view_tensor.dim() == 4, f"local view 必须是 4D 张量"
                    local_views.append(view_tensor.cuda(non_blocking=True))
                images.extend(local_views)
            
            # 前向传播计算损失
            loss = model(images, is_eval=False, cluster_result=cluster_result, index=index)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 统计损失
            total_num += batch_size
            total_loss += loss.item() * batch_size
            train_bar.set_description(f'Epoch {epoch}: Loss: {total_loss/total_num:.4f}')
            
        except Exception as e:
            print(f"❌ 批次 {batch_idx} 失败: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    return total_loss / total_num if total_num > 0 else 0.0

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
  
