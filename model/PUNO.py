import torch
import torch.nn as nn
import pytorch3d.ops
import numpy as np
from utils.common import *

import torch.nn.init as init

def knn(x, k):
    xt = x.transpose(2, 1)
    return pytorch3d.ops.knn_points(xt, xt, K=k)[1]

def get_graph_feature(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)  # (batch_size, num_points, k)
    device = x.device

    idx_base = torch.arange(
        0, batch_size, device=device).view(-1, 1, 1) * num_points

    idx = idx + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2,
                    1).contiguous()  # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()  # (B,C,N,K)

    return feature


class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5):
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)

        out = (x - mean) / (std + self.eps)
        out = self.weight * out + self.bias
        return out

class galerkinAtten(nn.Module):
    def __init__(self, midc, heads):
        super().__init__()

        self.headc = midc // heads
        self.heads = heads
        self.midc = midc  # feat_dim

        self.qkv_proj = nn.Conv1d(midc, 3 * midc, 1)
        self.o_proj1 = nn.Conv1d(midc, midc, 1)
        self.o_proj2 = nn.Conv1d(midc, midc, 1)

        self.kln = LayerNorm((self.heads, 1, self.headc))
        self.vln = LayerNorm((self.heads, 1, self.headc))

        self.act = nn.GELU()

    def forward(self, x, name='0'):
        B, C, N = x.shape
        bias = x

        qkv = self.qkv_proj(x)  # B, 3*C, N
        qkv = qkv.permute(0, 2, 1).reshape(B, N, self.heads, 3 * self.headc)  # B, N, hd, 3*C/hd
        qkv = qkv.permute(0, 2, 1, 3)  # B, hd, N, 3*C/hd
        q, k, v = qkv.chunk(3, dim=-1)  # B, hd, N, C/hd

        k = self.kln(k)
        v = self.vln(v)

        v = torch.matmul(k.transpose(-2, -1), v) / (N)  # B, hd, C/hd, C/hd
        v = torch.matmul(q, v)  # B, hd, N, C/hd
        v = v.permute(0, 2, 1, 3).reshape(B, N, C)  # B, N, C

        ret = v.permute(0, 2, 1) + bias  # B, C, N
        bias = self.o_proj2(self.act(self.o_proj1(ret))) + bias

        return bias

class Lifting(nn.Module):
    def __init__(self, hidden_dim=48, dim=128):
        super().__init__()

        assert hidden_dim % 6 == 0

        self.embedding_dim = hidden_dim
        e = torch.pow(2, torch.arange(self.embedding_dim // 6)).float() * np.pi
        e = torch.stack([
            torch.cat([e, torch.zeros(self.embedding_dim // 6),
                       torch.zeros(self.embedding_dim // 6)]),
            torch.cat([torch.zeros(self.embedding_dim // 6), e,
                       torch.zeros(self.embedding_dim // 6)]),
            torch.cat([torch.zeros(self.embedding_dim // 6),
                       torch.zeros(self.embedding_dim // 6), e]),
        ])
        self.register_buffer('basis', e)  # 3 x 16

        self.mlp = nn.Linear(self.embedding_dim + 3, dim)

    @staticmethod
    def embed(input, basis):
        projections = torch.einsum(
            'bnd,de->bne', input, basis)
        embeddings = torch.cat([projections.sin(), projections.cos()], dim=2)
        return embeddings

    def forward(self, input):
        # input: B x N x 3
        return self.mlp(torch.cat([self.embed(input, self.basis), input], dim=2))  # B x N x C

class DisplaceNet2v2(nn.Module):
    def __init__(self, fd=64, num_knn=10, L=10, weight_init_scale=1e-5, dp_scale=1e-3):
        super(DisplaceNet2v2, self).__init__()
        self.num_knn = num_knn
        self.L = L
        self.dp_scale = dp_scale
        self.patch_feature_net = nn.Sequential(nn.Conv2d(3, fd, 1), nn.BatchNorm2d(fd),
                                               nn.LeakyReLU(negative_slope=0.2),
                                               nn.Conv2d(fd, fd * 2, 1), nn.BatchNorm2d(fd * 2),
                                               nn.LeakyReLU(negative_slope=0.2),
                                               nn.Conv2d(fd * 2, fd * 2, 1), nn.BatchNorm2d(fd * 2),
                                               nn.LeakyReLU(negative_slope=0.2),
                                               nn.Conv2d(fd * 2, fd * 2, 1)
                                               )
        self.displace_net = nn.Sequential(nn.Conv1d(3 + 2 * 3 * self.L + fd * 4, fd * 4, 1), nn.BatchNorm1d(fd * 4),
                                          nn.LeakyReLU(negative_slope=0.2),
                                          nn.Conv1d(fd * 4, fd * 2, 1), nn.BatchNorm1d(fd * 2),
                                          nn.LeakyReLU(negative_slope=0.2),
                                          nn.Conv1d(fd * 2, fd, 1), nn.BatchNorm1d(fd),
                                          nn.LeakyReLU(negative_slope=0.2),
                                          nn.Conv1d(fd, 3, 1)
                                          )
        for m in self.displace_net.modules():
            if isinstance(m, (torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Linear)):
                m.weight.data = m.weight.data * weight_init_scale

    def forward(self, pc):
        # B, N, 3
        B, N, _ = pc.shape

        _, idx, knn_pc = pytorch3d.ops.knn_points(pc, pc, K=self.num_knn, return_nn=True, return_sorted=False)
        # (B,M,K)    (B,M,K,3)

        knn_pc_local = pc.unsqueeze(2) - knn_pc  # (B,M,K,3)

        feature = self.patch_feature_net(knn_pc_local.permute(0, 3, 1, 2))  # (B,128,M,K)

        max_patch_feature = torch.max(feature, dim=3, keepdim=False)[0]  # (B,128,M)
        mean_patch_feature = torch.mean(feature, dim=3, keepdim=False)  # (B,128,M)

        enc_bases = (2. ** torch.arange(0, self.L)).unsqueeze(0).unsqueeze(0).unsqueeze(3).repeat(B, 1, 1, N).to(pc)
        # B, 1, L, N
        pc2 = pc.transpose(2, 1)
        pc_enc = (pc2[:, :, None, :] * enc_bases)  # B, 3, 1, N -> B, 3, L, N
        pc_enc = pc_enc.reshape(B, -1, N)  # B, 3 * L, N
        pc_enc = torch.cat([pc2, pc_enc.sin(), pc_enc.cos()], dim=1)  # B, 3 + 3 * 2L, N
        concat_vector2 = torch.concat(
            (pc_enc, max_patch_feature, mean_patch_feature), dim=1)
        # (B, 6 + 1 + 128 + 128, M, K)
        dis_pc = self.dp_scale * self.displace_net(concat_vector2)
        return pc + dis_pc.transpose(2, 1)  # B, N, 3

class PolyPatch(nn.Module):
    def __init__(self, knn=64, fd=128, train_up_ratio=16):
        super(PolyPatch, self).__init__()
        self.knn = knn

        self.dgcnn_conv1 = nn.Sequential(nn.Conv2d(6, fd, kernel_size=1),
                                         nn.BatchNorm2d(fd),
                                         nn.LeakyReLU(negative_slope=0.2))
        self.dgcnn_conv2 = nn.Sequential(nn.Conv2d(fd, fd, kernel_size=1),
                                         nn.BatchNorm2d(fd),
                                         nn.LeakyReLU(negative_slope=0.2))
        self.dgcnn_conv3 = nn.Sequential(nn.Conv1d(fd + fd, fd, kernel_size=1),
                                         nn.BatchNorm1d(fd),
                                         nn.LeakyReLU(negative_slope=0.2))
        self.dgcnn_conv4 = nn.Sequential(nn.Conv2d(fd + fd, fd, kernel_size=1),
                                         nn.BatchNorm2d(fd),
                                         nn.LeakyReLU(negative_slope=0.2))
        self.dgcnn_conv5 = nn.Sequential(nn.Conv1d(fd + fd, fd, kernel_size=1),
                                         nn.BatchNorm1d(fd),
                                         nn.LeakyReLU(negative_slope=0.2))
        self.dgcnn_conv6 = nn.Sequential(nn.Conv2d(fd + fd, fd, kernel_size=1),
                                         nn.BatchNorm2d(fd),
                                         nn.LeakyReLU(negative_slope=0.2))
        self.dgcnn_conv7 = nn.Sequential(nn.Conv1d(fd + fd, fd, kernel_size=1),
                                         nn.BatchNorm1d(fd),
                                         nn.LeakyReLU(negative_slope=0.2))

        self.uv_2order_coefficient_conv = nn.Sequential(
            nn.Conv1d(fd * 9, 256, kernel_size=1), nn.BatchNorm1d(256), nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(256, 128, kernel_size=1), nn.BatchNorm1d(128), nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(128, 64, kernel_size=1), nn.BatchNorm1d(64), nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(64, 6 * 3, kernel_size=1))

        for m in self.uv_2order_coefficient_conv.modules():
            if isinstance(m, (torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Linear)):
                # torch.nn.init.xavier_uniform_(m.weight)
                m.weight.data = m.weight.data * 1e-5
                # print(m.weight.data)

        grid_size = train_up_ratio

        u = torch.from_numpy(np.arange(0, grid_size, dtype=float).reshape(grid_size, 1, 1)).repeat(1, grid_size, 1) / (
                grid_size - 1)
        v = torch.from_numpy(np.arange(0, grid_size, dtype=float).reshape(1, grid_size, 1)).repeat(grid_size, 1, 1) / (
                grid_size - 1)
        uv = torch.cat((u, v, torch.zeros_like(u)), dim=2).reshape(-1, 3)  # (grid_size*grid_size,3)

        uv = uv.unsqueeze(0).repeat(grid_size * grid_size, 1,
                                    1)  # (grid_size*grid_size,grid_size*grid_size,3)  ,  regard as (B,N,3)

        first_p_index = torch.arange(0, uv.size(1)).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 3)

        first_uv = torch.gather(uv, dim=1, index=first_p_index)

        grid = torch.cat((first_uv, uv), dim=1).cuda().float()  # (B,N+1,3)

        index = utils.furthest_point_sample(grid.contiguous(), train_up_ratio)
        uv = utils.gather_operation(grid.transpose(1, 2).contiguous(), index).transpose(1, 2)[:, :,
             0:2]  # (grid_size*grid_size,up_ratio,2)

        self.uv_set = (0.1 * (uv * 2 - 1)).cpu()

    def forward(self, x, up_ratio=16, poisson=True):
        # x:(B,3,N)

        batch_size = x.size(0)
        num_point = x.size(2)

        edge_feature = get_graph_feature(x, k=self.knn)  # (B,6,N,20)
        out1 = self.dgcnn_conv1(edge_feature)  # (B,128,N,20)
        out2 = self.dgcnn_conv2(out1)  # (B,128,N,20)
        net_max_1 = out2.max(dim=-1, keepdim=False)[0]  # (B,128,N)
        net_mean_1 = out2.mean(dim=-1, keepdim=False)  # (B,128,N)

        out3 = self.dgcnn_conv3(torch.cat((net_max_1, net_mean_1), 1))  # (B,128,N)

        edge_feature = get_graph_feature(out3, k=self.knn)  # (B,256,N,20)
        out4 = self.dgcnn_conv4(edge_feature)  # (B,128,N,20)

        net_max_2 = out4.max(dim=-1, keepdim=False)[0]  # (B,128,N)
        net_mean_2 = out4.mean(dim=-1, keepdim=False)  # (B,128,N)

        out5 = self.dgcnn_conv5(torch.cat((net_max_2, net_mean_2), 1))  # (B,128,N)

        edge_feature = get_graph_feature(out5, k=self.knn)  # (B,256,N,20)
        out6 = self.dgcnn_conv6(edge_feature)  # (B,128,N,20)

        net_max_3 = out6.max(dim=-1, keepdim=False)[0]  # (B,128,N)
        net_mean_3 = out6.mean(dim=-1, keepdim=False)  # (B,128,N)

        out7 = self.dgcnn_conv7(torch.cat((net_max_3, net_mean_3), dim=1))

        concat = torch.cat((net_max_1,  # 128
                            net_mean_1,  # 128
                            out3,  # 128
                            net_max_2,  # 128
                            net_mean_2,  # 128
                            out5,  # 128
                            net_max_3,  # 128
                            net_mean_3,  # 128
                            out7,  # 128
                            ), dim=1)  # (B,C,N)

        sel_uv_index = torch.randint(0, int(up_ratio * up_ratio), size=(int(batch_size * num_point), 1, 1)).repeat(1,
                                                                                                                   self.uv_set.size(
                                                                                                                       1),
                                                                                                                   2).to(
            concat)

        uv = torch.gather(self.uv_set.to(concat), dim=0, index=sel_uv_index.type(torch.int64)).reshape(batch_size,
                                                                                                       num_point, -1,
                                                                                                       2).to(concat)

        u = uv[:, :, :, 0:1]
        v = uv[:, :, :, 1:]  # (B,N,U,1)

        # [1, u, v , u*u, u*v, v*v]
        uv_vector = torch.concat((torch.ones_like(u), u, v, u * u, u * v, v * v), dim=-1)  # (B,N,U,6)

        # [0, 1, 0, 2*u, v, 0]      grad u
        uv_vector_grad_u = torch.concat(
            (torch.zeros_like(u), torch.ones_like(u), torch.zeros_like(u), 2 * u, v, torch.zeros_like(u)), dim=-1)

        # [0, 0, 1, 0, u, 2*v]      grad v
        uv_vector_grad_v = torch.concat(
            (torch.zeros_like(u), torch.zeros_like(u), torch.ones_like(u), torch.zeros_like(u), u, 2 * v), dim=-1)

        # [0, 0, 0, 2, 0, 0]        grad uu
        # uv_vector_grad_uu=torch.concat((torch.zeros_like(u),torch.zeros_like(u),torch.zeros_like(u),2*torch.ones_like(u),torch.zeros_like(u),torch.zeros_like(u)),dim=-1)
        # [0, 0, 0, 0, 1, 0]        grad uv
        # uv_vector_grad_uv=torch.concat((torch.zeros_like(u),torch.zeros_like(u),torch.zeros_like(u),torch.zeros_like(u),torch.ones_like(u),torch.zeros_like(u)),dim=-1)
        # [0,0,0,0,0,2]
        # uv_vector_grad_vv=torch.concat((torch.zeros_like(u),torch.zeros_like(u),torch.zeros_like(u),torch.zeros_like(u),torch.zeros_like(u),2*torch.ones_like(u)),dim=-1)

        coefficient = self.uv_2order_coefficient_conv(concat).transpose(2, 1).reshape(batch_size, num_point, 6, 3)
        # (B,6*3,N)->(B,N,6*3)->(B,N,6,3)

        xyz_offset = torch.matmul(uv_vector, coefficient)  # (B,N,U,6)@(B,N,6,3) -> (B,N,U,3)

        xyz = x.transpose(2, 1).unsqueeze(2) + xyz_offset  # (B,N,U,3)

        xyz_grad_u = torch.matmul(uv_vector_grad_u, coefficient)  # (B,N,U,3)
        xyz_grad_v = torch.matmul(uv_vector_grad_v, coefficient)

        normal = torch.cross(xyz_grad_u, xyz_grad_v)
        normal = F.normalize(normal, dim=-1)  # (B,N,U,3)

        return {'dense_xyz': xyz.reshape(batch_size, -1, 3),  # (B,N,U,3)
                'dense_normal': normal.reshape(batch_size, -1, 3),
                # (B,N,U,3)            #'sparse_normal':normal_sparse,
                }

class MP(nn.Module):
    def __init__(self):
        super(MP, self).__init__()
        self.vertice_disp = DisplaceNet2v2(fd=128, num_knn=16, dp_scale=1.)
        self.manifole_mp = PolyPatch(knn=8, train_up_ratio=16)
    def forward(self, x):
        # x: B, N, 3
        return self.manifole_mp(self.vertice_disp(x).transpose(2, 1))

class NO(nn.Module):
    def __init__(self, fd=128):
        super(NO, self).__init__()

        self.lifting = Lifting(dim=fd * 2)

        self.attn_1 = galerkinAtten(2 * fd, 8)
        self.attn_2 = galerkinAtten(2 * fd, 8)
        # self.mapping = nn.Sequential(
        #     galerkinAtten(2 * fd, 8),
        #     galerkinAtten(2 * fd, 8)
        # )
        self.to_outputs = nn.Sequential(
            nn.Conv1d(fd * 2, 128, 1),
            nn.GELU(),
            nn.Conv1d(128, 3, 1)
        )
        self.displace = DisplaceNet2v2(fd=64, num_knn=8, dp_scale=1.)

    def projecting(self, x):
        # B, d, NU
        return self.displace((self.to_outputs(x).transpose(2, 1)))  # B, N, 3

    def forward(self, x):
        # x:(B,3,N)
        B, _, N = x.shape
        pc = x
        x = self.lifting(x.transpose(2, 1)).transpose(2, 1)
        x = self.attn_1(x)
        x = self.attn_2(x)
        # x = self.mapping(x)
        return self.projecting(x) + pc.transpose(2, 1)  # B, N, 3

class PUNO(nn.Module):
    def __init__(self, fd=128):
        super(PUNO, self).__init__()
        self.MP = MP()
        self.NO = NO()
    def forward(self, x):
        return self.NO(self.MP(x)['dense_xyz'].transpose(2, 1))