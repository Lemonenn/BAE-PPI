import dgl
import numpy as np
import copy
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

from dgl.nn.pytorch import GraphConv, GINConv, HeteroGraphConv

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.autograd.set_detect_anomaly(True)


class GIN(torch.nn.Module):
    def __init__(self,  param):
        super(GIN, self).__init__()

        self.num_layers = param['ppi_num_layers']
        self.dropout = nn.Dropout(param['dropout_ratio'])
        self.layers = nn.ModuleList()
        
        self.layers.append(GINConv(nn.Sequential(nn.Linear(param['prot_hidden_dim'], param['ppi_hidden_dim']), 
                                                 nn.ReLU(), 
                                                 nn.Linear(param['ppi_hidden_dim'], param['ppi_hidden_dim']), 
                                                 nn.ReLU(), 
                                                 nn.BatchNorm1d(param['ppi_hidden_dim'])), 
                                                 aggregator_type='sum', 
                                                 learn_eps=True))

        for i in range(self.num_layers - 1):
            self.layers.append(GINConv(nn.Sequential(nn.Linear(param['ppi_hidden_dim'], param['ppi_hidden_dim']), 
                                                     nn.ReLU(), 
                                                     nn.Linear(param['ppi_hidden_dim'], param['ppi_hidden_dim']), 
                                                     nn.ReLU(), 
                                                     nn.BatchNorm1d(param['ppi_hidden_dim'])), 
                                                     aggregator_type='sum', 
                                                     learn_eps=True))

        self.linear = nn.Linear(param['ppi_hidden_dim'], param['ppi_hidden_dim'])
        self.fc = nn.Linear(param['ppi_hidden_dim'], param['output_dim'])

    def forward(self, g, x, ppi_list, idx):

        for l, layer in enumerate(self.layers):
            x = layer(g, x)
            x = self.dropout(x)

        x = F.dropout(F.relu(self.linear(x)), p=0.5, training=self.training)

        node_id = np.array(ppi_list)[idx]
        x1 = x[node_id[:, 0]]
        x2 = x[node_id[:, 1]]

        x = self.fc(torch.mul(x1, x2))
        
        return x


class HGNN(nn.Module):
    def __init__(self, param):
        super(HGNN, self).__init__()
        self.num_layers = param['ppi_num_layers']
        self.dropout = nn.Dropout(param['dropout_ratio'])
        self.hidden_dim = param['ppi_hidden_dim']
        self.output_dim = param['output_dim']
        self.input_dim = param['prot_hidden_dim']  # 和 GIN 保持一致

        self.relation_names = ['reaction', 'binding', 'ptmod', 'activation', 'inhibition', 'catalysis', 'expression', 'unknown']
        self.num_relations = len(self.relation_names)

        self.layers = nn.ModuleList()
        for i in range(self.num_layers):
            in_dim = self.input_dim if i == 0 else self.hidden_dim
            conv = HeteroGraphConv({
                rel: GraphConv(in_dim, self.hidden_dim)
                for rel in self.relation_names
            }, aggregate='sum')
            self.layers.append(conv)

        self.linear = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc = nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, g, x, ppi_list, idx):
        unknown_ratio = 0.15
        # print("Before randomize:", g.num_nodes('protein'), g.num_edges())
        if self.training and unknown_ratio > 0:
            g = self._randomize_edges(g, unknown_ratio)
        #     print("After randomize:", g.num_nodes('protein'), g.num_edges())
        # print("Before encoding:", x.shape)

        # 输入是异构图g，初始x为 {'protein': [N, dim]}
        h = {'protein': x}

        for l, layer in enumerate(self.layers):
            h = layer(g, h)
            h = {k: self.dropout(F.relu(v)) for k, v in h.items()}

        x = F.dropout(F.relu(self.linear(h['protein'])), p=0.5, training=self.training)

        node_id = torch.tensor(ppi_list, device=x.device)[idx]
        x1 = x[node_id[:, 0]]
        x2 = x[node_id[:, 1]]
        x = self.fc(torch.mul(x1, x2))  

        return x

    def _randomize_edges(self, g, ratio):
        """
        返回一个新图，其中部分边类型被替换成 'unknown'
        """
        # 复制图结构（浅复制节点 & 边数据）
        g_aug = copy.deepcopy(g)

        for rel in self.relation_names:
            if rel == 'unknown':
                continue
            # 获取该关系下的所有边
            num_edges = g_aug.num_edges(rel)
            if num_edges == 0:
                continue

            # 随机选出部分边
            num_unknown = int(num_edges * ratio)
            edges_to_change = random.sample(range(num_edges), num_unknown)

            # 将这些边移动到 'unknown' 关系
            src, dst = g_aug.edges(form='uv', etype=rel)
            src_unknown = src[edges_to_change]
            dst_unknown = dst[edges_to_change]

            # 删除这些边的原关系
            mask = torch.ones(num_edges, dtype=torch.bool)
            mask[edges_to_change] = False
            g_aug.remove_edges(edges_to_change, etype=rel)

            # 添加到 unknown 关系
            g_aug.add_edges(src_unknown, dst_unknown, etype='unknown')

        return g_aug



class RBFExpansion(nn.Module):
    """将距离展开为 RBF 特征"""
    def __init__(self, K=16, cutoff=20.0):
        super().__init__()
        self.K = K
        self.centers = torch.linspace(0, cutoff, K)
        self.width = (self.centers[1] - self.centers[0]).item()

    def forward(self, dist):
        # dist: [E]
        return torch.exp(-((dist.unsqueeze(-1) - self.centers.to(dist.device)) ** 2) / self.width ** 2)

class EdgeBias(nn.Module):
    def __init__(self, num_edge_types=3, num_rbf=16):
        super().__init__()
        self.rbf = RBFExpansion(num_rbf)
        self.edge_emb = nn.Embedding(num_edge_types, num_rbf)
        self.mlp = nn.Linear(num_rbf, 1)

    def forward(self, dist, etype_id):
        dist_feat = self.rbf(dist)       # [E, num_rbf]
        type_feat = self.edge_emb(etype_id)  # [E, num_rbf]
        return self.mlp(dist_feat + type_feat)  # [E,1]

class AttentionPooling(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.query = nn.Parameter(torch.randn(dim))

    def forward(self, h):  # h: [N, D]
        attn = torch.matmul(h, self.query)  # [N]
        attn = torch.softmax(attn, dim=0)
        return torch.sum(h * attn.unsqueeze(-1), dim=0)  # [D]

#-----------------------------------
# Multi-Head Graph Transformer
#-----------------------------------
class MultiHeadGraphTransformerLayer(nn.Module):
    """
    多头图 Transformer 层。
    包括多头自注意力（MHSA）、残差连接、LayerNorm 和前馈网络（FFN）。
    """
    def __init__(self, dim, num_heads=8, dropout=0.1, ffn_scale=4):
        super().__init__()
        assert dim % num_heads == 0, "dim 必须能被 num_heads 整除"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        # Q K V 投影
        self.q_lin = nn.Linear(dim, dim, bias=False)
        self.k_lin = nn.Linear(dim, dim, bias=False)
        self.v_lin = nn.Linear(dim, dim, bias=False)

        # 注意力输出投影
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(dropout)

        # 残差 + 归一化
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        # FFN
        hidden = dim * ffn_scale
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def _mhsa(self, g: dgl.DGLHeteroGraph, h: torch.Tensor, edge_bias: dict):
        """
        多头自注意力（Multi-Head Self-Attention）
        g: 异构图
        h: [N, dim]
        edge_bias: {etype: [E,1]} —— 按边的标量偏置（由距离+类型得到）
        return: [N, dim]
        """
        with g.local_scope():
            N = h.size(0)
            # 投影并拆多头
            Q = self.q_lin(h).view(N, self.num_heads, self.head_dim)  # [N, H, Dh]
            K = self.k_lin(h).view(N, self.num_heads, self.head_dim)
            V = self.v_lin(h).view(N, self.num_heads, self.head_dim)

            g.ndata['Q'] = Q
            g.ndata['K'] = K
            g.ndata['V'] = V

            # 将预先计算好的 edge_bias 写入各边类型，广播到每个 head
            for i, etype in enumerate(g.canonical_etypes):
                # [E,1] -> [E,H]
                # 这里我们使用 .expand() 来确保是广播操作，而不是原地修改
                # g.edges[etype].data['attn_bias'] = edge_bias[etype].expand(-1, self.num_heads)
                g.edges[etype].data['attn_bias'] = edge_bias[etype]

            # 定义消息与聚合
            def message_func(edges):
                # 点积注意力（逐 head）
                # src K, dst Q: [E, H, Dh]
                score = (edges.src['K'] * edges.dst['Q']).sum(-1, keepdim=True)  # [E, H, 1]
                # 添加偏置，注意是加法，这不会引起原地修改
                score = score / (self.head_dim ** 0.5) + edges.data['attn_bias'].unsqueeze(-1)
                
                return {'score': score, 'V': edges.src['V']}

            def reduce_func(nodes):
                # mailbox['score']: [N, deg, H, 1]
                attn = F.softmax(nodes.mailbox['score'], dim=1)
                # 使用 dropout
                attn = self.attn_drop(attn)
                # 加权求和
                out = (attn * nodes.mailbox['V']).sum(dim=1)  # [N, H, Dh]
                return {'h_out': out}

            # 多关系安全聚合
            g.multi_update_all(
                {etype: (message_func, reduce_func) for etype in g.etypes},
                cross_reducer='sum'
            )

            # 拼头并线性投影
            h_out = g.ndata['h_out'].contiguous().view(N, self.num_heads * self.head_dim)  # [N, dim]
            h_out = self.proj(h_out)
            return h_out

    def _mhca(self, g: dgl.DGLHeteroGraph, h_q: torch.Tensor, h_kv: torch.Tensor, edge_bias: dict):
        """
        多头交叉注意力（Multi-Head Cross-Attention）
        g: 异构图
        h_q: [Nq, dim]
        h_kv: [Nk, dim] —— K/V 来自编码器，Q 来自解码器
        edge_bias: {etype: [E,1]} —— 按边的标量偏置（由距离+类型得到）
        return: [N, dim]
        """
        with g.local_scope():
            Nq = h_q.size(0)
            Nk = h_kv.size(0)

            # 线性映射并拆多头
            Q = self.q_lin(h_q).view(Nq, self.num_heads, self.head_dim)   # [N, H, Dh]
            K = self.k_lin(h_kv).view(Nk, self.num_heads, self.head_dim)
            V = self.v_lin(h_kv).view(Nk, self.num_heads, self.head_dim)

            # 写入图节点特征：dst 用 Q（decoder），src 用 K/V（encoder）
            # 这里默认 decoder 和 encoder 节点集相同（同一个批图）；若不同，需要在构图时提供跨图边
            g.ndata['Q_cross'] = Q
            g.ndata['K_cross'] = K
            g.ndata['V_cross'] = V

            # 写入每种关系的边偏置（支持 [E,1] 或 [E,H]）
            for etype in g.canonical_etypes:
                eb = edge_bias[etype]                      # [E,1] 或 [E,H]
                if eb.dim() == 2 and eb.size(-1) == 1:
                    eb = eb.expand(-1, self.num_heads)     # -> [E,H]
                g.edges[etype].data['attn_bias'] = eb      # [E,H]

            # 消息与聚合：K/V 从 src 来，Q 从 dst 来
            def message_func(edges):
                # src K, dst Q: [E, H, Dh]
                score = (edges.src['K_cross'] * edges.dst['Q_cross']).sum(-1, keepdim=True)  # [E,H,1]
                score = score / (self.head_dim ** 0.5) + edges.data['attn_bias'].unsqueeze(-1)  # +bias
                return {'score': score, 'Vmsg': edges.src['V_cross']}

            def reduce_func(nodes):
                # score: [N, deg, H, 1]
                attn = F.softmax(nodes.mailbox['score'], dim=1)
                attn = self.attn_drop(attn)
                out = (attn * nodes.mailbox['Vmsg']).sum(dim=1)   # [N, H, Dh]
                return {'h_out': out}

            g.multi_update_all(
                {etype: (message_func, reduce_func) for etype in g.etypes},
                cross_reducer='sum'
            )

            h_out = g.ndata['h_out'].contiguous().view(Nq, self.num_heads * self.head_dim)  # [N, D]
            h_out = self.proj(h_out)
            return h_out

    def forward(self, g: dgl.DGLHeteroGraph, h: torch.Tensor, edge_bias: dict):
        """Encoder Block"""
        # MHSA + 残差 + LN
        attn_out = self._mhsa(g, h, edge_bias)
        # h = h + attn_out 是正确的非原地操作
        h = self.norm1(h + attn_out)

        # FFN + 残差 + LN
        ffn_out = self.ffn(h)
        # h = h + ffn_out 也是正确的非原地操作
        h = self.norm2(h + ffn_out)
        return h

class GraphTransformer_Decoder(nn.Module):
    """
    Graph Transformer 解码器。
    用于将量化后的隐向量解码回原始节点特征。
    """
    def __init__(self, param):
        super().__init__()
        self.num_layers = param['prot_num_layers']
        self.dim = param['prot_hidden_dim']

        # 边偏置模块
        self.edge_bias_module = EdgeBias(num_edge_types=4, num_rbf=16)
        # 用于 mask 节点
        self.mask_token = nn.Parameter(torch.randn(1, self.dim))

        # 堆叠 Transformer blocks 自注意力
        self.layers = nn.ModuleList([
            MultiHeadGraphTransformerLayer(
                dim=self.dim,
                num_heads=4,
                dropout=param.get('dropout_ratio', 0.1),
                ffn_scale=4
            )
            for _ in range(self.num_layers)
        ])

        self.cross_norm = nn.ModuleList([
            nn.LayerNorm(self.dim) for _ in range(self.num_layers)
        ])

        # 额外的 FC + BN + Dropout
        self.post_fc = nn.ModuleList([
            nn.Linear(self.dim, self.dim) for _ in range(self.num_layers)
        ])
        self.post_norm = nn.ModuleList([
            nn.BatchNorm1d(self.dim) for _ in range(self.num_layers)
        ])
        self.dropout = nn.Dropout(param.get('dropout_ratio', 0.1))

        # 输出层
        self.out_proj = nn.Linear(self.dim, param['input_dim'])
        

    @torch.no_grad()
    def _build_edge_bias(self, g: dgl.DGLHeteroGraph):
        """
        与编码器中相同，用于生成注意力偏置。
        """
        edge_bias = {}
        for i, etype in enumerate(g.canonical_etypes):
            dist_feat = g.edges[etype].data['dist'].float().to(g.device).view(-1)  # [E]
            etype_id = torch.full((dist_feat.size(0),), i, device=g.device, dtype=torch.long)
            edge_bias[etype] = self.edge_bias_module(dist_feat, etype_id)  # [E,1]
        return edge_bias

    #     return x_recon
    def decoding(self, g: dgl.DGLHeteroGraph, encoder_out: torch.Tensor, visible_idx, mask):
        N = g.num_nodes()
        device = encoder_out.device

        # ====== 初始化输入 ======
        h = torch.zeros(N, self.dim, device=device)
        h[visible_idx] = encoder_out                     # visible 节点
        h[mask] = self.mask_token.expand(mask.sum(), -1) # mask 节点 → [MASK]
        h = h.clone()

        # ====== Transformer Decoder ======
        edge_bias = self._build_edge_bias(g)
        # 核心修改：在进入 cross-attention 之前对 h_kv 进行填充
        N_total = g.number_of_nodes()
        dim = encoder_out.size(-1)
        
        # 创建一个全零张量作为占位符
        h_kv_padded = torch.zeros((N, dim), device=device, dtype=encoder_out.dtype)
        # 将 encoder 的输出 h_kv 填充到正确的位置
        h_kv_padded[visible_idx] = encoder_out
        for l, layer in enumerate(self.layers):
            # MHSA + 残差 + LN
            attn_out = layer._mhsa(g, h, edge_bias)
            # h = h + attn_out 是正确的非原地操作
            h = self.cross_norm[l](h + attn_out)

            # Cross-Attention (Q = h, K/V = encoder_out)
            cross_attn = layer._mhca(g, h, h_kv_padded, edge_bias)
            h = self.cross_norm[l](h + cross_attn)

            h = self.post_fc[l](h)
            h = self.post_norm[l](F.relu(h))
            if l != self.num_layers - 1:
                h = self.dropout(h)

        # ====== 重建 ======
        x_recon = self.out_proj(h)
        return x_recon


# # -----------------------------------
# Encoder
# -----------------------------------
class GraphTransformer_Encoder(nn.Module):
    """
    兼容你原先的接口：
      - __init__(param, data_loader)
      - forward(vq_layer) -> [num_proteins, 2*hidden]
    需要：
      g.ndata['x']   : 节点原始特征（如 one-hot/ESM/AA property 等），维度 = param['input_dim']
      g.ndata['seq_id'] (long) : 0..L-1；若缺失会自动生成
      每种边类型都有 g.edges[etype].data['dist'] : [E] 的距离（float）
    """
    def __init__(self, param, data_loader):
        super().__init__()
        self.data_loader = data_loader
        self.num_layers = param['prot_num_layers']
        self.dim = param['prot_hidden_dim']

        # 输入线性映射
        self.in_proj = nn.Linear(param['input_dim'], self.dim)
        self.pooling = AttentionPooling(dim=self.dim)

        # 边偏置（距离 RBF + 边类型）
        self.edge_bias_module = EdgeBias(num_edge_types=4, num_rbf=16)

        # Transformer blocks
        self.layers = nn.ModuleList([
            MultiHeadGraphTransformerLayer(
                dim=self.dim,
                num_heads=4,
                dropout= param.get('dropout_ratio', 0.1),
                ffn_scale=4
            )
            for _ in range(self.num_layers)
        ])

        # 额外的 FC + BN + Dropout （模仿 GCN）
        self.post_fc = nn.ModuleList([
            nn.Linear(self.dim, self.dim) for _ in range(self.num_layers)
        ])
        self.post_norm = nn.ModuleList([
            nn.BatchNorm1d(self.dim) for _ in range(self.num_layers)
        ])
        self.dropout = nn.Dropout(param.get('dropout_ratio', 0.1))

    @torch.no_grad()
    def _build_edge_bias(self, g: dgl.DGLHeteroGraph):
        edge_bias = {}

        for i, etype in enumerate(g.canonical_etypes):
            dist_feat = g.edges[etype].data['dist'].float().to(g.device).view(-1)   # [E]
            etype_id  = torch.full((dist_feat.size(0),), i, device=g.device, dtype=torch.long)
            edge_bias[etype] = self.edge_bias_module(dist_feat, etype_id)          # [E,1]
        return edge_bias

    def encoding(self, g: dgl.DGLHeteroGraph):
        # 节点输入
        x = g.ndata['x'].float()
        h = self.in_proj(x)

        # 边偏置
        edge_bias = self._build_edge_bias(g)

        # 堆叠 Transformer 层
        # for layer in self.layers:
        #     h = layer(g, h, edge_bias)
        for l, layer in enumerate(self.layers):
            h = layer(g, h, edge_bias)           # Transformer
            h = self.post_fc[l](h)               # 额外线性
            h = self.post_norm[l](F.relu(h))     # BN + ReLU
            if l != self.num_layers - 1:         # 最后一层不dropout
                h = self.dropout(h)

        return h  # [N, dim]

    def encoding_mask(self, g: dgl.DGLHeteroGraph):
        N = g.num_nodes()
        device = g.device

        # mask_ratio = 0.15
        mask_ratio = 0.0000001

        # ====== 随机采样 mask 节点 ======
        num_mask = int(N * mask_ratio)
        perm = torch.randperm(N, device=device)
        mask_idx = perm[:num_mask]
        visible_idx = perm[num_mask:]

        mask = torch.zeros(N, dtype=torch.bool, device=device)
        mask[mask_idx] = True
        g.ndata['mask'] = mask  # 存起来给 decoder 用

        # ====== 输入特征 ======
        x = g.ndata['x'].float()
        x_visible = x[visible_idx]
        h = self.in_proj(x_visible)

        # ====== 子图 ======
        g_visible = dgl.node_subgraph(g, visible_idx)

        # ====== transformer encoder ======
        edge_bias = self._build_edge_bias(g_visible)
        for l, layer in enumerate(self.layers):
            h = layer(g_visible, h, edge_bias)
            h = self.post_fc[l](h)
            h = self.post_norm[l](F.relu(h))
            if l != self.num_layers - 1:
                h = self.dropout(h)

        return h, visible_idx, mask


    def forward(self):
        prot_embed_list = []
        device = next(self.parameters()).device

        for batch_graph in self.data_loader:
            batch_graph = batch_graph.to(device)

            # 节点编码
            h = self.encoding(batch_graph)             # [N_total, dim]

            # VQ
            # z, _, _ = vq_layer(h)                      # [N_total, dim_z]
            # batch_graph.ndata['h'] = torch.cat([h, h], dim=-1)
            batch_graph.ndata['h'] = h

            # # 按子图取平均
            # for g in dgl.unbatch(batch_graph):
            #     prot_embed = g.ndata['h'].mean(dim=0)  # [2*dim]
            #     prot_embed_list.append(prot_embed.detach().cpu())
            for g in dgl.unbatch(batch_graph):
                h = g.ndata['h']
                prot_embed = self.pooling(h)   # attention pooling
                prot_embed_list.append(prot_embed.detach().cpu())

        return torch.stack(prot_embed_list, dim=0)     # [num_proteins, 2*dim]


class CodeBook(nn.Module):
    def __init__(self, param, data_loader):
        super(CodeBook, self).__init__()

        self.param = param

        self.Protein_Encoder = GraphTransformer_Encoder(param, data_loader)
        self.Protein_Decoder = GraphTransformer_Decoder(param)
      
    def forward(self, batch_graph):
        # --- 编码 ---
        z, visible_idx, mask = self.Protein_Encoder.encoding_mask(batch_graph)   # [N, hidden_dim]

        # --- 解码 ---
        x_recon = self.Protein_Decoder.decoding(batch_graph, z, visible_idx, mask)

        # --- 重建损失 ---
        recon_loss = F.mse_loss(x_recon, batch_graph.ndata['x'])

        # MSE Loss
        # mask_loss = F.mse_loss(x_recon[mask], g.ndata['x'][mask])

        # Cosine similarity loss
        x_norm = F.normalize(x_recon[mask], dim=-1, eps=1e-12)
        y_norm = F.normalize(batch_graph.ndata['x'][mask], dim=-1, eps=1e-12)
        mask_loss = (1 - (x_norm * y_norm).sum(dim=-1)).pow(self.param['sce_scale'])

        return z, x_recon, recon_loss, mask_loss.mean()
