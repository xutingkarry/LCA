import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import torch
from torch import nn
import torch_geometric.nn as gnn
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GCNConv, global_max_pool as gmp,GATConv,GINConv
from torch_geometric.nn import GCNConv, global_max_pool as gmp,GATConv,GINConv,global_mean_pool,global_add_pool,global_max_pool
from torch_geometric.nn import GCNConv, GCN2Conv, GATConv, global_max_pool as gmp, global_add_pool as gap, global_mean_pool as gep, global_sort_pool
from einops.layers.torch import Rearrange, Reduce
# -*- coding: utf-8 -*-

from .layers2coss_gCN_GRAPH import TransformerEncoderLayer
from einops import repeat
#这个文件用于将蛋白质序列分割




class GraphTransformerEncoder(nn.TransformerEncoder):
    def forward(self, x, edge_index, complete_edge_index,
                subgraph_node_index=None, subgraph_edge_index=None,
                subgraph_edge_attr=None, subgraph_indicator_index=None, edge_attr=None, degree=None,
                ptr=None, return_attn=False):
        #
        output = x

        for mod in self.layers:
            output = mod(output, edge_index, complete_edge_index,
                         edge_attr=edge_attr, degree=degree,
                         subgraph_node_index=subgraph_node_index,
                         subgraph_edge_index=subgraph_edge_index,
                         subgraph_indicator_index=subgraph_indicator_index,
                         subgraph_edge_attr=subgraph_edge_attr,
                         ptr=ptr,
                         return_attn=return_attn
                         )
        if self.norm is not None:
            output = self.norm(output)
        return output


class GraphTransformer(nn.Module):
    def __init__(self, in_size=78, in_size_prot=33,num_class=2, d_model=128, num_heads=8,num_features_xc=92,
                 dim_feedforward=512, dropout=0.0, num_layers=1,
                 batch_norm=True, abs_pe=False, abs_pe_dim=0,
                 n_output=1, n_filters=32, embed_dim=128, num_features_xt=25, output_dim=128,
                 gnn_type="gcn", se="gnn", use_edge_attr=False, num_edge_features=4,
                 in_embed=False, edge_embed=True, use_global_pool=True, max_seq_len=None,
                 global_pool='mean', **kwargs):
        super().__init__()

        # compound:gcn+Transformer
        # protein:cnn block+Transformer

        self.abs_pe = abs_pe
        self.abs_pe_dim = abs_pe_dim
        self.embedding= nn.Linear(in_features=in_size,out_features=d_model,bias=False)#（78，128，false）


        self.use_edge_attr = use_edge_attr#边的属性,false
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

        kwargs['edge_dim'] = None
        self.gnn_type = gnn_type#graph
        self.se = se#gnn
        #Transformer
        encoder_layer = TransformerEncoderLayer(
            d_model, num_heads, dim_feedforward, dropout, batch_norm=batch_norm,
            gnn_type=gnn_type, se=se, **kwargs)#(128,8,512,0.0,true,graph,gnn,none)
        self.encoder = GraphTransformerEncoder(encoder_layer, num_layers)#(encoder_layer,4)
        self.global_pool = global_pool#mean
        if global_pool == 'mean':
            self.pooling = gnn.global_mean_pool
        elif global_pool == 'add':
            self.pooling = gnn.global_add_pool
        elif global_pool == 'cls':
            self.cls_token = nn.Parameter(torch.randn(1, d_model))
            self.pooling = None
        self.use_global_pool = use_global_pool#true

        self.max_seq_len = max_seq_len#none
        if max_seq_len is None:
            self.classifier = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.ReLU(True),
                nn.Linear(d_model, num_class)
            )#((128,128),relu,(128,2))
        else:
            self.classifier = nn.ModuleList()
            for i in range(max_seq_len):
                self.classifier.append(nn.Linear(d_model, num_class))

        self.n_output = n_output#1




        #药物的1489
        # self.conlstm = nn.LSTM(256, 256, 1, dropout=0.2, bidirectional=True)
        self.com_fc1 = nn.Linear(2 * output_dim, 1024)
        self.com_bn1 = nn.BatchNorm1d(1024)
        self.com_fc2 = nn.Linear(1024, 512)
        self.com_bn2 = nn.BatchNorm1d(512)
        self.out = nn.Linear(512, self.n_output)
        #对蛋白质序列：将蛋白质序列分割成两个部分，再使用卷积和交叉注意力机制提取特征

        self.p_embed = nn.Embedding(num_features_xt + 1, embed_dim)
        # target：512，1000 ，512，1000，128
        self.p_cnn1=nn.Conv1d(500,128,kernel_size=3,padding=1,stride=1)
        self.cnn_bn1=nn.BatchNorm1d(128)
        self.p_cnn2=nn.Conv1d(128,64,kernel_size=3,padding=1,stride=1)
        self.cnn_bn2 = nn.BatchNorm1d(64)
        self.p_cnn3 = nn.Conv1d(64, 16, kernel_size=8, padding=0, stride=1)
        self.cnn_bn3 = nn.BatchNorm1d(16)
        self.target_fc1=nn.Sequential(
            nn.Linear(121,128),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        self.maxpool=nn.MaxPool1d(kernel_size=3,padding=1,stride=1)
        self.avgpool=nn.AvgPool1d(kernel_size=3,padding=1,stride=1)
        self.CAA=CAA_Block(128,8,1)
        self.CAA2=CAA_Block2()
        self.p_fc4 = nn.Linear(16 * 122, output_dim)
        self.p_bn4 = nn.BatchNorm1d(output_dim)









    def forward(self, data,return_attn=False):
        # get graph input
        x, edge_index, batch, edge_attr = data.x, data.edge_index, data.batch, data.edge_attr


        target=data.target


        node_depth = data.node_depth if hasattr(data, "node_depth") else None

        if self.se == "khopgnn":
            subgraph_node_index = data.subgraph_node_idx
            subgraph_edge_index = data.subgraph_edge_index
            subgraph_indicator_index = data.subgraph_indicator
            subgraph_edge_attr = data.subgraph_edge_attr if hasattr(data, "subgraph_edge_attr") \
                else None
        else:
            subgraph_node_index = None
            subgraph_edge_index = None
            subgraph_indicator_index = None
            subgraph_edge_attr = None

        complete_edge_index = data.complete_edge_index if hasattr(data, 'complete_edge_index') else None
        abs_pe = data.abs_pe if hasattr(data, 'abs_pe') else None
        degree = data.degree if hasattr(data, 'degree') else None

        # x = x.long()
        output=x
        output=self.embedding(output)
        #

        if self.abs_pe and abs_pe is not None:
            abs_pe = self.embedding_abs_pe(abs_pe)
            output = output + abs_pe
        if self.use_edge_attr and edge_attr is not None:
            edge_attr = self.embedding_edge(edge_attr)
            if subgraph_edge_attr is not None:
                subgraph_edge_attr = self.embedding_edge(subgraph_edge_attr)
        else:
            edge_attr = None
            subgraph_edge_attr = None

        if self.global_pool == 'cls' and self.use_global_pool:
            bsz = len(data.ptr) - 1
            if complete_edge_index is not None:
                new_index = torch.vstack((torch.arange(data.num_nodes).to(data.batch), data.batch + data.num_nodes))
                new_index2 = torch.vstack((new_index[1], new_index[0]))
                idx_tmp = torch.arange(data.num_nodes, data.num_nodes + bsz).to(data.batch)
                new_index3 = torch.vstack((idx_tmp, idx_tmp))
                complete_edge_index = torch.cat((
                    complete_edge_index, new_index, new_index2, new_index3), dim=-1)
            if subgraph_node_index is not None:
                idx_tmp = torch.arange(data.num_nodes, data.num_nodes + bsz).to(data.batch)
                subgraph_node_index = torch.hstack((subgraph_node_index, idx_tmp))
                subgraph_indicator_index = torch.hstack((subgraph_indicator_index, idx_tmp))
            degree = None
            cls_tokens = repeat(self.cls_token, '() d -> b d', b=bsz)
            output = torch.cat((output, cls_tokens))

        output = self.encoder(
            output,
            edge_index,
            complete_edge_index,
            edge_attr=edge_attr,
            degree=degree,
            subgraph_node_index=subgraph_node_index,
            subgraph_edge_index=subgraph_edge_index,
            subgraph_indicator_index=subgraph_indicator_index,
            subgraph_edge_attr=subgraph_edge_attr,
            ptr=data.ptr,
            return_attn=return_attn
        )
        # readout step
        if self.use_global_pool:
            if self.global_pool == 'cls':
                output = output[-bsz:]
            else:
                output = self.pooling(output, data.batch)
        if self.max_seq_len is not None:
            pred_list = []
            for i in range(self.max_seq_len):
                pred_list.append(self.classifier[i](output))
            return pred_list
        # return self.classifier(output)

       #药物的1489





        #对蛋白质序列进行分割，最后再使用深层次的CNN

        p_embed = self.p_embed(target)

        # target: 512,1000--> 512,1000,128

        # 第一个矩阵
        target_1 = p_embed[:, :500, :]

        # 第二个矩阵
        target_2 = p_embed[:, 500:1000, :]

        #每个矩阵大小为512*250*128
        #1.对每个矩阵进行cnn
        target_1=self.relu(self.cnn_bn1(self.p_cnn1(target_1)))
        target_1=self.relu(self.cnn_bn2(self.p_cnn2(target_1)))
        target_1=self.relu(self.cnn_bn3(self.p_cnn3(target_1)))
        target_1=self.target_fc1(target_1)
        query1=self.relu(self.maxpool(target_1)+self.avgpool(target_1))

        target_2 = self.relu(self.cnn_bn1(self.p_cnn1(target_2)))
        target_2 = self.relu(self.cnn_bn2(self.p_cnn2(target_2)))
        target_2 = self.relu(self.cnn_bn3(self.p_cnn3(target_2)))
        target_2=self.target_fc1(target_2)
        query2 = self.relu(self.maxpool(target_2) + self.avgpool(target_2))

        caa1=self.CAA(query2,target_1)
        caa2=self.CAA(query1,target_2)
        protein=self.CAA2(query1,caa1,query2,caa2)
        protein=protein.view(-1,16*122)
        protein=self.dropout(self.relu(self.p_bn4(self.p_fc4(protein))))

        con=torch.cat((output,protein),1)



        xc = self.com_fc1(con)
        xc = self.relu(xc)
        xc = self.com_bn1(xc)
        xc = self.dropout(xc)

        xc = self.com_fc2(xc)
        xc = self.relu(xc)
        xc = self.com_bn2(xc)
        xc = self.dropout(xc)

        out = self.out(xc)

        return out





class CAA_Block2(nn.Module):
    def __init__(self):
        super(CAA_Block2, self).__init__()

        self.mlp=nn.Sequential(
            nn.Conv1d(32, 16, kernel_size=7, stride=1),
            nn.BatchNorm1d(16),
            nn.ReLU()
        )
    def forward(self, query1,target1,query2,target2):
        protein1=target1+query1
        protein2=target2+query2
        protein=torch.cat((protein1,protein2),dim=1)
        protein=self.mlp(protein)
        return protein



class CAA_Block(nn.Module):
    def __init__(self, d_model, n_head, nlayers, dropout=0.1, activation="relu"):
        super(CAA_Block, self).__init__()
        self.encoder = nn.ModuleList([CAA_Block_Layer(d_model, n_head, dropout, activation)
                                      for _ in range(nlayers)])
    def forward(self, q,kv, atten_mask=None):
        for layer in self.encoder:
            x = layer.forward(q,kv, atten_mask)
        return x


class CAA_Block_Layer(nn.Module):
    def __init__(self, d_model, n_head, dropout=0.1, activation="relu"):
        super().__init__()
        self.attn = CAttention(h=n_head, d_model=d_model, dropout=dropout)
        self.dropout = nn.Dropout(dropout)


        if activation == "relu":
            self.activation = F.relu
        if activation == "gelu":
            self.activation = F.gelu


    def forward(self, q,kv, atten_mask):

        # add & norm 1
        attn = self.dropout(self.attn(q, kv, kv, attn_mask=atten_mask))
        return attn
class CAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(CAttention, self).__init__()
        assert d_model % h == 0

        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h

        self.linear_layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        self.output_linear = nn.Linear(d_model, d_model)
        self.attention = ScaledDotProductAttention()

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, attn_mask=None):
        batch_size = query.size(0)
        query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linear_layers, (query, key, value))]
        x, atten = self.attention(query, key, value, attn_mask=attn_mask, dropout=self.dropout)#这个X就是公式中的Z，atten就是softmax中的那一坨内积

        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)#view函数表示要重新定义矩阵的形状。
        return self.output_linear(x)




class ScaledDotProductAttention(nn.Module):#Attention(Q,K,V) = softmax(Q*Kt/sqrt(dk)) *V
    def forward(self, query, key, value, attn_mask=None, dropout=None):
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask == 0, -1e9)  # 保留位置为0的值，其他位置填充极小的数
        p_attn = F.softmax(scores, dim=-1)
        if dropout is not None:
            p_attn = dropout(p_attn)
        return torch.matmul(p_attn, value), p_attn  # (batch, n_head, seqLen, dim)







