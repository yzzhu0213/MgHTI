import torch
import torch.nn as nn
from dgllife.model.gnn import GCN
import torch.nn.functional as F
import os

os.environ["CUDA_VISIBLE_DEVICES"] = '4'


class HerbEncoder(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(HerbEncoder, self).__init__()

        self.fc = nn.Linear(input_dim, output_dim)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.Linear(output_dim, input_dim),
            nn.Sigmoid()
        )

    def forward(self, target):
        encoded = self.encoder(target)
        decoded = self.decoder(encoded)
        return encoded, decoded


class ProteinCNN(nn.Module):
    def __init__(self, embedding_dim, num_filters, kernel_size, padding=True):
        super(ProteinCNN, self).__init__()
        if padding:
            self.embedding = nn.Embedding(26, embedding_dim, padding_idx=0)
        else:
            self.embedding = nn.Embedding(26, embedding_dim)
        in_ch = [embedding_dim] + num_filters
        self.in_ch = in_ch[-1]
        kernels = kernel_size
        self.conv1 = nn.Conv1d(in_channels=in_ch[0], out_channels=in_ch[1], kernel_size=kernels[0])
        self.bn1 = nn.BatchNorm1d(in_ch[1])
        self.conv2 = nn.Conv1d(in_channels=in_ch[1], out_channels=in_ch[2], kernel_size=kernels[1])
        self.bn2 = nn.BatchNorm1d(in_ch[2])
        self.conv3 = nn.Conv1d(in_channels=in_ch[2], out_channels=in_ch[3], kernel_size=kernels[2])
        self.bn3 = nn.BatchNorm1d(in_ch[3])
        self.dropout = torch.nn.Dropout(0.2)

    def forward(self, v):
        v = self.embedding(v.long())
        v = v.transpose(1, 2)
        v = self.bn1(F.relu(self.conv1(v)))
        v = self.bn2(F.relu(self.conv2(v)))
        v = self.bn3(F.relu(self.conv3(v)))

        v = v.view(v.size(0), v.size(2), -1)
        v, _ = torch.max(v, dim=1)

        return v


class MolecularGCN(nn.Module):
    def __init__(self, in_feats, dim_embedding, hidden_feats, padding=True, activation=None):
        super(MolecularGCN, self).__init__()
        self.init_transform = nn.Linear(in_feats, dim_embedding, bias=False)
        if padding:
            with torch.no_grad():
                self.init_transform.weight[-1].fill_(0)
        self.gnn = GCN(in_feats=dim_embedding, hidden_feats=hidden_feats, activation=activation)

    def forward(self, mol_graph):
        node_feats = mol_graph.ndata['feat']
        node_feats = self.init_transform(node_feats)
        node_feats = self.gnn(mol_graph, node_feats)
        molecule_features, _ = torch.max(node_feats, dim=0)
        return molecule_features



class AttentionPooling(nn.Module):
    def __init__(self, input_dim):
        super(AttentionPooling, self).__init__()
        self.input_dim = input_dim
        self.attention = nn.Sequential(
            nn.Linear(input_dim, 128, bias=False),
            nn.Tanh(),
            nn.Linear(128, 1, bias=False),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        attention_scores = self.attention(x).squeeze(-1)
        pooled_output = torch.sum(attention_scores.unsqueeze(-1) * x, dim=-2)
        return pooled_output


class MultiHeadAttentionPooling(nn.Module):
    def __init__(self, input_dim, heads):
        super(MultiHeadAttentionPooling, self).__init__()
        self.input_dim = input_dim
        self.heads = heads
        self.attention_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, input_dim, bias=False),
                nn.Tanh(),
                nn.Linear(input_dim, 1, bias=False),
                nn.Softmax(dim=-1)
            ) for _ in range(heads)
        ])

    def forward(self, x):
        attention_scores = torch.cat([head(x) for head in self.attention_heads], dim=-1)
        averaged_attention = torch.mean(attention_scores, dim=-1, keepdim=True)
        pooled_output = torch.sum(averaged_attention * x, dim=-1)
        return pooled_output


class Prediction(torch.nn.Module):
    def __init__(self, input_dim, drop_out):
        super(Prediction, self).__init__()
        self.fc1 = nn.Linear(input_dim, int(input_dim / 2))
        self.fc2 = nn.Linear(int(input_dim / 2), int(input_dim / 4))
        self.fc3 = nn.Linear(int(input_dim / 4), 1)

        self.relu = nn.ReLU(drop_out)
        self.dropout = nn.Dropout(drop_out)

    def forward(self, feature):
        feature = self.dropout(self.relu(self.fc1(feature)))
        feature = self.dropout(self.relu(self.fc2(feature)))
        prediction = torch.sigmoid(self.fc3(feature))

        return prediction


class MILHTI(nn.Module):
    def __init__(self,
                 herb_input_dim,
                 herb_output_dim,
                 target_input_dim,
                 target_output_dim,
                 ingredient_input_dim,
                 ingredient_output_dim,
                 drop_out
                 ):
        super(MILHTI, self).__init__()

        self.herb_model = HerbEncoder(herb_input_dim, herb_output_dim)
        self.criterion = nn.MSELoss()
        self.target_model = ProteinCNN(embedding_dim=target_output_dim,
                                       num_filters=[target_output_dim, target_output_dim, target_output_dim],
                                       kernel_size=[3, 6, 9], padding=True)
        self.ingredient_model = MolecularGCN(ingredient_input_dim, ingredient_output_dim, [ingredient_output_dim])
        self.attention = AttentionPooling(ingredient_output_dim)
        self.attention2 = MultiHeadAttentionPooling(76, 2)
        self.pre = Prediction(herb_output_dim + target_output_dim + ingredient_output_dim, drop_out)
        self.pre_ins = Prediction(target_output_dim + ingredient_output_dim, drop_out)
        self.maxins = 76

    def forward(self, herb, target, ingredient):
        herb_fearure, decoded = self.herb_model(herb)
        loss_e = self.criterion(decoded, herb)
        target_feature = self.target_model(target)
        ingredient = ingredient.to(dtype=torch.float32)
        ingredient_feature = self.attention(ingredient)

        # herb-target interactions predict
        com_feature = torch.cat((herb_fearure, ingredient_feature, target_feature), 1)
        prediction_bag = self.pre(com_feature)

        # ingredient-target interactions predict
        target_feature = target_feature.unsqueeze(1)
        target_feature = target_feature.repeat(1, self.maxins, 1)
        ingredient_target = torch.cat((ingredient, target_feature), 2)
        ingredient_target = ingredient_target.view(len(herb_fearure) * self.maxins, len(ingredient_feature[0]) + len(target_feature[0]))
        ingredient_target = ingredient_target.to(dtype=torch.float32)
        # if known labels
        # return
        prediction_ins = self.pre_ins(ingredient_target)
        # if unknown labels
        # continue
        prediction_ins = prediction_ins.view(len(herb_fearure), 1, self.maxins)
        prediction_ins = self.attention2(prediction_ins)

        return prediction_bag, prediction_ins, loss_e

