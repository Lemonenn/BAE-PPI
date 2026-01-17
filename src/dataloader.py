import os
import csv
import json
import pickle
import random
import numpy as np
from tqdm import tqdm

import dgl 
import torch

from utils import get_bfs_sub_graph, get_dfs_sub_graph

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_data(dataset, split_mode, seed, skip_head=True):

    name = 0
    ppi_name = 0
    
    protein_name = {}
    ppi_dict = {}
    ppi_list = []
    ppi_label_list = []

    class_map = {'reaction':0, 'binding':1, 'ptmod':2, 'activation':3, 'inhibition':4, 'catalysis':5, 'expression':6}

    ppi_path = '../data/processed_data/protein.actions.{}.txt'.format(dataset)
    prot_seq_path = '../data/processed_data/protein.{}.sequences.dictionary.csv'.format(dataset)
    prot_r_edge_path = '../data/processed_data/protein.rball.edges.{}.npy'.format(dataset)
    prot_k_edge_path = '../data/processed_data/protein.knn.edges.{}.npy'.format(dataset)
    prot_s_edge_path = '../data/processed_data/protein.seq.edges.{}.npy'.format(dataset) # sequence edges
    prot_node_path = '../data/processed_data/protein.nodes.{}.pt'.format(dataset)

    if os.path.exists("../data/processed_data/{}_ppi.pkl".format(dataset)):
        with open("../data/processed_data/{}_ppi.pkl".format(dataset), "rb") as tf:
            ppi_list = pickle.load(tf)
        with open("../data/processed_data/{}_ppi_label.pkl".format(dataset), "rb") as tf:
            ppi_label_list = pickle.load(tf)

    else:
        
        # get node and node name
        with open(prot_seq_path) as f:
            reader = csv.reader(f)
            for row in reader:
                protein_name[row[0]] = name
                name += 1

        for line in tqdm(open(ppi_path)):
            if skip_head:
                skip_head = False
                continue
            line = line.strip().split('\t')
        
            # if line[0] not in protein_name.keys():
            #     protein_name[line[0]] = name
            #     name += 1
            
            # if line[1] not in protein_name.keys():
            #     protein_name[line[1]] = name
            #     name += 1

            # get edge and its label
            if line[0] < line[1]:
                temp_data = line[0] + "__" + line[1]
            else:
                temp_data = line[1] + "__" + line[0]

            if temp_data not in ppi_dict.keys():
                ppi_dict[temp_data] = ppi_name
                temp_label = [0, 0, 0, 0, 0, 0, 0]
                temp_label[class_map[line[2]]] = 1
                ppi_label_list.append(temp_label)
                ppi_name += 1
            else:
                index = ppi_dict[temp_data]
                temp_label = ppi_label_list[index]
                temp_label[class_map[line[2]]] = 1
                ppi_label_list[index] = temp_label

        for ppi in tqdm(ppi_dict.keys()):
            temp = ppi.strip().split('__')
            ppi_list.append(temp)

        ppi_num = len(ppi_list)
        for i in tqdm(range(ppi_num)):
            seq1_name = ppi_list[i][0]
            seq2_name = ppi_list[i][1]
            ppi_list[i][0] = protein_name[seq1_name]
            ppi_list[i][1] = protein_name[seq2_name]

        with open("../data/processed_data/{}_ppi.pkl".format(dataset), "wb") as tf:
            pickle.dump(ppi_list, tf)
        with open("../data/processed_data/{}_ppi_label.pkl".format(dataset), "wb") as tf:
            pickle.dump(ppi_label_list, tf)

    # ppi_g = dgl.to_bidirected(dgl.graph(ppi_list))
    ppi_edges_by_type = {interaction: [] for interaction in class_map.keys()}
    for idx, (u, v) in enumerate(ppi_list):
        label = ppi_label_list[idx]
        for interaction, i in class_map.items():
            if label[i] == 1:
                ppi_edges_by_type[interaction].append((u, v))
                ppi_edges_by_type[interaction].append((v, u))  # 双向
                break

    ppi_g = dgl.heterograph({
        ('protein', rel_type, 'protein'): edges
        for rel_type, edges in ppi_edges_by_type.items()
    }).to(device)
    print("ppi_g num_nodes, num_edges:", ppi_g.num_nodes('protein'), ppi_g.num_edges())
    protein_data = ProteinDatasetDGL(prot_r_edge_path, prot_k_edge_path, prot_s_edge_path, prot_node_path, dataset)
    ppi_split_dict = split_dataset(ppi_list, dataset, split_mode, seed)

    # ========================================
    # 1. 创建训练图 (ppi_g_train): 只包含训练集中的边及其真实类型。
    ppi_edges_train_by_type = {interaction: [] for interaction in class_map.keys()}
    ppi_edges_train_by_type['unknown'] = []

    for idx in ppi_split_dict['train_index']:
        u, v = ppi_list[idx]
        label = ppi_label_list[idx]
        for interaction, i in class_map.items():
            if label[i] == 1:
                ppi_edges_train_by_type[interaction].append((u, v))
                ppi_edges_train_by_type[interaction].append((v, u))  # 双向
                break

    # ppi_g_train = dgl.heterograph({
    #     ('protein', rel_type, 'protein'): edges
    #     for rel_type, edges in ppi_edges_train_by_type.items()}).to(device)
    ppi_g_train = dgl.heterograph({
        ('protein', rel_type, 'protein'): edges
        for rel_type, edges in ppi_edges_train_by_type.items()}, num_nodes_dict={'protein': ppi_g.num_nodes('protein')}).to(device)
    print("ppi_g_train num_nodes, num_edges:", ppi_g_train.num_nodes('protein'), ppi_g_train.num_edges())

    # 2. 创建测试图 (ppi_g_test): 包含所有边。训练集和验证集边有真实类型，测试集边类型为 'unknown'。
    ppi_edges_test_by_type = {interaction: [] for interaction in class_map.keys()}
    ppi_edges_test_by_type['unknown'] = []

    all_known_indices = ppi_split_dict['train_index'] + ppi_split_dict['val_index']
    unknown_indices = ppi_split_dict['test_index']
    for idx in all_known_indices:
        u, v = ppi_list[idx]
        label = ppi_label_list[idx]
        for interaction, i in class_map.items():
            if label[i] == 1:
                ppi_edges_test_by_type[interaction].append((u, v))
                ppi_edges_test_by_type[interaction].append((v, u))
                break

    for idx in unknown_indices:
        u, v = ppi_list[idx]
        ppi_edges_test_by_type['unknown'].append((u, v))
        ppi_edges_test_by_type['unknown'].append((v, u))

    ppi_g_test = dgl.heterograph({
        ('protein', rel_type, 'protein'): edges
        for rel_type, edges in ppi_edges_test_by_type.items()}).to(device)
    print("ppi_g_test num_nodes, num_edges:", ppi_g_test.num_nodes('protein'), ppi_g_test.num_edges())

    # 返回两个图
    return protein_data, ppi_g_train.to(device), ppi_g_test.to(device), ppi_list, torch.FloatTensor(np.array(ppi_label_list)).to(device), ppi_split_dict
    # ========================================


class ProteinDatasetDGL(torch.utils.data.Dataset):
    def __init__(self, prot_r_edge_path, prot_k_edge_path, prot_s_edge_path, prot_node_path, dataset):
        
        if os.path.exists("../data/processed_data/{}_protein_graphs.pkl".format(dataset)):
            with open("../data/processed_data/{}_protein_graphs.pkl".format(dataset), "rb") as tf:
        # if os.path.exists("../data/processed_data_ori/{}_protein_graphs.pkl".format(dataset)):
        #     with open("../data/processed_data_ori/{}_protein_graphs.pkl".format(dataset), "rb") as tf:
                self.prot_graph_list = pickle.load(tf)
                print("graphs list length:", len(self.prot_graph_list))
        
        else:

            prot_r_edge = np.load(prot_r_edge_path, allow_pickle=True)
            prot_k_edge = np.load(prot_k_edge_path, allow_pickle=True)
            prot_s_edge = np.load(prot_s_edge_path, allow_pickle=True)
            prot_node = torch.load(prot_node_path)

            self.prot_graph_list = []

            for i in range(len(prot_r_edge)):
                r_src, r_dst, r_dist = zip(*prot_r_edge[i])
                k_src, k_dst, k_dist = zip(*prot_k_edge[i])
                s_src, s_dst, s_dist = zip(*prot_s_edge[i])

                # prot_g = dgl.graph(prot_edge[i]).to(device)
                prot_g = dgl.heterograph({
                    # ('amino_acid', 'SEQ', 'amino_acid'): (s_src, s_dst),
                    ('amino_acid', 'STR_KNN', 'amino_acid'): (k_src, k_dst),
                    ('amino_acid', 'STR_DIS', 'amino_acid'): (r_src, r_dst)
                }).to(device)                

                # 存边特征
                # prot_g.edges['SEQ'].data['dist'] = torch.FloatTensor(s_dist).to(device)
                prot_g.edges['STR_KNN'].data['dist'] = torch.FloatTensor(k_dist).to(device)
                prot_g.edges['STR_DIS'].data['dist'] = torch.FloatTensor(r_dist).to(device)

                # 序列 embedding 直接作为节点特征，或者单独保存
                prot_g.ndata['x'] = torch.FloatTensor(prot_node[i]).to(device)
                prot_g.ndata['seq_id'] = torch.arange(prot_g.num_nodes(), dtype=torch.long, device=prot_g.device)
                prot_g.ndata['seq_feat'] = torch.FloatTensor(prot_node[i]).to(device)

                self.prot_graph_list.append(prot_g)

            print("graphs loaded from file:{}", len(self.prot_graph_list))
            with open("../data/processed_data/{}_protein_graphs.pkl".format(dataset), "wb") as tf:
                pickle.dump(self.prot_graph_list, tf)

    def __len__(self):
        return len(self.prot_graph_list)

    def __getitem__(self, idx):
        return self.prot_graph_list[idx]
        
def collate(samples):
    return dgl.batch_hetero(samples)


def split_dataset(ppi_list, dataset, split_mode, seed):
    if not os.path.exists("../data/processed_data/{}_{}.json".format(dataset, split_mode)):
        if split_mode == 'random':
            ppi_num = len(ppi_list)
            random_list = [i for i in range(ppi_num)]
            random.shuffle(random_list)

            ppi_split_dict = {}
            ppi_split_dict['train_index'] = random_list[: int(ppi_num*0.6)]
            ppi_split_dict['val_index'] = random_list[int(ppi_num*0.6) : int(ppi_num*0.8)]
            ppi_split_dict['test_index'] = random_list[int(ppi_num*0.8) :]

            jsobj = json.dumps(ppi_split_dict)
            with open("../data/processed_data/{}_{}.json".format(dataset, split_mode), 'w') as f:
                f.write(jsobj)
                f.close()

        elif split_mode == 'bfs' or split_mode == 'dfs':
            node_to_edge_index = {}
            ppi_num = len(ppi_list)

            for i in range(ppi_num):
                edge = ppi_list[i]
                if edge[0] not in node_to_edge_index.keys():
                    node_to_edge_index[edge[0]] = []
                node_to_edge_index[edge[0]].append(i)

                if edge[1] not in node_to_edge_index.keys():
                    node_to_edge_index[edge[1]] = []
                node_to_edge_index[edge[1]].append(i)
            
            node_num = len(node_to_edge_index)
            sub_graph_size = int(ppi_num * 0.4)

            if split_mode == 'bfs':
                selected_edge_index = get_bfs_sub_graph(ppi_list, node_num, node_to_edge_index, sub_graph_size)
            elif split_mode == 'dfs':
                selected_edge_index = get_dfs_sub_graph(ppi_list, node_num, node_to_edge_index, sub_graph_size)
            
            all_edge_index = [i for i in range(ppi_num)]
            unselected_edge_index = list(set(all_edge_index).difference(set(selected_edge_index)))

            random_list = [i for i in range(len(selected_edge_index))]
            random.shuffle(random_list)

            ppi_split_dict = {}
            ppi_split_dict['train_index'] = unselected_edge_index
            ppi_split_dict['val_index'] = [selected_edge_index[i] for i in random_list[:int(ppi_num*0.2)]]
            ppi_split_dict['test_index'] = [selected_edge_index[i] for i in random_list[int(ppi_num*0.2):]]

            jsobj = json.dumps(ppi_split_dict)
            with open("../data/processed_data/{}_{}.json".format(dataset, split_mode), 'w') as f:
                f.write(jsobj)
                f.close()
        
        else:
            print("your mode is {}, you should use bfs, dfs or random".format(split_mode))
            return
    else:
        with open("../data/processed_data/{}_{}.json".format(dataset, split_mode), 'r') as f:
            ppi_split_dict = json.load(f)
            f.close()

    print("Train_PPI: {} | Val_PPI: {} | Test_PPI: {}".format(len(ppi_split_dict['train_index']), len(ppi_split_dict['val_index']), len(ppi_split_dict['test_index'])))

    return ppi_split_dict
    

def load_pretrain_data(dataset):

    prot_r_edge_path = '../data/processed_data/protein.rball.edges.{}.npy'.format(dataset)
    prot_k_edge_path = '../data/processed_data/protein.knn.edges.{}.npy'.format(dataset)
    prot_node_path = '../data/processed_data/protein.nodes.{}.pt'.format(dataset)

    protein_data = ProteinDatasetDGL(prot_r_edge_path, prot_k_edge_path, prot_node_path, dataset)

    return protein_data