import dgl
import dgl.function as fn
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph
import networkx as nx


gcn_msg = fn.copy_src(src='h', out='m')
gcn_reduce = fn.sum(msg='m', out='h')


class NodeApplyModule(nn.Module):
    def __init__(self, in_feats, out_feats, activation):
        super(NodeApplyModule, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)
        self.activation = activation

    def forward(self, node):
        h = self.linear(node.data['h'])
        if self.activation is not None:
            h = self.activation(h)
        return {'h' : h}


class GCN(nn.Module):
    def __init__(self, in_feats, out_feats, activation):
        super(GCN, self).__init__()
        self.apply_mod = NodeApplyModule(in_feats, out_feats, activation)

    def forward(self, g, feature):
        g.ndata['h'] = feature.to(next(self.parameters()).device)
        g.update_all(gcn_msg, gcn_reduce)
        g.apply_nodes(func=self.apply_mod)
        return g.ndata.pop('h')


class Net(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Net, self).__init__()
        self.gcn1 = GCN(input_size, hidden_size, F.relu)
        self.gcn2 = GCN(hidden_size, output_size, None)

    def forward(self, g, features):
        x = self.gcn1(g, features)
        x = self.gcn2(g, x)
        return x

# # Load URL vocab from S3
# import os
# from simpletransformers.data.data_utils import load_url_vocab
#
#
# file_path = os.path.join('jie-faq', 'faq', 'outreach_support_url_meta.json')
# urlvocab = load_url_vocab(file_path,
#                           bert_encode=True,
#                           bert_encoding_filename='./simpletransformers/data/data_from_bert/embedding')
# print('Number of recommendation candidate URLs: %d' % urlvocab.vocab_size)
#
# # nx_dg = nx.DiGraph()
# nx_dg = nx.from_numpy_matrix(urlvocab.out_connectivity)
# g = dgl.DGLGraph()
# g.from_networkx(nx_dg)
# features = urlvocab.bert_embedding

if __name__ == '__main__':

    # Example training a GCN from:
    # https://docs.dgl.ai/tutorials/models/1_gnn/1_gcn.html#sphx-glr-tutorials-models-1-gnn-1-gcn-py
    import time
    import numpy as np
    from dgl.data import citation_graph as citegrh

    net = Net(1433, 16, 7)
    print(net)


    def load_cora_data():
        data = citegrh.load_cora()
        features = th.FloatTensor(data.features)
        labels = th.LongTensor(data.labels)
        train_mask = th.BoolTensor(data.train_mask)
        test_mask = th.BoolTensor(data.test_mask)
        g = data.graph
        # add self loop
        g.remove_edges_from(nx.selfloop_edges(g))
        g = DGLGraph(g)
        g.add_edges(g.nodes(), g.nodes())
        return g, features, labels, train_mask, test_mask


    def evaluate(model, g, features, labels, mask):
        model.eval()
        with th.no_grad():
            logits = model(g, features)
            logits = logits[mask]
            labels = labels[mask]
            _, indices = th.max(logits, dim=1)
            correct = th.sum(indices == labels)
            return correct.item() * 1.0 / len(labels)


    g_, features_, labels, train_mask, test_mask = load_cora_data()
    optimizer = th.optim.Adam(net.parameters(), lr=1e-3)
    dur = []
    for epoch in range(50):
        if epoch >=3:
            t0 = time.time()

        net.train()
        logits = net(g, features)
        logp = F.log_softmax(logits, 1)
        loss = F.nll_loss(logp[train_mask], labels[train_mask])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch >=3:
            dur.append(time.time() - t0)

        acc = evaluate(net, g, features, labels, test_mask)
        print("Epoch {:05d} | Loss {:.4f} | Test Acc {:.4f} | Time(s) {:.4f}".format(
                epoch, loss.item(), acc, np.mean(dur)))








