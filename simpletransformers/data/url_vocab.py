import os
import json
import torch
import pickle
import numpy as np
import networkx as nx
from node2vec import Node2Vec
from node2vec.edges import HadamardEmbedder
from transformers import BertConfig, BertTokenizer
from transformers.modeling_bert import BertPreTrainedModel, BertModel


CURR_PATH = os.path.split(os.path.abspath(__file__))[0]
NODE2VEC_EMBEDDING_FILENAME = os.path.join(CURR_PATH, './data_from_node2vec/node_embedding.txt')
NODE2VEC_EMBEDDING_MODEL_FILENAME = os.path.join(CURR_PATH, './data_from_node2vec/embedding_model')
NODE2VEC_EDGES_EMBEDDING_FILENAME = os.path.join(CURR_PATH, './data_from_node2vec/edge_embedding.txt')


class UrlVocab:
    """A vocabulary of entities (articles identifiable by URL) that are recommendation candidates."""
    def __init__(self, df_url,
                 node2vec_encode=False,
                 bert_encode=False,
                 bert_embedding_filename=None):
        self.df_url = df_url
        self.url_vocab_list = list(self.df_url['url'])
        self.vocab_size = len(self.url_vocab_list)
        self.url_to_idx = {url: i for i, url in enumerate(self.url_vocab_list)}
        self.url_to_context = {x['url']: x
                               for x in json.loads(self.df_url.to_json(orient='records'))}

        # Navigation nodes
        self.nav_url_list = sorted(list(set(sum(self.df_url['nav_hrefs'], []))))
        self.all_url_list = self.url_vocab_list + self.nav_url_list
        self.nav_url_to_idx = {url: i for i, url in enumerate(self.nav_url_list)}
        self.url_to_nav_urls = {x: y for x, y in zip(self.df_url['url'], self.df_url['nav_hrefs'])}

        # Find connectivity
        # Connectivity between article nodes
        self.out_connectivity, self.in_connectivity = self._find_url_connectivity()
        # Connectivity between all nodes (article + nav)
        self.all_connectivity = self._find_all_connectivity()

        # Embed structural information
        self.url_embedding, self.nav_url_embedding = None, None
        if node2vec_encode:
            self.url_embedding, self.nav_url_embedding = self._node2vec_encode()

        # Embed using BERT
        self.bert_embedding = None
        self.bert_embedding_filename = bert_embedding_filename
        if bert_encode and bert_embedding_filename is not None:
            self.bert_embedding = self._get_bert_encoding(self.bert_embedding_filename)


    def _find_url_connectivity(self):
        """Find related entities (articles) through hyperlinks"""
        # Find cross edges
        out_url = self.df_url['hrefs']
        connectivity = []
        for ou in out_url:
            conn = [0 for _ in range(self.vocab_size)]
            for u in ou:
                if u in self.url_to_idx:
                    conn[self.url_to_idx[u]] = 1
            connectivity.append(conn)
        out_connectivity = np.array(connectivity)  # Cross edge adjacent matrix
        in_connectivity = np.transpose(out_connectivity)
        print(f'Number of articles with OUT hyperlinks: {sum(out_connectivity.sum(1) > 0)}')
        print(f'Number of articles with IN hyperlinks:  {sum(in_connectivity.sum(1) > 0)}')
        print(f'Number of articles with OUT or IN hyperlinks:',
              f'{sum((out_connectivity.sum(1) > 0) | (in_connectivity.sum(1) > 0))}')
        return out_connectivity, in_connectivity

    def _find_all_connectivity(self):
        """Find related articles and navigation pages."""
        if self.url_vocab_list is None or self.nav_url_list is None or self.out_connectivity is None:
            return None
        all_connectivity = np.zeros([len(self.all_url_list), len(self.all_url_list)])
        all_connectivity[0:len(self.url_vocab_list), 0:len(self.url_vocab_list)] = np.array(self.out_connectivity)
        for x, y in zip(self.df_url['url'], self.df_url['nav_hrefs']):
            nav_path = [x] + y[::-1]
            for i in range(len(nav_path) - 1):
                start = self.all_url_list.index(nav_path[i])
                end = self.all_url_list.index(nav_path[i+1])
                all_connectivity[start, end] = 1
        return all_connectivity

    def _node2vec_encode(self, save=False):
        if self.all_connectivity is None:
            print("Need to run function _find_all_connectivityd() first.")
            return
        graph = nx.from_numpy_matrix(self.all_connectivity, create_using=nx.DiGraph)
        node2vec = Node2Vec(graph, dimensions=64, walk_length=30, num_walks=200, workers=4)
        model = node2vec.fit(window=10, min_count=1, batch_words=4)

        # Save Node2vec results
        if save:
            model.wv.save_word2vec_format(NODE2VEC_EMBEDDING_FILENAME)
            model.save(NODE2VEC_EMBEDDING_MODEL_FILENAME)
            edges_embs = HadamardEmbedder(keyed_vectors=model.wv)
            edges_kv = edges_embs.as_keyed_vectors()
            edges_kv.save_word2vec_format(NODE2VEC_EDGES_EMBEDDING_FILENAME)

        url_embedding = model.wv.vectors[0: len(self.url_vocab_list)]
        nav_url_embedding = model.wv.vectors[len(self.url_vocab_list) :]
        return url_embedding, nav_url_embedding

    def idx2url(self, i):
        return self.url_vocab_list[i]

    def url2idx(self, url):
        return self.url_to_idx[url]

    def encode_urls(self, urls):
        return [1 if x in urls else 0
                for x in self.url_vocab_list]

    def in_vocab(self, url):
        return url in self.url_to_idx

    def get_name(self, url):
        return self.url_to_context[url]['name']

    def get_email_context(self, url):
        return self.url_to_context[url]['context']

    def get_email_anchor(self, url):
        return self.url_to_context[url]['context']

    def get_title(self, url):
        return self.url_to_context[url]['header']

    def get_text(self, url):
        return self.url_to_context[url]['article']

    def url2outconn(self, url):
        return self.out_connectivity[self.url2idx(url)].tolist()

    def url2inconn(self, url):
        return self.in_connectivity[self.url2idx(url)].tolist()

    def url2nav(self, url):
        """Return a binary representation of the navigation nodes of a URL."""
        return self.encode_nav_urls(self.url_to_nav_urls[url])

    def encode_nav_urls(self, nav_urls):
        if self.nav_url_list is None:
            return None
        return [1 if x in nav_urls else 0
                for x in self.nav_url_list]

    def url2node2vec(self, url):
        if url in self.url_vocab_list:
            return self.url_embedding[self.url_to_idx[url]]
        if url in self.nav_url_list:
            return self.nav_url_embedding[self.nav_url_to_idx[url]]
        return None

    def _get_bert_encoding(self, filename):
        if os.path.exists(filename):
            print(f'Loading bert encoding from {filename}.')
            with open(filename, 'rb') as f:
                bert_embedding = pickle.loads(f)
                assert bert_embedding.shape[0] == self.vocab_size
        else:
            print(f'BERT encoding not found, generating.')
            bert_embedding = self._bert_encode()
            with open(filename, 'wb') as f:
                pickle.dump(bert_embedding, f, protocol=pickle.HIGHEST_PROTOCOL)
                print(f'Saved BERT encoding to {filename}')
        return bert_embedding

    def _bert_encode(self,
                     max_seq_length=128,
                     sequence_a_segment_id=0,
                     sequence_b_segment_id=1,
                     cls_token_segment_id=1,
                     pad_token_segment_id=0,
                     mask_padding_with_zero=True):

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        bert_config = BertConfig.from_pretrained('bert-base-uncased')
        model = BertModel(bert_config)

        all_input_ids, all_input_mask, all_segment_ids = [], [], []
        for header, article in zip(self.df_url['header'], self.df_url['article']):
            text = header + '. ' + article
            tokens = tokenizer.tokenize(text)
            special_tokens_count = 2
            if len(tokens) > max_seq_length - special_tokens_count:
                tokens = tokens[: (max_seq_length - special_tokens_count)]
            segment_ids = [sequence_a_segment_id] * len(tokens)
            tokens = [tokenizer.cls_token] + tokens + [tokenizer.sep_token]
            segment_ids = [cls_token_segment_id] + segment_ids + [sequence_a_segment_id]
            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

            # Padding
            padding_length = max_seq_length - len(input_ids)
            pad_token = tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0]
            input_ids = input_ids + ([pad_token] * padding_length)
            input_mask = input_mask + [0] * padding_length
            segment_ids = segment_ids + ([pad_token_segment_id] * padding_length)

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length
            all_input_ids.append(input_ids)
            all_input_mask.append(input_mask)
            all_segment_ids.append(segment_ids)

        all_input_ids = torch.tensor(all_input_ids)
        all_input_mask = torch.tensor(all_input_mask)
        all_segment_ids = torch.tensor(all_segment_ids)

        model.eval()
        outputs = model(all_input_ids,
                        attention_mask=all_input_mask,
                        token_type_ids=all_segment_ids)
        embedding = outputs[1].data.numpy()
        del model
        return embedding

