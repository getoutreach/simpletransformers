import os
import json
import numpy as np
import networkx as nx
from node2vec import Node2Vec
from node2vec.edges import HadamardEmbedder


CURR_PATH = os.path.split(os.path.abspath(__file__))[0]
NODE2VEC_EMBEDDING_FILENAME = os.path.join(CURR_PATH, './data_from_node2vec/node_embedding.txt')
NODE2VEC_EMBEDDING_MODEL_FILENAME = os.path.join(CURR_PATH, './data_from_node2vec/embedding_model')
NODE2VEC_EDGES_EMBEDDING_FILENAME = os.path.join(CURR_PATH, './data_from_node2vec/edge_embedding.txt')


class UrlVocab:
    """A vocabulary of entities (articles identifiable by URL) that are recommendation candidates."""
    def __init__(self, df_url):
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
        self.url_embedding, self.nav_url_embedding = self._node2vec_encode()
        pass

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
            return self.url_embedding[self.url_to_idx(url)]
        if url in self.nav_url_list:
            return self.nav_url_embedding[self.nav_url_to_idx(url)]
        return None