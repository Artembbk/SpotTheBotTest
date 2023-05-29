from __future__ import annotations
import numpy as np
from scipy.spatial.distance import euclidean
import networkx as nx
from copy import deepcopy
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from itertools import combinations
import pydot
from tqdm import tqdm 
import pickle
from sklearn.neighbors import NearestNeighbors
from itertools import chain
from typing import Union
import networkx as nx
import io
from networkx.readwrite import json_graph
import ray
from ray.util import ActorPool
import time
import math


def save(graph, path):
    with open(path, 'wb') as f:
        pickle.dump(graph, f)

def read(path):
    with open(path, 'rb') as f:
        graph = pickle.load(f)

    return graph

class Node:
    def __init__(self, word: str = "", vector: np.ndarray = None, neighbors: set[str] = set()):        
        self.word: str = deepcopy(word)
        self.vector: np.ndarray = deepcopy(vector)
        self.neighbors: set[str] = deepcopy(neighbors)
        self.nearest_neighbor: str = ""

    def add_neighbor(self, neighbor: str):
        self.neighbors.add(neighbor)

    def remove_neighbor(self, neighbor: str):
        self.neighbors.discard(neighbor)

    def dist_to(self, other: Node, dist_func=euclidean):
        return dist_func(self.vector, other.vector)

    def is_inside_sphere(self, node1: Node, node2: Node):
        center = (node1.vector + node2.vector) / 2
        radius = euclidean(center, node1.vector)
        return euclidean(self.vector, center) < radius
    
    def __str__(self) -> str:
        return self.word

    def __eq__(self, other: Node) -> bool:
        return self.word == other.word



class Graph:
    def __init__(self):
        self.vertices: dict[str, Node] = {}
        self.nx_graph: nx.Graph = None
        self.vecs: np.array = None
        self.word2num: dict[str, int] = {}
        self.num2word: list[str] = None
        self.nrst_nbrs: dict[str, str] = {}
        self.shortest_distances = {}

    @classmethod
    def from_word2vec(grp, word2vec: dict[str, np.ndarray], k=16) -> Graph:
        graph = grp()
        graph.vecs = np.array(list(word2vec.values()))
        graph.num2word = list(word2vec.keys())
        for i, word in enumerate(graph.num2word):
            graph.word2num[word] = i
        for word, vector in word2vec.items():
            graph.add_word(word, vector[-k:])
        return graph

    def add_node(self, node: Node):
        self.vertices[node.word] = node
        for neighbor in node.neighbors:
            self.vertices[neighbor].neighbors.add_neighbor(node.word)

    def add_word(self, word: str, vector: np.ndarray):
        node = Node(word, vector)
        self.add_node(node)

    def remove_word(self, word: str):
        node = self.vertices[word]
        for neighbor_word in node.neighbors:
            neighbor_node = self.vertices[neighbor_word]
            neighbor_node.neighbors.remove(word)

        self.vertices.pop(word, None)

    def create_sub_graph(self, words):
        words_to_remove = []
        for word in self.vertices.keys():
            if word not in words:
                words_to_remove.append(word)

        for word in words_to_remove:
            self.remove_word(word)

    def create_sub_graph_from_text(self, text):
        words = text.split()
        unique_words = set(words)
        self.create_sub_graph(unique_words)

    def text_to_word_pairs(self, text):
        words = text.split()  # Разделение текста на слова
        pairs = [(words[i], words[i + 1]) for i in range(len(words) - 1)]  # Формирование пар соседних слов
        return pairs

    def go_through_text(self, text: str, graph: nx.classes.graph.Graph) -> list:
        text = text.split()
        path_edges = []
        current_word = text[0]
        file_ = open("stream_go_through_text.txt", 'w')
        for word in tqdm(text[1:], file=file_):
            if current_word not in graph:
                current_word = word
                continue
            if word not in graph:
                continue
            if word in graph[current_word]:
                path_edges.append((current_word, word))
                file_.write('in path_edges!!!')
            else:
                try:
                    shortest_path = tuple(nx.dijkstra_path(graph, current_word, word))
                    path_edges.append(shortest_path)
                except:
                    path_edges.append(f"No path {current_word} -- {word}")
            current_word = word
        file_.close()
        return path_edges
    
    def convert_text_shortest_paths_to_text_edges_path(self, shortest_paths: list) -> list:
        edges_path = []
        for short_path in shortest_paths:
            current_word = short_path[0]
            for word in short_path[1:]:
                edges_path.append((current_word, word))
                current_word = word
        return edges_path

    def create_sub_graph_from_text_like_path(self, text):
        self.create_nx_graph()
        path_edges = self.go_through_text(text, self.nx_graph)
        edges = self.convert_text_shortest_paths_to_text_edges_path(path_edges)
        self.nx_graph = self.nx_graph.subgraph(edges)


    def neighbors(self, word: str) -> set[str]:
        return self.vertices[word].neighbors

    def __iter__(self):
        self._iter_obj = iter(self.__vertices)
        return self._iter_obj
    
    def __next__(self):
        return next(self._iter_obj)

    def get_edges(self):
        edges = list()
        for node in self.vertices.values():
            for neighbor in node.neighbors:
                edges.append((node.word, neighbor, node.dist_to(self.vertices[neighbor])))
        return edges

    def get_nodes(self):
        return list(self.vertices.values())
    
    def get_words(self):
        return list(self.vertices.keys())
    
    def knn(self, k = 10):
        nbrs = NearestNeighbors(n_neighbors=k).fit(self.vecs)
        _, indices = nbrs.kneighbors(self.vecs)
        for i, nbrs_nums in enumerate(indices):
            word = self.num2word[i]
            self.nrst_nbrs[word] = []
            for nbr_num in nbrs_nums:
                self.nrst_nbrs[word].append(self.num2word[nbr_num])

    def compute_nearest_neighbors(self):
        self.knn(2)
        for node in self.get_nodes():
            node.nearest_neighbor = self.nrst_nbrs[node.word][-1]

    def does_spheres_intersecting(self, node: Node, other: Node):
        if node.nearest_neighbor == "" or other.nearest_neighbor == "":
            raise ValueError
        node_nearest_node = self.vertices[node.nearest_neighbor]
        other_nearest_node = self.vertices[other.nearest_neighbor]
        return node.dist_to(other) < node.dist_to(node_nearest_node) + other.dist_to(other_nearest_node)

    def create_nx_graph(self):
        if self.nx_graph is not None:
            return
        self.nx_graph = nx.Graph()
        edges = self.get_edges()
        self.nx_graph.add_nodes_from(self.get_words())
        for node in self.get_nodes():
            self.nx_graph.nodes[node.word]['pos'] = node.vector
        self.nx_graph.add_weighted_edges_from(edges)

    def create_nx_sub_graph(self, words):
        self.nx_sub_graph = self.nx_graph.subgraph(words)


        
    def plot(self):
        if self.nx_graph is None:
            self.create_nx_graph()

        plt.figure(figsize=(30, 30))
        ax = plt.gca()
        pos = {word:(vec[0], vec[1]) for (word, vec) in nx.get_node_attributes(self.nx_graph, 'pos').items()}
        nx.draw(self.nx_graph, ax=ax, font_size=30, pos=pos, with_labels=True)
    
    def diametr(self):
        self.create_nx_graph()
        largest_component = max(nx.connected_components(self.nx_graph), key=len)
        return nx.radius(self.nx_graph.subgraph(largest_component))
    
    def radius(self):
        self.create_nx_graph()
        largest_component = max(nx.connected_components(self.nx_graph), key=len)
        return nx.radius(self.nx_graph.subgraph(largest_component))
    
    def average_shortest_path_length(self):
        self.create_nx_graph()
        largest_component = max(nx.connected_components(self.nx_graph), key=len)
        return nx.average_shortest_path_length(self.nx_graph.subgraph(largest_component))
    
    def closeness_centrality(self):
        self.create_nx_graph()
        return np.mean(list(nx.closeness_centrality(self.nx_graph).values()))
    
    def betweenness_centrality(self):
        self.create_nx_graph()
        return np.mean(list(nx.betweenness_centrality(self.nx_graph).values()))
    
    def edge_betweenness_centrality(self):
        self.create_nx_graph()
        return np.mean(list(nx.edge_betweenness_centrality(self.nx_graph).values()))
    
    def load_centrality(self):
        self.create_nx_graph()
        return np.mean(list(nx.edge_betweenness_centrality(self.nx_graph).values()))
        

@ray.remote
class SubEpsilonGraph(Graph):
    def __init__(self, graph):
        super().__init__()
        self.sub_words = set()
        self.eps = graph.eps
        self.vertices = graph.vertices
        self.nx_graph = graph.nx_graph
        self.vecs = graph.vecs
        self.word2num = graph.word2num
        self.num2word = graph.num2word
        self.nrst_nbrs = graph.nrst_nbrs
        
    async def create_sub_epsilon_graph(self, index, language):
        for word in tqdm(self.sub_words, file = open(f'./tmp/eps_{language}_{index}.log', 'w')):
            vertex = self.vertices[word]
            for other in self.get_nodes():
                if vertex == other:
                    continue
                if vertex.dist_to(other) < self.eps:
                    vertex.neighbors.add(other.word)
                    other.neighbors.add(vertex.word)
        return self.vertices
    
    def add_sub_word(self, word):
        self.sub_words.add(word)


class EpsilonGraph(Graph):
    def __init__(self):
        super().__init__()

    @classmethod
    def from_word2vec(grp, word2vec, eps: float = 1e-3, num_cpus=1, k=16, language='ru'):
        graph = super().from_word2vec(word2vec, k)
        graph.language = language
        graph.eps = eps
        if num_cpus == 1:
            graph.create_epsilon_graph()
        else:
            graph.create_parallel_epsilon_graph(num_cpus)
        return graph

    def create_epsilon_graph(self):
        for vertex in tqdm(self.get_nodes()):
            for other in self.get_nodes():
                if vertex == other:
                    continue
                if vertex.dist_to(other) < self.eps:
                    vertex.neighbors.add(other.word)
                    other.neighbors.add(vertex.word)

    def create_parallel_epsilon_graph(self, num_cpus):
        ray.init(num_cpus=num_cpus)
        streaming_actors = [SubEpsilonGraph.remote(self) for _ in range(num_cpus)]
        not_done_ids = []
        for i, word in enumerate(self.get_words()):
            not_done_ids.append(streaming_actors[i % num_cpus].add_sub_word.remote(word))

        while not_done_ids:
            if len(not_done_ids) % 1000 == 0:
                print(len(not_done_ids))
            done_ids, not_done_ids = ray.wait(not_done_ids, timeout=1)

        results = []
        not_done_ids = []
        for index, actor in enumerate(streaming_actors):
            not_done_ids.append(actor.create_sub_epsilon_graph.remote(index, self.language))
        
        while not_done_ids:
            done_ids, not_done_ids = ray.wait(not_done_ids, timeout=1)
            results.extend(ray.get(done_ids))
        

        for sub_vertices in results:
            for word in self.get_words():
                self.vertices[word].neighbors = self.vertices[word].neighbors.union(sub_vertices[word].neighbors)
        
        ray.shutdown()

@ray.remote
class SubGabrielGraph(Graph):
    def __init__(self, graph):
        super().__init__()
        self.sub_words = set()
        self.vertices = deepcopy(graph.vertices)
        self.nx_graph = graph.nx_graph
        self.vecs = graph.vecs
        self.word2num = graph.word2num
        self.num2word = graph.num2word
        self.nrst_nbrs = graph.nrst_nbrs
        
    async def create_sub_gabriel_graph(self, index, language):
        for p_word in tqdm(self.sub_words, file = open(f'./tmp/gg_{language}_{index}.log', 'w')):
            p_nbrs = self.nrst_nbrs[p_word]
            for q_word in self.get_words():
                if q_word == p_word:
                    continue
                q_nbrs = self.nrst_nbrs[q_word]
                is_edge = True
                p_node = self.vertices[p_word]
                q_node = self.vertices[q_word]
                for x_word in chain(p_nbrs, q_nbrs):
                    if x_word == p_word or x_word == q_word:
                        continue
                    x_node = self.vertices[x_word]
                    if x_node.is_inside_sphere(p_node, q_node):
                        is_edge = False
                if is_edge:
                    p_node.add_neighbor(q_node.word)
                    q_node.add_neighbor(p_node.word)
        return self.vertices
    
    def add_sub_word(self, word):
        self.sub_words.add(word)


class GabrielGraph(Graph):
    def __init__(self):
        super().__init__()

    @classmethod
    def from_word2vec(grp, word2vec, num_cpus=1, k=16, language='ru'):
        graph = super().from_word2vec(word2vec, k)
        graph.language = language
        graph.knn(k=10)
        if num_cpus == 1:
            graph.create_gabriel_graph()
        else:
            graph.create_parallel_gabriel_graph(num_cpus)
        return graph


    def create_gabriel_graph(self):
        for p_word in tqdm(self.get_words()):
            p_nbrs = self.nrst_nbrs[p_word]
            for q_word in self.get_words():
                if q_word == p_word:
                    continue
                q_nbrs = self.nrst_nbrs[q_word]
                is_edge = True
                p_node = self.vertices[p_word]
                q_node = self.vertices[q_word]
                for x_word in chain(p_nbrs, q_nbrs):
                    if x_word == p_word or x_word == q_word:
                        continue
                    x_node = self.vertices[x_word]
                    if x_node.is_inside_sphere(p_node, q_node):
                        is_edge = False
                if is_edge:
                    p_node.add_neighbor(q_node.word)
                    q_node.add_neighbor(p_node.word)

    def create_parallel_gabriel_graph(self, num_cpus):
        ray.init(num_cpus=num_cpus)
        streaming_actors = [SubGabrielGraph.remote(self) for _ in range(num_cpus)]
        not_done_ids = []
        for i, word in enumerate(self.get_words()):
            not_done_ids.append(streaming_actors[i % num_cpus].add_sub_word.remote(word))

        while not_done_ids:
            if len(not_done_ids) % 1000 == 0:
                print(len(not_done_ids))
            done_ids, not_done_ids = ray.wait(not_done_ids, timeout=1)

        results = []
        not_done_ids = []
        for index, actor in enumerate(streaming_actors):
            not_done_ids.append(actor.create_sub_gabriel_graph.remote(index, self.language))
        
        while not_done_ids:
            done_ids, not_done_ids = ray.wait(not_done_ids, timeout=1)
            results.extend(ray.get(done_ids))
        

        for sub_vertices in results:
            for word in self.get_words():
                self.vertices[word].neighbors = self.vertices[word].neighbors.union(sub_vertices[word].neighbors)

        ray.shutdown()

@ray.remote
class SubInfluenceGraph(Graph):
    def __init__(self, graph):
        super().__init__()
        self.sub_words = set()
        self.vertices = graph.vertices
        self.nx_graph = graph.nx_graph
        self.vecs = graph.vecs
        self.word2num = graph.word2num
        self.num2word = graph.num2word
        self.nrst_nbrs = graph.nrst_nbrs
        
    async def create_sub_influence_graph(self, index, language):
        for word1 in tqdm(self.sub_words, file=open(f'./tmp/inf_{language}_{index}.log', 'w')):
            node1 = self.vertices[word1]
            for node2 in self.vertices.values():
                if node1 != node2:
                    if self.does_spheres_intersecting(node1, node2):
                        node1.add_neighbor(node2.word)
                        node2.add_neighbor(node1.word)
        return self.vertices
    
    def add_sub_word(self, word):
        self.sub_words.add(word)

class InfluenceGraph(Graph):
    def __init__(self):
        super().__init__()

    @classmethod
    def from_word2vec(grp, word2vec, num_cpus=1, k=16, language='ru'):
        graph = super().from_word2vec(word2vec, k)
        graph.language = language
        if num_cpus == 1:
            graph.create_influence_graph()
        else:
            graph.create_parallel_influence_graph(num_cpus)
        return graph

    def create_influence_graph(self):
        self.compute_nearest_neighbors()
        for node1 in tqdm(self.vertices.values()):
            for node2 in self.vertices.values():
                if node1 != node2:
                    if self.does_spheres_intersecting(node1, node2):
                        node1.add_neighbor(node2.word)
                        node2.add_neighbor(node1.word)

    def create_parallel_influence_graph(self, num_cpus):
        self.compute_nearest_neighbors()
        ray.init(num_cpus=num_cpus)
        streaming_actors = [SubInfluenceGraph.remote(self) for _ in range(num_cpus)]
        not_done_ids = []
        for i, word in enumerate(self.get_words()):
            not_done_ids.append(streaming_actors[i % num_cpus].add_sub_word.remote(word))

        while not_done_ids:
            if len(not_done_ids) % 1000 == 0:
                print(len(not_done_ids))
            done_ids, not_done_ids = ray.wait(not_done_ids, timeout=1)

        results = []
        not_done_ids = []
        for index, actor in enumerate(streaming_actors):
            not_done_ids.append(actor.create_sub_influence_graph.remote(index, self.language))
        
        while not_done_ids:
            done_ids, not_done_ids = ray.wait(not_done_ids, timeout=1)
            results.extend(ray.get(done_ids))
        

        for sub_vertices in results:
            for word in self.get_words():
                self.vertices[word].neighbors = self.vertices[word].neighbors.union(sub_vertices[word].neighbors)

        ray.shutdown()

class KNNGraph(Graph):
    def __init__(self):
        super().__init__()

    @classmethod
    def from_word2vec(grp, word2vec, knn, k=16):
        graph = super().from_word2vec(word2vec, k)
        graph.create_KNN_Graph(knn)
        return graph

    def create_KNN_Graph(self, k):
        self.knn(k+1)
        for node in self.get_nodes():
            for nbr_word in self.nrst_nbrs[node.word]:
                if nbr_word == node.word:
                    continue
                nbr_node = self.vertices[nbr_word]
                node.add_neighbor(nbr_word)
                nbr_node.add_neighbor(node.word)

class RNGGraph(GabrielGraph):
    def __init__(self):
        super().__init__()

    @classmethod
    def from_word2vec(grp, word2vec, num_cpus=1, k=16):
        graph = super().from_word2vec(word2vec, num_cpus=num_cpus, k=k)
        graph.create_RNG_Graph()
        return graph
    
    @classmethod
    def from_gabriel_graph(grp, gabriel_graph, k=16):
        graph = grp()

        graph.vertices = gabriel_graph.vertices
        graph.vecs = gabriel_graph.vecs
        graph.word2num = gabriel_graph.word2num
        graph.num2word = gabriel_graph.num2word
        graph.nrst_nbrs = gabriel_graph.nrst_nbrs
        graph.create_RNG_Graph()
        return graph
        
                        
    def create_RNG_Graph(self):
        edges_to_remove = []
        for word1 in tqdm(self.get_words()):
            node1 = self.vertices[word1]
            for word2 in node1.neighbors:
                node2 = self.vertices[word2]
                for other_word in chain(self.nrst_nbrs[word1], self.nrst_nbrs[word2]):
                    other_node = self.vertices[other_word]
                    dist = node1.dist_to(node2)
                    if max(node1.dist_to(other_node), node2.dist_to(other_node)) < dist:
                        edges_to_remove.append((word1, word2))

        for word1, word2 in edges_to_remove:
            self.vertices[word1].remove_neighbor(word2)
            self.vertices[word2].remove_neighbor(word1)
