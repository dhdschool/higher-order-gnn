from pathlib import Path
import pandas as pd
import numpy as np
import mat4py
import os
import toponetx as tnx
import torch_geometric
from time import time
import igraph as ig

EPINION_DIR = Path('data/epinions') # Epinions dataset directory path
CIAO_DIR = Path('data/ciao')        # Ciao dataset direcotyr path


class Reader:
    
    
    def __init__(self, dir: os.PathLike):
        """Data object for the Epinions or the Ciao dataset. Writes data to file in numpy format if not present.

        Args:
            dir (os.PathLike): The path of the directory containg the categories.txt, rating.mat, and trustnetwork.mat files
        """
        self.dir = dir
        
        self.item_edges = self._load_item_edges()
        self.person_edges = self._load_person_edges()
        
        self.person_vertices = self._load_person_vertices()
        self.item_vertices = self._load_item_vertices()

        categories = self._read_categories(dir / 'categories.txt')
        self.categories = pd.DataFrame.from_dict(categories, orient='index').rename(columns={0:'Category'})
    
    def _load_item_edges(self):
        if not os.path.exists(dir / 'item_edges.npy'):
            rating = mat4py.loadmat(str(dir / 'rating.mat'))
            item_edges = np.array(rating['rating'])
            np.save(dir / 'item_edges.npy', item_edges)
        else:
            item_edges = np.load(dir / 'item_edges.npy')
        return item_edges
    
    def _load_person_edges(self):
        if not os.path.exists(dir / 'person_edges.npy'):
            trustnetwork = mat4py.loadmat(str(dir / 'trustnetwork.mat'))
            person_edges = np.array(trustnetwork['trustnetwork'])
            np.save(dir / 'person_edges.npy', person_edges)
        else:
            person_edges = np.load(dir / 'person_edges.npy')
        return person_edges
    
    def _load_person_vertices(self, person_edges=None, item_edges=None):
        if person_edges is None: person_edges = self.person_edges
        if item_edges is None: item_edges = self.item_edges
        
        if not os.path.exists(dir / 'person_vertices.npy'):
            person_vertices = np.unique(np.concatenate([person_edges[:, 0], person_edges[:, 1], item_edges[:, 0]], axis=0))
            np.save(dir / 'person_vertices.npy', person_vertices)
        else:
            person_vertices = np.load(dir / 'person_vertices.npy')
            
        return person_vertices

    def _load_item_vertices(self, item_edges=None):
        if item_edges is None: item_edges = self.item_edges
        
        if not os.path.exists(dir / 'item_vertices.npy'):
            item_vertices = np.unique(item_edges[:, 1])
            np.save(dir / 'item_vertices.npy', item_vertices)
        else:
            item_vertices = np.load(dir / 'item_vertices.npy')  
    
        return item_vertices
        
    def _read_categories(self, fp: os.PathLike):
        """Internal method for reading the categories.txt file as a dictionary

        Args:
            fp (os.PathLike): The file path of the categories file

        Returns:
            dict: Indices based dictionary object mapping from index to category
        """
        categories = {}
        with open(fp, mode='r') as categories_file:
            while (data:=categories_file.readline()) != '': # While not EOF
                index, value = data.split(' ', maxsplit=1)
                index = int(index)
                categories[index] = value.rstrip() # Remove newline on end of string
        return categories


class TnnDataset(torch_geometric.data.Dataset):
    def __init__(self, reader: Reader, max_two_cell_size=3):
        super().__init__()
        
        self.reader: Reader = reader

        self.x0 = reader.person_vertices
        self.x1 = reader.person_edges
        self.x2, self.x2_groups = self._load_cliques(max_two_cell_size)
        
        self.x0_attr = reader.item_edges[:, :3]
        self.y = reader.item_edges[:, 3]

    def _load_cliques(self, max_size: int):
        """Generates/loads all cliques up to the maximum size from the data and saves them to disk if not present.

        Args:
            max_size (int): The maximum clique size that will be generated

        Returns:
            List, int: Returns a list and the size of the list. The list contains groups of the same size that have edges between all vertices in that group
        """
        cliques = []
        for size in range(3, max_size + 1):
            if not os.path.exists(self.reader.dir / f'person_cliques_{size}.npy'):
                clique = ig.Graph(edges=self.reader.person_edges).cliques(min=size, max=size)
                if len(clique) == 0: return cliques, size
                clique = np.vstack(clique)
                np.save(self.reader.dir / f'person_cliques_{size}.npy', clique)
            else:
                clique = np.load(self.reader.dir / f'person_cliques_{size}.npy')
            cliques.append(clique)
        return cliques, size     
    
    def _create_cc(self):
        complex = tnx.CombinatorialComplex()
        
        # 0-cells (people)
        for vertice in self.x0:
            complex.add_cell(vertice, rank=0)
            
        # 1-cells (connections)
        for edge in self.x1:
            complex.add_cell(edge, rank=1)

        # 2-cells (cliques)
        for arr in self.x2:
            for cell in arr:
                complex.add_cell(cell, rank=2)
                
        return complex
    
    
def _testing(dataset_dir: os.PathLike):
    """Testing suite for this file

    Args:
        dataset_dir (os.PathLike): Directory of the dataset we are testing
    """
    
    reader_start_time = time()
    reader = Reader(dataset_dir)
    categories = np.unique(reader.item_edges[:, 2])
    rating_vals = np.unique(reader.item_edges[:, 3])
            
    print(f"Categories: {reader.categories['Category'].to_list()}")
    print(f"Category index numbers: {categories}")
    print(f"Rating values in dataset: {rating_vals}")
    print()
    print(f"Number of item vertices present: {len(reader.item_vertices)}")
    print(f"Number of item-people edges present: {len(reader.item_edges)}")
    print(f"Number of person vertices: {len(reader.person_vertices)}")
    print(f"Number of person-person edges present: {len(reader.person_edges)}")
    
    reader_end_time = time()
    print(f"Time to load reader: {round(reader_end_time - reader_start_time, 4)}s")
    print()
    
    method1_start_time = time()
    dataset = TnnDataset(reader, max_two_cell_size=3)
    print(f"Total number of cliques: {sum(map(lambda x: len(x), dataset.cliques))}")
    for idx, clique in enumerate(dataset.cliques):
        print(f"Number of cliques of size {idx + 3}: {len(clique)}")
    method1_end_time = time()
    print(f"Time to load dataset: {round(method1_end_time - method1_start_time, 4)}s")
    
    
if __name__ == '__main__':
    _testing(CIAO_DIR)
    