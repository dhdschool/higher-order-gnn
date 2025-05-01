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
        
        if not os.path.exists(dir / 'item_edges.npy'):
            rating = mat4py.loadmat(str(dir / 'rating.mat'))
            self.item_edges = np.array(rating['rating'])
            np.save(dir / 'item_edges.npy', self.item_edges)
        else:
            self.item_edges = np.load(dir / 'item_edges.npy')

        if not os.path.exists(dir / 'person_edges.npy'):
            trustnetwork = mat4py.loadmat(str(dir / 'trustnetwork.mat'))
            self.person_edges = np.array(trustnetwork['trustnetwork'])
            np.save(dir / 'person_edges.npy', self.person_edges)
        else:
            self.person_edges = np.load(dir / 'person_edges.npy')
        
        if not os.path.exists(dir / 'person_vertices.npy'):
            self.person_vertices = np.unique(np.concatenate([self.person_edges[:, 0], self.person_edges[:, 1], self.item_edges[:, 0]], axis=0))
            np.save(dir / 'person_vertices.npy', self.person_vertices)
        else:
            self.person_vertices = np.load(dir / 'person_vertices.npy')
            
        if not os.path.exists(dir / 'item_vertices.npy'):
            self.item_vertices = np.unique(self.item_edges[:, 1])
            np.save(dir / 'item_vertices.npy', self.item_vertices)
        else:
            self.item_vertices = np.load(dir / 'item_vertices.npy')        

        categories = self._read_categories(dir / 'categories.txt')
        self.categories = pd.DataFrame.from_dict(categories, orient='index').rename(columns={0:'Category'})
        
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
        self.complex = tnx.CombinatorialComplex()
        
        # 0-cells (people)
        for vertice in reader.person_vertices:
            self.complex.add_cell(vertice, rank=0)
        
        # 1-cells (connections)
        for edge in reader.person_edges:
            self.complex.add_cell(edge, rank=1)

        # 2-cells (cliques)
        self.cliques = self._load_cliques(max_two_cell_size)
        for arr in self.cliques:
            for cell in arr:
                self.complex.add_cell(cell, rank=2)
                
    def _load_cliques(self, max_size: int):
        cliques = []
        for size in range(3, max_size + 1):
            if not os.path.exists(self.reader.dir / f'person_cliques_{size}.npy'):
                clique = ig.Graph(edges = self.reader.person_edges).cliques(min=size, max=size)
                if len(clique) == 0: return cliques
                clique = np.vstack(clique)
                np.save(self.reader.dir / f'person_cliques_{size}.npy', clique)
            else:
                clique = np.load(self.reader.dir / f'person_cliques_{size}.npy')
            cliques.append(clique)
        return cliques
        
            
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
    dataset = TnnDataset(reader, max_two_cell_size=14)
    print(f"Total number of cliques: {sum(map(lambda x: len(x), dataset.cliques))}")
    for idx, clique in enumerate(dataset.cliques):
        print(f"Number of cliques of size {idx + 3}: {len(clique)}")
    method1_end_time = time()
    print(f"Time to load three rank complex: {round(method1_end_time - method1_start_time, 4)}s")
    
if __name__ == '__main__':
    _testing(CIAO_DIR)
    