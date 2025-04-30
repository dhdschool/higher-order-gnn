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
        """Data object for the Epinions or the Ciao dataset

        Args:
            dir (os.PathLike): The path of the directory containg the categories.txt, rating.mat, and trustnetwork.mat files
        """
        rating = mat4py.loadmat(str(dir / 'rating.mat'))
        self.rating = np.array(rating['rating'])
        
        trustnetwork = mat4py.loadmat(str(dir / 'trustnetwork.mat'))
        self.trustnetwork = np.array(trustnetwork['trustnetwork'])
        
        categories = self._read_categories(dir / 'categories.txt')
        self.categories = pd.DataFrame.from_dict(categories, orient='index').rename(columns={0:'Category'})
                
        self.person_vertices = np.unique(np.concatenate([self.trustnetwork[:, 0], self.trustnetwork[:, 1], self.rating[:, 0]], axis=0))
        self.items_vertices = np.unique(self.rating[:, 1])

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
        
        self.complex = tnx.CombinatorialComplex()
        for vertice in reader.person_vertices:
            self.complex.add_cell(vertice, rank=0)
        
        for edge in reader.trustnetwork:
            self.complex.add_cell(edge, rank=1)

        cliques = ig.Graph(edges = reader.trustnetwork).cliques(min=3, max=max_two_cell_size)
        for cell in cliques:
            self.complex.add_cell(cell, rank=2)
            
            
def _testing(dataset_dir: os.PathLike):
    """Testing suite for this file

    Args:
        dataset_dir (os.PathLike): Directory of the dataset we are testing
    """
    
    reader_start_time = time()
    reader = Reader(dataset_dir)
    categories = np.unique(reader.rating[:, 2])
    rating_vals = np.unique(reader.rating[:, 3])
            
    print(f"Categories: {reader.categories['Category'].to_list()}")
    print(f"Category index numbers: {categories}")
    print(f"Rating values in dataset: {rating_vals}")
    print()
    print(f"Number of item vertices present: {len(reader.items_vertices)}")
    print(f"Number of item-people edges present: {len(reader.rating)}")
    print(f"Number of person vertices: {len(reader.person_vertices)}")
    print(f"Number of person-person edges present: {len(reader.trustnetwork)}")
    print()
    
    reader_end_time = time()
    print(f"Time to load reader: {round(reader_end_time - reader_start_time, 4)}s")
    
    method1_start_time = time()
    dataset = TnnDataset(reader)
    method1_end_time = time()
    print(f"Time to load three rank complex: {round(method1_end_time - method1_start_time, 4)}s")
    
if __name__ == '__main__':
    _testing(CIAO_DIR)
    