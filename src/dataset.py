from pathlib import Path
import pandas as pd
import numpy as np
import mat4py
import os
import toponetx as tnx

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


def _testing(dataset_dir: os.PathLike):
    """Testing suite for this file

    Args:
        dataset_dir (os.PathLike): Directory of the dataset we are testing
    """
    dataset = Reader(dataset_dir)
    categories = np.unique(dataset.rating[:, 2])
    rating_vals = np.unique(dataset.rating[:, 3])
        
    all_person_vertices = np.unique(np.concatenate([dataset.trustnetwork[:, 0], dataset.trustnetwork[:, 1], dataset.rating[:, 0]], axis=0))
    items_vertices = np.unique(dataset.rating[:, 1])
    
    print(f"Categories: {dataset.categories['Category'].to_list()}")
    print(f"Category index numbers: {categories}")
    print(f"Rating values in dataset: {rating_vals}")
    print()
    print(f"Number of item vertices present: {len(items_vertices)}")
    print(f"Number of item-people edges present: {len(dataset.rating)}")
    print(f"Number of person vertices: {len(all_person_vertices)}")
    print(f"Number of person-person edges present: {len(dataset.trustnetwork)}")
if __name__ == '__main__':
    _testing(CIAO_DIR)
