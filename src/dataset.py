from pathlib import Path
import pandas as pd
import numpy as np
import mat4py
import os

EPINION_DIR = Path('data/epinions') # Epinions dataset directory path
CIAO_DIR = Path('data/ciao')        # Ciao dataset direcotyr path


class Dataset:
    
    
    def __init__(self, dir: os.PathLike):
        """Dataset object for the Epinions or the Ciao dataset

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
            while (data:=categories_file.readline()) != '':
                index, value = data.split(' ', maxsplit=1)
                index = int(index)
                categories[index] = value.rstrip()
        return categories
    
if __name__ == '__main__':
    epinions = Dataset(CIAO_DIR)
    print(np.unique(epinions.rating[:, 2]))
    print(np.unique(epinions.rating[:, 3]))