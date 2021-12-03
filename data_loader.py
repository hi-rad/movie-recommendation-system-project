from torch.utils.data import Dataset as TorchDataset
import pandas as pd
import numpy as np
import os


class Dataset(TorchDataset):
    """Characterizes a dataset for PyTorch"""

    def __init__(self, files_directory, file, users_list, movie_mappings_dict):
        """Initialization"""
        self.file = os.path.join(files_directory, file)
        self.users_list = users_list
        self.movie_mappings_dict = movie_mappings_dict

    def __len__(self):
        """Denotes the total number of samples"""
        return len(self.users_list)

    def __getitem__(self, ind):
        """Generates one sample of data"""
        user_id = self.users_list[ind]
        user_item_interaction = np.full(len(self.movie_mappings_dict), 0.0)
        user_ratings = pd.read_csv(os.path.join(self.file, str(user_id) + '.csv'))
        if len(user_ratings) > 0:
            movie_indices = [self.movie_mappings_dict[movie_id] for movie_id in user_ratings['movieId']]
            rating_values = [rating for rating in user_ratings['rating']]
            np.put(user_item_interaction, movie_indices, rating_values)
        return user_item_interaction