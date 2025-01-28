import os
import sys
from torch.utils.data import  Dataset
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from auditing.feature_extractor import FeatureExtractor


class BasicModelSet(FeatureExtractor, Dataset):
    def __init__(self, args, num_models, data_type="target") -> None:
        super(BasicModelSet, self).__init__(args, num_models, data_type)

    def __getitem__(self, index):    
        dirname = self.model_dir_set[index]
        label = self.labels[index]
        return dirname, label
    
    def __len__(self):
        return len(self.labels)
    