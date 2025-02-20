import torch 

from collections import abc as abc
from torch.utils.data import Subset



class CustomSubset(Subset):

    def __init__(self, dataset, indices):
        """Customising the Subset for dataset splitting
        The original Subset cannot inherite the attributes of original dataset 
        Therefore we add the function on

        Args:
            dataset (torch.utils.data.dataset): the original dataset
            indices (list): the indices of splitted examples 
        """
        super().__init__(dataset, indices)
        
        # attributes inheritance
        attrs = [attr for attr in dir(dataset) if not attr.startswith('__')]
        for attr in attrs:
            if isinstance(getattr(dataset, attr), abc.Sequence) and not isinstance(getattr(dataset, attr), str):
                setattr(self, attr, getattr(dataset, attr).__class__([getattr(dataset, attr)[i] for i in indices]))
            else:
                setattr(self, attr, getattr(dataset, attr))
 
    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]      
        
    def __len__(self):
        return len(self.indices)


    
def Split_dataset(args, dataset):
    """Split the dataset into two parts (train & validation)

    Args:
        args (Easydict): the config of dataset
        dataset (torch.utils.data.dataset): the original dataset

    Returns:
        CustomSubset: (trainset, valset)
    """
    num_samples = len(dataset)
    n_val = int(args.val_ratio * num_samples)
    print(f"We split {num_samples - n_val} examples for training, {n_val} for on-training validation")
    if n_val == 0:
        return dataset, None

    if args.get('random_split', False):
        ids = torch.randperm(num_samples)
    else:
        # select the images in the end for validation
        ids = torch.arange(num_samples)
        
    trainset = CustomSubset(dataset, ids[:num_samples - n_val])
    valset = CustomSubset(dataset, ids[num_samples - n_val:])
    return trainset, valset


def Check_dataset(dataset):
    """Verify wether the dataset is complete (each examples get an label)

    Args:
        dataset (torch.utils.data.dataset): the original dataset

    Raises:
        ValueError: error info when dataset is incomplete
    """
    num_examples = len(dataset)
    print(f"there are {num_examples} legs for {dataset.mode}")
    
    try:
        for i in range(len(dataset)):
            _ = dataset[i]
    except:
        raise ValueError("the dataset is not complete, please verify whether any image lack of labels")