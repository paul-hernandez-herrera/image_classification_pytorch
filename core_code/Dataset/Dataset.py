from ..util import util
from torch.utils.data import Dataset
from torch.nn.functional import one_hot
from pathlib import Path
import numpy as np
from torch import tensor
from ..util.preprocess import preprocess_image
from torchvision.transforms import Resize
import warnings

class CustomImageDataset(Dataset):
    """
    The purpose of this class is to load input and target image pairs from two separate directories, 
    preprocess the input image if required, and return them as a tuple of PyTorch tensors.    
    """
    def __init__(self, list_folders, enable_preprocess = False):
        valid_suffix = {".tif", ".tiff"}
        self.enable_preprocess = enable_preprocess        
        self.data_augmentation_flag = False
        self.list_folders = list_folders
        #reading all files in target folder
        
        self.path_files = []
        self.labels = np.zeros((0,1))
        for idx, folder_path in enumerate(list_folders):
            current_image_paths = [p for p in Path(folder_path).iterdir() if p.suffix in valid_suffix] 
            self.path_files += current_image_paths
            self.labels = np.vstack((self.labels, idx * np.ones((len(current_image_paths), 1))))
        
        #check if we need to resize images to a commun shape
        self.resize = True ## setting resize to True, since model ResNet50 requires image shape (224,224)
        if not self.resize:
            self.resize = not check_training_set_equal_img_sizes(self.path_files)
        
        self.input_size = np.array([224, 224]) if self.resize else util.imread(self.path_files[idx]).shape[-2:]
        self.transform_resize = Resize((int(self.input_size[0]), int(self.input_size[1])), antialias='True') if self.resize else None
        
        self.num_classes = len(self.list_folders)
        
    def __len__(self):
        return len(self.path_files)
    
    def __n_classes__(self):
        return len(self.list_folders)
        
    def __getitem__(self, idx):
        #reading input and target images
        input_img = util.imread(self.path_files[idx])
        target = self.labels[idx]
        
        #preprocess image if required
        if self.enable_preprocess:
            input_img = preprocess_image(input_img)
            
        #converting numpy to tensor
        input_img = tensor(input_img.astype(np.float32)).float()
        target = tensor(target).long()
        
        #converting targets to one_hot encoding. [1,C]. Loss functions requires [C,1]
        target = one_hot(target, num_classes= self.num_classes).permute(1,0)
        
        if self.resize:
            input_img = self.transform_resize(input_img)


        #U-Net requires dimensions to be [C,W,H]. Make sure that we have Channel dimension
        input_img = input_img.unsqueeze(0) if input_img.dim() == 2 else input_img
            
        
        if self.data_augmentation_flag:
            input_img = self.data_augmentation_object.run(input_img)
        
        
        return input_img, target


    def set_data_augmentation(self, augmentation_flag = False, data_augmentation_object = None):
        """
        this method is used to set a data augmentation flag and object. 
        The data_augmentation_flag is a boolean indicating whether data augmentation should be performed or not
        data_augmentation_object is an object containing the data augmentation methods to be applied.
        """
        self.data_augmentation_flag = augmentation_flag
        self.data_augmentation_object = data_augmentation_object
        if not(augmentation_flag):
            self.data_augmentation_object = None
        
        

def check_training_set_equal_img_sizes(path_files):
    #verify that every image has the same shape
    img_shape = util.imread(path_files[0]).shape
    
    for f in path_files:
        current_img_shape = util.imread(f).shape
        if current_img_shape!=img_shape:
            warnings.warn("Training set have images with different shape. Resizing to default size: (224x224)")
            return False
    return True