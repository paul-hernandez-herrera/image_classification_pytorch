from torchvision.transforms.functional import affine, hflip, vflip
from torchvision.transforms import Resize
import numpy as np

class augmentation_classification_task():
    def __init__(self, zoom_range = [0.8, 1.2],
                 shear_angle = [-5, 5], 
                 img_resize = [224,224],
                 enable_shear = True, 
                 enable_hflip = True, 
                 enable_vflip = True, 
                 enable_zoom = True,
                 enable_resize = False):
        self.zoom_range = zoom_range
        self.shear_angle = shear_angle
        
        #flag to compute specific transformations
        self.enable_shear = enable_shear
        self.enable_hflip = enable_hflip
        self.enable_vflip = enable_vflip
        self.enable_zoom = enable_zoom
        self.enable_resize = enable_resize
        
        self.transform_resize = Resize((img_resize[0],img_resize[1]), antialias='True')
        
    def horizontal_flip(self, image):        
        #random horizontal flip
        if np.random.uniform(0, 1) > 0.5:
            image = hflip(image)
        return image
    
    def vertical_flip(self, image):        
        #random vertical flip    
        if np.random.uniform(0, 1) > 0.5:
            image = vflip(image)
        return image 
    
    def affine_transform(self, image, scale=1, angle=0, translate=[0, 0], shear=0):
        image = affine(image, scale=scale, angle=angle, translate=translate, shear=shear)
        return image    
    
    def affine_zoom(self, image):        
        #random zoom
        if np.random.uniform(0, 1) > 0.5:
            zoom = np.random.uniform(*self.zoom_range)
            image = self.affine_transform(image, scale=zoom)
        return image
    
    def affine_shear(self, image):        
        #random shear
        if np.random.uniform(0, 1) > 0.5:
            shear = np.random.uniform(*self.shear_angle)
            image = self.affine_transform(image, shear=shear)
        return image
        
    def run(self, image):
        if self.enable_resize:
            image = self.transform_resize(image)
        
        if self.enable_hflip:
            image = self.horizontal_flip(image)
            
        if self.enable_vflip:
            image = self.vertical_flip(image)
            
        if self.enable_zoom:
            image = self.affine_zoom(image)
            
        if self.enable_shear:
            image = self.affine_shear(image)
        
        return image   