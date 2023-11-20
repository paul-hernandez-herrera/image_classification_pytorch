import matplotlib.pyplot as plt
import numpy as np
from ..parameters_interface.ipwidget_basic import set_dropdown
from ipywidgets import widgets
from core_code.parameters_interface.ipwidget_basic import set_IntSlider, set_checkbox
import napari
from IPython.display import display
from .util import imread
from pathlib import Path

########################################################################################################
########################################################################################################

def show_images_from_Dataset(custom_dataset, n_images_to_display=3):
    # Create figure with subplots for each image and its channels
    n_channels = custom_dataset[0][0].shape[0]
    fig, axs = plt.subplots(n_images_to_display, n_channels, figsize=(10, 20), dpi=80)
    
    for i in range(n_images_to_display):
        img, label = custom_dataset[np.random.randint(len(custom_dataset))]
        
        img, label = img.numpy(), label.numpy()
        img = convert_img_shape_to_C_W_H(img)
        label = np.argmax(label)
        
        for j in range(n_channels):
            axs[i,j].imshow(img[j], cmap='gray')
            axs[i,j].set_title(f'Image {i} - Ch{j} ---- LABEL = {label}')
            axs[i,j].axis('off')
            fig.colorbar(axs[i,j].imshow(img[j], cmap='gray'), ax=axs[i,j])
        
    plt.tight_layout()
    plt.show()    
    
########################################################################################################
########################################################################################################
    
def convert_img_shape_to_C_W_H(img):
    if img.ndim == 4:
        img = np.max(img, axis=1)
    return img
    
########################################################################################################
########################################################################################################

def show_images_predicted_class_interactive(img_file_paths, predicted_class):
    global main_container
    main_container = widgets.HBox()
    
    dropdown_options = [(Path(file_name).name, str(idx)) for idx, file_name in enumerate(img_file_paths)] 
    
    dropdown_w = set_dropdown('Image to show: ', dropdown_options)

    def dropdown_handler(change):
        index = int(change.new)
        show_image(index)      
    
    def show_image(index):        
        img  = imread(img_file_paths[index])
        label_class = str(predicted_class[index])

        try:
            show_image_napari(img, label_class)            
        except:
            main_container.close()
            plt_imshow(img, label_class) 
            
    def show_image_napari(img, label_class):
        viewer = napari.Viewer(title = f"Class predicted = {label_class}")
        viewer.add_image(img)
            
    def plt_imshow(image, label_class):
        if (len(image.shape) not in [2]) or (len(image.shape) == 3 and image.shape[0] not in [3]):
            raise ValueError('Can not display current image.')            
            
        plt.imshow(image, cmap='gray')
        plt.axis('off')
        plt.title(f"Class predicted = {label_class}") 
        plt.show()

    dropdown_w.observe(dropdown_handler, names='value')    
    show_image(0)   