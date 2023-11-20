from .util import util
import numpy as np
from pathlib import Path
import torch
from .util.deeplearning_util import load_model
from .parameters_interface import ipwidget_basic
from .parameters_interface.parameters_widget import parameters_device
from .util.preprocess import preprocess_image
from torchvision.transforms import Resize

def predict_model(input_path, model= None, model_path = None, output_folder=None, device = 'cpu', enable_preprocess = False):
    img_file_paths = util.get_image_file_paths(input_path)

    
    # Set output folder path
    output_folder = output_folder or Path(input_path) / 'output'
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    
    model = load_model(model_path = model_path, device = device)
    print(model_path)

    # evaluation of the model
    model.eval()

    resize_img = Resize((224, 224), antialias='True') 

    # we are assuming that model is RESNET50
    n_classes = model.fc.out_features

    # make folders to save the images assigned to each class
    for i in range(0, n_classes):
        Path(output_folder, 'class_' + str(i)).mkdir(parents=True, exist_ok=True)

    
    # Disable gradient calculations during evaluation
    output_predicted_class = []
    with torch.no_grad():  
        for current_img_path in img_file_paths:
            
            # reading image
            input_img = util.imread(current_img_path)

            #preprocess image if required
            if enable_preprocess:
                input_img = preprocess_image(input_img)               

            # convert image to tensor
            input_img = torch.tensor(input_img.astype(np.float32)).float()

            # resize img
            img = resize_img(input_img)
            
            # convert to shape [B,C,W,H]
            img = torch.tensor(input_img).unsqueeze(0).to(device=device)

            # Apply model to input image
            network_output = model(img) 
            
            #output from model is [B,C]. Calculate network labels
            output_class = torch.argmax(torch.sigmoid(network_output), dim=1).cpu().numpy()
            
            # Save output probability map as image
            output_file_path = Path(output_folder,  'class_' + str(output_class[0].astype(int)), current_img_path.name)
            util.imwrite(output_file_path, input_img.cpu().numpy())
            
            print(f"{output_file_path.name}  DONE")
            output_predicted_class.append(output_class)
    return {"inputs": img_file_paths, "predicted_class": output_predicted_class}

class PredictClassInteractive:
    def __init__(self):
        #setting the parameters required to predict images
        # Set parameters required to predict images
        self.model_path_w = ipwidget_basic.set_text('Model path:', 'Insert path here')
        self.folder_path_w = ipwidget_basic.set_text('Folder path:', 'Insert path here')
        self.folder_output_w = ipwidget_basic.set_text('Output path:', 'Insert path here')
        self.device_w = parameters_device()
        
    def run(self):
        device = self.device_w.get_device()
    
        # Predict images and return list of output file paths
        output = predict_model(self.folder_path_w.value, 
                               model_path = self.model_path_w.value, 
                               output_folder = self.folder_output_w.value, 
                               device = device,
                               enable_preprocess = False)
        
        return output

if __name__== '__main__':
    
    #to define
    print('to define command line options')