import sys
sys.path.append('../Classification')
import ClassifierModel

import matplotlib.pyplot as plt
from vis.visualization import visualize_activation
from vis.utils import utils
from keras import activations
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (18,6)

import numpy as np
import matplotlib.cm as cm
from vis.visualization import visualize_cam


class MyVisualization():

    def __init__(self, model_builder, classes = 2):

        self.new_model = model_builder.get_transferred_model('../../Models/vgg768trunkcropped-16-acc 0.86-loss 0.8742.hdf5', classes)

        # Utility to search for layer index by name. 
        # Alternatively we can specify this as -1 since it corresponds to the last layer.
        layer_idx = -1 #utils.find_layer_idx(new_model, 'dense_3')

        # Swap softmax with linear
        self.new_model.layers[layer_idx].activation = activations.linear
        self.new_model = utils.apply_modifications(self.new_model)


    def plot_gradCAM(self, img_path, width, height, layer_idx=-1, title="", modifier=None, filter_indices=0, alpha=0.2): # filter_indices=0 -> class=infested; filter_indices=1 -> class=healthy
        img = utils.load_img(img_path, target_size=(width, height))

        grads = visualize_cam(self.new_model, layer_idx, filter_indices=filter_indices, seed_input=img, backprop_modifier=modifier)
        f, ax = plt.subplots(1, 2, figsize=(20,10))

        ax[0].imshow(img)
        ax[0].imshow(grads,cmap='jet', alpha = alpha)
        plt.title(title)
        ax[1].imshow(img)
        plt.show()
    