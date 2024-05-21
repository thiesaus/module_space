import matplotlib.pyplot as plt 
import os
import numpy as np

class Visualize:
    def __init__(self):
        self.losses={
            "cross_image_text": [],
            "cross_text_image": []
        }
        self.num_epoch=len(self.losses["cross_image_text"])

        self.path="./figs/training_loss_{}.png"
    def add_loss(self,loss):
        self.losses["cross_image_text"].append(loss["cross_image_text"].item())
        self.losses["cross_text_image"].append(loss["cross_text_image"].item())
        self.num_epoch=len(self.losses["cross_image_text"])

    def plot_loss(self):
        plt.plot(np.array(range(self.num_epoch)),np.array(self.losses["cross_image_text"]),label="cross_image_text")
        plt.plot(np.array(range(self.num_epoch)),np.array(self.losses["cross_text_image"]),label="cross_text_image")
        plt.legend()
        
        plt.grid()
        plt.show()
        
        if not os.path.exists("./figs"):
            os.makedirs("./figs")
        plt.savefig(self.path.format(self.num_epoch))
        plt.clf()
        plt.close()
    