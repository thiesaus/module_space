import matplotlib.pyplot as plt 
import os
import numpy as np

class Visualize:
    def __init__(self):
        self.losses={
            "mse_loss": [],
            "mae_loss": []
        }
        self.num_epoch=len(self.losses["mse_loss"])

        self.path="./figs/training_loss_{}.png"
    def add_loss(self,loss):
        self.losses["mse_loss"].append(loss["mse_loss"].item())
        self.losses["mae_loss"].append(loss["mae_loss"].item())
        self.num_epoch=len(self.losses["mse_loss"])

    def plot_loss(self):
        plt.plot(np.array(range(self.num_epoch)),np.array(self.losses["mse_loss"]),label="mse_loss")
        plt.plot(np.array(range(self.num_epoch)),np.array(self.losses["mae_loss"]),label="mae_loss")
        plt.legend()
        
        plt.grid()
        plt.show()
        
        if not os.path.exists("./figs"):
            os.makedirs("./figs")
        plt.savefig(self.path.format(self.num_epoch))
        plt.clf()
        plt.close()
    