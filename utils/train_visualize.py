import matplotlib.pyplot as plt 

class Visualize:
    def __init__(self):
        self.num_epoch=0
        self.losses={
            "mse_loss": [],
            "mae_loss": []
        }
        self.path="../figs/training_loss_{}.png"
    def add_loss(self,loss):
        self.losses["mse_loss"].append(loss["mse_loss"])
        self.losses["mae_loss"].append(loss["mae_loss"])
        self.num_epoch+=1

    def plot_loss(self):
        plt.plot(range(self.num_epoch),self.losses["mse_loss"],label="mse_loss")
        plt.plot(range(self.num_epoch),self.losses["mae_loss"],label="mae_loss")
        plt.legend()
        plt.show()
    
    def save_plot(self,path):
        plt.plot(range(self.num_epoch),self.losses["mse_loss"],label="mse_loss")
        plt.plot(range(self.num_epoch),self.losses["mae_loss"],label="mae_loss")
        plt.legend()
        plt.savefig(path.format(self.num_epoch))