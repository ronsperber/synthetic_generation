from typing import Type, Callable
import torch
import torch.nn as nn
from .models import Generator, Discriminator
from .training import train_gan, train_wgan_gp 
from .model_saving import load_gan_checkpoint, save_gan_checkpoint

class BaseGanProcess:
    def __init__(self, G:Generator, D:Discriminator):
        self.G = G
        self.D = D
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.G.to(self.device)
        self.D.to(self.device)

    def generate_samples(self, num_samples, c=None):
        with torch.no_grad():
            return self.G.generate_samples(num_samples=num_samples, c=c).cpu()

    def save(
            self,
            path: str,
            training_configs : dict | None = None
    ):
        save_gan_checkpoint(save_path=path, G=self.G, D=self.D, training_configs=training_configs)

    @classmethod
    def load(cls,
             path: str,
             G_cls: Type[nn.Module] = Generator,
             D_cls: Type[nn.Module] = Discriminator,
             map_location: Callable | str | dict | None = None
    ):
        G,D,_ = load_gan_checkpoint(
            path=path,
            generator_cls=G_cls,
            discriminator_cls=D_cls,
            map_location=map_location
        )
        return cls(G, D)
        
class GANProcess(BaseGanProcess):
    def train(self, X, c=None, **train_args):
        train_gan(G=self.G, D=self.D, X=X, c=c, **train_args)


class WGANProcess(BaseGanProcess):
    def train(self, X, c=None, **train_args):
        train_wgan_gp( G=self.G, D=self.D, X=X, c=c, **train_args)
