"""
module that holds GAN processes (Classes to hold models and train/save)
"""
from typing import Type, Callable
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from .models import Generator, Discriminator
from .training import train_gan, train_wgan_gp 
from .model_saving import load_gan_checkpoint, save_gan_checkpoint

class BaseGanProcess:
    """
    Base class for GAN processes
    """
    def __init__(self, G:Generator, D:Discriminator):
        """
        Create process with Generator and Discriminator/Critic
        G : Generator
            Generator model
        D: Discriminator
            Discriminator/Critic model
        """
        self.G = G
        self.D = D
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.G.to(self.device)
        self.D.to(self.device)

    def generate_samples(self, num_samples: int, c: torch.Tensor | None = None) -> torch.Tensor:
        """
        Generate samples from model
        Parameters
        ---------
        num_samples : int
            number of samples to produce
        c: torch.Tensor | None
            when the model uses a conditional, conditional used to create samples
        """
        with torch.no_grad():
            return self.G.generate_samples(num_samples=num_samples, c=c).cpu()

    def save(
            self,
            path: str,
            training_configs : dict | None = None
    ):
        """
        saving a process
        Parameters
        ----------
        path: str
            location to save
        training_configs: dict | None
            optional dictionary of training configs
        """
        save_gan_checkpoint(save_path=path, G=self.G, D=self.D, training_configs=training_configs)

    @classmethod
    def load(cls,
             path: str,
             G_cls: Type[nn.Module] = Generator,
             D_cls: Type[nn.Module] = Discriminator,
             map_location: Callable | str | dict | None = None
    ):
        """
        class method to create a class from a saved checkpoint
        path : str
            where to load the class from
        G_cls : Type[nn.Module]:
            class for the Generator
        D_cls : Type[nn.Module]:
            class for the Discriminator/Critic
        map_location: Callable | str | dict | None
            information used for location for torch.load
        """
        G,D,_ = load_gan_checkpoint(
            path=path,
            generator_cls=G_cls,
            discriminator_cls=D_cls,
            map_location=map_location
        )
        return cls(G, D)
        
class GANProcess(BaseGanProcess):
    """
    class for GAN process
    """
    def train(self, X: torch.Tensor | DataLoader, c: torch.Tensor | None = None, **train_args):
        """
        method to train the process on given data
        and storing the training configs

        Parameters
        ----------
        X: torch.Tensor | DataLoader
            data to be trained on
        c : torch.Tensor | None
            optional conditional tensor
        **train_args:
            arguments used for training
        """
        self.train_configs = train_gan(G=self.G, D=self.D, X=X, c=c, return_configs=True, **train_args)

    def train_save(self, save_path: str, X: torch.Tensor | DataLoader,c: torch.Tensor | None = None, **train_args):
        """
        method to train and save the process
        Parameters
        save_path: str
            where to save the data
        X : torch.Tensor | DataLoader
            data to be trained on
        c: torch.Tensor | None
            optional conditional tensor
        **train_args
            additional arguments to use in training
        """
        history = self.train(X=X, c=c, **train_args)
        self.save(path=save_path, training_configs=self.train_configs)
        return history

class WGANProcess(BaseGanProcess):
    """
    WGAN process class
    """
    def train(self, X:torch.Tensor | DataLoader, c: torch.Tensor | None = None, **train_args):
        """
        training the Generator and Discriminator on a dataset
        and saving the training configs
        Parameters
        ---------
        X : torch.Tensor | DataLoader
            data used to train
        c : torch.Tensor | None
            optional conditional tensor
        **train_args:
            additional arguments for training
        """
        self.train_configs = train_wgan_gp( G=self.G, D=self.D, X=X, c=c, return_configs=True, **train_args)

    def train_save(self, save_path: str, X: torch.Tensor,c: torch.Tensor | None = None, **train_args):
        """
        training the Generator and Discriminator on a dataset
        and saving the training configs and saving the process
        Parameters
        ---------
        save_path: str
            where to save the process
        X : torch.Tensor | DataLoader
            data used to train
        c : torch.Tensor | None
            optional conditional tensor
        **train_args:
            additional arguments for training
        """
        history = self.train(X=X, c=c, **train_args)
        self.save(path=save_path, training_configs=self.train_configs)
        return history