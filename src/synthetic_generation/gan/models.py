"""
model with classes for Generator and Discriminator (or Critic)
for a GAN/WGAN-GP
"""

from typing import Sequence, TypeAlias
from dataclasses import dataclass
from collections.abc import Callable
import torch
import torch.nn as nn

# types used for the classes
LayerDims = tuple[int, int]
HiddenDims: TypeAlias = LayerDims | Sequence[LayerDims]
ActivationFactory: TypeAlias = Callable[[], nn.Module] | nn.Module

# dataclass for output head info
@dataclass
class OutputHead:
    dim: int | tuple[int,...]  # output dimension(s) for this head, tuple is for potential future use
    activation: ActivationFactory = nn.Identity # activation to be applied during training
    decode : ActivationFactory  = nn.Identity # activation to be applied for inference
    name: str = ""               # optional name to identify this head

class ImageGenerator(nn.Module):
    """
    class for Generator for images in GAN or WGAN
    """

    def __init__(
        self,
        noise_dim: int,
        out_dim: tuple[int,...],
        conv_channels: int | list[int],
        num_conv_layers: int | None = None,
        hidden_activation: ActivationFactory = nn.ReLU,
        output_activation: ActivationFactory = nn.Tanh,
        conditional_dim: int = 0,
    ):
        """
        initialization for image generation class
        Parameters
        ----------
        noise_dim : int
            number of noise dimensions to use as inputs
        out_dim : tuple[int,...]
            size of output dimensions (C, H, W)
            note : currently only square images supported
            resize first if images are not square
        conv_channels : int | list[int]
            number of input convolutional channels for first layer or for all layers when a list
        num_conv_layers : int
            when conv_channels is an int, the number of layers must be specified
        hidden_activation : ActivationFactory
            activation used before output layer
        output_activation: ActivationFactory
            activation used for final layer
        conditional_dim : int
            when not 0, the dimension of the conditional 
        """
        super().__init__()
        self.noise_dim = noise_dim
        self.conditional_dim = conditional_dim
        self.out_dim = out_dim
        if isinstance(conv_channels, int):
            if num_conv_layers is None:
                raise ValueError("Number of convolutional layers must be specified when conv_channels is an int")
            # when an int is given halve the number of input channels each time
            conv_channels = [conv_channels // (2 ** n) for n in range(num_conv_layers)]
        image_size = out_dim[-1] # size of the square that will be output
        if image_size % (2 ** len(conv_channels)) != 0:
            raise ValueError(
                f"image_size {image_size} is not divisible by 2^num_conv_layers "
                f"(2^{len(conv_channels)}={2 ** len(conv_channels)}). "
                f"e.g. use num_conv_layers=2 for 28x28 images."
                )
        initial_size = image_size // (2 ** len(conv_channels))
        self.initial_size = initial_size
        self.initial_channels = conv_channels[0]
        image_channels = out_dim[0] #number of channels in the image
        # construct the output channels
        conv_out_channels = conv_channels[1:] + [image_channels]
        linear_out_size = conv_channels[0] * initial_size ** 2
        self.input_layer = nn.Linear(noise_dim + conditional_dim, linear_out_size)
        self.bn_1 = nn.BatchNorm1d(linear_out_size)
        self.conv_layers = nn.ModuleList()
        self.bn_2d = nn.ModuleList()
        for i, in_channels in enumerate(conv_channels):
            self.conv_layers.append(
                nn.ConvTranspose2d(
                    in_channels=in_channels,
                    out_channels=conv_out_channels[i],
                    kernel_size=4,
                    stride=2,
                    padding=1)
                    )
        for out_channels in conv_out_channels[:-1]:
            self.bn_2d.append(nn.BatchNorm2d(out_channels))
        if callable(hidden_activation) and not isinstance(hidden_activation,nn.Module):
            self.activation = hidden_activation()
        else:
            self.activation = hidden_activation
        if callable(output_activation) and not isinstance(output_activation, nn.Module):
            self.output_activation = output_activation()
        else:
            self.output_activation = output_activation
    
    def forward(self, z: torch.Tensor, c: torch.Tensor | None = None) -> torch.Tensor:
        """
        forward method for the network
        Parameters
        ----------
        z : torch.Tensor
            input noise
        c : torch.Tensor
            optional conditional tensor if there is a conditional
        Returns
        -------
        torch.Tensor
            the result of passing (z,c) through the network
        """
        if self.conditional_dim > 0 and c is None:
            raise ValueError(f"Conditional dimension is {self.conditional_dim} and no conditional was passed.")
        if c is not None:
            x = torch.cat([z, c], dim = 1)
        else:
            x = z
        x = self.input_layer(x)
        x = self.bn_1(x)
        x = self.activation(x)
        B = x.size(0)
        x = x.view(B, self.initial_channels, self.initial_size, self.initial_size)
        for conv, bn in zip(self.conv_layers[:-1], self.bn_2d):
            x = conv(x)
            x = bn(x)
            x = self.activation(x)
        x = self.conv_layers[-1](x)
        x = self.output_activation(x)
        return x

    def generate(self, num_samples: int, c: torch.Tensor | None = None):
        """
        method to generate a 'fake' data set
        Parameters
        ----------
        num_samples : int
            how many samples to be generated
        c : torch.Tensor | None
            when not none, the conditional to be used fot generating data
        Returns
        -------
        torch.Tensor
            a 'fake' dataset of shape (num_samples, *self.output_dim)
        """
        device = next(self.parameters()).device
        # if there is a conditional validate that the length of the conditional
        # matches the number of samples
        if c is not None:
            if c.shape[0] != num_samples:
                raise ValueError(
                    "Number of samples must equal length of conditional input"
                )
            c = c.to(device)
        # generate random noise and output data
        z = torch.randn(num_samples, self.noise_dim, device=device)
        return self.forward(z, c)

    def generate_samples(self, num_samples: int, c: torch.Tensor | None = None):
        """
        function to generate inference samples in eval mode with no gradient tracking
        Parameters
        ----------
        num_samples : int
            number of samples to generate
        c : torch.Tensor | None
            when not None conditional to use to generate the samples
        Returns
        -------
        torch.Tensor
            sample of size num_samples of sample data with post-training decode used
        """
        # save current training status
        training = self.training
        # put in eval to use decode on output heads
        self.eval()
        with torch.no_grad():
            # get the output using decoders in output heads
            output = self.generate(num_samples, c)
        # restore training state
        if training:
            self.train()
        return output.cpu()




class Generator(nn.Module):
    """
    class for Generator in a GAN or WGAN
    """

    def __init__(
        self,
        noise_dim: int,
        num_hidden_layers: int,
        hidden_dims: HiddenDims,
        out_dim: int | None = None,
        output_heads: list[OutputHead] | None = None,  
        hidden_activation: ActivationFactory = nn.LeakyReLU,
        conditional_dim: int = 0,
    ):
        """
        initialization of Generator class
        Parameters
        ----------
        noise_dim : int
            dimension of noise to use as inputs
        num_hidden_layers : int
            number of linear layers between input and output layers
        out_dim : int | None
            dimension of output if no output heads specified
        output_heads : list | None
            output heads to use when specified
        hidden_dims : HiddenDims
            either a single (in_dim, out_dim) tuple reused for all hidden layers
            or a sequence of such tuples specifying each layer explicitly
        hidden_activation : ActivationFactory
            class used for activation after each layer other than the output layer
        conditional_dim : int
            represents the dimension of the conditional, 0 indicates no conditional
        """
        super().__init__()
        # save initialization parameters for recreating model
        self.init_args = {
            "noise_dim": noise_dim,
            "num_hidden_layers": num_hidden_layers,
            "out_dim": out_dim,
            "hidden_dims": hidden_dims,
            "hidden_activation": hidden_activation,
            "output_heads": output_heads,
            "conditional_dim": conditional_dim
        }
        if output_heads is None:
            if out_dim is None:
                raise ValueError("Either out_dim or output_heads must be provided.")
            # Default: single identity head
            output_heads = [
                OutputHead(dim=out_dim, activation=nn.Identity, name="default")
            ]
        if isinstance(hidden_dims, tuple):
            hidden_dims = [hidden_dims] * num_hidden_layers
        # validate length of hidden_dims
        if len(hidden_dims) != num_hidden_layers:
            raise ValueError(
                "Number of hidden layers and length of hidden_dims must be equal"
            )
        # validate that all hidden dims are ordered pairs
        for i, h in enumerate(hidden_dims):
            if not (isinstance(h, tuple) and len(h) == 2):
                raise ValueError(f"hidden_dims[{i} must be a tuple (in_dim, out_dim), got {h}]")
        # validate that output dimension of a layer matches input dimension of next
        for i in range(len(hidden_dims) - 1):
            if hidden_dims[i][1] != hidden_dims[i + 1][0]:
                raise ValueError(
                    f"hidden_dims[{i}][1] ({hidden_dims[i][1]}) "
                    f"!= hidden_dims[{i + 1}][0] ({hidden_dims[i + 1][0]})"
                )
        self.noise_dim = noise_dim
        self.output_dim = out_dim
        if callable(hidden_activation) and not isinstance(hidden_activation,nn.Module):
            self.activation = hidden_activation()
        else:
            self.activation = hidden_activation
        self.conditional_dim = conditional_dim
        self.output_heads = output_heads
        self.input_layer = nn.Linear(
            self.noise_dim + self.conditional_dim, hidden_dims[0][0]
        )
        self.output_dim = sum(head.dim for head in output_heads)
        self.hidden_layers = nn.ModuleList()
        for i in range(num_hidden_layers):
            self.hidden_layers.append(nn.Linear(*hidden_dims[i]))

        trunk_out_dim = hidden_dims[-1][1]
        # Output heads
        self.head_layers = nn.ModuleList()
        self.head_activations = nn.ModuleList()
        self.head_decodes = nn.ModuleList()
        for head in self.output_heads:
            self.head_layers.append(nn.Linear(trunk_out_dim,head.dim))
            if callable(head.activation) and not isinstance(head.activation, nn.Module):
                self.head_activations.append(head.activation())
            else:
                self.head_activations.append(head.activation)
            if callable(head.decode) and not isinstance(head.decode, nn.Module):
                self.head_decodes.append(head.decode())
            else:
                self.head_decodes.append(head.decode)


    def forward(self, z: torch.Tensor, c: torch.Tensor | None = None) -> torch.Tensor :
        """
        forward method for the network
        Parameters
        ----------
        z : torch.Tensor
            input noise
        c : torch.Tensor
            optional conditional tensor if there is a conditional
        Returns
        -------
        torch.Tensor
            the result of passing (z,c) through the network
        """
        if self.conditional_dim > 0 and c is None:
            raise ValueError(f"Conditional dimension is {self.conditional_dim} and no conditional was passed.")
        if c is not None:
            x = torch.cat([z, c], dim=1)
        else:
            x = z

        x = self.activation(self.input_layer(x))

        for layer in self.hidden_layers:
            x = self.activation(layer(x))

        outputs = []
        if self.training:
            for layer, activation in zip(self.head_layers, self.head_activations):
                outputs.append(activation(layer(x)))
        else:
            for layer, decode in zip(self.head_layers, self.head_decodes):
                outputs.append(decode(layer(x)))


        return torch.cat(outputs, dim=1)

    def generate(self, num_samples: int, c: torch.Tensor | None = None):
        """
        method to generate a 'fake' data set
        Parameters
        ----------
        num_samples : int
            how many samples to be generated
        c : torch.Tensor | None
            when not none, the conditional to be used fot generating data
        Returns
        -------
        torch.Tensor
            a 'fake' dataset of shape (num_samples, self.output_dim)
        """
        device = next(self.parameters()).device
        # if there is a conditional validate that the length of the conditional
        # matches the number of samples
        if c is not None:
            if c.shape[0] != num_samples:
                raise ValueError(
                    "Number of samples must equal length of conditional input"
                )
            c = c.to(device)
        # generate random noise and output data
        z = torch.randn(num_samples, self.noise_dim, device=device)
        return self.forward(z, c)

    def generate_samples(self, num_samples: int, c: torch.Tensor | None = None):
        """
        function to generate inference samples in eval mode to use the decode on output heads
        Parameters
        ----------
        num_samples : int
            number of samples to generate
        c : torch.Tensor | None
            when not None conditional to use to generate the samples
        Returns
        -------
        torch.Tensor
            sample of size num_samples of sample data with post-training decode used
        """
        # save current training status
        training = self.training
        # put in eval to use decode on output heads
        self.eval()
        with torch.no_grad():
            # get the output using decoders in output heads
            output = self.generate(num_samples, c)
        # restore training state
        if training:
            self.train()
        return output.cpu()


class Discriminator(nn.Module):
    """
    class for a Discriminator or Critic for a
    GAN / WGAN respectively
    """

    def __init__(
        self,
        feature_dim: int,
        num_hidden_layers: int,
        hidden_dims: HiddenDims,
        hidden_activation: ActivationFactory = nn.LeakyReLU,
        use_sigmoid: bool = False,
        conditional_dim: int = 0,
    ):
        """
        Initialization for class
        Parameters
        ----------
        feature_dim : int
            number of features in the data
        num_hidden_layers : int
            number of linear layers between input and output layers
        hidden_dims : HiddenDims
                either a single (in_dim, out_dim) tuple reused for all hidden layers
                or a sequence of such tuples specifying each layer explicitly
        hidden_activation : ActivationFactory
            activation function to be used on all layers other than output layer
        use_sigmoid : bool
            whether or not a final sigmoid activation is to be applied. This should
            always be False for a WGAN
        conditional_dim : int
           dimension of conditional, 0 indicates no conditional used
        """
        super().__init__()
        # save initialization parameters for recreating model
        self.init_args={
            "feature_dim": feature_dim,
            "num_hidden_layers": num_hidden_layers,
            "hidden_dims": hidden_dims,
            "hidden_activation": hidden_activation,
            "use_sigmoid": use_sigmoid,
            "conditional_dim": conditional_dim
        }
        if isinstance(hidden_dims, tuple):
            hidden_dims = [hidden_dims] * num_hidden_layers
        # valitdate that we have the right number of hidden_dims
        if len(hidden_dims) != num_hidden_layers:
            raise ValueError(
                "Number of hidden layers and length of hidden_dims must be equal"
            )
        # validate that the output of each layer has same dimension as the input of the
        # next layer
        for i in range(len(hidden_dims) - 1):
            if hidden_dims[i][1] != hidden_dims[i + 1][0]:
                raise ValueError(
                    f"hidden_dims[{i}][1] ({hidden_dims[i][1]}) "
                    f"!= hidden_dims[{i + 1}][0] ({hidden_dims[i + 1][0]})"
                )

        self.feature_dim = feature_dim
        if isinstance(hidden_activation, nn.Module):
            self.activation = hidden_activation
        else:
            self.activation = hidden_activation()  
        self.conditional_dim = conditional_dim 

        self.input_layer = nn.Linear(
            self.feature_dim + self.conditional_dim, hidden_dims[0][0]
        )

        self.hidden_layers = nn.ModuleList()
        for i in range(num_hidden_layers):
            self.hidden_layers.append(nn.Linear(*hidden_dims[i]))

        self.output_layer = nn.Linear(hidden_dims[-1][1], 1)
        self.out_activation = nn.Sigmoid() if use_sigmoid else nn.Identity()

    def forward(self, x: torch.Tensor, c: torch.Tensor | None = None, return_features: bool = False):
        """
        forward method for the network
        Parameters
        ----------
        x : torch.Tensor
            data that will be evaluated by the Discriminator/Critic
        c : optional torch.Tensor
            When a conditional is being used, the conditional to be evaluated
        return_features : bool
            whether or not to return the output of the last hidden layer as well as final output
        Returns
        -------
        torch.Tensor
            output of the Discriminator/Critic
            when a Discriminator either the logit or probability,
            when a Critic, critic score
        Optional torch.Tensor
            when return_features is True, the output of the last hidden layer
        """
        if self.conditional_dim > 0 and c is None:
            raise ValueError(f"Conditional dimension is {self.conditional_dim}, but no conditional was passed")
        if c is not None:
            x = torch.cat([x, c], dim=1)
        x = self.activation(self.input_layer(x))
        for layer in self.hidden_layers:
            x = self.activation(layer(x))
        feature_space = x
        x = self.output_layer(x)
        final = self.out_activation(x)
        if return_features:
            return final, feature_space
        return final
    
    
