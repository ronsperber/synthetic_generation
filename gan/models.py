"""
model with classes for Generator and Discriminator (or Critic)
for a GAN/WGAN-GP
"""

from typing import Sequence, TypeAlias
from collections.abc import Callable
import torch
import torch.nn as nn

# types used for the classes
LayerDims = tuple[int, int]
HiddenDims: TypeAlias = LayerDims | Sequence[LayerDims]
ActivationFactory: TypeAlias = Callable[[], nn.Module]


class Generator(nn.Module):
    """
    class for Generator in a GAN or WGAN
    """

    def __init__(
        self,
        noise_dim: int,
        num_hidden_layers: int,
        out_dim: int,
        hidden_dims: HiddenDims,
        hidden_activation: ActivationFactory = nn.LeakyReLU,
        output_activation: ActivationFactory = nn.Identity,
        use_conditional: bool = False,
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
        out_dim : int
            dimension of output
        hidden_dims : HiddenDims
            either a single (in_dim, out_dim) tuple reused for all hidden layers
            or a sequence of such tuples specifying each layer explicitly
        hidden_activation : ActivationFactory
            class used for activation after each layer other than the output layer
        output_activation : ActivationFactory
            class used for output activation function
        use_conditional : bool
            Boolean of whether or not an additional conditional is being used
        conditional_dim : int
            when use_conditional is True, represents the dimension of the conditional
        """
        super().__init__()
        # save initialization parameters for recreating model
        self.init_args = {
            "noise_dim": noise_dim,
            "num_hidden_layers": num_hidden_layers,
            "out_dim": out_dim,
            "hidden_dims": hidden_dims,
            "hidden_activation": hidden_activation,
            "output_activation": output_activation,
            "use_conditional": use_conditional,
            "conditional_dim": conditional_dim
        }
        if isinstance(hidden_dims, tuple):
            hidden_dims = [hidden_dims] * num_hidden_layers
        # validate length of hidden_dims
        if len(hidden_dims) != num_hidden_layers:
            raise ValueError(
                "Number of hidden layers and length of hidden_dims must be equal"
            )
        # validate that output dimension of a layer matches input dimension of next
        for i in range(len(hidden_dims) - 1):
            if hidden_dims[i][1] != hidden_dims[i + 1][0]:
                raise ValueError(
                    f"hidden_dims[{i}][1] ({hidden_dims[i][1]}) "
                    f"!= hidden_dims[{i + 1}][0] ({hidden_dims[i + 1][0]})"
                )

        self.noise_dim = noise_dim
        self.output_dim = out_dim
        self.activation = hidden_activation()
        self.conditional_dim = conditional_dim if use_conditional else 0
        self.output_activation = output_activation()
        self.input_layer = nn.Linear(
            self.noise_dim + self.conditional_dim, hidden_dims[0][0]
        )

        self.hidden_layers = nn.ModuleList()
        for i in range(num_hidden_layers):
            self.hidden_layers.append(nn.Linear(*hidden_dims[i]))

        self.output_layer = nn.Linear(hidden_dims[-1][1], self.output_dim)

    def forward(self, z: torch.Tensor, c: torch.Tensor | None = None):
        """
        forward method for the network
        Parameters
        ----------
        z : torch.Tensor
            input noise
        c : torch.Tensor
            optional conditional tensor if there is a conditional
        """
        if c is not None:
            x = torch.cat([z, c], dim=1)
        else:
            x = z

        x = self.activation(self.input_layer(x))

        for layer in self.hidden_layers:
            x = self.activation(layer(x))

        return self.output_activation(self.output_layer(x))

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
        use_conditional: bool = False,
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
        use_conditional : bool
            Boolean on whether or not a conditional is being used
        use_sigmoid : bool
            whether or not a final sigmoid activation is to be applied. This should
            always be False for a WGAN
        condtional_dim : int
            when use_conditional is True, the dimension of the conditional
        """
        super().__init__()
        # save initialization parameters for recreating model
        self.init_args={
            "feature_dim": feature_dim,
            "num_hidden_layers": num_hidden_layers,
            "hidden_dims": hidden_dims,
            "hidden_activation": hidden_activation,
            "use_conditional": use_conditional,
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
        self.activation = hidden_activation()
        self.conditional_dim = conditional_dim if use_conditional else 0

        self.input_layer = nn.Linear(
            self.feature_dim + self.conditional_dim, hidden_dims[0][0]
        )

        self.hidden_layers = nn.ModuleList()
        for i in range(num_hidden_layers):
            self.hidden_layers.append(nn.Linear(*hidden_dims[i]))

        self.output_layer = nn.Linear(hidden_dims[-1][1], 1)
        self.out_activation = nn.Sigmoid() if use_sigmoid else nn.Identity()

    def forward(self, x: torch.Tensor, c: torch.Tensor | None = None):
        """
        forward method for the network
        Parameters
        ----------
        x : torch.Tensor
            data that will be evaluated by the Discriminator/Critic
        c : optional torch.Tensor
            When a condtional is being used, the conditional to be evaluated
        Returns
        -------
        torch.Tensor
            output of the Discriminator/Critic
            when a Discriminator either the logit or probability,
            when a Critic, critic score
        """
        if c is not None:
            x = torch.cat([x, c], dim=1)
        x = self.activation(self.input_layer(x))
        for layer in self.hidden_layers:
            x = self.activation(layer(x))
        x = self.output_layer(x)
        return self.out_activation(x)
