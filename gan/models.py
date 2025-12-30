import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(
        self,
        noise_dim: int,
        num_hidden_layers: int,
        out_dim: int,
        hidden_dims,
        hidden_activation = nn.LeakyReLU,
        output_activation = nn.Identity,
        use_conditional: bool = False,
        conditional_dim: int = 0
    ):
        super().__init__()

        if isinstance(hidden_dims, tuple):
            hidden_dims = [hidden_dims] * num_hidden_layers

        if len(hidden_dims) != num_hidden_layers:
            raise ValueError("Number of hidden layers and length of hidden_dims must be equal")
        for i in range(len(hidden_dims) - 1):
            if hidden_dims[i][1] != hidden_dims[i + 1][0]:
                raise ValueError(
                    f"hidden_dims[{i}][1] ({hidden_dims[i][1]}) "
                    f"!= hidden_dims[{i+1}][0] ({hidden_dims[i+1][0]})"
                    )

        self.noise_dim = noise_dim
        self.output_dim = out_dim
        self.activation = hidden_activation()
        self.conditional_dim = conditional_dim if use_conditional else 0
        self.output_activation = output_activation()
        self.input_layer = nn.Linear(
            self.noise_dim + self.conditional_dim,
            hidden_dims[0][0]
        )

        self.hidden_layers = nn.ModuleList()
        for i in range(num_hidden_layers):
            self.hidden_layers.append(nn.Linear(*hidden_dims[i]))

        self.output_layer = nn.Linear(hidden_dims[-1][1], self.output_dim)

    def forward(self, z, c=None):
        if c is not None:
            x = torch.cat([z, c], dim=1)
        else:
            x = z

        x = self.activation(self.input_layer(x))

        for layer in self.hidden_layers:
            x = self.activation(layer(x))

        return self.output_activation(self.output_layer(x))
    
    def generate(
        self,
        num_samples: int,
        c: torch.Tensor | None = None
    ):
        device = next(self.parameters()).device
        if c is not None:
            if c.shape[0] != num_samples:
                raise ValueError("Number of samples must equal length of conditional input")
            c = c.to(device)
        z = torch.randn(num_samples, self.noise_dim, device=device)
        return self.forward(z, c)

    
class Discriminator(nn.Module):
    def __init__(
            self,
            feature_dim: int,
            num_hidden_layers: int,
            hidden_dims : list | tuple,
            hidden_activation = nn.LeakyReLU,
            use_conditional: bool = False,
            use_sigmoid: bool = False,
            conditional_dim: int = 0
            ):
            super().__init__()

            if isinstance(hidden_dims, tuple):
                hidden_dims = [hidden_dims] * num_hidden_layers

            if len(hidden_dims) != num_hidden_layers:
                raise ValueError("Number of hidden layers and length of hidden_dims must be equal")
            for i in range(len(hidden_dims) - 1):
                if hidden_dims[i][1] != hidden_dims[i + 1][0]:
                    raise ValueError(
                    f"hidden_dims[{i}][1] ({hidden_dims[i][1]}) "
                    f"!= hidden_dims[{i+1}][0] ({hidden_dims[i+1][0]})"
                    )

            self.feature_dim = feature_dim
            self.activation = hidden_activation()
            self.conditional_dim = conditional_dim if use_conditional else 0

            self.input_layer = nn.Linear(
                self.feature_dim + self.conditional_dim,
                hidden_dims[0][0]
        )

            self.hidden_layers = nn.ModuleList()
            for i in range(num_hidden_layers):
                self.hidden_layers.append(nn.Linear(*hidden_dims[i]))

            self.output_layer = nn.Linear(hidden_dims[-1][1], 1)
            self.out_activation = nn.Sigmoid() if use_sigmoid else nn.Identity()
    def forward(self, x, c=None):
        if c is not None:
            x = torch.cat([x, c], dim=1)
        x = self.activation(self.input_layer(x))
        for layer in self.hidden_layers:
            x = self.activation(layer(x))
        x = self.output_layer(x)
        return self.out_activation(x)
