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
        use_conditional: bool = False,
        conditional_dims: int = 0
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
        self.conditional_dims = conditional_dims if use_conditional else 0

        self.input_layer = nn.Linear(
            self.noise_dim + self.conditional_dims,
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

        return self.output_layer(x)
    
    class Discriminator(nn.Module):
        def __init__(
                self,
                feature_dim: int,
                num_hidden_layers: int,
                hidden_dims,
                hidden_activation = nn.LeakyReLU,
                use_conditional: bool = False,
                 conditional_dims: int = 0
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
            self.conditional_dims = conditional_dims if use_conditional else 0

            self.input_layer = nn.Linear(
                self.feature_dim + self.conditional_dims,
                hidden_dims[0][0]
        )

            self.hidden_layers = nn.ModuleList()
            for i in range(num_hidden_layers):
                self.hidden_layers.append(nn.Linear(*hidden_dims[i]))

            self.output_layer = nn.Linear(hidden_dims[-1][1], 1)

        def forward(self, x, c=None):
            if c is not None:
                x = torch.cat([x, c], dim=1)
            x = self.activation(self.input_layer(x))
            for layer in self.hidden_layers:
                x = self.activation(layer(x))
            return torch.sigmoid(self.output_layer(x))
