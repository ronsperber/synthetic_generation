import torch
import types
from .models import DiffusionProcess, MLPTimeEmbedding, SinusoidalTimeEmbedding, DiffusionNet

BASE_MODEL_CLASSES = {
    "MLPTimeEmbedding" : MLPTimeEmbedding,
    "SinusoidalTimeEmbedding" : SinusoidalTimeEmbedding,
    "DiffusionNet" : DiffusionNet
}

BASE_ACTIVATIONS = {
    "ReLU": torch.nn.ReLU,
    "Tanh": torch.nn.Tanh,
    "Sigmoid": torch.nn.Sigmoid,
    "Identity": torch.nn.Identity,
    "GELU": torch.nn.GELU,
    "SiLU": torch.nn.SiLU,  # same as Swish
}


def save_diffusion_checkpoint(process, path: str):
    """
    Save a DiffusionProcess checkpoint with config + state_dict
    """
    model = process.model

    # Build config dict for the model
    config = {
        "class_name": type(model).__name__,
        "data_dim": model.data_dim,
        "conditional_dim": model.conditional_dim,
        "time_embedding": None,          # will store class and init_args if needed
        "conditional_embedding": None,   # will store class and init_args if needed
        "num_hidden_layers": model.num_hidden_layers,
        "hidden_dims": model.hidden_dims,
        "activation": type(model.activation).__name__ if isinstance(model.activation, torch.nn.Module) else model.activation,
    }

    # Handle time_embedding
    if hasattr(model.time_embedding, "init_args"):
        config["time_embedding"] = {
            "class_name": type(model.time_embedding).__name__,
            "init_args": model.time_embedding.init_args
        }
    elif isinstance(model.time_embedding, torch.nn.Module):
        config["time_embedding"] = {
            "class_name": type(model.time_embedding).__name__,
            "init_args": {}  # no args stored
        }

    # Handle conditional embedding
    cond_emb = model.conditional_embedding
    if cond_emb is None or isinstance(cond_emb, torch.nn.Identity):
        config["conditional_embedding"] = None
    elif isinstance(cond_emb, torch.nn.Embedding):
        config["conditional_embedding"] = {
            "class_name": "nn.Embedding",
            "num_embeddings": cond_emb.num_embeddings,
            "embedding_dim": cond_emb.embedding_dim
        }
    elif hasattr(cond_emb, "init_args"):
        config["conditional_embedding"] = {
            "class_name": type(cond_emb).__name__,
            "init_args": cond_emb.init_args
        }
    else:
        # fallback if we don’t know how to reconstruct
        raise ValueError(f"Cannot save conditional_embedding of type {type(cond_emb)} without pickling")

    # Save checkpoint
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "config": config,
        "betas": process.betas.cpu(),  # save on CPU
        "num_timesteps": process.num_timesteps,
        "data_dim": process.data_dim
    }

    torch.save(checkpoint, path)


def load_diffusion_checkpoint(path: str, model_classes: dict = None, activation_dict: dict = None):
    """
    Load a DiffusionProcess checkpoint.
    
    model_classes: optional dictionary mapping class_name -> class
        used if you have custom MLPTimeEmbedding or conditional embedding classes
    """
    model_classes = BASE_MODEL_CLASSES | (model_classes or {})
    activation_dict = BASE_ACTIVATIONS | (activation_dict or {})
    checkpoint = torch.load(path, map_location="cpu")
    config = checkpoint["config"]

    # Rebuild time_embedding
    time_embedding = None
    if config["time_embedding"] is not None:
        te_class_name = config["time_embedding"]["class_name"]
        te_args = config["time_embedding"]["init_args"]
        if te_class_name in model_classes:
            te_class = model_classes[te_class_name]
        else:
            te_class = globals()[te_class_name]  # assumes it's imported
        time_embedding = te_class(**te_args)

    # Rebuild conditional embedding
    cond_emb_cfg = config["conditional_embedding"]
    if cond_emb_cfg is None:
        conditional_embedding = None
    elif cond_emb_cfg["class_name"] == "nn.Embedding":
        conditional_embedding = torch.nn.Embedding(
            num_embeddings=cond_emb_cfg["num_embeddings"],
            embedding_dim=cond_emb_cfg["embedding_dim"]
        )
    else:
        class_name = cond_emb_cfg["class_name"]
        init_args = cond_emb_cfg["init_args"]
        if class_name in model_classes:
            cls = model_classes[class_name]
        else:
            cls = globals()[class_name]
        conditional_embedding = cls(**init_args)

    # Rebuild model
    if config["class_name"] in model_classes:
        model_cls = model_classes[config["class_name"]]
    else:
        model_cls = globals()[config["class_name"]]
    if config["activation"] in activation_dict:
        activation = activation_dict[config["activation"]]
    else:
        raise ValueError(f"{config["activation"]} is not in the dictionary of activations")
    model = model_cls(
        data_dim=config["data_dim"],
        conditional_dim=config["conditional_dim"],
        conditional_embedding=conditional_embedding,
        time_embedding=time_embedding,
        num_hidden_layers=config["num_hidden_layers"],
        hidden_dims=config["hidden_dims"],
        activation=activation
    )
    model.load_state_dict(checkpoint["model_state_dict"])

    # Rebuild DiffusionProcess
    process = DiffusionProcess(
        model=model,
        betas=checkpoint["betas"],
        num_timesteps=checkpoint["num_timesteps"],
        data_dim=checkpoint["data_dim"]
    )

    return process, config
