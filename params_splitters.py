import torch
from finetune_utils import list_parameters


def hf_roberta_param_splitter(model: torch.nn.Module, wsc_trick=False):

    base = "base_model" if not wsc_trick else f"base_model.electra"
    embed_name = "embeddings" # if not wsc_trick else f"embeddings"
    #scaler_name = "embeddings_project"
    layers_name = "layer"
    output_name = (
        "classifier" if not wsc_trick else f"base_model.classifier"
    )

    groups = [list_parameters(model, f"{base}.{embed_name}")]
    for i in range(model.base_model.config.num_hidden_layers):
        groups.append(list_parameters(model, f"{base}.encoder.{layers_name}[{i}]"))

    groups.append(list_parameters(model, output_name))
    #if model.base_model.config.hidden_size != model.base_model.config.embedding_size:
    #    groups[0] += list_parameters(model, f"{base}.{scaler_name}")
    # if c.my_model and hparam["pre_norm"]:
    #    groups[-2] += list_parameters(model, f"{base}.encoder.norm")

    assert len(list(model.parameters())) == sum([len(g) for g in groups])
    for i, (p1, p2) in enumerate(
        zip(model.parameters(), [p for g in groups for p in g])
    ):
        assert torch.equal(p1, p2), f"The {i} th tensor"
    
    #print(groups)
    return groups


def hf_electra_param_splitter(model: torch.nn.Module, wsc_trick=False):

    base = "base_model" if not wsc_trick else f"base_model.electra"
    embed_name = "embeddings" # if not wsc_trick else f"embeddings"
    scaler_name = "embeddings_project"
    layers_name = "layer"
    output_name = (
        "classifier" if not wsc_trick else f"base_model.classifier"
    )

    groups = [list_parameters(model, f"{base}.{embed_name}")]
    for i in range(model.base_model.config.num_hidden_layers):
        groups.append(list_parameters(model, f"{base}.encoder.{layers_name}[{i}]"))
    groups.append(list_parameters(model, output_name))
    if model.base_model.config.hidden_size != model.base_model.config.embedding_size:
        groups[0] += list_parameters(model, f"{base}.{scaler_name}")
    # if c.my_model and hparam["pre_norm"]:
    #    groups[-2] += list_parameters(model, f"{base}.encoder.norm")

    assert len(list(model.parameters())) == sum([len(g) for g in groups])
    for i, (p1, p2) in enumerate(
        zip(model.parameters(), [p for g in groups for p in g])
    ):
        assert torch.equal(p1, p2), f"The {i} th tensor"

    #print(groups)
    return groups