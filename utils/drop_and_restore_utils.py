from dataclasses import dataclass, field
from typing import Optional, List, Callable
import inspect

import numpy as np
import math
import torch


def sample_length_configuration(
    max_seq_length,
    num_hidden_layers,
    layer_config=None,
    length_drop_prob=None,
    length_drop_ratio=None,
    length_drop_ratio_bound=None,
    min_length=2,
):
    length = max_seq_length
    length_configuration = ()
    for i in range(num_hidden_layers):
        if layer_config is None or i in layer_config:
            if length_drop_prob is not None:
                length = length - np.random.binomial(length, length_drop_prob)
            elif length_drop_ratio is not None:
                length = int(np.ceil(length * (1 - length_drop_ratio)))
            elif length_drop_ratio_bound is not None:
                length = np.random.randint(int(np.ceil(length * (1 - length_drop_ratio_bound))), length + 1)
        length = max(length, min_length)
        length_configuration += (length,)
    return length_configuration


def sample_layer_configuration(
    num_hidden_layers,
    layer_dropout_prob=None,
    layer_dropout=None,
    layer_dropout_bound=None,
):
    if layer_dropout_prob is not None:
        return tuple(i for i in range(num_hidden_layers) if np.random.random() >= layer_dropout_prob)
    elif layer_dropout is not None:
        layer_dropout = min(layer_dropout, num_hidden_layers - 1)
        return tuple(range(num_hidden_layers - layer_dropout))
    elif layer_dropout_bound is not None:
        layer_dropout_bound = min(layer_dropout_bound, num_hidden_layers - 1)
        return tuple(range(num_hidden_layers - np.random.randint(0, layer_dropout_bound + 1)))
    return None


def sample_head_configuration(
    num_heads,
    num_hidden_layers,
    layer_config=None,
    max_head_pruning=False,
    random_head_pruning=False,
    min_head=1,
    prune_ratio=None,
):
    if prune_ratio is not None:
        max_pruning_configuration = [math.floor(r) for r in np.linspace(min_head, prune_ratio, num_hidden_layers)]
    else:
        max_pruning_configuration = [math.floor(r) for r in np.linspace(min_head, num_heads, num_hidden_layers)]
        max_pruning_configuration[-1] -= 1
    
    head = 0
    head_configuration = ()
    for i in range(num_hidden_layers):
        if layer_config is None or i in layer_config:
            if max_head_pruning:
                head = max_pruning_configuration[i]
            elif random_head_pruning:
                head = np.random.randint(head, max_pruning_configuration[i])
        head_configuration += (head,)
    return head_configuration

def what_to_prune(
    head_importance,
    gene,
    to_prune=None,
    at_least_x_heads_per_layer=0,
    rescale_by_number=False,
):
    head_importance = head_importance.clone()
    n_layers, n_heads = head_importance.size()
    to_prune = to_prune or {}
    if rescale_by_number:
        for layer in to_prune:
            #head_importance[layer] *= sqrt(n_layers / len(to_prune[layer]))
            head_importance[layer] *= math.sqrt(len(to_prune[layer]) / n_layers)
    # Sort heads by score
    heads_and_score = [
        ((layer, head), head_importance[layer, head])
        for layer in range(n_layers)
        for head in range(n_heads)
    ]
    heads_and_score = sorted(heads_and_score, key=lambda x: x[1])
    sorted_heads = [head_and_score[0]
                    for head_and_score in heads_and_score]
    # Ensure we don't delete all heads in a layer
    if at_least_x_heads_per_layer:
        # Remove the top scoring head in each layer
        to_protect = {l: 0 for l in range(n_layers)}
        filtered_sorted_heads = []
        for layer, head in reversed(sorted_heads):
            if layer in to_protect:
                if to_protect[layer] < at_least_x_heads_per_layer:
                    to_protect[layer] += 1
                    continue
                else:
                    to_protect.pop(layer)
            filtered_sorted_heads.insert(0, (layer, head))
        sorted_heads = filtered_sorted_heads
    # layer/heads that were already pruned
    # Prune the lowest scoring heads
    sorted_heads = [
        (layer, head)
        for (layer, head) in sorted_heads
        if layer not in to_prune or head not in to_prune[layer]
    ]
    # Update heads to prune
    for layer, head in sorted_heads:
        if layer not in to_prune:
            to_prune[layer] = []
        if len(to_prune[layer]) < gene[layer]:
            to_prune[layer].append(head)
    return to_prune

def apply_chunking_to_forward(
    forward_fn: Callable[..., torch.Tensor], chunk_size: int, chunk_dim: int, *input_tensors
) -> torch.Tensor:
    """
    This function chunks the :obj:`input_tensors` into smaller input tensor parts of size :obj:`chunk_size` over the
    dimension :obj:`chunk_dim`. It then applies a layer :obj:`forward_fn` to each chunk independently to save memory.

    If the :obj:`forward_fn` is independent across the :obj:`chunk_dim` this function will yield the same result as
    directly applying :obj:`forward_fn` to :obj:`input_tensors`.

    Args:
        forward_fn (:obj:`Callable[..., torch.Tensor]`):
            The forward function of the model.
        chunk_size (:obj:`int`):
            The chunk size of a chunked tensor: :obj:`num_chunks = len(input_tensors[0]) / chunk_size`.
        chunk_dim (:obj:`int`):
            The dimension over which the :obj:`input_tensors` should be chunked.
        input_tensors (:obj:`Tuple[torch.Tensor]`):
            The input tensors of ``forward_fn`` which will be chunked.
    Returns:
        :obj:`torch.Tensor`: A tensor with the same shape as the :obj:`foward_fn` would have given if applied`.


    Examples::

        # rename the usual forward() fn to forward_chunk()
        def forward_chunk(self, hidden_states):
            hidden_states = self.decoder(hidden_states)
            return hidden_states

        # implement a chunked forward function
        def forward(self, hidden_states):
            return apply_chunking_to_forward(self.forward_chunk, self.chunk_size_lm_head, self.seq_len_dim, hidden_states)
    """

    assert len(input_tensors) > 0, "{} has to be a tuple/list of tensors".format(input_tensors)
    tensor_shape = input_tensors[0].shape
    assert all(
        input_tensor.shape == tensor_shape for input_tensor in input_tensors
    ), "All input tenors have to be of the same shape"

    # inspect.signature exist since python 3.5 and is a python method -> no problem with backward compability
    num_args_in_forward_chunk_fn = len(inspect.signature(forward_fn).parameters)
    assert num_args_in_forward_chunk_fn == len(
        input_tensors
    ), "forward_chunk_fn expects {} arguments, but only {} input tensors are given".format(
        num_args_in_forward_chunk_fn, len(input_tensors)
    )

    if chunk_size > 0:
        assert (
            input_tensors[0].shape[chunk_dim] % chunk_size == 0
        ), "The dimension to be chunked {} has to be a multiple of the chunk size {}".format(
            input_tensors[0].shape[chunk_dim], chunk_size
        )

        num_chunks = input_tensors[0].shape[chunk_dim] // chunk_size

        # chunk input tensor into tuples
        input_tensors_chunks = tuple(input_tensor.chunk(num_chunks, dim=chunk_dim) for input_tensor in input_tensors)
        # apply forward fn to every tuple
        output_chunks = tuple(forward_fn(*input_tensors_chunk) for input_tensors_chunk in zip(*input_tensors_chunks))
        # concatenate output at same dimension
        return torch.cat(output_chunks, dim=chunk_dim)

    return forward_fn(*input_tensors)