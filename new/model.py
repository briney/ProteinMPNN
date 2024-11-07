# Copyright (c) 2024 brineylab @ scripps
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT

from typing import Optional

import torch
import torch.nn as nn


def gather_edges(edges: torch.Tensor, neighbor_idx: torch.Tensor) -> torch.Tensor:
    """
    Gather edge features from a neighbor index.

    Parameters
    ----------
    edges: torch.Tensor
        Edge features with shape [B, N, N, C].

    neighbor_idx: torch.Tensor
        Neighbor indices with shape [B, N, K].

    Returns
    -------
    edge_features: torch.Tensor
        Gathered edge features with shape [B, N, K, C].
    """
    # expand neighbor indices [B, N, K] => [B, N, K, C]
    neighbors = neighbor_idx.unsqueeze(-1).expand(-1, -1, -1, edges.size(-1))
    # gather edges [B, N, N, C] => [B, N, K, C]
    edge_features = torch.gather(edges, 2, neighbors)
    return edge_features


def gather_nodes(nodes: torch.Tensor, neighbor_idx: torch.Tensor) -> torch.Tensor:
    """
    Gather node features from a neighbor index.

    Parameters
    ----------
    nodes: torch.Tensor
        Node features with shape [B, N, C].

    neighbor_idx: torch.Tensor
        Neighbor indices with shape [B, N, K].

    Returns
    -------
    neighbor_features: torch.Tensor
        Gathered neighbor features with shape [B, N, K, C].

    """
    # flatten neighbor indices [B, N, K] => [B, NK]
    neighbors_flat = neighbor_idx.view((neighbor_idx.shape[0], -1))
    # expand flattened indices [B, NK] => [B, NK, C]
    neighbors_flat = neighbors_flat.unsqueeze(-1).expand(-1, -1, nodes.size(2))
    # gather nodes [B, N, C] => [B, NK, C]
    neighbor_features = torch.gather(nodes, 1, neighbors_flat)
    # reshape to [B, N, K, C]
    neighbor_features = neighbor_features.view(list(neighbor_idx.shape)[:3] + [-1])
    return neighbor_features


def cat_neighbors_nodes(
    h_nodes: torch.Tensor, h_edges: torch.Tensor, E_idx: torch.Tensor
) -> torch.Tensor:
    """
    Concatenate node/edge features of neighbors.

    Parameters
    ----------
    h_nodes: torch.Tensor
        Node features with shape [B, N, C].

    h_edges: torch.Tensor
        Edge features with shape [B, N, N, C].

    E_idx: torch.Tensor
        Neighbor indices with shape [B, N, K].

    Returns
    -------
    h_nn: torch.Tensor
        Concatenated node and neighbor features with shape [B, N, K, 2C].

    """
    # gather node features [B, N, C] with neighbor indices [B, N, K] => [B, N, K, C]
    h_nodes = gather_nodes(h_nodes, E_idx)
    # concatenate edge and node features [B, N, N, C] + [B, N, K, C] => [B, N, K, 2C]
    h_nn = torch.cat([h_edges, h_nodes], -1)
    return h_nn


class PositionWiseFeedForward(nn.Module):
    """
    Position-wise feedforward network.

    Parameters
    ----------
    num_hidden: int
        Number of hidden features.

    num_ff: int
        Number of feedforward features.

    activation: str
        Activation function.

    """

    def __init__(self, num_hidden: int, num_ff: int, activation: str = "gelu"):
        super(PositionWiseFeedForward, self).__init__()
        self.W_in = nn.Linear(num_hidden, num_ff, bias=True)
        self.W_out = nn.Linear(num_ff, num_hidden, bias=True)
        if activation.lower() == "gelu":
            self.act = torch.nn.GELU()
        else:
            raise ValueError(f"Activation {activation} not supported")

    def forward(self, h_V: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        h_V: torch.Tensor
            Node features with shape [B, N, C].

        Returns
        -------
        h: torch.Tensor
            Output features with shape [B, N, C].

        """
        h = self.act(self.W_in(h_V))
        h = self.W_out(h)
        return h


class EncoderLayer(nn.Module):
    """
    MPNN encoder layer.

    Parameters
    ----------
    num_hidden: int
        Number of hidden features.

    num_in: int
        Number of input features. This is typically 3x the number of hidden features,
        as the input is the concatenation of node features and the node/edge features of neighbors.

    dropout: float
        Dropout rate.

    activation: str
        Activation function.

    scale: float
        Scale factor.

    """

    def __init__(
        self,
        num_hidden: int,
        num_in: int,
        dropout: float = 0.1,
        activation: str = "gelu",
        scale: float = 30,
    ):
        super(EncoderLayer, self).__init__()
        self.num_hidden = num_hidden
        self.num_in = num_in
        self.scale = scale

        if activation.lower() == "gelu":
            self.act = torch.nn.GELU()
        else:
            raise ValueError(f"Activation {activation} not supported")

        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(num_hidden)
        self.norm2 = nn.LayerNorm(num_hidden)
        self.norm3 = nn.LayerNorm(num_hidden)

        self.W1 = nn.Linear(num_hidden + num_in, num_hidden, bias=True)
        self.W2 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W3 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W11 = nn.Linear(num_hidden + num_in, num_hidden, bias=True)
        self.W12 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W13 = nn.Linear(num_hidden, num_hidden, bias=True)

        self.dense = PositionWiseFeedForward(num_hidden, num_hidden * 4, activation)

    def forward(
        self,
        h_V: torch.Tensor,
        h_E: torch.Tensor,
        E_idx: torch.Tensor,
        mask_V: Optional[torch.Tensor] = None,
        mask_attend: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        h_V: torch.Tensor
            Node features with shape [B, N, C].

        h_E: torch.Tensor
            Edge features with shape [B, N, K, C].

        E_idx: torch.Tensor
            Neighbor indices with shape [B, N, K].

        mask_V: Optional[torch.Tensor]
            Node mask with shape [B, N].

        mask_attend: Optional[torch.Tensor]
            Attention mask with shape [B, N, K].

        Returns
        -------
        h_V: torch.Tensor
            Output node features with shape [B, N, C].

        h_E: torch.Tensor
            Output edge features with shape [B, N, K, C].

        """
        # concatenate node and edge features of neighbors => [B, N, K, 2C]
        h_EV = cat_neighbors_nodes(h_V, h_E, E_idx)
        # expand node features to match the first three dimensions of the concatenated features => [B, N, K, C]
        h_V_expand = h_V.unsqueeze(-2).expand(-1, -1, h_EV.size(-2), -1)
        # concatenate node features with the node/edge features of neighbors => [B, N, K, 3C]
        h_EV = torch.cat([h_V_expand, h_EV], -1)
        h_message = self.W3(self.act(self.W2(self.act(self.W1(h_EV)))))
        if mask_attend is not None:
            # apply mask
            h_message = mask_attend.unsqueeze(-1) * h_message
        dh = torch.sum(h_message, -2) / self.scale
        h_V = self.norm1(h_V + self.dropout(dh))

        # feedforward network
        dh = self.dense(h_V)
        h_V = self.norm2(h_V + self.dropout(dh))
        if mask_V is not None:
            # apply node mask
            mask_V = mask_V.unsqueeze(-1)
            h_V = mask_V * h_V

        # concatenate node features with the edge features of their neighbors => [B, N, K, 2C]
        h_EV = cat_neighbors_nodes(h_V, h_E, E_idx)
        # expand node features to match the shape of the concatenated features => [B, N, K, C]
        h_V_expand = h_V.unsqueeze(-2).expand(-1, -1, h_EV.size(-2), -1)
        # concatenate expanded node features with the concatenated features => [B, N, K, 3C]
        h_EV = torch.cat([h_V_expand, h_EV], -1)
        h_message = self.W13(self.act(self.W12(self.act(self.W11(h_EV)))))
        h_E = self.norm3(h_E + self.dropout(h_message))

        return h_V, h_E


class DecoderLayer(nn.Module):
    """
    MPNN decoder layer.
    """

    def __init__(
        self,
        num_hidden: int,
        num_in: int,
        dropout: float = 0.1,
        activation: str = "gelu",
        scale: float = 30.0,
    ):
        super(DecoderLayer, self).__init__()
        self.num_hidden = num_hidden
        self.num_in = num_in
        self.scale = scale

        if activation.lower() == "gelu":
            self.act = torch.nn.GELU()
        else:
            raise ValueError(f"Activation {activation} not supported")

        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(num_hidden)
        self.norm2 = nn.LayerNorm(num_hidden)

        self.W1 = nn.Linear(num_hidden + num_in, num_hidden, bias=True)
        self.W2 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W3 = nn.Linear(num_hidden, num_hidden, bias=True)

        self.dense = PositionWiseFeedForward(num_hidden, num_hidden * 4, activation)

    def forward(
        self,
        h_V: torch.Tensor,
        h_E: torch.Tensor,
        mask_V: Optional[torch.Tensor] = None,
        mask_attend: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        h_V: torch.Tensor
            Node features with shape [B, N, C].

        h_E: torch.Tensor
            Edge features with shape [B, N, K, C].

        mask_V: Optional[torch.Tensor]
            Node mask with shape [B, N].

        mask_attend: Optional[torch.Tensor]
            Attention mask with shape [B, N, K].

        Returns
        -------
        h_V: torch.Tensor
            Output node features with shape [B, N, C].

        """
        # expand node features to match the shape of the edge features => [B, N, K, C]
        h_V_expand = h_V.unsqueeze(-2).expand(-1, -1, h_E.size(-2), -1)
        # concatenate expanded node features with the edge features => [B, N, K, 2C]
        h_EV = torch.cat([h_V_expand, h_E], -1)
        h_message = self.W3(self.act(self.W2(self.act(self.W1(h_EV)))))
        if mask_attend is not None:
            # apply mask
            h_message = mask_attend.unsqueeze(-1) * h_message
        dh = torch.sum(h_message, -2) / self.scale
        h_V = self.norm1(h_V + self.dropout(dh))

        # feedforward network
        dh = self.dense(h_V)
        h_V = self.norm2(h_V + self.dropout(dh))

        if mask_V is not None:
            # apply node mask
            mask_V = mask_V.unsqueeze(-1)
            h_V = mask_V * h_V

        return h_V
