# Copyright (c) 2024 brineylab @ scripps
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .features import ProteinFeatures


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


class ProteinMPNN(nn.Module):
    """
    ProteinMPNN model.
    """

    def __init__(
        self,
        num_letters: int,  # what's the difference between num_letters and vocab_size?
        node_features: int,
        edge_features: int,
        hidden_dim: int,
        num_encoder_layers: int = 3,
        num_decoder_layers: int = 3,
        vocab_size: int = 21,
        k_neighbors: int = 64,
        augment_eps: float = 0.05,
        dropout: float = 0.1,
        ca_only: bool = False,
    ):
        """
        ProteinMPNN model.

        Parameters
        ----------
        num_letters: int
            Number of letters in the alphabet.

        node_features: int
            Number of node features.

        edge_features: int
            Number of edge features.

        hidden_dim: int
            Number of hidden features.

        num_encoder_layers: int, default=3
            Number of encoder layers.

        num_decoder_layers: int, default=3
            Number of decoder layers.

        vocab: int, default=21
            Number of vocabulary items.

        k_neighbors: int, default=64
            Number of neighbors.

        augment_eps: float, default=0.05
            Augmentation epsilon, used to add noise.

        dropout: float, default=0.1
            Dropout rate.

        ca_only: bool, default=False
            Whether to use only Ca features.

        """
        super(ProteinMPNN, self).__init__()

        # Hyperparameters
        self.node_features = node_features
        self.edge_features = edge_features
        self.hidden_dim = hidden_dim

        # features
        #
        # NOTE: the actual ProteinMPNN code inverts node and edge features
        # not sure whether that's a mistake or not, but probably doesn't matter since they're
        # the same value by default (node_features == edge_features == hidden_dim)
        #
        # the node/edge inversion is also present in the original Ingraham 2019 code, see
        # ProteinFeatures here: https://github.com/jingraham/neurips19-graph-protein-design/blob/22e497a2b565fe82f17e12ea37e89dcf4e50e92f/struct2seq/protein_features.py#L44
        # and instanttiation in the model here: https://github.com/jingraham/neurips19-graph-protein-design/blob/22e497a2b565fe82f17e12ea37e89dcf4e50e92f/struct2seq/struct2seq.py#L28
        self.features = ProteinFeatures(
            edge_features=edge_features,
            node_features=node_features,
            top_k=k_neighbors,
            augment_eps=augment_eps,
            ca_only=ca_only,
        )

        self.W_e = nn.Linear(edge_features, hidden_dim, bias=True)
        self.W_s = nn.Embedding(vocab_size, hidden_dim)
        self.W_out = nn.Linear(hidden_dim, num_letters, bias=True)

        # encoder layers
        self.encoder_layers = nn.ModuleList(
            [
                EncoderLayer(hidden_dim, hidden_dim * 2, dropout=dropout)
                for _ in range(num_encoder_layers)
            ]
        )

        # decoder layers
        self.decoder_layers = nn.ModuleList(
            [
                DecoderLayer(hidden_dim, hidden_dim * 3, dropout=dropout)
                for _ in range(num_decoder_layers)
            ]
        )

        # initialize weights
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        X: torch.Tensor,
        S: torch.Tensor,
        mask: torch.Tensor,
        chain_M: torch.Tensor,
        residue_idx: torch.Tensor,
        chain_encoding_all: torch.Tensor,
        randn: torch.Tensor,
        use_input_decoding_order: bool = False,
        decoding_order: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        X: torch.Tensor
            Input node features with shape [B, L, C], where B is the batch size,
            L is the sequence length, and C is the number of features.

        S: torch.Tensor
            Target sequence with shape [B, L].

        mask: torch.Tensor
            Mask with shape [B, L].

        chain_M: torch.Tensor
            Chain mask with shape [B, L].

        residue_idx: torch.Tensor
            Residue indices with shape [B, L].

        chain_encoding_all: torch.Tensor
            Chain encoding with shape [B, L, C].

        randn: torch.Tensor
            Random noise with shape [B, L].

        use_input_decoding_order: bool, default=False
            Whether to use input decoding order.

        decoding_order: Optional[torch.Tensor], default=None
            Decoding order with shape [B, L].

        Returns
        -------
        log_probs: torch.Tensor
            Log probabilities with shape [B, L, V], where V is the vocabulary size.

        """
        device = X.device

        # embed edge features => [B, L, K, C] and neighbor indices => [B, L, K]
        E, E_idx = self.features(X, mask, residue_idx, chain_encoding_all)
        h_E = self.W_e(E)  # => [B, L, K, C]
        # initialize node features => [B, L, C]
        h_V = torch.zeros((E.shape[0], E.shape[1], E.shape[-1]), device=device)

        # embed the target sequence => [B, L, C]
        h_S = self.W_s(S)

        # create encoder mask => [B, L, K]
        mask_attend = gather_nodes(mask.unsqueeze(-1), E_idx).squeeze(-1)
        mask_attend = mask.unsqueeze(-1) * mask_attend

        # encoder
        for layer in self.encoder_layers:
            h_V, h_E = layer(h_V, h_E, E_idx, mask, mask_attend)

        # concatenate sequence embeddings with edge features => [B, L, K, 2C]
        h_ES = cat_neighbors_nodes(h_S, h_E, E_idx)

        # concatenate target sequence embeddings with edge features => [B, L, K, 2C]
        # but with zeros for the target sequence embeddings to keep the target hidden
        h_EX_encoder = cat_neighbors_nodes(torch.zeros_like(h_S), h_E, E_idx)
        # concatenate node features with the target/edge features => [B, L, K, 3C]
        h_EXV_encoder = cat_neighbors_nodes(h_V, h_EX_encoder, E_idx)

        # update chain mask to include only regions specified by the mask
        chain_M = chain_M * mask
        if not use_input_decoding_order:
            # pseudo-randomly sort the chain mask in a way that ensures masked regions
            # (i.e. where chain_M = 1.0) will come before unmasked regions
            decoding_order = torch.argsort((chain_M + 0.0001) * (torch.abs(randn)))
        mask_size = E_idx.shape[1]
        permutation_matrix_reverse = torch.nn.functional.one_hot(
            decoding_order, num_classes=mask_size
        ).float()
        order_mask_backward = torch.einsum(
            "ij, biq, bjp->bqp",
            (1 - torch.triu(torch.ones(mask_size, mask_size, device=device))),
            permutation_matrix_reverse,
            permutation_matrix_reverse,
        )
        mask_attend = torch.gather(order_mask_backward, 2, E_idx).unsqueeze(-1)
        mask_1D = mask.view([mask.size(0), mask.size(1), 1, 1])
        mask_bw = mask_1D * mask_attend
        mask_fw = mask_1D * (1.0 - mask_attend)

        h_EXV_encoder_fw = mask_fw * h_EXV_encoder
        for layer in self.decoder_layers:
            # Masked positions attend to encoder information, unmasked see.
            h_ESV = cat_neighbors_nodes(h_V, h_ES, E_idx)
            h_ESV = mask_bw * h_ESV + h_EXV_encoder_fw
            h_V = layer(h_V, h_ESV, mask)

        logits = self.W_out(h_V)
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs
