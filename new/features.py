# Copyright (c) 2024 brineylab @ scripps
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT


from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import gather_edges


class ProteinFeatures(nn.Module):
    def __init__(
        self,
        edge_features: int,
        node_features: int,
        num_positional_embeddings: int = 16,
        num_rbf: int = 16,
        top_k: int = 30,
        features_type: str = "all",
        augment_eps: float = 0.0,
        dropout: float = 0.1,
    ):
        """
        Initialize the ProteinFeatures class.

        Parameters
        ----------
        edge_features: int
            Number of edge features.

        node_features: int
            Number of node features.

        num_positional_embeddings: int, default=16
            Number of positional embeddings.

        num_rbf: int, default=16
            Number of radial basis functions.

        top_k: int, default=30
            Top k neighbors to consider.

        features_type: str, default="all"
            Type of features to use. Options are "all" or "ca_only".

        augment_eps: float, default=0.0
            Epsilon for augmentation.

        dropout: float, default=0.1
            Dropout rate.

        """
        super(ProteinFeatures, self).__init__()
        self.edge_features = edge_features
        self.node_features = node_features
        self.num_positional_embeddings = num_positional_embeddings
        self.num_rbf = num_rbf
        self.top_k = top_k
        self.features_type = features_type
        self.augment_eps = augment_eps
        self.dropout = dropout

        if features_type.lower() == "all":
            node_in = 6
            edge_in = num_positional_embeddings + num_rbf * 25
        elif features_type.lower() in ["ca_only", "ca"]:
            node_in = 3
            edge_in = num_positional_embeddings + num_rbf * 9 + 7
        else:
            raise ValueError(f"Invalid features type: {features_type}")

        self.embeddings = PositionalEncodings(num_positional_embeddings)
        self.node_embedding = nn.Linear(node_in, node_features, bias=False)
        self.edge_embedding = nn.Linear(edge_in, edge_features, bias=False)
        self.norm_nodes = nn.LayerNorm(node_features)
        self.norm_edges = nn.LayerNorm(edge_features)

    def _dist(
        self, X: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute distances and indices of top k neighbors.

        Parameters
        ----------
        X: torch.Tensor
            Input tensor.

        mask: torch.Tensor
            Mask tensor.

        eps: float, default=1e-6
            Epsilon value.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Distances and indices of top k neighbors.

        """
        # compute mask for 2D distances
        mask_2D = torch.unsqueeze(mask, 1) * torch.unsqueeze(mask, 2)
        # compute pairwise distance between all nodes
        dX = torch.unsqueeze(X, 1) - torch.unsqueeze(X, 2)
        # convert to Euclidean distance and mask
        D = mask_2D * torch.sqrt(torch.sum(dX**2, 3) + eps)
        # find the max distance for each node and set distance for masked nodes to this value
        D_max, _ = torch.max(D, -1, keepdim=True)
        D_adjust = D + (1.0 - mask_2D) * D_max
        # get top k neighbors
        D_neighbors, E_idx = torch.topk(
            D_adjust, np.minimum(self.top_k, X.shape[1]), dim=-1, largest=False
        )
        return D_neighbors, E_idx

    def _rbf(self, D: torch.Tensor) -> torch.Tensor:
        """
        Compute radial basis functions.

        Parameters
        ----------
        D: torch.Tensor
            Distance tensor.

        Returns
        -------
        torch.Tensor
            Radial basis functions tensor.

        """
        D_min = 2.0
        D_max = 22.0
        D_count = self.num_rbf
        # create RBF centers
        D_mu = torch.linspace(D_min, D_max, D_count, device=D.device)
        D_mu = D_mu.view([1, 1, 1, -1])  # => [1, 1, 1, D_count]
        # standard deviation of the RBFs
        D_sigma = (D_max - D_min) / D_count
        D_expand = torch.unsqueeze(D, -1)  # => [B, L, L, 1]
        # compute RBFs
        rbf = torch.exp(-(((D_expand - D_mu) / D_sigma) ** 2))  # => [B, L, L, D_count]
        return rbf

    def _get_rbf(
        self, A: torch.Tensor, B: torch.Tensor, E_idx: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute RBFs for a pair of nodes.

        Parameters
        ----------
        A: torch.Tensor
            Node tensor. [B, L, F]

        B: torch.Tensor
            Node tensor. [B, L, F]

        E_idx: torch.Tensor
            Edge indices. [B, L, K]

        Returns
        -------
        torch.Tensor
            RBFs tensor. [B, L, L, D_count]

        """
        # compute pairwise distance between nodes in A and B => [B, L, L]
        D_A_B = torch.sqrt(
            torch.sum((A[:, :, None, :] - B[:, None, :, :]) ** 2, -1) + 1e-6
        )
        # gather the top k neighbors for each node in A and B => [B, L, K]
        D_A_B_neighbors = gather_edges(D_A_B[:, :, :, None], E_idx)[:, :, :, 0]
        # compute RBFs for the top k neighbors
        rbf = self._rbf(D_A_B_neighbors)  # => [B, L, L, D_count]
        return rbf


class PositionalEncodings(nn.Module):
    """
    Positional encodings for nodes and edges.
    """

    def __init__(self, num_embeddings: int, max_relative_feature: int = 32):
        """
        Initialize the PositionalEncodings class.

        Parameters
        ----------
        num_embeddings: int
            Number of embeddings.

        max_relative_feature: int, default=32
            Maximum relative feature.
        """
        super(PositionalEncodings, self).__init__()
        self.num_embeddings = num_embeddings
        self.max_relative_feature = max_relative_feature
        self.linear = nn.Linear(2 * max_relative_feature + 1 + 1, num_embeddings)

    def forward(self, offset: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        offset: torch.Tensor
            Offset tensor.

        mask: torch.Tensor
            Mask tensor.

        Returns
        -------
        torch.Tensor
            Embeddings tensor.
        """
        # clip values to be between 0 and 2 * max_relative_feature
        d = torch.clip(
            offset + self.max_relative_feature, 0, 2 * self.max_relative_feature
        )
        # apply mask
        d = d * mask
        # set all masked values to 2 * max_relative_feature + 1 (out of range value)
        d = d + (1 - mask) * (2 * self.max_relative_feature + 1)
        # one hot encode (2 * max_relative_feature + 1 + 1 classes)
        d_onehot = F.one_hot(d, 2 * self.max_relative_feature + 1 + 1)
        E = self.linear(d_onehot.float())
        return E
