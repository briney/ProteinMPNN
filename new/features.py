# Copyright (c) 2024 brineylab @ scripps
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT


from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import gather_edges, gather_nodes


class ProteinFeatures(nn.Module):
    def __init__(
        self,
        edge_features: int,
        node_features: int,
        num_positional_embeddings: int = 16,
        num_rbf: int = 16,
        top_k: int = 30,
        ca_only: bool = False,
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

        ca_only: bool, default=False
            Whether to use Ca-only features.

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
        self.ca_only = ca_only
        self.augment_eps = augment_eps
        self.dropout = dropout

        if ca_only:
            node_in = 3
            edge_in = num_positional_embeddings + num_rbf * 9 + 7
            self.forward_fn = self._forward_ca_only
        else:
            node_in = 6
            edge_in = num_positional_embeddings + num_rbf * 25
            self.forward_fn = self._forward_all

        self.embeddings = PositionalEncodings(num_positional_embeddings)
        self.node_embedding = nn.Linear(node_in, node_features, bias=False)
        self.edge_embedding = nn.Linear(edge_in, edge_features, bias=False)
        self.norm_nodes = nn.LayerNorm(node_features)
        self.norm_edges = nn.LayerNorm(edge_features)

    def forward(
        self,
        X: torch.Tensor,
        mask: torch.Tensor,
        residue_idx: torch.Tensor,
        chain_labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Parameters
        ----------
        X: torch.Tensor
            Input tensor. [B, L, 4, 3]

        mask: torch.Tensor
            Mask tensor. [B, L]

        residue_idx: torch.Tensor
            Residue indices tensor. [B, L]

        chain_labels: torch.Tensor
            Chain labels tensor. [B, L]

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Node features and edge indices.
            ``Node features``: [B, L, node_features]
            ``Edge indices``: [B, L, K]

        """
        return self.forward_fn(
            X=X, mask=mask, residue_idx=residue_idx, chain_labels=chain_labels
        )

    def _forward_all(
        self,
        X: torch.Tensor,
        mask: torch.Tensor,
        residue_idx: torch.Tensor,
        chain_labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for all-atom features.

        Parameters
        ----------
        X: torch.Tensor
            Input tensor. [B, L, 4, 3]

        mask: torch.Tensor
            Mask tensor. [B, L]

        residue_idx: torch.Tensor
            Residue indices tensor. [B, L]

        chain_labels: torch.Tensor
            Chain labels tensor. [B, L]

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Node features and edge indices.
            ``Node features``: [B, L, node_features]
            ``Edge indices``: [B, L, K]

        """
        if self.augment_eps > 0:
            # add random noise to the input tensor
            X = X + self.augment_eps * torch.randn_like(X)

        # compute vectors for each bond => [B, L, 3]
        b = X[:, :, 1, :] - X[:, :, 0, :]  # N-Ca
        c = X[:, :, 2, :] - X[:, :, 1, :]  # Ca-C
        # cross product of b and c => [B, L, 3]
        a = torch.cross(b, c, dim=-1)
        # all atoms coordinates, including "virtual" Cb => [B, L, 3]
        Cb = -0.58273431 * a + 0.56802827 * b - 0.54067466 * c + X[:, :, 1, :]
        Ca = X[:, :, 1, :]
        N = X[:, :, 0, :]
        C = X[:, :, 2, :]
        O = X[:, :, 3, :]

        # compute distances and indices of top k neighbors => [B, L, K]
        D_neighbors, E_idx = self._dist(Ca, mask)

        # compute RBFs for all pairs of nodes
        RBFs = []
        RBFs.append(self._rbf(D_neighbors))  # Ca-Ca
        RBFs.append(self._get_rbf(N, N, E_idx))  # N-N
        RBFs.append(self._get_rbf(C, C, E_idx))  # C-C
        RBFs.append(self._get_rbf(O, O, E_idx))  # O-O
        RBFs.append(self._get_rbf(Cb, Cb, E_idx))  # Cb-Cb
        RBFs.append(self._get_rbf(Ca, N, E_idx))  # Ca-N
        RBFs.append(self._get_rbf(Ca, C, E_idx))  # Ca-C
        RBFs.append(self._get_rbf(Ca, O, E_idx))  # Ca-O
        RBFs.append(self._get_rbf(Ca, Cb, E_idx))  # Ca-Cb
        RBFs.append(self._get_rbf(N, C, E_idx))  # N-C
        RBFs.append(self._get_rbf(N, O, E_idx))  # N-O
        RBFs.append(self._get_rbf(N, Cb, E_idx))  # N-Cb
        RBFs.append(self._get_rbf(Cb, C, E_idx))  # Cb-C
        RBFs.append(self._get_rbf(Cb, O, E_idx))  # Cb-O
        RBFs.append(self._get_rbf(O, C, E_idx))  # O-C
        RBFs.append(self._get_rbf(N, Ca, E_idx))  # N-Ca
        RBFs.append(self._get_rbf(C, Ca, E_idx))  # C-Ca
        RBFs.append(self._get_rbf(O, Ca, E_idx))  # O-Ca
        RBFs.append(self._get_rbf(Cb, Ca, E_idx))  # Cb-Ca
        RBFs.append(self._get_rbf(C, N, E_idx))  # C-N
        RBFs.append(self._get_rbf(O, N, E_idx))  # O-N
        RBFs.append(self._get_rbf(Cb, N, E_idx))  # Cb-N
        RBFs.append(self._get_rbf(C, Cb, E_idx))  # C-Cb
        RBFs.append(self._get_rbf(O, Cb, E_idx))  # O-Cb
        RBFs.append(self._get_rbf(C, O, E_idx))  # C-O

        # concatenate all RBFs => [B, L, K, num_rbf * 25]
        RBFs = torch.cat(tuple(RBFs), dim=-1)

        # determine the relative position ("offset") of each pair of nodes => [B, L, L]
        offset = residue_idx[:, :, None] - residue_idx[:, None, :]
        # gather the relative postion of the top k neighbors for each node => [B, L, K]
        offset = gather_edges(offset[:, :, :, None], E_idx)[..., 0]

        # determine whether interactions are within or between chains
        d_chains = (
            (chain_labels[:, :, None] - chain_labels[:, None, :]) == 0
        ).long()  # find self vs non-self interaction
        E_chains = gather_edges(d_chains[:, :, :, None], E_idx)[..., 0]

        # positional embeddings
        E_positional = self.embeddings(offset.long(), E_chains)

        # concatenate positional embeddings and RBFs => [B, L, K, num_positional_embeddings + num_rbf * 25]
        E = torch.cat((E_positional, RBFs), -1)

        # edge embeddings
        E = self.edge_embedding(E)
        E = self.norm_edges(E)  # => [B, L, K, edge_features]
        return E, E_idx

    def _forward_ca_only(
        self,
        X: torch.Tensor,
        mask: torch.Tensor,
        residue_idx: torch.Tensor,
        chain_labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for CA-only features.

        Parameters
        ----------
        X: torch.Tensor
            Input tensor. [B, L, 4, 3]

        mask: torch.Tensor
            Mask tensor. [B, L]

        residue_idx: torch.Tensor
            Residue indices tensor. [B, L]

        chain_labels: torch.Tensor
            Chain labels tensor. [B, L]

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Node features and edge indices.
            ``Node features``: [B, L, node_features]
            ``Edge indices``: [B, L, K]

        """
        if self.augment_eps > 0:
            # add random noise to the input tensor
            X = X + self.augment_eps * torch.randn_like(X)

        # compute distances and indices of top k neighbors => [B, L, K]
        D_neighbors, E_idx, _ = self._dist(X, mask)

        # compute vectors for each bond => [B, L, 3]
        Ca_0 = torch.zeros(X.shape, device=X.device)
        Ca_2 = torch.zeros(X.shape, device=X.device)
        Ca_0[:, 1:, :] = X[:, :-1, :]
        Ca_1 = X
        Ca_2[:, :-1, :] = X[:, 1:, :]

        # compute backbone features => [B, L-3, 3]
        _, O_features = self._orientations_coarse(X, E_idx)

        # compute RBFs for all pairs of nodes (Ca-only)
        RBFs = []
        RBFs.append(self._rbf(D_neighbors))  # Ca_1-Ca_1
        RBFs.append(self._get_rbf(Ca_0, Ca_0, E_idx))
        RBFs.append(self._get_rbf(Ca_2, Ca_2, E_idx))
        RBFs.append(self._get_rbf(Ca_0, Ca_1, E_idx))
        RBFs.append(self._get_rbf(Ca_0, Ca_2, E_idx))
        RBFs.append(self._get_rbf(Ca_1, Ca_0, E_idx))
        RBFs.append(self._get_rbf(Ca_1, Ca_2, E_idx))
        RBFs.append(self._get_rbf(Ca_2, Ca_0, E_idx))
        RBFs.append(self._get_rbf(Ca_2, Ca_1, E_idx))
        RBFs = torch.cat(tuple(RBFs), dim=-1)

        # determine the relative position ("offset") of each pair of nodes => [B, L, L]
        offset = residue_idx[:, :, None] - residue_idx[:, None, :]
        # gather the relative postion of the top k neighbors for each node => [B, L, K]
        offset = gather_edges(offset[:, :, :, None], E_idx)[..., 0]

        # determine whether interactions are within or between chains
        d_chains = ((chain_labels[:, :, None] - chain_labels[:, None, :]) == 0).long()
        E_chains = gather_edges(d_chains[:, :, :, None], E_idx)[..., 0]

        # positional embeddings
        E_positional = self.embeddings(offset.long(), E_chains)

        # concatenate positional embeddings and RBFs => [B, L, K, num_positional_embeddings + num_rbf * 9 + 7]
        E = torch.cat((E_positional, RBFs, O_features), -1)

        E = self.edge_embedding(E)
        E = self.norm_edges(E)  # => [B, L, K, edge_features]
        return E, E_idx

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

    def _quaternions(self, R):
        """
        Convert a batch of 3D rotations [R] to quaternions [Q]

        Parameters
        ----------
        R: torch.Tensor
            Rotation matrix tensor. [..., 3, 3]

        Returns
        -------
        torch.Tensor
            Quaternions tensor. [..., 4]

        """
        # extract the diagonal elements of the rotation matrix => [..., 3]
        diag = torch.diagonal(R, dim1=-2, dim2=-1)
        # split the diagonal elements into separate tensors => [...]
        Rxx, Ryy, Rzz = diag.unbind(-1)
        # compute the magnitudes of the rotation vectors => [..., 3]
        magnitudes = torch.stack(
            [Rxx - Ryy - Rzz, -Rxx + Ryy - Rzz, -Rxx - Ryy + Rzz], -1
        )
        magnitudes = 0.5 * torch.sqrt(torch.abs(1 + magnitudes))

        # compute the signs of the rotation vectors => [..., 3]
        signs = torch.sign(
            torch.stack(
                [
                    R[:, :, :, 2, 1] - R[:, :, :, 1, 2],
                    R[:, :, :, 0, 2] - R[:, :, :, 2, 0],
                    R[:, :, :, 1, 0] - R[:, :, :, 0, 1],
                ],
                -1,
            )
        )
        xyz = signs * magnitudes  # => [..., 3]

        # The relu enforces a non-negative trace
        w = torch.sqrt(F.relu(1 + diag.sum(-1, keepdim=True))) / 2.0  # => [..., 1]
        Q = torch.cat((xyz, w), -1)
        Q = F.normalize(Q, dim=-1)
        return Q

    def _orientations_coarse(
        self, X: torch.Tensor, E_idx: torch.Tensor, eps: float = 1e-6
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute coarse orientations.

        Parameters
        ----------
        X: torch.Tensor
            Input tensor of shape ``[B, L, 3]``, where ``B`` is the batch size, ``L`` is the
            sequence length, and ``3`` represents the 3D coordinates of the Ca atoms.

        E_idx: torch.Tensor
            Edge indices of the shape ``[B, L, K]``, where ``K`` is the number of neighbors.

        eps: float, default=1e-6
            Epsilon value to avoid division by zero.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            ``AD_features``: backbone features of shape ``[B, L, 3]``
            ``O_features``: orientation features of shape ``[B, L, K, 7]``

        """
        # compute difference between consecutive Ca atoms => [B, L-1, 3]
        dX = X[:, 1:, :] - X[:, :-1, :]
        dX_norm = torch.norm(dX, dim=-1)
        # exclude Ca-Ca jumps => [B, L-1, 3]
        dX_mask = (3.6 < dX_norm) & (dX_norm < 4.0)
        dX = dX * dX_mask[:, :, None]
        # normalize differences to unit vectors => [B, L-1, 3]
        U = F.normalize(dX, dim=-1)
        # extract three consecutive unit vectors, each => [B, L-3, 3]
        u_2 = U[:, :-2, :]
        u_1 = U[:, 1:-1, :]
        u_0 = U[:, 2:, :]
        # compute backbone normals => [B, L-3, 3]
        n_2 = F.normalize(torch.cross(u_2, u_1), dim=-1)
        n_1 = F.normalize(torch.cross(u_1, u_0), dim=-1)
        # compute bond angles => [B, L-3]
        cosA = -(u_1 * u_0).sum(-1)
        cosA = torch.clamp(cosA, -1 + eps, 1 - eps)  # clamp to roughly [-1, 1]
        A = torch.acos(cosA)
        # compute angle between normals => [B, L-3]
        cosD = (n_2 * n_1).sum(-1)
        cosD = torch.clamp(cosD, -1 + eps, 1 - eps)  # clamp to roughly [-1, 1]
        D = torch.sign((u_2 * n_1).sum(-1)) * torch.acos(cosD)

        # compute backbone features => [B, L-3, 3]
        AD_features = torch.stack(
            (torch.cos(A), torch.sin(A) * torch.cos(D), torch.sin(A) * torch.sin(D)),
            dim=2,
        )
        # pad with zeros to match the input tensor shape => [B, L, 3]
        AD_features = F.pad(AD_features, (0, 0, 1, 2), "constant", 0)

        # build relative orientations
        # normalize consecutive unit vectors => [B, L-3, 3]
        o_1 = F.normalize(u_2 - u_1, dim=-1)
        # stack o_1 and n_2 vectors and their cross product => [B, L-3, 3, 3]
        O = torch.stack((o_1, n_2, torch.cross(o_1, n_2)), 2)
        O = O.view(list(O.shape[:2]) + [9])  # => [B, L-3, 9]
        O = F.pad(O, (0, 0, 1, 2), "constant", 0)  # => [B, L, 9]
        O_neighbors = gather_nodes(O, E_idx)  # => [B, L, K, 9]
        X_neighbors = gather_nodes(X, E_idx)  # => [B, L, K, 3]

        # reshape orientations and orientation neighbors as rotation matrices
        # O => [B, L, 3, 3]
        # O_neighbors => [B, L, K, 3, 3]
        O = O.view(list(O.shape[:2]) + [3, 3])
        O_neighbors = O_neighbors.view(list(O_neighbors.shape[:3]) + [3, 3])

        # compute difference between each Ca atom and its neighbors > [B, L, K, 3]
        dX = X_neighbors - X.unsqueeze(-2)
        # rotate differences into local reference frames => [B, L, K, 3]
        dU = torch.matmul(O.unsqueeze(2), dX.unsqueeze(-1)).squeeze(-1)
        dU = F.normalize(dU, dim=-1)
        # compute relative rotations between each Ca atom and its neighbors => [B, L, K, 3, 3]
        R = torch.matmul(O.unsqueeze(2).transpose(-1, -2), O_neighbors)
        # convert relative rotations to quaternions => [B, L, K, 4]
        Q = self._quaternions(R)

        # orientation features => [B, L, K, 7]
        O_features = torch.cat((dU, Q), dim=-1)
        return AD_features, O_features


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
