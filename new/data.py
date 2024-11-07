# Copyright (c) 2024 brineylab @ scripps
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT


import json
import time
from typing import Optional

import numpy as np

from .pdb import parse_PDBs


class StructureDataset:
    def __init__(
        self,
        data_path: str,
        verbose: bool = True,
        truncate: Optional[int] = None,
        max_length: int = 100,
        alphabet: str = "ACDEFGHIKLMNPQRSTVWYX-",
        fmt: str = "pdb",
    ):
        """
        Initialize the dataset.

        Parameters
        ----------
        data_path: str
            Path to the data. Can be a JSONL file, a PDB file, or a directory containing one or more PDB files.

        verbose: bool
            Whether to print verbose output.

        truncate: Optional[int]
            Number of entries to truncate the dataset to.

        max_length: int
            Maximum length of the sequence.

        alphabet: str
            Alphabet to use for the sequence.

        fmt: str
            Format of the data. Can be either "pdb" or "jsonl".

        """
        self.data_path = data_path
        self.verbose = verbose
        self.truncate = truncate
        self.max_length = max_length
        self.alphabet = alphabet
        self.alphabet_set = set([a for a in self.alphabet])
        self.fmt = fmt

        if self.fmt == "pdb":
            self.data = self._process_pdb(self.data_path)
        elif self.fmt == "jsonl":
            self.data = self._process_jsonl(self.data_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        return self.data[idx]

    def __iter__(self):
        return iter(self.data)

    def _process_pdb(self, pdb_path: str):
        """
        Process a PDB file or a directory containing PDB files.

        Parameters
        ----------
        pdb_path: str
            Path to the PDB file or a directory containing PDB files.

        Returns
        -------
        list[dict]
            List of dictionaries containing the data. If a single PDB file is provided, the list will
            contain one dictionary. If a directory is provided, the list will contain multiple dictionaries.

        """
        start = time.time()
        discard_count = {"bad_chars": 0, "too_long": 0}
        data = []
        pdb_dict_list = parse_PDBs(pdb_path, verbose=self.verbose)
        for i, entry in enumerate(pdb_dict_list):
            # check for problems
            problems = self._check_for_problems(entry["seq"])
            if problems is None:
                data.append(entry)
            else:
                discard_count[problems] += 1
            # truncate the dataset size
            if self.truncate is not None and len(data) == self.truncate:
                return
            # progress
            if self.verbose and (i + 1) % 1000 == 0:
                elapsed = time.time() - start
                print(f"{len(data)} entries ({i + 1} loaded) in {elapsed:.1f} seconds")
        if self.verbose:
            print("discarded", discard_count)
        return data

    def _process_jsonl(self, jsonl_path: str):
        """
        Process a JSONL file.

        Parameters
        ----------
        jsonl_path: str
            Path to the JSONL file.

        Returns
        -------
        list[dict]
            List of dictionaries containing the data.

        """
        start = time.time()
        discard_count = {"bad_chars": 0, "too_long": 0}
        data = []
        with open(jsonl_path) as f:
            for i, line in enumerate(f):
                entry = json.loads(line.strip())
                # check for problems
                problems = self._check_for_problems(entry["seq"])
                if problems is None:
                    data.append(entry)
                else:
                    discard_count[problems] += 1
                # truncate the dataset size
                if self.truncate is not None and len(data) == self.truncate:
                    return
                # progress
                if self.verbose and (i + 1) % 1000 == 0:
                    elapsed = time.time() - start
                    print(
                        f"{len(data)} entries ({i + 1} loaded) in {elapsed:.1f} seconds"
                    )
            if self.verbose:
                print("discarded", discard_count)
        return data

    def _check_for_problems(self, sequence: str) -> Optional[str]:
        """
        Check a sequence for problems: non-alphabet characters or too long.

        Parameters
        ----------
        sequence: str
            Sequence to check.

        Returns
        -------
        Optional[str]
            Problem type if there is a problem, otherwise None.

        """
        bad_chars = set([s for s in sequence]).difference(self.alphabet_set)
        if len(bad_chars) > 0:
            return "bad_chars"
        elif len(sequence) > self.max_length:
            return "too_long"
        return None


class StructureLoader:
    def __init__(
        self,
        dataset: StructureDataset,
        batch_size: int = 10000,
        shuffle: bool = True,
        drop_last: bool = False,
    ):
        """
        Initialize the loader.

        Parameters
        ----------
        dataset: StructureDataset
            Dataset to load.

        batch_size: int
            Batch size (in tokens, not sequences).

        shuffle: bool
            Whether to shuffle the data.

        collate_fn: Callable
            Collation function.

        drop_last: bool
            Whether to drop the last batch if it is not full.

        """
        self.dataset = dataset
        self.size = len(dataset)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.batches = []

        # cluster into batches of similar sizes
        batch = []
        for sequence in sorted(self.dataset, key=lambda x: len(x["seq"]), reverse=True):
            size = len(sequence["seq"])
            if size * (len(batch) + 1) <= self.batch_size:
                batch.append(sequence)
            else:
                self.batches.append(batch)
                batch = []
        if batch:
            if len(batch) < self.batch_size and not self.drop_last:
                self.batches.append(batch)

    def __len__(self):
        return len(self.batches)

    def __iter__(self):
        if self.shuffle:
            np.random.shuffle(self.batches)
        for batch in self.batches:
            yield batch
