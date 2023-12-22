import copy
import os
import pathlib
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from einops import repeat

from artefact_nca.utils.minecraft.block_loader import get_color_dict
from artefact_nca.utils.minecraft.minecraft_client import MinecraftClient


def pad_target(x, target_size):
    # assumes equal padding
    diff = target_size - x.shape[-1]
    padding = [(diff // 2, diff // 2) for i in range(3)]
    padding.insert(0, (0, 0))
    return np.pad(x, padding)


class VoxelDataset:
    def __init__(
            self,
            entity_name: Optional[str] = None,
            target_voxel: Optional[Any] = None,
            target_color_dict: Optional[Dict[Any, Any]] = None,
            target_unique_val_dict: Optional[Dict[Any, Any]] = None,
            nbt_path: Optional[str] = None,
            load_coord: List[int] = [0, 0, 0],
            load_entity_config: Dict[Any, Any] = {},
            pool_size: int = 48,
            sample_specific_pool: bool = False,
            load_embeddings: bool = True,
            num_hidden_channels: Any = 10,
            half_precision: bool = False,
            spawn_at_bottom: bool = True,
            use_random_seed_block: bool = False,
            device: Optional[Any] = None,
            input_shape: Optional[List[int]] = None,
            padding_by_power: Optional[int] = None,
            verbose: bool = True
    ):
        self.verbose = verbose
        self.entity_name = entity_name
        self.load_entity_config = load_entity_config
        self.load_coord = load_coord
        self.nbt_path = nbt_path
        self.load_embeddings = load_embeddings
        self.target_voxel = target_voxel
        self.target_color_dict = target_color_dict
        self.target_unique_val_dict = target_unique_val_dict
        self.pool_size = pool_size
        self.sample_specific_pools = sample_specific_pool
        self.last_sample = 0
        self.input_shape = input_shape
        if self.target_voxel is None and self.target_unique_val_dict is None:
            (
                _,
                _,
                self.target_voxel,
                self.target_color_dict,
                self.target_unique_val_dict,
            ) = MinecraftClient.load_entity(
                self.entity_name,
                nbt_path=self.nbt_path,
                load_coord=self.load_coord,
                load_entity_config=self.load_entity_config,
            )
        else:
            self.target_color_dict = get_color_dict(
                self.target_unique_val_dict.values()
            )
        if padding_by_power is not None:
            p = padding_by_power
            current_power = 0
            while p < self.target_voxel.shape[-1]:
                p = padding_by_power ** current_power
                current_power += 1
            self.target_voxel = pad_target(self.target_voxel, p)
            self.current_power = current_power - 1
        self.num_categories = len(self.target_unique_val_dict)
        if self.input_shape is not None:
            self.num_samples = 1
            self.width = self.input_shape[0]
            self.depth = self.input_shape[1]
            self.height = self.input_shape[2]
        if self.target_voxel is not None:
            self.num_samples = self.target_voxel.shape[0]
            self.width = self.target_voxel.shape[1]
            self.depth = self.target_voxel.shape[2]
            self.height = self.target_voxel.shape[3]
            self.targets = repeat(
                self.target_voxel, "t w d h -> t b d h w", b=self.pool_size
            ).astype(np.int)
        else:
            self.targets = np.zeros(
                (self.num_samples, self.pool_size, self.depth, self.height, self.width)
            ).astype(np.int)
        self.embedding_channels = 0
        if self.load_embeddings:
            self.embeddings = self.setup_embeddings()
        self.num_channels = num_hidden_channels + self.num_categories + 1
        self.living_channel_dim = self.num_categories
        self.half_precision = half_precision
        self.spawn_at_bottom = spawn_at_bottom
        self.use_random_seed_block = use_random_seed_block
        if self.half_precision:
            self.data = self.get_seed(self.pool_size).astype(np.float16)
        else:
            self.data = self.get_seed(self.pool_size).astype(np.float32)

    def to_device(self, device):
        self.data = torch.from_numpy(self.data).to(device)
        self.targets = torch.from_numpy(self.targets).to(device)
        self.targets = self.targets.long()

    def get_seed(self, batch_size=1):
        if self.sample_specific_pools:
            seed = np.zeros((self.num_samples, batch_size, self.depth, self.height, self.width, self.num_channels))
        else:
            seed = np.zeros((1, batch_size, self.depth, self.height, self.width, self.num_channels))
        # random_class_arr = np.eye(self.num_categories)[np.random.choice(np.arange(1,self.num_categories), batch_size)]
        randint = np.random.randint(1, self.num_categories, batch_size)
        if self.spawn_at_bottom:
            seed[:, :, self.depth // 2, 0, self.width // 2, self.num_categories:] = 1.0
            if self.use_random_seed_block:
                for i in range(randint.shape[0]):
                    seed[:, i, self.depth // 2, 0, self.width // 2, randint[i]] = 1.0
        else:
            seed[:, :, self.depth // 2, self.height // 2, self.width // 2, self.num_categories:
            ] = 1.0
            if self.use_random_seed_block:
                for i in range(randint.shape[0]):
                    seed[:, i, self.depth // 2, self.height // 2, self.width // 2, randint[i]] = 1.0
        return seed

    def sample(self, tree, batch_size):
        indices = random.sample(range(self.pool_size), batch_size)
        return self.get_data(tree, indices)

    def get_data(self, tree, indices):
        if self.sample_specific_pools:
            return self.data[tree, indices], self.targets[tree, indices], self.embeddings[tree], tree, indices
        return self.data[0, indices], self.targets[tree, indices], self.embeddings[tree], tree, indices

    def update_dataset_function(self, out, tree, indices, embedding=None, saveToFile=False):
        if self.sample_specific_pools:
            self.data[tree, indices] = out
        else:
            self.data[0, indices] = out
        if embedding is not None and self.load_embeddings:
            self.embeddings[tree] = embedding
            if saveToFile:
                np.savetxt(os.path.join(self.nbt_path, 'embeddings.csv'), self.embeddings, delimiter=',', fmt='%10.5f')

    def setup_embeddings(self):
        if self.nbt_path is None:
            raise ValueError("Must provide an nbt_path")
        if self.nbt_path is not None:
            if '.' in self.nbt_path:
                raise ValueError("Path must be of folder type or turn of embedding loading")
            else:
                p = Path(self.nbt_path)
                if not p.exists():
                    raise Exception("failed to find the data folder")

                try:
                    embeddings = np.genfromtxt(os.path.join(self.nbt_path, 'embeddings.csv'), delimiter=",").astype(np.float32)
                except Exception:
                    raise Exception("embeddings.txt does not exists in data folder")
                shape = embeddings.shape
                if shape[0] != self.num_samples:
                    raise ValueError("Number of embedding does not match number of loaded trees")
                else:
                    self.embedding_channels = shape[1]
                    if self.verbose: print(f'Loaded {shape[0]} trees with each {shape[1]} embeddings')
                    if self.verbose: print(embeddings)
                    return embeddings
