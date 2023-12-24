import copy
import os
import pathlib
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from IPython.core.display import clear_output
from einops import repeat, rearrange
from matplotlib import pyplot as plt

from artefact_nca.utils.minecraft import replace_colors
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
            sample_specific_pool: bool = True,
            random_tree_sampling: bool = True,
            equal_sized_samples: bool = False,
            load_embeddings: bool = False,
            embedding_dim: Optional[int] = None,
            num_hidden_channels: Any = 10,
            half_precision: bool = False,
            spawn_at_bottom: bool = True,
            use_random_seed_block: bool = False,
            device: Optional[Any] = None,
            input_shape: Optional[List[int]] = None,
            padding_by_power: Optional[int] = None,
            verbose: bool = False
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
        self.sample_random_tree = random_tree_sampling
        self.equal_sized_samples = equal_sized_samples
        self.load_entity_config['same_size'] = self.equal_sized_samples
        self.last_sample = 0
        self.input_shape = input_shape
        if not self.equal_sized_samples and not self.sample_specific_pools:
            raise ValueError("Sample specific pools are needed if padding is not equal for all trees")
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
            raise NotImplementedError
        #     p = padding_by_power
        #     current_power = 0
        #     while p < self.target_voxel.shape[-1]:
        #         p = padding_by_power ** current_power
        #         current_power += 1
        #     self.target_voxel = pad_target(self.target_voxel, p)
        #     self.current_power = current_power - 1
        self.num_categories = len(self.target_unique_val_dict)
        self.num_samples = 1
        if self.input_shape is not None:
            self.width = self.input_shape[0]
            self.depth = self.input_shape[1]
            self.height = self.input_shape[2]
        if self.target_voxel is not None:
            self.num_samples = self.target_voxel.shape[0] if self.equal_sized_samples else len(self.target_voxel)
            self.width = self.target_voxel.shape[1] if self.equal_sized_samples else self.target_voxel[0].shape[0]
            self.depth = self.target_voxel.shape[2] if self.equal_sized_samples else self.target_voxel[0].shape[1]
            self.height = self.target_voxel.shape[3] if self.equal_sized_samples else self.target_voxel[0].shape[2]
            self.targets = repeat(
                self.target_voxel, "t w d h -> t b d h w", b=self.pool_size
            ).astype(np.int) if self.equal_sized_samples else [repeat(
                t, "w d h -> b d h w", b=self.pool_size
            ).astype(np.int) for t in self.target_voxel]
        else:
            self.targets = np.zeros(
                (self.num_samples, self.pool_size, self.depth, self.height, self.width)
            ).astype(np.int) if self.equal_sized_samples else [np.zeros(
                (self.pool_size, self.depth, self.height, self.width)
            ).astype(np.int) for t in range(self.num_samples)]
        self.embedding_dim = embedding_dim
        self.embeddings = self.setup_embeddings()
        self.num_channels = num_hidden_channels + self.num_categories + 1
        self.living_channel_dim = self.num_categories
        self.half_precision = half_precision
        self.spawn_at_bottom = spawn_at_bottom
        self.use_random_seed_block = use_random_seed_block
        if self.half_precision:
            self.data = self.get_seed(self.pool_size).astype(np.float16) if self.equal_sized_samples else [s.astype(np.float16) for s in self.get_seed(self.pool_size)]
        else:
            self.data = self.get_seed(self.pool_size).astype(np.float32) if self.equal_sized_samples else [s.astype(np.float32) for s in self.get_seed(self.pool_size)]

    def to_device(self, device):
        self.data = torch.from_numpy(self.data).to(device) if self.equal_sized_samples else [torch.from_numpy(t).to(device) for t in self.data]
        self.targets = torch.from_numpy(self.targets).to(device).long() if self.equal_sized_samples else [torch.from_numpy(t).to(device).long() for t in self.targets]
        self.embeddings = torch.from_numpy(self.embeddings).to(device)

    def get_seed(self, batch_size=1):
        if self.sample_specific_pools:
            seed = np.zeros((self.num_samples, batch_size, self.depth, self.height, self.width, self.num_channels)) if self.equal_sized_samples else \
                [np.zeros((batch_size, t.shape[1], t.shape[2], t.shape[3], self.num_channels)) for t in self.targets]
        else:
            seed = np.zeros((1, batch_size, self.depth, self.height, self.width, self.num_channels)) if self.equal_sized_samples else \
                [np.zeros((batch_size, t.shape[1], t.shape[2], t.shape[3], self.num_channels)) for t in self.targets]
        # random_class_arr = np.eye(self.num_categories)[np.random.choice(np.arange(1,self.num_categories), batch_size)]
        randint = np.random.randint(1, self.num_categories, batch_size)
        if self.equal_sized_samples:
            if self.spawn_at_bottom:
                seed[:, :, self.depth // 3: 2*self.depth // 3, 0:3, self.width // 3: 2*self.width // 3, self.num_categories:] = 1.0
                if self.use_random_seed_block:
                    for i in range(randint.shape[0]):
                        seed[:, i, self.depth // 3: 2*self.depth // 3, 0:3, self.width // 3: 2*self.width // 3, randint[i]] = 1.0
            else:
                seed[:, :, self.depth // 3: 2*self.depth // 3, self.height // 3: 2*self.height // 3, self.width // 3: 2*self.width // 3, self.num_categories:] = 1.0
                if self.use_random_seed_block:
                    for i in range(randint.shape[0]):
                        seed[:, i, self.depth // 3: 2*self.depth // 3, self.height // 3: 2*self.height // 3, self.width // 3: 2*self.width // 3, randint[i]] = 1.0
        else:
            for j in range(len(seed)):
                if self.spawn_at_bottom:
                    seed[j][:, seed[j].shape[0] // 3: 2*seed[j].shape[0] // 3, 0:3, seed[j].shape[2] // 3: 2*seed[j].shape[2] // 3, self.num_categories:] = 1.0
                    if self.use_random_seed_block:
                        for i in range(randint.shape[0]):
                            seed[j][i, seed[j].shape[0] // 3: 2*seed[j].shape[0] // 3, 0:3, seed[j].shape[2] // 3: 2*seed[j].shape[2] // 3, randint[i]] = 1.0
                else:
                    seed[j][:, seed[j].shape[0] // 3: 2*seed[j].shape[0] // 3, seed[j].shape[1] // 3: 2*seed[j].shape[1] // 3, seed[j].shape[2] // 3: 2*seed[j].shape[2] // 3, self.num_categories:] = 1.0
                    if self.use_random_seed_block:
                        for i in range(randint.shape[0]):
                            seed[j][i, seed[j].shape[0] // 3: 2*seed[j].shape[0] // 3, seed[j].shape[1] // 3: 2*seed[j].shape[1] // 3, seed[j].shape[2] // 3: 2*seed[j].shape[2] // 3, randint[i]] = 1.0

        return seed

    def visualize_seed(self):
        raise NotImplementedError('needs non equal size adaptation!')
        if self.verbose:
            post_batch = rearrange(self.data[0], "b d h w c -> b w d h c")
            post_batch = replace_colors(
                np.argmax(post_batch[:, :, :, :, : self.num_categories], -1),
                self.target_color_dict,
            )
            clear_output()
            vis1 = post_batch[:5]
            num_cols = len(vis1)
            num_rows = 1
            vis1[vis1 == "_empty"] = None
            fig = plt.figure(figsize=(15, 10))

            for i in range(1, num_cols + 1):
                ax1 = fig.add_subplot(num_rows, num_cols, i, projection="3d")
                ax1.voxels(vis1[i - 1], facecolors=vis1[i - 1], edgecolor="k")
                ax1.set_title("Pool {}".format(i))
            plt.subplots_adjust(bottom=0.005)
            plt.show()

    def sample(self, batch_size):
        if self.sample_random_tree:
            tree = random.sample(range(self.num_samples), 1)
        else:
            tree = (self.last_sample + 1) % self.num_samples
        indices = random.sample(range(self.pool_size), batch_size)
        return self.get_data(tree, indices)

    def get_data(self, tree, indices):
        tree = tree[0] if isinstance(tree, list) else tree
        target = self.targets[tree]
        pools = self.data[tree] if self.sample_specific_pools else self.data[0]
        return pools[indices], target[indices], self.embeddings[tree], tree, indices

    def update_dataset_function(self, out, tree, indices, embedding=None, saveToFile=False):
        if self.sample_specific_pools:
            if self.equal_sized_samples:
                self.data[tree, indices] = out
            else:
                self.data[tree][indices] = out
        else:
            if self.equal_sized_samples:
                self.data[0, indices] = out
            else:
                self.data[0][indices] = out
        if not self.sample_random_tree:
            self.last_sample = tree
        if embedding is not None:
            self.embeddings[tree] = embedding
            if saveToFile:
                np.savetxt(os.path.join(self.nbt_path, 'embeddings.csv'), self.embeddings.detach().cpu().numpy().reshape((self.num_samples, -1)), delimiter=',', fmt='%10.5f')

    def save_embeddings(self, path):
        if self.embeddings is not None:
            np.savetxt(os.path.join(path, 'embeddings.csv'),
                       self.embeddings.detach().cpu().numpy().reshape((self.num_samples, -1)), delimiter=',',
                       fmt='%10.5f')

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
                if self.load_embeddings:
                    try:
                        embeddings = np.genfromtxt(os.path.join(self.nbt_path, 'embeddings.csv'), delimiter=",").astype(np.float32)
                    except Exception:
                        raise Exception("embeddings.txt does not exists in data folder")
                    shape = embeddings.shape
                    embeddings = embeddings.reshape((shape[0], 2, -1))
                    if shape[0] != self.num_samples:
                        raise ValueError("Number of embedding does not match number of loaded trees")
                    elif embeddings.shape[2] != self.embedding_dim:
                        raise ValueError("Number of embedding does not match num embeddings is csv file")
                    else:
                        if self.verbose: print(f'Loaded {shape[0]} trees with each {shape[1]} embeddings')
                        if self.verbose: print(embeddings)
                        return embeddings
                else:
                    embeddings = np.random.normal(loc=0, scale=0.2, size=(self.num_samples, 2, self.embedding_dim))
                    if self.verbose: print(embeddings)
                    return embeddings
