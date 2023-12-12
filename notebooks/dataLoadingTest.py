from os import listdir
from os.path import isfile, join
import numpy as np
import torch
from matplotlib.colors import rgb2hex
import matplotlib.pyplot as plt
from einops import rearrange, repeat
import numpy as np
from IPython.display import clear_output
import torch
from hydra.utils import instantiate
from hydra.experimental import initialize, initialize_config_dir, compose

from artefact_nca.trainer.voxel_ca_trainer import VoxelCATrainer
from artefact_nca.utils.minecraft import MinecraftClient, convert_to_color, Blockloader, spawn_entities
from artefact_nca.utils.minecraft.voxel_utils import replace_colors

base_nbt_path = "C:\\Users\\cedri\\Desktop\\Code\\ETH\\DLProject\\Neural_Cellular_Automata_for_diverse_Tree_growing\\artefact_nca\\data\\structs_dataset\\acacia_trees".replace(
    '\\', '/')


def visualize_output(ct, out):
    clear_output()
    out = rearrange(out, 'b d h w c -> b w d h c')
    argmax = np.argmax(out[:, :, :, :, :ct.num_categories], -1)
    out = replace_colors(argmax, ct.dataset.target_color_dict)[0]
    ax = plt.figure().add_subplot(projection='3d')
    ax.voxels(out, facecolors=out, edgecolor='k')

    plt.show()
    return argmax


if __name__ == '__main__':
    nbt_path = "{}/acacia_001.nbt".format(base_nbt_path)
    blocks, unique_vals, target, color_dict, unique_val_dict = MinecraftClient.load_entity("trees", nbt_path=base_nbt_path,
                                                                                           load_coord=(0, 0, 0), load_entity_config={'load_range': (6, 14, 6)})
    for i in range(target.shape[0]):
        color_arr = convert_to_color(target[i], color_dict)

        ax = plt.figure().add_subplot(projection='3d')
        ax.voxels(color_arr, facecolors=color_arr, edgecolor='k')

        plt.show()