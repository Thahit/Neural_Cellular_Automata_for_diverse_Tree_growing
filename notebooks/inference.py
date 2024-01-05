from os import listdir
from os.path import isfile, join
import numpy as np
import torch
from matplotlib.colors import rgb2hex
import matplotlib.pyplot as plt
from einops import rearrange, repeat
import numpy as np
from IPython.display import clear_output

from artefact_nca.trainer.voxel_ca_trainer import VoxelCATrainer
from artefact_nca.utils.minecraft import MinecraftClient, convert_to_color, Blockloader, spawn_entities
from artefact_nca.utils.minecraft.voxel_utils import replace_colors

base_nbt_path = "/home/thahit/github/Neural_Cellular_Automata_for_diverse_Tree_growing/artefact_nca/data/structs_dataset"
#base_nbt_path = "C:/Users/cedri/Desktop/Code/ETH/DLProject/Neural_Cellular_Automata_for_diverse_Tree_growing/artefact_nca/data/structs_dataset"
# base_nbt_path = "/mnt/c/Users/olezh/ETH/Deep Learning/Neural_Cellular_Automata_for_diverse_Tree_growing/artefact_nca/data/structs_dataset/acacia_trees"


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
    # nbt_path = "{}/acacia_trees/acacia_003.nbt".format(base_nbt_path)
    nbt_path = "{}/acacia_trees".format(base_nbt_path)
    # blocks, unique_vals, target, color_dict, unique_val_dict = MinecraftClient.load_entity("trees",
    #                                                                                        nbt_path=base_nbt_path,
    #                                                                                        load_coord=(0, 0, 0),
    #                                                                                        load_entity_config={
    #                                                                                            'load_range': (
    #                                                                                                8, 15, 8)})
    ct = VoxelCATrainer.from_config(
        "{}/acacia_trees/config.yaml".format(base_nbt_path),
        config={
            "dataset_config": {"nbt_path": nbt_path},
            "pretrained_path":"artefact_nca/data/structs_dataset/acacia_trees/2023-12-25-22-50-30_AcaciaTrees_8tree_final_darkgreen/checkpoints/20000/AcaciaTrees_8tree_iteration_20000.pt",
        }
    )
    # targets = ct.dataset.targets
    # color_dict = ct.dataset.target_color_dict
    #
    # for i in range(len(targets)):
    #     target = targets[i].cpu().numpy()
    #     color_arr = convert_to_color(target[0, :, :, : , :3], color_dict)
    #
    #     ax = plt.figure().add_subplot(projection='3d')
    #     ax.voxels(color_arr, facecolors=color_arr, edgecolor='k')
    #
    #     plt.show()
    #ct.train()
    embedding = [0.,  0.,   0.,   0.,]
    embedding_params= [   0.01914,   0.07834,  -0.07155,   0.07578,  -0.00835,  -0.15394,  -0.10814,   0.26071]
    ct.infer(embeddings = embedding,
            #embedding_params = embedding_params,
               #dimensions=[12,20,12],
                )
