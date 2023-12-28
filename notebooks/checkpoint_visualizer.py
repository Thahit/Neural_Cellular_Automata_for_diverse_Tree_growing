import os

from einops import rearrange
import numpy as np
import torch
import matplotlib.pyplot as plt

from artefact_nca.trainer.voxel_ca_trainer import VoxelCATrainer
from artefact_nca.utils.minecraft.voxel_utils import replace_colors


def visualize_output(ct, out):
    out = rearrange(out, 'b d h w c -> b w d h c')
    argmax = np.argmax(out[:, :, :, :, :ct.num_categories], -1)
    out = replace_colors(argmax, ct.dataset.target_color_dict)[0]
    ax = plt.figure().add_subplot(projection='3d')
    ax.voxels(out, facecolors=out, edgecolor='k')

    #plt.show()
    plt.savefig("visualize.png")
    return argmax

def find_file(directory, extension):
    for file in os.listdir(directory):
        if file.endswith(extension):
            return os.path.join(directory, file)

def rollout_model(checkpoints_path, checkpoint_number, nbt_path):
    directory = f"{checkpoints_path}/{checkpoint_number}"
    config_path = find_file(directory, ".yaml")
    pretrained_path = find_file(directory, ".pt")
    ct = VoxelCATrainer.from_config(
                    config_path,
                    config={
                        "pretrained_path": pretrained_path,
                        "use_cuda":False,
                        "dataset_config":{"nbt_path": nbt_path},
                    }
                )
    with torch.no_grad():
        final, states, life_masks = ct.rollout(steps=100)
    _ = visualize_output(ct, final.cpu().numpy())
    

if __name__=='__main__':
    nbt_path = "/mnt/c/Users/olezh/ETH/Deep Learning/Neural_Cellular_Automata_for_diverse_Tree_growing/artefact_nca/data/structs_dataset/acacia_trees"
    path = "/mnt/c/Users/olezh/ETH/Deep Learning/Neural_Cellular_Automata_for_diverse_Tree_growing/checkpoints/2023-12-27-02-05-38_AcaciaTrees_8tree_final_pink/checkpoints"
    rollout_model(path, 1200, nbt_path)