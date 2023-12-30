import os

from einops import rearrange
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

from artefact_nca.trainer.voxel_ca_trainer import VoxelCATrainer
from artefact_nca.utils.minecraft.voxel_utils import replace_colors


def visualize_output(ct, out, step, directory=None):
    out = rearrange(out, 'b d h w c -> b w d h c')
    argmax = np.argmax(out[:, :, :, :, :ct.num_categories], -1)
    out = replace_colors(argmax, ct.dataset.target_color_dict)[0]
    ax = plt.figure().add_subplot(projection='3d')
    ax.voxels(out, facecolors=out, edgecolor='k')

    # plt.show()
    plt.savefig(f'{directory}/visualize_{step}.png')
    return argmax


def find_file(directory, extension):
    for file in os.listdir(directory):
        if file.endswith(extension):
            return os.path.join(directory, file)


def rollout_model(checkpoints_path, checkpoint_number, nbt_path, steps=100):
    directory = f"{checkpoints_path}/{checkpoint_number}"
    config_path = find_file(directory, ".yaml")
    pretrained_path = find_file(directory, ".pt")
    embedding_path = find_file(directory, ".csv")
    ct = VoxelCATrainer.from_config(
        config_path,
        config={
            "pretrained_path": pretrained_path,
            "use_cuda": torch.cuda.is_available(),
            "wandb": False,
            "dataset_config": {"nbt_path": nbt_path, "embedding_path": embedding_path},
        }
    )
    final, states, life_masks = ct.rollout(steps=steps)
    ratio = steps//10
    bar = tqdm(np.arange(10))

    for i in bar:
        visualize_output(ct, states[i * ratio].cpu().numpy(), i * ratio, directory=directory)

    _ = visualize_output(ct, final.cpu().numpy(), len(states), directory=directory)


if __name__ == '__main__':
    # nbt_path = "/mnt/c/Users/olezh/ETH/Deep Learning/Neural_Cellular_Automata_for_diverse_Tree_growing/artefact_nca/data/structs_dataset/acacia_trees"
    # path = "/mnt/c/Users/olezh/ETH/Deep Learning/Neural_Cellular_Automata_for_diverse_Tree_growing/checkpoints/2023-12-27-02-05-38_AcaciaTrees_8tree_final_pink/checkpoints"
    nbt_path = "C:/Users/cedri/Desktop/Code/ETH/DLProject/Neural_Cellular_Automata_for_diverse_Tree_growing/artefact_nca/data/structs_dataset/acacia_trees"
    path = "C:/Users/cedri/Desktop/Code/ETH/DLProject/Neural_Cellular_Automata_for_diverse_Tree_growing/checkpoints/2023-12-26-19-03-11_AcaciaTrees_8tree_final_orange/checkpoints"
    rollout_model(path, 3000, nbt_path, steps=10000)
