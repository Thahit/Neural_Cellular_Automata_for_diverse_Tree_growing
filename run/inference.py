import os

from artefact_nca.trainer.voxel_ca_trainer import VoxelCATrainer
from artefact_nca.utils.minecraft.voxel_utils import replace_colors

base_nbt_path = "C:/Users/cedri/Desktop/Code/ETH/DLProject/Neural_Cellular_Automata_for_diverse_Tree_growing/artefact_nca/data/structs_dataset"


def find_file(directory, extension):
    for file in os.listdir(directory):
        if file.endswith(extension):
            return os.path.join(directory, file)


if __name__ == '__main__':
    dataset = 'various_trees'
    nbt_path = "{}/{}".format(base_nbt_path, dataset)
    checkpoints_path = "C:/Users/cedri/Desktop/Code/ETH/DLProject/Neural_Cellular_Automata_for_diverse_Tree_growing/checkpoints/2024-01-05-00-40-39_VariousTrees_20tree_final_pink/checkpoints"
    directory = f"{checkpoints_path}/{2600}"
    config_path = find_file(directory, ".yaml")
    pretrained_path = find_file(directory, ".pt")
    embedding_path = find_file(directory, ".csv")

    ct = VoxelCATrainer.from_config(
        config_path,
        config={
            "dataset_config": {"nbt_path": nbt_path, "embedding_path": embedding_path},
            "pretrained_path": pretrained_path,
            "wandb": False,
        }
    )
    embedding = [0, 0, 0, 0]
    embedding_params = [0.01914, 0.07834, -0.07155, 0.07578, -0.00835, -0.15394, -0.10814, 0.26071]
    ct.infer(embeddings=embedding,
             # embedding_params = embedding_params,
             dimensions=[10, 30, 10],
             steps=80
             )
