import os
import torch

from artefact_nca.trainer.voxel_ca_trainer import VoxelCATrainer

# base_nbt_path = "/home/thahit/github/Neural_Cellular_Automata_for_diverse_Tree_growing/artefact_nca/data/structs_dataset/acacia_trees"
base_nbt_path = "C:/Users/cedri/Desktop/Code/ETH/DLProject/Neural_Cellular_Automata_for_diverse_Tree_growing/artefact_nca/data/structs_dataset"
# base_nbt_path = "/mnt/c/Users/olezh/ETH/Deep Learning/Neural_Cellular_Automata_for_diverse_Tree_growing/artefact_nca/data/structs_dataset"

def find_file(directory, extension):
    for file in os.listdir(directory):
        if file.endswith(extension):
            return os.path.join(directory, file)

if __name__ == '__main__':
    dataset = 'various_trees'
    nbt_path = "{}/{}".format(base_nbt_path, dataset)
    checkpoints_path = "C:/Users/cedri/Desktop/Code/ETH/DLProject/Neural_Cellular_Automata_for_diverse_Tree_growing/checkpoints/2024-01-05-00-40-39_VariousTrees_20tree_final_pink/checkpoints"
    # checkpoints_path = "/mnt/c/Users/olezh/ETH/Deep Learning/Neural_Cellular_Automata_for_diverse_Tree_growing/checkpoints/2023-12-27-02-05-38_AcaciaTrees_8tree_final_pink/checkpoints"
    directory = f"{checkpoints_path}/{2600}"
    config_path = find_file(directory, ".yaml")
    pretrained_path = find_file(directory, ".pt")
    embedding_path = find_file(directory, ".csv")

    ct = VoxelCATrainer.from_config(
        "{}/{}/config.yaml".format(base_nbt_path, dataset),
        config={
            # "pretrained_path": pretrained_path,
            "use_cuda": torch.cuda.is_available(),
            "wandb": True,
            # "dataset_config": {"nbt_path": nbt_path, "embedding_path": embedding_path},
            "dataset_config": {"nbt_path": nbt_path},
        }
    )
    ct.train()
