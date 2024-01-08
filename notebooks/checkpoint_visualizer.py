import os

from einops import rearrange
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import nbtlib
import open3d as o3d
from matplotlib.colors import rgb2hex, hex2color

from artefact_nca.trainer.voxel_ca_trainer import VoxelCATrainer
from artefact_nca.utils.minecraft.voxel_utils import replace_colors


def read_nbt(nbt_path):
    return nbtlib.load(find_file(nbt_path, ".nbt"))


def output_nbt(ct, nbt_data, output_data, step=0, directory=None):
    w = int(nbt_data.root['Width'])
    d = int(nbt_data.root['Length'])
    h = int(nbt_data.root['Height'])
    print(output_data.shape)
    for y in range(h):
        for x in range(w):
            for z in range(d):
                nbt_data.root['Blocks'][x + z * w + y * w * d] = ct.dataset.target_unique_val_dict[output_data[0, x, z, y]]
                nbt_data.root['Data'][x + z * w + y * w * d] = ct.dataset.block_to_data_dict[ct.dataset.target_unique_val_dict[output_data[x, z, y]]]
    nbt_data.save(f'{directory}/visualize_{step}.nbt')


def visualize_output(ct, out, nbt_data, step, directory=None):
    out = rearrange(out, 'b d h w c -> b w d h c')
    argmax = np.argmax(out[:, :, :, :, :ct.num_categories], -1)
    output_nbt(ct, nbt_data, argmax, step, directory)
    out = replace_colors(argmax, ct.dataset.target_color_dict)[0]
    ax = plt.figure().add_subplot(projection='3d')
    ax.voxels(out, facecolors=out, edgecolor='k')
    print(out)

    # plt.show()
    plt.savefig(f'{directory}/visualize_{step}.png')
    return argmax


def visualize_o3d(ct, out, step, directory=None):
    pcd = o3d.geometry.PointCloud()
    out = rearrange(out, 'b d h w c -> b w d h c')
    argmax = np.argmax(out[:, :, :, :, :ct.num_categories], -1)

    out = replace_colors(argmax, ct.dataset.target_color_dict)[0]
    print(out.shape)
    points = []
    colors = []
    for x in range(out.shape[0]):
        for y in range(out.shape[1]):
            for z in range(out.shape[2]):
                if out[x, y, z]:
                    points.append([x, y, z])
                    colors.append(hex2color(out[x, y, z]))
    pcd.points = o3d.utility.Vector3dVector(points)
    # fit to unit cube
    # pcd.scale(1 / np.max(pcd.get_max_bound() - pcd.get_min_bound()),
    #           center=pcd.get_center())
    pcd.colors = o3d.utility.Vector3dVector(colors) # np.random.uniform(0, 1, size=(len(points), 3))

    print('voxelization')
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd,
                                                                voxel_size=1)
    o3d.visualization.draw_geometries([voxel_grid])
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(voxel_grid)
    ctr = vis.get_view_control()
    # print("Field of view (before changing) %.2f" % ctr.get_field_of_view())
    # ctr.change_field_of_view(step=fov_step)
    # print("Field of view (after changing) %.2f" % ctr.get_field_of_view())
    # vis.run()
    vis.destroy_window()

    # vis = o3d.visualization.Visualizer()
    # vis.create_window()
    # vis.add_geometry(voxel_grid)
    # vis.update_geometry(voxel_grid)
    # vis.poll_events()
    # vis.update_renderer()
    # vis.capture_screen_image(path + '/test.png')
    # vis.destroy_window()


def find_file(directory, extension):
    for file in os.listdir(directory):
        if file.endswith(extension):
            return os.path.join(directory, file)


def rollout_model(checkpoints_path, checkpoint_number, nbt_path, steps=100):
    directory = f"{checkpoints_path}/{checkpoint_number}"
    config_path = find_file(directory, ".yaml")
    pretrained_path = find_file(directory, ".pt")
    embedding_path = find_file(directory, ".csv")
    nbt_data = read_nbt(nbt_path)
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
    ratio = steps // 10
    bar = tqdm(np.arange(10))

    # for i in bar:
    #     visualize_output(ct, states[i * ratio].cpu().numpy(), nbt_data, i * ratio, directory=directory)
    #
    # _ = visualize_output(ct, final.cpu().numpy(), nbt_data, len(states), directory=directory)
    visualize_o3d(ct, final.cpu().numpy(), len(states), directory=directory)


if __name__ == '__main__':
    # nbt_path = "/mnt/c/Users/olezh/ETH/Deep Learning/Neural_Cellular_Automata_for_diverse_Tree_growing/artefact_nca/data/structs_dataset/acacia_trees"
    # path = "/mnt/c/Users/olezh/ETH/Deep Learning/Neural_Cellular_Automata_for_diverse_Tree_growing/checkpoints/2023-12-27-02-05-38_AcaciaTrees_8tree_final_pink/checkpoints"
    nbt_path = "C:/Users/cedri/Desktop/Code/ETH/DLProject/Neural_Cellular_Automata_for_diverse_Tree_growing/artefact_nca/data/structs_dataset/acacia_trees"
    path = "C:/Users/cedri/Desktop/Code/ETH/DLProject/Neural_Cellular_Automata_for_diverse_Tree_growing/checkpoints/2023-12-26-19-03-11_AcaciaTrees_8tree_final_orange/checkpoints"
    rollout_model(path, 3000, nbt_path, steps=100)
