import random
from enum import Enum
from typing import Any, Dict, List, Optional

import typing
import attr
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from einops import rearrange
from hydra.utils import instantiate
from IPython.display import clear_output
from matplotlib.colors import hex2color
from tqdm import tqdm
from loguru import logger

# internal
from artefact_nca.base.base_torch_trainer import BaseTorchTrainer
from artefact_nca.dataset.voxel_dataset import VoxelDataset
from artefact_nca.model.voxel_ca_model import VoxelCAModel
from artefact_nca.utils.minecraft import *  # noqa
from artefact_nca.utils.minecraft.voxel_utils import replace_colors
import wandb
from omegaconf import OmegaConf


# zero out a cube
def damage_cube(state, x, y, z, half_width):
    damaged = state.clone()

    x_dim = state.shape[1]
    y_dim = state.shape[2]
    z_dim = state.shape[3]

    from_x = np.clip(x - half_width, a_min=0, a_max=x_dim)
    from_y = np.clip(y - half_width, a_min=0, a_max=y_dim)
    from_z = np.clip(z - half_width, a_min=0, a_max=z_dim)

    to_x = np.clip(x + half_width, a_min=0, a_max=x_dim)
    to_y = np.clip(y + half_width, a_min=0, a_max=y_dim)
    to_z = np.clip(z + half_width, a_min=0, a_max=z_dim)

    damaged[:, from_x:to_x, from_y:to_y, from_z:to_z, :] = 0
    return damaged


# zero out a sphere
def damage_sphere(state, x, y, z, radius):
    damaged = state.clone()

    x_ = list(range(state.shape[1]))
    y_ = list(range(state.shape[2]))
    z_ = list(range(state.shape[3]))

    x_, y_, z_ = np.meshgrid(x_, y_, z_, indexing="ij")

    dist_squared = (x_ - x) * (x_ - x) + (y_ - y) * (y_ - y) + (z_ - z) * (z_ - z)
    not_in_sphere = dist_squared > (radius * radius)
    damaged = state * not_in_sphere
    return damaged


@attr.s
class VoxelCATrainer(BaseTorchTrainer):
    _config_name_: str = "voxel"

    damage_radius_denominator: int = attr.ib(default=5)
    use_iou_loss: bool = attr.ib(default=True)
    use_bce_loss: bool = attr.ib(default=False)
    use_sample_pool: bool = attr.ib(default=True)
    num_hidden_channels: Optional[int] = attr.ib(default=12)
    embedding_dim: Optional[int] = attr.ib(default=None)
    num_categories: Optional[int] = attr.ib(default=None)
    variational: bool = attr.ib(default=True)
    use_dataset: bool = attr.ib(default=True)
    use_model: bool = attr.ib(default=True)
    half_precision: bool = attr.ib(default=False)
    min_steps: int = attr.ib(default=48)
    max_steps: int = attr.ib(default=64)
    damage: bool = attr.ib(default=False)
    num_damaged: int = attr.ib(default=2)
    torch_seed: Optional[int] = attr.ib(default=None)
    update_dataset: bool = attr.ib(default=True)
    seed: Optional[Any] = attr.ib(default=None)
    var_lr: float = attr.ib(default=None)
    var_loss_weight: float = attr.ib(default=1.)
    clip_gradients: bool = attr.ib(default=False)
    use_index: bool = attr.ib(default=False)
    random: bool = attr.ib(default=False)
    learnable_embeddings = False
    num_channels = 0

    def post_dataset_setup(self):
        print("Post dataset setup!")
        print(
            f'Target: #Trees: {self.dataset.num_samples} | #DifBlocks: {self.dataset.num_categories} \n| Target shape: {[t.shape for t in self.dataset.targets]} \n| Target voxel shape: {[t.shape for t in self.dataset.target_voxel]}')
        print(
            f'Data: #Pools: {self.dataset.pool_size} | #PoolsPerTree: {self.dataset.sample_specific_pools} | Data shape: {[t.shape for t in self.dataset.data]}')

        # self.dataset.visualize_seed()
        self.num_samples = self.dataset.num_samples
        self.num_categories = self.dataset.num_categories
        self.num_channels = self.dataset.num_channels
        self.model_config["living_channel_dim"] = self.num_categories

        self.learnable_embeddings = self.variational or (self.embedding_dim and (not self.random) and self.use_index)

    def get_seed(self, batch_size=1, tree=None):
        if tree is None:
            return self.dataset.get_seeds(batch_size)
        else:
            return self.dataset.get_seed(tree, batch_size)

    def sample_batch(self, tree: int, batch_size: int):
        return self.dataset.sample(tree, batch_size)

    # def log_epoch(self, train_metrics, epoch):
    #     for metric in train_metrics:
    #         self.tensorboard_logger.log_scalar(
    #             train_metrics[metric], metric, step=epoch
    #         )

    def rank_loss_function(self, x, targets):
        x = rearrange(x, "b d h w c -> b c d h w").to(self.device)
        out = torch.mean(
            F.cross_entropy(
                x[:, : self.num_categories, :, :, :].float(), targets, reduction="none"
            ),
            dim=[-2, -3, -1],
        )
        return out

    def apply_damage(self, batch, coords=None, num_damaged=None):

        x_ = list(range(batch.shape[1]))
        y_ = list(range(batch.shape[2]))
        z_ = list(range(batch.shape[3]))
        r = np.max(batch.shape) // self.damage_radius_denominator
        if num_damaged is None:
            num_damaged = self.num_damaged
        for i in range(1, num_damaged + 1):
            if coords is None:
                x = np.random.randint(0, batch.shape[1])
                y = np.random.randint(0, batch.shape[2])
                z = np.random.randint(0, batch.shape[3])
            center = [x, y, z]
            coords = np.ogrid[: batch.shape[1], : batch.shape[2], : batch.shape[3]]
            distance = np.sqrt(
                (coords[0] - center[0]) ** 2
                + (coords[1] - center[1]) ** 2
                + (coords[2] - center[2]) ** 2
            )
            not_in_sphere = 1 * (distance > r)
            ones = (
                    np.ones((batch.shape[1], batch.shape[2], batch.shape[3]))
                    - not_in_sphere
            )
            batch[-i] *= torch.from_numpy(not_in_sphere[:, :, :, None]).to(self.device)
            batch[-i][:, :, :, 0] += torch.from_numpy(ones).to(self.device)
        return batch

    def setup_trainer(self):
        self.current_iteration = 0
        if self.name is None:
            self.name = self.__class__.__name__
        self.additional_tune_config = self.tune_config["additional_config"]
        self.tune_config = {
            **{
                k: self.tune_config[k]
                for k in self.tune_config
                if k != "additional_config"
            },
            **self.additional_tune_config,
        }
        self.config = OmegaConf.to_container(self.config)
        self.config["trainer"]["name"] = self.name
        self.device = torch.device(
            "cuda:{}".format(self.device_id) if self.use_cuda else "cpu"
        )
        self.setup()
        self.setup_logging_and_checkpoints()
        self._setup_dataset()
        self.setup_dataloader()
        self._setup_model()
        self._setup_optimizer()
        self.load(self.pretrained_path)
        self.setup_device()
        self.post_setup()
        if self.wandb:
            wandb.init(
                # set the wandb project where this run will be logged
                project="NCA",
                name=self.name,
                # track hyperparameters and run metadata
                config={
                    "dataset_name": self.name.split('_')[0],
                    "num_samples": self.num_samples,
                    "epochs": self.epochs,
                    "batch_size": self.batch_size,
                    "min_steps": self.min_steps,
                    "max_steps": self.max_steps,
                    "damage": self.damage,
                    "num_hidden_channels": self.num_hidden_channels,
                    "num_categories": self.num_categories,
                    "num_embedding_channels": self.embedding_dim,
                    "num_channels": self.num_channels,
                    "variational": self.variational,
                    "use_index": self.use_index,
                    "random": self.random,
                    "var_lr": self.var_lr,
                    "var_loss_weight": self.var_loss_weight,
                    "dataset_conf": {"dimensions": self.dataset.dimensions,
                                     "num_channels": self.dataset.num_channels,
                                     "living_channel": self.dataset.living_channel_dim,
                                     "spawn_at_bottom": self.dataset.spawn_at_bottom,
                                     "use_random_seed_block": self.dataset.use_random_seed_block,
                                     "sample_specific_pools": self.dataset.sample_specific_pools,
                                     "equal_sized_samples": self.dataset.equal_sized_samples,
                                     "load_embeddings": self.dataset.load_embeddings,
                                     },
                    "half_precision": self.half_precision
                })

    def visualize_one(self, out, step=0, method='o3d'):
        out = rearrange(out, "b d h w c -> b w d h c")
        out = replace_colors(
            np.argmax(out[:, :, :, :, : self.num_categories], -1),
            self.dataset.target_color_dict,
        )
        if method == 'o3d':
            out = out[0]
            import open3d as o3d
            points = []
            colors = []
            for x in range(out.shape[0]):
                for y in range(out.shape[1]):
                    for z in range(out.shape[2]):
                        if out[x, y, z]:
                            points.append([x, z, y])
                            ran = (random.random() - 0.5) * 0.1
                            colors.append([min(255, max(c + ran, 0)) for c in hex2color(out[x, y, z])])
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.colors = o3d.utility.Vector3dVector(colors)  # np.random.uniform(0, 1, size=(len(points), 3))

            print('voxelization step: ', step)
            voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd,
                                                                        voxel_size=1)
            # o3d.visualization.draw_geometries([voxel_grid])
            vis = o3d.visualization.Visualizer()
            vis.create_window()
            vis.add_geometry(voxel_grid)
            ctr = vis.get_view_control()
            ctr.set_zoom(0.9)
            ctr.change_field_of_view(step=1)
            ctr.rotate(-200, 30)
            vis.run()
            vis.destroy_window()
        else:
            clear_output()
            print(f'step: ', step)
            fig = plt.figure(figsize=(15, 10))
            ax = fig.add_subplot(projection='3d')
            vis0 = out[:5]
            vis0[vis0 == "_empty"] = None
            ax.voxels(vis0[0], facecolors=vis0[0], edgecolor="k")
            # ax.voxels(voxelarray, facecolors=colors, edgecolor='k')

            plt.show()

    def visualize(self, out):
        # for tree in range(self.dataset.num_samples):
        for tree in range(1):
            prev_batch = out["prev_batch"][tree]
            post_batch = out["post_batch"][tree]
            prev_batch = rearrange(prev_batch, "b d h w c -> b w d h c")
            post_batch = rearrange(post_batch, "b d h w c -> b w d h c")
            prev_batch = replace_colors(
                np.argmax(prev_batch[:, :, :, :, : self.num_categories], -1),
                self.dataset.target_color_dict,
            )
            post_batch = replace_colors(
                np.argmax(post_batch[:, :, :, :, : self.num_categories], -1),
                self.dataset.target_color_dict,
            )
            clear_output()
            vis0 = prev_batch[:5]
            vis1 = post_batch[:5]
            num_cols = len(vis0)
            vis0[vis0 == "_empty"] = None
            vis1[vis1 == "_empty"] = None
            print(f'Before --- After --- Tree {tree}')
            fig = plt.figure(figsize=(15, 10))
            for i in range(1, num_cols + 1):
                ax0 = fig.add_subplot(1, num_cols, i, projection="3d")
                ax0.voxels(vis0[i - 1], facecolors=vis0[i - 1], edgecolor="k")
                ax0.set_title("Index {}".format(i))
            for i in range(1, num_cols + 1):
                ax1 = fig.add_subplot(2, num_cols, i + num_cols, projection="3d")
                ax1.voxels(vis1[i - 1], facecolors=vis1[i - 1], edgecolor="k")
                ax1.set_title("Index {}".format(i))
            plt.subplots_adjust(bottom=0.005)
            plt.show()

    def rollout(self, initial=None, steps=100, tree=0):
        """
            Deprecated & Buggy
        """
        if initial is None:
            initial, _, embedding, _, _ = self.sample_batch(tree, 1)
        if not isinstance(initial, torch.Tensor):
            initial = torch.from_numpy(initial).to(self.device)

        embedding_input = None
        if self.embedding_dim:
            embedding = embedding.reshape(2, -1)  # dont come in correct shape
            if self.random:
                embedding_input = torch.normal(mean=0, std=1, size=(self.batch_size, self.embedding_dim))
                embedding_input = embedding_input.to(self.device)
            elif self.variational:
                embedding_input = torch.normal(mean=0, std=1, size=(self.batch_size, self.embedding_dim))
                embedding_input = embedding_input.to(self.device)
                embedding_input *= torch.exp(0.5 * embedding[1])  # var
                embedding_input += embedding[0]  # mean
            else:
                embedding_input = torch.ones((self.batch_size, self.embedding_dim))
                embedding_input = embedding_input.to(self.device)
                if self.use_index:
                    embedding_input *= tree / self.num_samples
                else:  # encoder
                    embedding = embedding[0]  # because wrong shape
                    embedding_input *= embedding

            shape_to_emulate = [dim_i for dim_i in initial.shape[1:-1]]
            shape_to_emulate.extend([-1, -1])
            embedding_input = embedding_input.expand(shape_to_emulate)  # l,h,w, batch, channel
            embedding_input = embedding_input.permute(-2, 0, 1, 2, -1)  # back to b d h w c
        bar = tqdm(np.arange(steps))
        out = [initial]
        x = initial
        life_masks = [None]
        with torch.no_grad():
            for i in bar:
                x, life_mask = self.model(x, embeddings=embedding_input[tree:tree + 1], steps=1, return_life_mask=True)
                life_masks.append(life_mask)
                o = rearrange(x, "b c d h w -> b d h w c")
                out.append(o)
        return out[-1], out, life_masks

    def train(
            self, batch_size=None, epochs=None, checkpoint_interval=None, visualize=None
    ) -> typing.Dict[str, Any]:
        """Main training function, should call train_iter
        """
        if batch_size is not None:
            self.batch_size = batch_size
        if epochs is not None:
            self.epochs = epochs
        if checkpoint_interval is not None:
            self.checkpoint_interval = checkpoint_interval
        self.pre_train()
        self.setup_logging_and_checkpoints()
        logger.info(
            "Follow tensorboard logs with: tensorboard --logdir {}".format(
                self.tensorboard_log_path
            )
        )
        self.setup_dataloader()
        bar = tqdm(np.arange(self.epochs))
        for i in bar:
            save = i % self.checkpoint_interval == 0
            self.pre_train_iter()
            output = self.train_iter(self.batch_size, i, save)
            self.post_train_iter(output)
            metrics = output.get("metrics", {})
            loss = output["loss"]
            self.log_epoch(metrics, i)

            description = "--".join(["{}:{}".format(k, metrics[k]) for k in metrics])
            bar.set_description(description)
            if save:
                if self.visualize_output:
                    self.visualize(output)
                self.save(step=i)
            if self.early_stoppage:
                if loss <= self.loss_threshold:
                    self.save(step=i)
                    break
        self.post_train()
        return metrics

    def update_dataset_function(self, out, tree, indices, embedding=None, save_emb=False):
        with torch.no_grad():
            if self.half_precision:
                self.dataset.update_dataset_function(out.detach().type(torch.float16), tree, indices, embedding,
                                                     save_emb)
            else:
                self.dataset.update_dataset_function(out.detach(), tree, indices, embedding, save_emb)

    def iou(self, out, targets):
        targets = torch.clamp(targets, min=0, max=1)
        out = torch.clamp(
            torch.argmax(out[:, : self.num_categories, :, :, :], 1), min=0, max=1
        )
        intersect = torch.sum(out & targets).float()
        union = torch.sum(out | targets).float()
        o = (union - intersect) / (union + 1e-8)
        return o

    def get_loss(self, x, targets, embedding_params=None):
        iou_loss = 0
        if self.use_iou_loss:
            iou_loss = self.iou(x, targets)
        if self.use_bce_loss:
            weight_class = torch.ones(self.num_categories)
            weight_class[0] = 0.01
            class_loss = F.cross_entropy(
                x[:, : self.num_categories, :, :, :], targets, weight=weight_class.to(self.device)
            )
            alive = torch.clip(x[:, self.num_categories, :, :, :], 0.0, 1.0)
            alive_target_cells = torch.clip(targets, 0, 1).float()
            alive_loss = torch.nn.MSELoss()(alive, alive_target_cells)
        else:
            weight_class = torch.ones(self.num_categories)
            weight_class[0] = 0.01
            class_loss = F.cross_entropy(
                x[:, : self.num_categories, :, :, :], targets, weight=weight_class.to(self.device)
            )
            weight = torch.zeros(self.num_categories)
            weight[0] = 1.0
            alive_loss = F.cross_entropy(
                x[:, : self.num_categories, :, :, :],
                targets,
                weight=weight.to(self.device),
            )

        variational_loss = 0

        if self.variational:  # kl div
            variational_loss = - 0.5 * torch.sum(
                1 + embedding_params[1] - embedding_params[0].pow(2) - embedding_params[1].exp())

        loss = (0.5 * class_loss + 0.5 * alive_loss + iou_loss) / 3.0
        loss += variational_loss * self.var_loss_weight
        return loss, iou_loss, class_loss, variational_loss

    def infer(self, embeddings=None, embedding_params=None, tree_id=0, steps=64, stepsize=8, dimensions=[11, 18, 11]):
        # batch, targets, embedding, tree, indices = self.sample_batch(0, 1)#batchsize
        batch = torch.Tensor(self.dataset.get_seed_custom(dimensions)).to(self.device)
        # batch = torch.zeros((1, dimensions[0], dimensions[1], dimensions[2], self.num_channels)).to(self.device)

        if embeddings != None:  # input values you are interested in
            embedding_input = torch.Tensor(embeddings).to(self.device)
            embedding_input = embedding_input.reshape(1, -1)
            shape_to_emulate = [dim_i for dim_i in batch.shape[1:-1]]
            shape_to_emulate.extend([-1, -1])
            # shape_to_emulate[:2] = embedding_input.shape
            embedding_input = embedding_input.expand(shape_to_emulate)  # l,h,w, batch, channel
            embedding_input = embedding_input.permute(-2, 0, 1, 2, -1)  # back to b d h w c
            # have to permute cannot do this direclty
        else:
            if embedding_params != None:
                embedding = torch.Tensor(embedding_params).to(self.device)
            else:
                _, _, embedding, _, _ = self.sample_batch(tree_id, 1)

            embedding_input = None
            if self.embedding_dim:
                embedding = embedding.reshape(2, -1)  # dont come in correct shape
                embedding_input = torch.ones((self.batch_size, self.embedding_dim))
                embedding_input = embedding_input.to(self.device)
                if self.random:
                    embedding_input = torch.normal(mean=0, std=1, size=(self.batch_size, self.embedding_dim))
                    embedding_input = embedding_input.to(self.device)
                elif self.variational:
                    embedding_input = torch.normal(mean=0, std=1, size=(self.batch_size, self.embedding_dim))
                    embedding_input = embedding_input.to(self.device)
                    embedding_input *= torch.exp(0.5 * embedding[1])  # var
                    embedding_input += embedding[0]  # mean
                else:
                    if self.use_index:
                        embedding_input *= tree_id / self.num_samples
                    else:  # encoder
                        embedding = embedding[0]  # because wrong shape
                        embedding_input *= embedding

                # shape_to_emulate = [num for num in batch.shape]#batch channel, l, h ,w
                shape_to_emulate = [dim_i for dim_i in batch.shape[1:-1]]
                shape_to_emulate.extend([-1, -1])
                # shape_to_emulate[:2] = embedding_input.shape
                embedding_input = embedding_input.expand(shape_to_emulate)  # l,h,w, batch, channel
                embedding_input = embedding_input.permute(-2, 0, 1, 2, -1)  # back to b d h w c
                # have to permute cannot do this direclty

        x = batch
        to_vis = x.detach().cpu().numpy()
        self.visualize_one(to_vis, 0, method='mat')
        print("input: ", x.shape)
        for i in range(steps // stepsize):
            with torch.no_grad():
                x = self.model(x, embeddings=embedding_input, steps=stepsize, rearrange_output=True)
                to_vis = x.detach().cpu().numpy()
                self.visualize_one(to_vis, (i + 1) * stepsize)

    def train_func(self, x, targets, embeddings=None, embedding_params=None, steps=1, save_emb=False):
        self.optimizer.zero_grad()
        # print(targets[0, 3:4, 3:5, 4:7])
        # print(x[0, :self.num_categories, 3:4, 3:5, 4:7])
        x = self.model(x, embeddings=embeddings, steps=steps, rearrange_output=False)

        loss, iou_loss, class_loss, var_loss = self.get_loss(x, targets, embedding_params)

        loss.backward()
        if self.clip_gradients:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)

        self.optimizer.step()
        self.scheduler.step()
        x = rearrange(x, "b c d h w -> b d h w c")
        out = {
            "out": x,
            "metrics": {"loss": loss.item(), "iou_loss": iou_loss.item(), "class_loss": class_loss.item()},
            "loss": loss,
        }

        if self.learnable_embeddings:  # optimizer cannot do this as it is nor part of the model and change
            out['metrics']["var_loss"] = var_loss.item()
            with torch.no_grad():
                embedding_params -= self.var_lr * embedding_params.grad
        return out

    def train_iter(self, batch_size=32, iteration=0, save_emb=False):
        output = {"prev_batch": [], "post_batch": [], "total_metrics": [], "total_loss": [], "metrics": {}}
        for tree in range(self.dataset.num_samples):
            batch, targets, embedding, tree, indices = self.sample_batch(tree, batch_size)
            # print(f'Batch Sampled: tree {tree} | bs: {batch.size()} | ts: {targets.size()} | emb: {embedding}')

            # _______________________________
            embedding_input = None
            if self.embedding_dim:
                embedding = embedding.reshape(2, -1)  # dont come in correct shape
                if self.random:
                    embedding_input = torch.normal(mean=0, std=1, size=(self.batch_size, self.embedding_dim))
                    embedding_input = embedding_input.to(self.device)
                elif self.variational:
                    embedding.requires_grad = True
                    embedding_input = torch.normal(mean=0, std=1, size=(self.batch_size, self.embedding_dim))
                    embedding_input = embedding_input.to(self.device)
                    embedding_input *= torch.exp(0.5 * embedding[1])  # var
                    embedding_input += embedding[0]  # mean
                else:
                    embedding_input = torch.ones((self.batch_size, self.embedding_dim))
                    embedding_input = embedding_input.to(self.device)
                    if self.use_index:
                        embedding_input *= tree / self.num_samples
                    else:  # encoder
                        embedding = embedding[0]  # because wrong shape
                        embedding.requires_grad = True
                        embedding_input *= embedding

                # shape_to_emulate = [num for num in batch.shape]#batch channel, l, h ,w
                shape_to_emulate = [dim_i for dim_i in batch.shape[1:-1]]
                shape_to_emulate.extend([-1, -1])
                # shape_to_emulate[:2] = embedding_input.shape
                embedding_input = embedding_input.expand(shape_to_emulate)  # l,h,w, batch, channel
                embedding_input = embedding_input.permute(-2, 0, 1, 2, -1)  # back to b d h w c
                # have to permute cannot do this direclty
            # _____________________________________

            if self.use_sample_pool:
                with torch.no_grad():
                    loss_rank = (
                        self.rank_loss_function(batch, targets)
                        .detach()
                        .cpu()
                        .numpy()
                        .argsort()[::-1]
                    )
                    batch = batch[loss_rank.copy()]
                    batch[:1] = torch.from_numpy(self.get_seed(tree=tree)).to(self.device)
                    # print(f'Rank: {loss_rank} | \ttargets: {targets.shape}, \tbatch: {batch.shape}')

                    if self.damage:
                        self.apply_damage(batch)

            steps = np.random.randint(self.min_steps, self.max_steps)
            if self.half_precision:
                with torch.cuda.amp.autocast():
                    out_dict = self.train_func(batch, targets, embeddings=embedding_input, embedding_params=embedding,
                                               steps=steps)
            else:
                # print(f'Batch Input: tree {tree} | bs: {batch.size()} | ts: {targets.size()} | steps: {steps}')
                out_dict = self.train_func(batch, targets, embeddings=embedding_input, embedding_params=embedding,
                                           steps=steps)
            out, loss, metrics = out_dict["out"], out_dict["loss"], out_dict["metrics"]

            if self.update_dataset and self.use_sample_pool:
                if self.variational:
                    embedding.grad.zero_()
                    self.update_dataset_function(out, tree, indices, embedding=embedding, save_emb=save_emb)
                else:
                    self.update_dataset_function(out, tree, indices)
                if not self.visualize_output:
                    del out
            if self.visualize_output:
                output["prev_batch"].append(batch.detach().cpu().numpy())
                output["post_batch"].append(out.detach().cpu().numpy())
            output["total_metrics"].append(metrics)
            output["total_loss"].append(loss)

        for metric in output["total_metrics"][0]:
            output["metrics"][metric] = sum([x[metric] for x in output["total_metrics"]]) / self.dataset.num_samples
        output["loss"] = torch.mean(torch.stack(output["total_loss"]))
        return output
