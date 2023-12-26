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
    variational: bool = attr.ib(default=False)
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

    def get_seed(self, batch_size=1, tree=None):
        if tree is None:
            return self.dataset.get_seeds(batch_size)
        else:
            return self.dataset.get_seed(tree, batch_size)

    def sample_batch(self, batch_size: int):
        return self.dataset.sample(batch_size)

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
                    "var_lr": self.var_lr,
                    "var_loss_weight": self.var_loss_weight,
                    "dataset_conf": {"dimensions": self.dataset.dimensions,
                                     "num_channels": self.dataset.num_channels,
                                     "living_channel": self.dataset.living_channel_dim,
                                     "spawn_at_bottom": self.dataset.spawn_at_bottom,
                                     "use_random_seed_block": self.dataset.use_random_seed_block,
                                     "sample_specific_pools": self.dataset.sample_specific_pools,
                                     "equal_sized_samples": self.dataset.equal_sized_samples,
                                     "sample_random_tree": self.dataset.sample_random_tree,
                                     "load_embeddings": self.dataset.load_embeddings,
                                     },
                    "half_precision": self.half_precision
                })

    def visualize(self, out):
        prev_batch = out["prev_batch"]
        post_batch = out["post_batch"]
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
        print("Before --- After")
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

    def rollout(self, initial=None, steps=100):
        if initial is None:
            initial, _, _, _, _ = self.sample_batch(1)
        if not isinstance(initial, torch.Tensor):
            initial = torch.from_numpy(initial).to(self.device)
        bar = tqdm(np.arange(steps))
        out = [initial]
        life_masks = [None]
        for i in bar:
            x, life_mask = self.model(out[-1], 1, return_life_mask=True)
            life_masks.append(life_mask)
            out.append(x)
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

        # with torch.no_grad():
        #     sum = torch.sum(targets)
        intersect = torch.sum(out & targets).float()
        # print(f'\nIntersection: {intersect} of {sum}')
        union = torch.sum(out | targets).float()
        o = (union - intersect) / (union + 1e-8)
        return o

    def get_loss(self, x, targets, embedding_params=None):
        iou_loss = 0
        if self.use_iou_loss:
            iou_loss = self.iou(x, targets)
        if self.use_bce_loss:
            weight_class = torch.ones(self.num_categories)
            weight_class[0] = 0.001
            class_loss = F.cross_entropy(
                x[:, : self.num_categories, :, :, :], targets, weight=weight_class.to(self.device)
            )
            alive = torch.clip(x[:, self.num_categories, :, :, :], 0.0, 1.0)
            alive_target_cells = torch.clip(targets, 0, 1).float()
            alive_loss = torch.nn.MSELoss()(alive, alive_target_cells)
        else:
            weight_class = torch.ones(self.num_categories)
            weight_class[0] = 0.001
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

    def get_loss_for_single_instance(self, x, rearrange_input=False):
        if rearrange_input:
            x = rearrange(x, "b d h w c -> b c d h w")
        batch, targets, embedding, tree, indices = self.sample_batch(1)
        return self.get_loss(x, targets)

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
            "metrics": {"loss": loss.item(), "iou_loss": iou_loss.item(), "class_loss": class_loss,
                        "var_loss": var_loss},
            "loss": loss,
        }

        if self.variational:  # optimizer cannot do this as it is nor part of the model and change
            with torch.no_grad():
                embedding_params -= self.var_lr * embedding_params.grad
        return out

    def train_iter(self, batch_size=32, iteration=0, save_emb=False):
        batch, targets, embedding, tree, indices = self.sample_batch(batch_size)
        # print(f'Batch Sampled: tree {tree} | bs: {batch.size()} | ts: {targets.size()} | emb: {embedding}')

        # _______________________________
        embedding_input = None
        if self.embedding_dim:

            embedding = embedding.reshape(2, -1)  # dont come in correct shape

            embedding_input = torch.normal(mean=0, std=1, size=(self.batch_size, self.embedding_dim)).to(self.device)
            if self.variational:
                embedding.requires_grad = True
                embedding_input *= torch.exp(0.5 * embedding[1])  # var
                embedding_input += embedding[0]  # mean

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
            with torch.cuda.amp.autocast():  # unused?
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

        out_dict["prev_batch"] = batch.detach().cpu().numpy()
        out_dict["post_batch"] = out.detach().cpu().numpy()
        out_dict["tree"] = tree
        return out_dict
