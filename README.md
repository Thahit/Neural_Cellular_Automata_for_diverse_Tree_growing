<div align="center">    

# Neural Cellular Automata for diverse Tree growing
Our Project aims to enhance Minecraft landscapes with AI-generated 3D trees. Utilizing neural cellular automata, each tree evolves independently, creating diverse and realistic forests. Instead of manually designing a number of tree templates to use again and again, the Automata creates random trees that can repair themselves/regrow naturally. <br>
The page is heavily inspired by [Growing 3D Artefacts and Functional Machines with Neural Cellular Automata.](https://github.com/real-itu/3d-artefacts-nca)<br>
[![Paper](https://img.shields.io/badge/paper-arxiv.2103.08737-B31B1B.svg)](https://arxiv.org/abs/2103.08737)

</div>

---

Requirements
----
- python3.8
- This package automatically install `test-evocraft-py` (https://github.com/shyamsn97/test-evocraft-py), but for further functionality follow installation steps here: https://github.com/real-itu/Evocraft-py

Installation
---------------
### For general installation
```
python setup.py install
pip install -r requirements.txt
```
Depending on your setup, you might have to run the setup.py script after every change in code.

If you want CUDA, run afterward
```
pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu118
```

Usage
-------------
### Configs
Each nca is trained on a specific structure w/ hyperparams and configurations defined in yaml config, which we use with [hydra](https://github.com/facebookresearch/hydra) to create the [NCA trainer class](artefact_nca/trainer/voxel_ca_trainer.py).

[Example Config](artefact_nca/data/structs_dataset/acacia_trees/config.yaml) :
```
trainer:
    name: AcaciaTrees_8tree
    min_steps: 48
    max_steps: 65
    visualize_output: true
    device_id: 0
    use_cuda: true
    num_hidden_channels: 12
    checkpoint_interval: 200
    wandb: true
    epochs: 3001
    embedding_dim: 1
    var_lr : .002
    var_loss_weight: 0.1
    batch_size: 8
    clip_gradients: false
    random: true
    variational: true
    #use_index: true
    model_config:
        normal_std: 0.1
        update_net_channel_dims: [64, 32, 32, 16]
    optimizer_config:
        lr: 0.002
    dataset_config:
        load_embeddings: false
        pool_size: 32
        spawn_at_bottom: false
        verbose: false
        sample_specific_pool: true
        equal_sized_samples: false
        load_entity_config: {'verbose': false}

defaults:
    - voxel
```


## Generation and Training
See [training Python file](run/train.py) for ways to train your models and [inference Python file](run/inference.py) to produce inference results using the pretrained model.

Authors
-------
Cedric Caspar <ccaspar@student.ethz.ch>, <https://github.com/CedricCaspar>

Nicolas Blumer <nblume@student.ethz.ch>, <https://github.com/Thahit>

Oleh Kuzyk <okuzyk@student.ethz.ch>, <https://github.com/Olezhko2001>

Piyushi Goyal <pgoyal@student.ethz.ch>, <https://github.com/piyushigoyal>
