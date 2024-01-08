import copy
import math
import os
from os import listdir
from os.path import isfile, join
from pathlib import Path

import grpc
import ast
import nbtlib
import numpy as np
from matplotlib.colors import rgb2hex
from nbtlib import serialize_tag, Byte
from test_evocraft_py.minecraft_pb2 import *
from test_evocraft_py.minecraft_pb2_grpc import *

from artefact_nca.utils.minecraft.block_utils import BlockBuffer


class Blockloader:
    @staticmethod
    def spawn_nbt_blocks(
        dataset_dir: str,
        filename: str = "Extra_dark_oak.nbt",
        load_coord=(0, 10, 0),
        block_priority=[],
        place_block_priority_first=True,
    ) -> None:
        nbt_filenames = [
            join(dataset_dir, f)
            for f in listdir(dataset_dir)
            if isfile(join(dataset_dir, f)) and f.endswith("nbt")
        ]
        if filename is not None:
            nbt_filenames = [
                f for f in nbt_filenames if f == join(dataset_dir, filename)
            ]
        block_buffer = BlockBuffer()
        for f in nbt_filenames:
            block_buffer.send_nbt_to_server(
                load_coord,
                f,
                block_priority=block_priority,
                place_block_priority_first=place_block_priority_first,
            )
            load_coord = [load_coord[0] + 30, load_coord[1], load_coord[2]]

    @staticmethod
    def clear_blocks(client, min_coords=(0,0,0), max_coords=(0,0,0)):

        client.fillCube(
            FillCubeRequest(  # Clear a 20x10x20 working area
                cube=Cube(
                    min=Point(x=min_coords[0], y=min_coords[1], z=min_coords[2]),
                    max=Point(x=max_coords[0], y=max_coords[1], z=max_coords[2],),
                ),
                type=5,
            )
        )

    @staticmethod
    def read_blocks(client, min_coords, max_coords):
        blocks = client.readCube(
            Cube(
                min=Point(x=min_coords[0], y=min_coords[1], z=min_coords[2]),
                max=Point(x=max_coords[0], y=max_coords[1], z=max_coords[2]),
            )
        )
        return blocks


def convert_to_color(arr, color_dict):
    new_arr = copy.deepcopy(arr).astype(object)
    for k in color_dict:
        new_arr[new_arr == k] = color_dict[k]
    return new_arr


def roundup(x):
    return int(math.ceil(x / 10.0)) * 10


def first_nonzero(arr, axis, invalid_val=-1):
    mask = arr != 0
    return np.where(mask.any(axis=axis), mask.argmax(axis=axis), invalid_val)


def get_color_dict(unique_vals):
    if len(unique_vals) == 3:
        print(f'Fixed Colors!')
        color_arr = [(64/255, 47/255, 30/255), (33/255, 100/255, 19/255)]
    elif len(unique_vals) == 8:
        print(f'Fixed Colors 7!')
        color_arr = [(64 / 255, 47 / 255, 30 / 255), (100 / 255, 60 / 255, 35 / 255),  # normal brown | light brown
                     (148 / 255, 117 / 255, 98 / 255), (66 / 255, 200 / 255, 38 / 255),  # dark brown | light green (66 / 255, 200 / 255, 38 / 255)
                     (75 / 255, 49 / 255, 31 / 255), (33 / 255, 100 / 255, 19 / 255),  # dark brown | normal green
                     (20 / 255, 15 / 255, 10 / 255)]  # black brown
    else:
        state = np.random.RandomState(0)
        color_arr = list(state.uniform(0, 1, (len(unique_vals), 3)))
    color_arr = [rgb2hex(color) for color in color_arr]
    color_arr = [None] + color_arr
    colors = color_arr[: len(unique_vals)]
    color_dict = {i: colors[i] for i in range(len(unique_vals))}
    return color_dict


def get_block_array(
    nbt_data: list,
    min_coords,
    max_coords,
    no_padding=True,
    unequal_padding=False,
    padding=None,
    same_size=False,
    verbose=False
):
    # for b in nbt_data[0]['palette']:
    #     print(b)
    # #
    # for b in nbt_data[0]['blocks']:
    #     print(b)
    # print(nbt_data[0]['Blocks'])

    num_trees = len(nbt_data)

    if verbose: print(f'Num Trees: {num_trees}')
    unique_set_blocks = set()
    unique_set_data = set()
    for i in range(len(nbt_data)):
        unique_set_blocks.update(nbt_data[i]['Blocks'])
        unique_set_data.update(nbt_data[i]['Data'])
    unique_vals = sorted(list(unique_set_blocks))
    unique_vals_data = sorted(list(unique_set_data))
    if verbose: print('Unique block values: ', unique_vals)
    if verbose: print('Unique data values: ', unique_vals_data)
    unique_vals.remove(0)
    unique_vals.insert(0, Byte(0))
    color_dict = get_color_dict(unique_vals)
    if verbose: print(color_dict)
    unique_val_to_int_dict = {unique_vals[i]: i for i in range(len(unique_vals))}
    if verbose: print(unique_val_to_int_dict)
    unique_val_dict = {i: unique_vals[i] for i in range(len(unique_vals))}
    if verbose: print(unique_val_dict)

    # print(nbt_data[0])
    block_to_data_conversion = {}
    for j in range(num_trees):

        for i in range(len(nbt_data[j]['Data'])):
            if nbt_data[j]['Blocks'][i] not in block_to_data_conversion:
                block_to_data_conversion[nbt_data[j]['Blocks'][i]] = {nbt_data[j]['Data'][i]}
            else:
                block_to_data_conversion[nbt_data[j]['Blocks'][i]].add(nbt_data[j]['Data'][i])
    if verbose: print(block_to_data_conversion)
    # pal = []
    # for j in range(num_trees):
    #     print('len', len(nbt_data[j]['palette']))
    #     print('palette', nbt_data[j]['palette'])
    #     pal.extend([e.unpack() for e in nbt_data[j]['palette']])
    # print(pal)
    # print('len Data', len(nbt_data[0]['Data']))
    # print('len Blocks', len(nbt_data[0]['Blocks']))
    # print('Height', nbt_data[0]['Height'])
    # print('Length', nbt_data[0]['Length'])
    # print('Width', nbt_data[0]['Width'])
    # print('Total', int(nbt_data[0]['Width']) * int(nbt_data[0]['Length']) * int(nbt_data[0]['Height']))

    if same_size:
        if verbose: print(f'Start equal size')
        min_coords_shifted = np.array(min_coords)
        max_coords_shifted = np.array(max_coords)
        size_arr = np.insert(np.array(max_coords_shifted) - np.array(min_coords_shifted) + 1, 0, num_trees)
        center = size_arr // 2
        if verbose: print(f'Min coords: {min_coords_shifted} | Max coords: {max_coords_shifted} => arr size: {size_arr}')
        arr = np.zeros(size_arr, dtype=object)

        for i in range(num_trees):
            internal_w = int(nbt_data[i]['Width'])
            internal_d = int(nbt_data[i]['Length'])
            internal_h = int(nbt_data[i]['Height'])
            internal_d_half = internal_d // 2
            internal_w_half = internal_w // 2
            if verbose: print(f'Internal w: {internal_w}, l: {internal_d}, h: {internal_h} | w: {internal_w_half}, l: {internal_d_half}')
            if internal_w > size_arr[1] or internal_d > size_arr[2] or internal_h+1 > size_arr[3]:
                raise Exception('Provided structure bounding box is bigger than the loading range')

            blocks = nbt_data[i]['Blocks']
            for y in range(internal_h):
                for x in range(internal_w):
                    for z in range(internal_d):

                        arr[i, x + center[1] - internal_w_half, z + center[2] - internal_d_half, y+1] = \
                            unique_val_to_int_dict[blocks[x + z*internal_w + y*internal_w*internal_d]]

        bounds = np.nonzero(arr)[1:]
        if unequal_padding and padding is not None:
            x_min = np.min(bounds[0]) - padding[0]
            x_max = np.max(bounds[0]) + padding[0]
            z_min = np.min(bounds[1]) - padding[2]
            z_max = np.max(bounds[1]) + padding[2]
            y_min = np.min(bounds[2]) - padding[1]
            y_max = np.max(bounds[2]) + padding[1]
        elif no_padding:
            x_min = np.min(bounds[0]) - 1
            x_max = np.max(bounds[0]) + 2
            z_min = np.min(bounds[1]) - 1
            z_max = np.max(bounds[1]) + 2
            y_min = np.min(bounds[2]) - 1
            y_max = np.max(bounds[2]) + 2
        else:
            x_min, y_min, z_min = 0, 0, 0
            x_max = arr.shape[1]
            z_max = arr.shape[2]
            y_max = arr.shape[3]

        res = []
        for i in range(num_trees):
            res.append(arr[i, x_min:x_max, z_min:z_max, y_min: y_max])
        dims = [r.shape for r in res]
        dimensions = np.array(dims)
        if verbose: print(f'Loaded {num_trees} trees with dimensions: {dimensions}')
        return block_to_data_conversion, dimensions, res, color_dict, unique_val_dict
    else:
        if verbose: print(f'Start unequal size')
        arr = []
        dimensions = []
        for i in range(num_trees):
            internal_w = int(nbt_data[i]['Width'])
            internal_d = int(nbt_data[i]['Length'])
            internal_h = int(nbt_data[i]['Height'])

            padding_w = padding[0] if padding else 2
            padding_d = padding[1] if padding else 2
            padding_h = padding[2] if padding else 2
            target = np.zeros((internal_w + padding_w*2, internal_d + padding_d*2, internal_h + padding_h), dtype=object)

            if verbose: print(f'Internal w: {internal_w}, l: {internal_d}, h: {internal_h} => size {target.shape} ({internal_w*internal_d*internal_h})')

            blocks = nbt_data[i]['Blocks']
            for y in range(internal_h):
                for x in range(internal_w):
                    for z in range(internal_d):

                        target[x + padding_w, z + padding_d, y+1] = \
                            unique_val_to_int_dict[blocks[x + z*internal_w + y*internal_w*internal_d]]
            arr.append(target)
            dimensions.append(target.shape)

        if verbose: print(f'Loaded {num_trees} trees with dimensions: {dimensions}')
        return block_to_data_conversion, dimensions, arr, color_dict, unique_val_dict


def read_nbt_target(
    nbt_path,
    load_coord=(0, 0, 0),
    load_range=(20, 50, 20),
    no_padding=True,
    unequal_padding=False,
    padding=None,
    block_priority=[],
    place_block_priority_first=True,
    same_size=False,
    verbose=False
):

    min_coords = (load_coord[0] - load_range[0], load_coord[2] - load_range[2], load_coord[1])
    max_coords = (
        load_coord[0] + load_range[0],
        load_coord[2] + load_range[2],
        load_coord[1] + load_range[1] * 2
    )

    if '.' in nbt_path:
        nbt_file = nbtlib.load(nbt_path)
        return get_block_array(
            [nbt_file.root], min_coords, max_coords, no_padding, unequal_padding, padding
        )
    else:

        p = Path(nbt_path)
        if not p.exists():
            raise Exception("failed to find the data folder")

        candidates = list(p.glob(f'*.nbt'))
        if len(candidates) == 0:
            raise Exception("directory is empty or has not nbt files")

        data = []
        for i, path in enumerate(candidates):
            nbt_file = nbtlib.load(path)
            # f_name = str(path).split('\\')[-1]
            # print(f'Tree {i}: {f_name}')
            data.append(nbt_file.root)

        return get_block_array(
            data, min_coords, max_coords, no_padding, unequal_padding, padding, same_size, verbose=verbose
        )


def create_flying_machine(load_coord=(50, 10, 10)):
    channel = grpc.insecure_channel("localhost:5001")
    client = MinecraftServiceStub(channel)
    min_coords = (load_coord[0] - 30, load_coord[1], load_coord[2] - 30)
    max_coords = (load_coord[0] + 30, load_coord[1] + 60, load_coord[2] + 30)

    Blockloader.clear_blocks(client, min_coords, max_coords)

    client.spawnBlocks(
        Blocks(
            blocks=[  # Spawn a flying machine
                # Lower layer
                Block(
                    position=Point(
                        x=load_coord[0] + 1, y=load_coord[1] + 5, z=load_coord[2] + 1
                    ),
                    type=PISTON,
                    orientation=NORTH,
                ),
                Block(
                    position=Point(
                        x=load_coord[0] + 1, y=load_coord[1] + 5, z=load_coord[2]
                    ),
                    type=SLIME,
                    orientation=NORTH,
                ),
                Block(
                    position=Point(
                        x=load_coord[0] + 1, y=load_coord[1] + 5, z=load_coord[2] - 1
                    ),
                    type=STICKY_PISTON,
                    orientation=SOUTH,
                ),
                Block(
                    position=Point(
                        x=load_coord[0] + 1, y=load_coord[1] + 5, z=load_coord[2] - 2
                    ),
                    type=PISTON,
                    orientation=NORTH,
                ),
                Block(
                    position=Point(
                        x=load_coord[0] + 1, y=load_coord[1] + 5, z=load_coord[2] - 4
                    ),
                    type=SLIME,
                    orientation=NORTH,
                ),
                # Upper layer
                # Activate
                Block(
                    position=Point(
                        x=load_coord[0] + 1, y=load_coord[1] + 6, z=load_coord[2] - 1
                    ),
                    type=QUARTZ_BLOCK,
                    orientation=NORTH,
                ),
                Block(
                    position=Point(
                        x=load_coord[0] + 1, y=load_coord[1] + 6, z=load_coord[2]
                    ),
                    type=REDSTONE_BLOCK,
                    orientation=NORTH,
                ),
                Block(
                    position=Point(
                        x=load_coord[0] + 1, y=load_coord[1] + 6, z=load_coord[2] - 4
                    ),
                    type=REDSTONE_BLOCK,
                    orientation=NORTH,
                ),
            ]
        )
    )
    return get_block_array(client, min_coords, max_coords, False, None)
