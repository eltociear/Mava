# Copyright 2022 InstaDeep Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
from operator import attrgetter
from typing import Any, Optional

import jax
import jax.numpy as jnp
import tensorstore as ts
from chex import Array
from etils import epath
from flashbax.buffers.trajectory_buffer import TrajectoryBufferState

# CURRENT LIMITATIONS / TODO LIST
# - Only works when fbx_state.is_full is False (i.e. can't handle wraparound)
# - Better async stuff
# - Only tested with flat buffers

DRIVER = "file://"
METADATA_FILE = "metadata.json"


# TODO: Typing
def _nested_subattr(path: Any) -> str:
    return str.join(".", jax.tree_util.tree_map(lambda s: s.name, path))


class Vault:
    def __init__(
        self,
        base_path: str,
        init_fbx_state: TrajectoryBufferState,
        metadata: Optional[dict] = None,
        vault_multiplier: int = 1_000_000,
    ) -> None:
        # Base path is the root folder for this vault. Must be absolute.
        self.base_path = os.path.abspath(base_path)

        # We use epath for metadata
        metadata_path = epath.Path(os.path.join(self.base_path, METADATA_FILE))

        # Check if the vault exists, & either read metadata or create it
        base_path_exists = os.path.exists(self.base_path)
        if base_path_exists:
            self.metadata = json.loads(metadata_path.read_text())
        else:
            # Create the vault root dir
            os.mkdir(self.base_path)
            self.metadata = {
                "version": 0.0,
                "vault_multiplier": vault_multiplier,
                "tree_struct": init_fbx_state.experience._fields,
                **(metadata or {}),
            }
            metadata_path.write_text(json.dumps(self.metadata))

        self.vault_multiplier = vault_multiplier

        self.vault_ds = ts.open(
            self._get_base_spec("vault_index"),
            dtype=jnp.int32,
            shape=(1,),
            create=not base_path_exists,
        ).result()
        self.vault_index = int(self.vault_ds.read().result()[0])

        self.all_ds = jax.tree_util.tree_map_with_path(
            lambda path, x: self._init_leaf(
                name=_nested_subattr(path),
                leaf=x,
                create_checkpoint=not base_path_exists,
            ),
            init_fbx_state.experience,
        )

        self.fbx_sample_experience = jax.tree_map(
            lambda x: x[:, 0:1, ...],
            init_fbx_state.experience,
        )
        self.last_received_fbx_index = 0

    def _get_base_spec(self, name: str) -> dict:
        return {
            "driver": "zarr",
            "kvstore": {
                "driver": "ocdbt",
                "base": f"{DRIVER}{self.base_path}",
                "path": name,
            },
        }

    def _init_leaf(self, name: str, leaf: Array, create_checkpoint: bool = False) -> ts.TensorStore:
        spec = self._get_base_spec(name)
        leaf_ds = ts.open(
            spec,
            dtype=leaf.dtype if create_checkpoint else None,
            shape=(leaf.shape[0], self.vault_multiplier * leaf.shape[1], *leaf.shape[2:])
            if create_checkpoint
            else None,
            create=create_checkpoint,
        ).result()
        return leaf_ds

    def _write_leaf(
        self,
        name: str,
        leaf: Array,
        source_interval: tuple,
        dest_interval: tuple,
    ) -> None:
        leaf_ds = attrgetter(name)(self.all_ds)
        leaf_ds[:, slice(*dest_interval), ...].write(
            leaf[:, slice(*source_interval), ...],
        ).result()

    def write(
        self,
        fbx_state: TrajectoryBufferState,
        source_interval: tuple = (None, None),
        dest_interval: tuple = (None, None),
    ) -> None:
        fbx_current_index = int(fbx_state.current_index)

        if source_interval == (None, None):
            source_interval = (self.last_received_fbx_index, fbx_current_index)
        write_length = source_interval[1] - source_interval[0]
        if write_length == 0:
            return

        if dest_interval == (None, None):
            dest_interval = (self.vault_index, self.vault_index + write_length)

        assert (source_interval[1] - source_interval[0]) == (dest_interval[1] - dest_interval[0])

        jax.tree_util.tree_map_with_path(
            lambda path, x: self._write_leaf(
                name=_nested_subattr(path),
                leaf=x,
                source_interval=source_interval,
                dest_interval=dest_interval,
            ),
            fbx_state.experience,
        )
        self.vault_index += write_length
        self.vault_ds.write(self.vault_index).result()

        self.last_received_fbx_index = fbx_current_index

    def _read_leaf(
        self,
        name: str,
        read_interval: tuple,
    ) -> Array:
        leaf_ds = attrgetter(name)(self.all_ds)
        return leaf_ds[:, slice(*read_interval), ...].read().result()

    def read(self, read_interval: tuple = (None, None)) -> Array:  # TODO typing
        if read_interval == (None, None):
            read_interval = (0, self.vault_index)

        return jax.tree_util.tree_map_with_path(
            lambda path, _: self._read_leaf(_nested_subattr(path), read_interval),
            self.fbx_sample_experience,
        )

    def get_buffer(
        self, size: int, key: Array, starting_index: Optional[int] = None
    ) -> TrajectoryBufferState:
        assert size <= self.vault_index
        if starting_index is None:
            starting_index = int(
                jax.random.randint(
                    key=key,
                    shape=(),
                    minval=0,
                    maxval=self.vault_index - size,
                )
            )
        return TrajectoryBufferState(
            experience=self.read((starting_index, starting_index + size)),
            current_index=starting_index + size,
            is_full=True,
        )
