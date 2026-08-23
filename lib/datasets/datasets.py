import json
import random
from collections import Counter, defaultdict
from copy import copy

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler

from constants.h2o_constants import h2o_obj_name
from constants.hot3d_constants import hot3d_obj_name
from constants.grab_constants import grab_obj_name
from lib.datasets.h2o import (
     MotionHOT3D, ContactHOT3D
)


def _resolve_seed(config):
    gaze2hoi_config = getattr(config, "gaze2hoi", None)
    if gaze2hoi_config is not None and hasattr(gaze2hoi_config, "exp"):
        return int(getattr(gaze2hoi_config.exp, "seed", 0))
    return int(getattr(config, "seed", 0))


def get_dataset(dataset_name, dataset_config, test_flag):
    assert dataset_name in [
        "Sequenceh2o", "Contacth2o", "Contacthot3d", "Motionh2o", "Motionhot3d",
        "Sequencegrab", "Contactgrab", "Motiongrab", 
        "Sequencearctic", "Contactarctic", "Motionarctic", "Sequencehot3d"
    ]
    dataset = MotionHOT3D(test_flag=test_flag, **dataset_config)
    
    return dataset

def get_dataloader(dataset_name, config, data_config, test=False):
    dataset = get_dataset(
        dataset_name, 
        data_config, 
        test
    )
    seed = _resolve_seed(config)
    generator = torch.Generator()
    generator.manual_seed(seed)

    def seed_worker(worker_id):
        worker_seed = seed + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)
    
    if config.balance_weights and not test:
        sampler = WeightedRandomSampler(
            torch.FloatTensor(dataset.balance_weights),
            len(dataset),
            generator=generator,
        )
        dataloader = DataLoader(
            dataset, 
            batch_size=config.batch_size, 
            sampler=sampler,
            num_workers=config.num_workers,
            drop_last=config.drop_last,
            worker_init_fn=seed_worker,
            generator=generator,
        )
    else:
        dataloader = DataLoader(
            dataset, 
            batch_size=config.batch_size, 
            shuffle=False,
            num_workers=config.num_workers,
            worker_init_fn=seed_worker,
            generator=generator,
        )
        
    return dataset, dataloader


def _stratified_train_validation_indices(dataset, validation_ratio, seed):
    """Split samples deterministically while preserving action/object coverage."""
    sample_count = len(dataset)
    validation_ratio = float(validation_ratio)
    if not 0.0 < validation_ratio < 1.0:
        raise ValueError(
            f"validation_ratio must be between 0 and 1, got {validation_ratio}"
        )
    if sample_count < 2:
        raise ValueError("At least two samples are required for a train/validation split.")

    actions = getattr(dataset, "action", None)
    object_names = getattr(dataset, "obj_names", None)
    groups = defaultdict(list)
    for index in range(sample_count):
        action = str(actions[index]) if actions is not None else ""
        object_name = str(object_names[index]) if object_names is not None else ""
        groups[(action, object_name)].append(index)

    rng = np.random.RandomState(int(seed))
    validation_indices = []
    deferred_indices = []
    for key in sorted(groups, key=lambda value: (str(value[0]), str(value[1]))):
        indices = np.asarray(groups[key], dtype=np.int64)
        rng.shuffle(indices)
        # Keep singleton/very small strata in training. The global fill below
        # still gives the requested validation size without emptying a stratum.
        group_validation_count = min(
            max(0, int(round(len(indices) * validation_ratio))),
            max(0, len(indices) - 1),
        )
        validation_indices.extend(indices[:group_validation_count].tolist())
        deferred_indices.extend(indices[group_validation_count:].tolist())

    target_validation_count = max(
        1, min(sample_count - 1, int(round(sample_count * validation_ratio)))
    )
    if len(validation_indices) < target_validation_count:
        candidates = np.asarray(deferred_indices, dtype=np.int64)
        rng.shuffle(candidates)
        selected = set(validation_indices)
        for index in candidates.tolist():
            if len(validation_indices) >= target_validation_count:
                break
            key = (
                str(actions[index]) if actions is not None else "",
                str(object_names[index]) if object_names is not None else "",
            )
            group_remaining = sum(
                candidate not in selected for candidate in groups[key]
            )
            if group_remaining <= 1:
                continue
            validation_indices.append(index)
            selected.add(index)
        # Highly sparse datasets can have only singleton strata. In that case
        # satisfy the requested global ratio while retaining at least one
        # training sample overall.
        for index in candidates.tolist():
            if len(validation_indices) >= target_validation_count:
                break
            if index not in selected:
                validation_indices.append(index)
                selected.add(index)

    if len(validation_indices) > target_validation_count:
        validation_indices = np.asarray(
            validation_indices, dtype=np.int64
        )
        rng.shuffle(validation_indices)
        validation_indices = validation_indices[
            :target_validation_count
        ].tolist()
    validation_set = set(validation_indices)
    train_indices = [
        index for index in range(sample_count) if index not in validation_set
    ]
    validation_indices = sorted(validation_set)
    if not train_indices or not validation_indices:
        raise RuntimeError(
            "The deterministic train/validation split produced an empty subset."
        )
    return train_indices, validation_indices


def get_train_validation_dataloaders(
    dataset_name,
    config,
    data_config,
    validation_ratio=0.1,
):
    """Return a shuffled/balanced train loader and deterministic eval loader.

    Separate dataset instances are intentional: ``test_flag=False`` enables
    training-time text processing, while ``test_flag=True`` preserves the
    canonical prompt and disables training-only augmentation for validation.
    """
    train_dataset_full = get_dataset(dataset_name, data_config, test_flag=False)
    # Only `test_flag` differs. A shallow dataset view avoids loading the same
    # meshes, point clouds, and motion arrays twice while keeping independent
    # train/eval behavior in __getitem__.
    validation_dataset_full = copy(train_dataset_full)
    validation_dataset_full.test_flag = True

    seed = _resolve_seed(config)
    train_indices, validation_indices = _stratified_train_validation_indices(
        train_dataset_full,
        validation_ratio=validation_ratio,
        seed=seed,
    )
    train_dataset = Subset(train_dataset_full, train_indices)
    validation_dataset = Subset(validation_dataset_full, validation_indices)

    train_generator = torch.Generator()
    train_generator.manual_seed(seed)
    validation_generator = torch.Generator()
    validation_generator.manual_seed(seed + 1)

    def seed_worker(worker_id):
        worker_seed = seed + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    loader_kwargs = {
        "num_workers": config.num_workers,
        "worker_init_fn": seed_worker,
    }
    if config.balance_weights:
        actions = getattr(train_dataset_full, "action", None)
        if actions is None:
            sample_weights = torch.as_tensor(
                train_dataset_full.balance_weights[train_indices],
                dtype=torch.float32,
            )
        else:
            train_action_keys = [
                str(actions[index]).capitalize() for index in train_indices
            ]
            train_action_counts = Counter(train_action_keys)
            sample_weights = torch.as_tensor(
                [
                    1.0 / train_action_counts[action]
                    for action in train_action_keys
                ],
                dtype=torch.float32,
            )
        train_sampler = WeightedRandomSampler(
            sample_weights,
            num_samples=len(train_indices),
            replacement=True,
            generator=train_generator,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            sampler=train_sampler,
            drop_last=config.drop_last,
            generator=train_generator,
            **loader_kwargs,
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            drop_last=config.drop_last,
            generator=train_generator,
            **loader_kwargs,
        )

    validation_batch_size = int(
        getattr(config.gaze2hoi.exp, "validation_batch_size", config.batch_size)
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=validation_batch_size,
        shuffle=False,
        drop_last=False,
        generator=validation_generator,
        **loader_kwargs,
    )
    return (
        train_dataset,
        train_loader,
        validation_dataset,
        validation_loader,
        train_indices,
        validation_indices,
    )
