#!/usr/bin/env python3
"""Mutable fixture transforms for focused P4-9 localization tests."""

from __future__ import annotations

import copy
from typing import Callable


TraceMutation = Callable[[dict], None]


def compose(*mutations: TraceMutation) -> TraceMutation:
    def mutate(raw: dict) -> None:
        for mutation in mutations:
            mutation(raw)

    return mutate


def select_stages(
    *stage_orders: int,
    state: str = "not_captured",
    detail: str = "fixture sparse coverage",
) -> TraceMutation:
    selected = frozenset(stage_orders)

    def mutate(raw: dict) -> None:
        layout = raw["intra_layer_checkpoint_layout"]
        requested = layout["requested_coordinates"]
        layout["captured_coordinates"] = [
            copy.deepcopy(coordinate)
            for coordinate in requested
            if coordinate["stage_order"] in selected
        ]
        layout["missing_coordinates"] = [
            {
                "coordinate": copy.deepcopy(coordinate),
                "state": state,
                "detail": detail,
            }
            for coordinate in requested
            if coordinate["stage_order"] not in selected
        ]
        raw["intra_layer_trace"] = [
            entry
            for entry in raw["intra_layer_trace"]
            if entry["stage_order"] in selected
        ]

    return mutate


def mismatch_digests(*stage_orders: int) -> TraceMutation:
    selected = frozenset(stage_orders)

    def mutate(raw: dict) -> None:
        for entry in raw["intra_layer_trace"]:
            if entry["stage_order"] in selected:
                entry["digest"]["value"] = "sha256:" + "f" * 64

    return mutate


def aggregate_only_change(stage_order: int) -> TraceMutation:
    def mutate(raw: dict) -> None:
        raw["intra_layer_trace"][stage_order].update(
            min=-7.0,
            max=7.0,
            mean=3.0,
            l2=8.0,
        )

    return mutate


def shape_mismatch(stage_order: int) -> TraceMutation:
    def mutate(raw: dict) -> None:
        entry = raw["intra_layer_trace"][stage_order]
        entry["shape"] = [2]
        entry["element_count"] = 2
        entry["digest"]["shape"] = [2]

    return mutate


def dtype_policy(observed_dtype: str) -> TraceMutation:
    def mutate(raw: dict) -> None:
        layout = raw["intra_layer_checkpoint_layout"]
        layout["digest_contract"]["observed_dtype"] = observed_dtype
        for entry in raw["intra_layer_trace"]:
            entry["observed_dtype"] = observed_dtype
            entry["digest"]["observed_dtype"] = observed_dtype

    return mutate


def precision_mismatch(stage_order: int) -> TraceMutation:
    def mutate(raw: dict) -> None:
        raw["intra_layer_trace"][stage_order][
            "precision_path"
        ] = "f32_accum;weights=f32;kv=f32"

    return mutate


def unknown_digest_policy(version: str = "unknown.digest/v2") -> TraceMutation:
    def mutate(raw: dict) -> None:
        layout = raw["intra_layer_checkpoint_layout"]
        layout["digest_contract"]["version"] = version
        for entry in raw["intra_layer_trace"]:
            entry["digest"]["version"] = version

    return mutate


def change_target_step(step: int) -> TraceMutation:
    def mutate(raw: dict) -> None:
        layout = raw["intra_layer_checkpoint_layout"]
        layout["runtime_checkpoint_step"] = step
        for collection in (
            layout["requested_coordinates"],
            layout["captured_coordinates"],
            raw["intra_layer_trace"],
        ):
            for item in collection:
                item["runtime_checkpoint_step"] = step

    return mutate
