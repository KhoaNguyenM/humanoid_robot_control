"""Apply an environment-only YAML config to the current Isaac Sim stage.

Usage in Isaac Sim 5.1:
1. Open any environment USD stage.
2. Open Window > Script Editor.
3. Paste/run this file, or execute it from the Script Editor.

The script updates stage/environment settings only. It does not add robots,
policies, action graphs, actuators, or collision schemas to scene geometry.
"""

from __future__ import annotations

import math
import os
from typing import Any

import omni.usd
import yaml
from pxr import Gf, PhysxSchema, Sdf, UsdGeom, UsdPhysics, UsdShade


CONFIG_PATH = "/home/khoa-ng/Job_Project/My_Code/humanoid_robot_control/RL_control/G1_control/env_config/g1_env_only.yaml"


_PHYSX_ATTRS: dict[str, tuple[list[str], Any]] = {
    "solver_type": (["physxScene:solverType"], Sdf.ValueTypeNames.Token),
    "enable_ccd": (["physxScene:enableCCD"], Sdf.ValueTypeNames.Bool),
    "enable_stabilization": (["physxScene:enableStabilization"], Sdf.ValueTypeNames.Bool),
    "bounce_threshold_velocity": (["physxScene:bounceThresholdVelocity"], Sdf.ValueTypeNames.Float),
    "friction_offset_threshold": (["physxScene:frictionOffsetThreshold"], Sdf.ValueTypeNames.Float),
    "friction_correlation_distance": (["physxScene:frictionCorrelationDistance"], Sdf.ValueTypeNames.Float),
    "gpu_found_lost_pairs_capacity": (["physxScene:gpuFoundLostPairsCapacity"], Sdf.ValueTypeNames.Int),
    "gpu_found_lost_aggregate_pairs_capacity": (
        ["physxScene:gpuFoundLostAggregatePairsCapacity"],
        Sdf.ValueTypeNames.Int,
    ),
    "gpu_total_aggregate_pairs_capacity": (
        ["physxScene:gpuTotalAggregatePairsCapacity"],
        Sdf.ValueTypeNames.Int,
    ),
    "gpu_collision_stack_size": (["physxScene:gpuCollisionStackSize"], Sdf.ValueTypeNames.Int),
    "gpu_heap_capacity": (["physxScene:gpuHeapCapacity"], Sdf.ValueTypeNames.Int),
    "gpu_temp_buffer_capacity": (["physxScene:gpuTempBufferCapacity"], Sdf.ValueTypeNames.Int),
    "gpu_max_num_partitions": (["physxScene:gpuMaxNumPartitions"], Sdf.ValueTypeNames.Int),
}

_SOLVER_TYPE_ALIASES = {
    0: "PGS",
    1: "TGS",
    "0": "PGS",
    "1": "TGS",
    "pgs": "PGS",
    "tgs": "TGS",
    "PGS": "PGS",
    "TGS": "TGS",
}


def _require_mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"Expected '{name}' to be a mapping, got {type(value).__name__}")
    return value


def _as_vec3(values: Any, name: str) -> tuple[float, float, float]:
    if not isinstance(values, (list, tuple)) or len(values) != 3:
        raise ValueError(f"Expected '{name}' to contain exactly 3 numbers")
    return (float(values[0]), float(values[1]), float(values[2]))


def _load_config(path: str) -> dict[str, Any]:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Config file does not exist: {path}")
    with open(path, "r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    return _require_mapping(data, "root YAML")


def _get_stage():
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError("No USD stage is open. Open an environment USD before running this script.")
    return stage


def _set_stage_metadata(stage, cfg: dict[str, Any]) -> None:
    stage_cfg = _require_mapping(cfg.get("stage", {}), "stage")

    up_axis = str(stage_cfg.get("up_axis", "Z")).upper()
    if up_axis not in ("Y", "Z"):
        raise ValueError("stage.up_axis must be either 'Y' or 'Z'")
    UsdGeom.SetStageUpAxis(stage, up_axis)

    meters_per_unit = float(stage_cfg.get("meters_per_unit", 1.0))
    kilograms_per_unit = float(stage_cfg.get("kilograms_per_unit", 1.0))
    UsdGeom.SetStageMetersPerUnit(stage, meters_per_unit)
    UsdPhysics.SetStageKilogramsPerUnit(stage, kilograms_per_unit)


def _get_time_steps_per_second(simulation_cfg: dict[str, Any]) -> float:
    if "time_steps_per_second" in simulation_cfg:
        tps = float(simulation_cfg["time_steps_per_second"])
    elif "time_codes_per_second" in simulation_cfg:
        tps = float(simulation_cfg["time_codes_per_second"])
    else:
        dt = float(simulation_cfg.get("dt", 0.005))
        if dt <= 0.0:
            raise ValueError("simulation.dt must be greater than zero")
        tps = 1.0 / dt
    if tps <= 0.0:
        raise ValueError("simulation.time_steps_per_second must be greater than zero")
    return tps


def _set_timing(stage, scene_prim, simulation_cfg: dict[str, Any]) -> None:
    tps = _get_time_steps_per_second(simulation_cfg)
    tps_int = int(round(tps))
    if abs(tps - tps_int) > 1.0e-6:
        raise ValueError("simulation.time_steps_per_second must be an integer value for Isaac Sim 5.1 PhysX")

    # Isaac Sim 5.1 exposes the UI field "Time Steps Per Second" through
    # PhysxSchema.PhysxSceneAPI, not UsdPhysics.Scene metadata.
    physx_scene_api = PhysxSchema.PhysxSceneAPI.Apply(scene_prim)
    time_steps_attr = physx_scene_api.GetTimeStepsPerSecondAttr()
    if not time_steps_attr:
        time_steps_attr = physx_scene_api.CreateTimeStepsPerSecondAttr()
    time_steps_attr.Set(tps_int)

    stage.SetTimeCodesPerSecond(tps)
    stage.SetFramesPerSecond(tps)
    print(f"[env-yaml] Set Time Steps Per Second on {scene_prim.GetPath()} to {tps_int}")


def _coerce_usd_value(type_name, value: Any) -> Any:
    if type_name == Sdf.ValueTypeNames.Token:
        return str(value)
    if type_name == Sdf.ValueTypeNames.Bool:
        return bool(value)
    if type_name in (Sdf.ValueTypeNames.Int, Sdf.ValueTypeNames.UInt):
        return int(value)
    if type_name in (Sdf.ValueTypeNames.Float, Sdf.ValueTypeNames.Double):
        return float(value)
    return value


def _apply_or_create_attr(prim, attr_name: str, type_name, value: Any) -> None:
    attr = prim.GetAttribute(attr_name)
    if not attr:
        attr = prim.CreateAttribute(attr_name, type_name)
        attr_type = type_name
    else:
        attr_type = attr.GetTypeName()
    attr.Set(_coerce_usd_value(attr_type, value))


def _set_physx_attrs(scene_prim, physx_cfg: dict[str, Any]) -> None:
    if not bool(physx_cfg.get("enabled", True)):
        return
    PhysxSchema.PhysxSceneAPI.Apply(scene_prim)

    for key, value in physx_cfg.items():
        if key == "enabled":
            continue
        mapping = _PHYSX_ATTRS.get(key)
        if mapping is None:
            print(f"[env-yaml] Skipping unsupported PhysX key: {key}")
            continue
        attr_names, type_name = mapping
        if key == "solver_type":
            value = _SOLVER_TYPE_ALIASES.get(value, value)
            if value not in ("PGS", "TGS"):
                raise ValueError("simulation.physx.solver_type must be 'PGS', 'TGS', 0, or 1")
        for attr_name in attr_names:
            _apply_or_create_attr(scene_prim, attr_name, type_name, value)


def _find_physics_scene_paths(stage) -> list[str]:
    return [str(prim.GetPath()) for prim in stage.Traverse() if prim.IsA(UsdPhysics.Scene)]


def _resolve_physics_scene_path(stage, configured_path: Any) -> str:
    configured_path = str(configured_path or "/physicsScene")
    configured_prim = stage.GetPrimAtPath(configured_path)
    if configured_prim and configured_prim.IsA(UsdPhysics.Scene):
        return configured_path

    existing_scene_paths = _find_physics_scene_paths(stage)
    if len(existing_scene_paths) == 1:
        existing_path = existing_scene_paths[0]
        print(
            "[env-yaml] Using existing PhysicsScene "
            f"{existing_path} instead of creating {configured_path}."
        )
        return existing_path

    return configured_path


def _set_physics_scene(stage, cfg: dict[str, Any]) -> None:
    simulation_cfg = _require_mapping(cfg.get("simulation", {}), "simulation")
    physics_scene_path = _resolve_physics_scene_path(stage, simulation_cfg.get("physics_scene_path", "/physicsScene"))

    scene = UsdPhysics.Scene.Define(stage, str(physics_scene_path))
    scene_prim = scene.GetPrim()

    gravity = _as_vec3(simulation_cfg.get("gravity", [0.0, 0.0, -9.81]), "simulation.gravity")
    magnitude = math.sqrt(sum(component * component for component in gravity))
    if magnitude > 0.0:
        direction = tuple(component / magnitude for component in gravity)
    else:
        direction = (0.0, 0.0, -1.0)
    scene.CreateGravityDirectionAttr(Gf.Vec3f(*direction))
    scene.CreateGravityMagnitudeAttr(float(magnitude))

    _set_timing(stage, scene_prim, simulation_cfg)
    _set_physx_attrs(scene_prim, _require_mapping(simulation_cfg.get("physx", {}), "simulation.physx"))
    _set_physics_material(stage, simulation_cfg.get("physics_material", {}))


def _set_physics_material(stage, material_cfg: Any) -> None:
    if not material_cfg:
        return
    material_cfg = _require_mapping(material_cfg, "simulation.physics_material")

    prim_path = str(material_cfg.get("prim_path", "/World/PhysicsMaterials/default_environment"))
    material = UsdShade.Material.Define(stage, prim_path)
    material_api = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    material_api.CreateStaticFrictionAttr(float(material_cfg.get("static_friction", 1.0)))
    material_api.CreateDynamicFrictionAttr(float(material_cfg.get("dynamic_friction", 1.0)))
    material_api.CreateRestitutionAttr(float(material_cfg.get("restitution", 0.0)))

    for yaml_key, attr_name in (
        ("friction_combine_mode", "physicsMaterial:frictionCombineMode"),
        ("restitution_combine_mode", "physicsMaterial:restitutionCombineMode"),
    ):
        if yaml_key in material_cfg:
            _apply_or_create_attr(
                material.GetPrim(),
                attr_name,
                Sdf.ValueTypeNames.Token,
                str(material_cfg[yaml_key]),
            )


def _save_copy(stage, cfg: dict[str, Any]) -> None:
    stage_cfg = _require_mapping(cfg.get("stage", {}), "stage")
    save_as = stage_cfg.get("save_as")
    if not save_as:
        print("[env-yaml] stage.save_as is empty; changes were applied to the current session only.")
        return

    save_as = os.path.abspath(os.path.expanduser(str(save_as)))
    os.makedirs(os.path.dirname(save_as), exist_ok=True)
    if not stage.GetRootLayer().Export(save_as):
        raise RuntimeError(f"Failed to export configured stage to: {save_as}")
    print(f"[env-yaml] Exported configured stage copy to: {save_as}")


def apply_environment_config(config_path: str = CONFIG_PATH) -> None:
    cfg = _load_config(config_path)
    stage = _get_stage()

    _set_stage_metadata(stage, cfg)
    _set_physics_scene(stage, cfg)
    _save_copy(stage, cfg)
    print(f"[env-yaml] Applied environment config from: {config_path}")


apply_environment_config()
