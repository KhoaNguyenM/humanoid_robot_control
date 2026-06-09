ROBOT_USD_PATH = "/home/khoa-ng/Job_Project/My_Code/humanoid_robot_control/unitree_sim_isaaclab/assets/robots/g1-29dof_wholebody_dex3/g1_29dof_with_dex3_rev_1_0.usd"
ROBOT_CONFIG_YAML_PATH = "/home/khoa-ng/Job_Project/My_Code/humanoid_robot_control/RL_control/G1_control/robot_avai_config/g1_29dof_wholebody_dex3_existing_robot.yaml"
ROBOT_PRIM_PATH = "/World/g1_29dof_with_dex3_rev_1_0"

import json
import math
import os
import re
from typing import Any

import omni.usd
import yaml
from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics


def _load_yaml(path: str) -> dict[str, Any]:
    path = os.path.abspath(os.path.expanduser(path))
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Robot YAML config does not exist: {path}")
    with open(path, "r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    if not isinstance(data, dict):
        raise TypeError(f"Expected robot YAML root to be a mapping, got {type(data).__name__}")
    return data


def _current_stage() -> Usd.Stage:
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError("No USD stage is open. Open an environment USD before running this script.")
    return stage


def _normalize_prim_path(path: str) -> Sdf.Path:
    raw = str(path or "").strip()
    if not raw:
        raise ValueError("ROBOT_PRIM_PATH is empty.")
    if not raw.startswith("/"):
        raw = "/" + raw
    path_obj = Sdf.Path(raw)
    if not path_obj.IsAbsolutePath() or not path_obj.IsPrimPath():
        raise ValueError(f"Invalid robot prim path: {raw}")
    return path_obj


def _set_attr(prim: Usd.Prim, attr_name: str, type_name, value: Any) -> None:
    if value is None:
        return
    attr = prim.GetAttribute(attr_name)
    if not attr:
        attr = prim.CreateAttribute(attr_name, type_name, custom=False)
    attr.Set(value)


def _set_schema_attr(schema_obj: Any, getter_name: str, creator_name: str, value: Any) -> bool:
    if value is None or not schema_obj:
        return False
    try:
        attr = getattr(schema_obj, getter_name)()
        if not attr:
            attr = getattr(schema_obj, creator_name)()
        attr.Set(value)
        return True
    except Exception:
        return False


def _apply_api(api_cls: Any, prim: Usd.Prim):
    if api_cls is None:
        return None
    try:
        api = api_cls(prim)
        if not api:
            api = api_cls.Apply(prim)
        return api
    except Exception:
        return None


def _group_enabled(cfg: dict[str, Any], group_name: str, fallback_option: str | None = None, default: bool = False) -> bool:
    groups = cfg.get("debug_apply_groups", {})
    if group_name in groups:
        return bool(groups[group_name])
    if fallback_option:
        return bool(cfg.get("script_options", {}).get(fallback_option, default))
    return default


def _enabled_groups(cfg: dict[str, Any]) -> dict[str, bool]:
    group_names = [
        "check_existing_prim_and_usd",
        "make_subtree_non_instanceable",
        "root_transform",
        "articulation_root",
        "rigid_body_props",
        "contact_report_api",
        "joint_drive_properties",
        "joint_drive_targets",
        "joint_state",
        "validation",
        "summary_attributes",
        "select_robot_after_load",
        "save_stage_after_load",
    ]
    return {name: _group_enabled(cfg, name, default=False) for name in group_names}


def _set_root_transform(prim: Usd.Prim, pos: list[float], rot: list[float]) -> None:
    xform = UsdGeom.Xformable(prim)
    translate_op = None
    orient_op = None

    for op in xform.GetOrderedXformOps():
        if op.IsInverseOp():
            continue
        if translate_op is None and op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
            translate_op = op
        elif orient_op is None and op.GetOpType() == UsdGeom.XformOp.TypeOrient:
            orient_op = op

    if translate_op is None:
        translate_op = xform.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble)
    translate_op.Set(Gf.Vec3d(float(pos[0]), float(pos[1]), float(pos[2])))

    if orient_op is None:
        orient_op = xform.AddOrientOp(UsdGeom.XformOp.PrecisionDouble)
    quat = Gf.Quatd(float(rot[0]), Gf.Vec3d(float(rot[1]), float(rot[2]), float(rot[3])))
    orient_op.Set(quat)


def _path_matches_asset(asset_path: str, expected_path: str) -> bool:
    if not asset_path:
        return False
    asset_path = os.path.expanduser(str(asset_path))
    expected_path = os.path.expanduser(str(expected_path))
    if os.path.isabs(asset_path) and os.path.exists(asset_path) and os.path.exists(expected_path):
        try:
            return os.path.samefile(asset_path, expected_path)
        except OSError:
            pass
    norm_asset = os.path.normpath(os.path.abspath(asset_path)) if os.path.isabs(asset_path) else os.path.normpath(asset_path)
    norm_expected = os.path.normpath(os.path.abspath(expected_path))
    if norm_asset == norm_expected:
        return True
    return norm_expected.endswith(norm_asset) or norm_asset.endswith(os.path.basename(norm_expected))


def _reference_asset_paths_from_spec(spec: Any) -> list[str]:
    paths: list[str] = []
    for list_name in ("referenceList", "payloadList"):
        list_op = getattr(spec, list_name, None)
        if list_op is None:
            continue
        for item_name in ("explicitItems", "prependedItems", "appendedItems", "addedItems"):
            for item in getattr(list_op, item_name, []) or []:
                asset_path = getattr(item, "assetPath", None)
                if asset_path:
                    paths.append(str(asset_path))
    return paths


def _prim_reference_asset_paths(prim: Usd.Prim) -> list[str]:
    paths: list[str] = []
    try:
        for spec in prim.GetPrimStack():
            paths.extend(_reference_asset_paths_from_spec(spec))
    except Exception:
        pass
    return paths


def _subtree_references_asset(root_prim: Usd.Prim, expected_path: str) -> tuple[bool, list[str]]:
    seen_paths: list[str] = []
    for prim in Usd.PrimRange(root_prim):
        for asset_path in _prim_reference_asset_paths(prim):
            seen_paths.append(asset_path)
            if _path_matches_asset(asset_path, expected_path):
                return True, seen_paths
    return False, seen_paths


def _make_subtree_non_instanceable(root_prim: Usd.Prim) -> int:
    changed = 0
    for prim in Usd.PrimRange(root_prim):
        try:
            if prim.IsInstanceable():
                prim.SetInstanceable(False)
                changed += 1
        except Exception:
            pass
    return changed


def _find_existing_robot_prim(stage: Usd.Stage, cfg: dict[str, Any]) -> Usd.Prim:
    robot_cfg = cfg.get("robot", {})
    target_path = _normalize_prim_path(ROBOT_PRIM_PATH or robot_cfg.get("target_prim_path", ""))
    robot_prim = stage.GetPrimAtPath(target_path)
    if not robot_prim or not robot_prim.IsValid():
        raise RuntimeError(
            f"Robot prim was not found at '{target_path}'. "
            "Add the robot USD to the open stage first, or update ROBOT_PRIM_PATH."
        )

    if _group_enabled(cfg, "make_subtree_non_instanceable", "make_subtree_non_instanceable", default=False):
        changed = _make_subtree_non_instanceable(robot_prim)
        if changed:
            print(f"[g1-existing-loader] Set instanceable=false on {changed} prim(s).")

    expected_usd = ROBOT_USD_PATH or robot_cfg.get("expected_usd_path", "")
    if _group_enabled(cfg, "check_existing_prim_and_usd", default=True) and bool(robot_cfg.get("require_matching_usd_reference", True)):
        matched, seen_paths = _subtree_references_asset(robot_prim, expected_usd)
        if not matched:
            if not seen_paths:
                detail = "No reference or payload asset path was found under this prim."
            else:
                detail = "Found asset paths: " + ", ".join(sorted(set(seen_paths)))
            raise RuntimeError(
                f"Prim '{target_path}' does not reference the expected robot USD: {expected_usd}. {detail}"
            )
    return robot_prim


def _matches_any(name: str, patterns: list[str]) -> bool:
    return any(re.fullmatch(str(pattern), name) for pattern in patterns)


def _resolve_pattern_value(value: Any, joint_name: str) -> Any:
    if value is None:
        return None
    if isinstance(value, dict):
        for pattern, item in value.items():
            if re.fullmatch(str(pattern), joint_name):
                return item
        return None
    return value


def _collect_joint_prims(root_prim: Usd.Prim) -> list[Usd.Prim]:
    joint_prims = []
    for prim in Usd.PrimRange(root_prim):
        if prim.IsA(UsdPhysics.RevoluteJoint) or prim.IsA(UsdPhysics.PrismaticJoint) or prim.IsA(UsdPhysics.Joint):
            joint_prims.append(prim)
    return joint_prims


def _collect_rigid_body_prims(root_prim: Usd.Prim) -> list[Usd.Prim]:
    rigid_prims = []
    for prim in Usd.PrimRange(root_prim):
        if UsdPhysics.RigidBodyAPI(prim):
            rigid_prims.append(prim)
    return rigid_prims


def _collect_articulation_root_prims(root_prim: Usd.Prim) -> list[Usd.Prim]:
    roots = []
    for prim in Usd.PrimRange(root_prim):
        if UsdPhysics.ArticulationRootAPI(prim):
            roots.append(prim)
    return roots


def _remove_api_schema(prim: Usd.Prim, api_cls: Any) -> None:
    if api_cls is None:
        return
    try:
        prim.RemoveAPI(api_cls)
    except Exception as exc:
        print(f"[g1-existing-loader] Warning: could not remove {api_cls} from {prim.GetPath()}: {exc}")


def _find_joint_init_value(joint_name: str, joint_pos_cfg: dict[str, Any]) -> Any:
    for pattern, value in joint_pos_cfg.items():
        if re.fullmatch(str(pattern), joint_name):
            return value
    return None


def _joint_drive_name(joint_prim: Usd.Prim) -> str:
    if joint_prim.IsA(UsdPhysics.PrismaticJoint):
        return "linear"
    return "angular"


def _to_usd_joint_position(joint_prim: Usd.Prim, value: float, unit: str) -> float:
    if joint_prim.IsA(UsdPhysics.PrismaticJoint):
        return float(value)
    if unit == "radians":
        return math.degrees(float(value))
    return float(value)


def _to_usd_angular_drive_gain(value: float, input_unit: str) -> float:
    normalized_unit = str(input_unit).strip().lower()
    if normalized_unit in {"radian", "radians", "per_radian"}:
        # USD angular drives use degree-based stiffness and damping units.
        return float(value) * math.pi / 180.0
    if normalized_unit in {"degree", "degrees", "per_degree"}:
        return float(value)
    raise ValueError(
        "Unsupported angular drive gain input unit "
        f"'{input_unit}'. Expected 'radians' or 'degrees'."
    )


def _to_usd_drive_gain(drive_name: str, value: float, angular_input_unit: str) -> float:
    if drive_name == "angular":
        return _to_usd_angular_drive_gain(value, angular_input_unit)
    return float(value)


def _apply_articulation_props(robot_prim: Usd.Prim, cfg: dict[str, Any]) -> None:
    if not _group_enabled(cfg, "articulation_root", "apply_articulation_root_to_robot_prim", default=False):
        return

    props = cfg.get("spawn", {}).get("articulation_props", {})
    for articulation_root in _collect_articulation_root_prims(robot_prim):
        if articulation_root != robot_prim:
            _remove_api_schema(articulation_root, UsdPhysics.ArticulationRootAPI)
            _remove_api_schema(articulation_root, getattr(PhysxSchema, "PhysxArticulationAPI", None))

    UsdPhysics.ArticulationRootAPI.Apply(robot_prim)
    physx_articulation = _apply_api(getattr(PhysxSchema, "PhysxArticulationAPI", None), robot_prim)
    _set_schema_attr(
        physx_articulation,
        "GetEnabledSelfCollisionsAttr",
        "CreateEnabledSelfCollisionsAttr",
        bool(props.get("enabled_self_collisions", False)),
    )
    _set_schema_attr(
        physx_articulation,
        "GetSolverPositionIterationCountAttr",
        "CreateSolverPositionIterationCountAttr",
        int(props.get("solver_position_iteration_count", 4)),
    )
    _set_schema_attr(
        physx_articulation,
        "GetSolverVelocityIterationCountAttr",
        "CreateSolverVelocityIterationCountAttr",
        int(props.get("solver_velocity_iteration_count", 1)),
    )

    _set_attr(robot_prim, "physxArticulation:enabledSelfCollisions", Sdf.ValueTypeNames.Bool, bool(props.get("enabled_self_collisions", False)))
    _set_attr(robot_prim, "physxArticulation:solverPositionIterationCount", Sdf.ValueTypeNames.Int, int(props.get("solver_position_iteration_count", 4)))
    _set_attr(robot_prim, "physxArticulation:solverVelocityIterationCount", Sdf.ValueTypeNames.Int, int(props.get("solver_velocity_iteration_count", 1)))
    if props.get("fix_root_link") is not None:
        _set_attr(robot_prim, "physxArticulation:fixRootLink", Sdf.ValueTypeNames.Bool, bool(props["fix_root_link"]))


def _apply_rigid_body_props(robot_prim: Usd.Prim, cfg: dict[str, Any]) -> int:
    if not _group_enabled(cfg, "rigid_body_props", "apply_rigid_props_to_all_rigid_bodies", default=False):
        return 0

    props = cfg.get("spawn", {}).get("rigid_props", {})
    rigid_prims = _collect_rigid_body_prims(robot_prim)
    for prim in rigid_prims:
        physx_rb = _apply_api(getattr(PhysxSchema, "PhysxRigidBodyAPI", None), prim)
        _set_schema_attr(physx_rb, "GetDisableGravityAttr", "CreateDisableGravityAttr", bool(props.get("disable_gravity", False)))
        _set_schema_attr(physx_rb, "GetRetainAccelerationsAttr", "CreateRetainAccelerationsAttr", bool(props.get("retain_accelerations", True)))
        _set_schema_attr(physx_rb, "GetLinearDampingAttr", "CreateLinearDampingAttr", float(props.get("linear_damping", 0.0)))
        _set_schema_attr(physx_rb, "GetAngularDampingAttr", "CreateAngularDampingAttr", float(props.get("angular_damping", 0.0)))
        _set_schema_attr(physx_rb, "GetMaxLinearVelocityAttr", "CreateMaxLinearVelocityAttr", float(props.get("max_linear_velocity", 1000.0)))
        _set_schema_attr(physx_rb, "GetMaxAngularVelocityAttr", "CreateMaxAngularVelocityAttr", float(props.get("max_angular_velocity", 1000.0)))
        _set_schema_attr(physx_rb, "GetMaxDepenetrationVelocityAttr", "CreateMaxDepenetrationVelocityAttr", float(props.get("max_depenetration_velocity", 1.0)))

        _set_attr(prim, "physxRigidBody:disableGravity", Sdf.ValueTypeNames.Bool, bool(props.get("disable_gravity", False)))
        _set_attr(prim, "physxRigidBody:retainAccelerations", Sdf.ValueTypeNames.Bool, bool(props.get("retain_accelerations", True)))
        _set_attr(prim, "physxRigidBody:linearDamping", Sdf.ValueTypeNames.Float, float(props.get("linear_damping", 0.0)))
        _set_attr(prim, "physxRigidBody:angularDamping", Sdf.ValueTypeNames.Float, float(props.get("angular_damping", 0.0)))
        _set_attr(prim, "physxRigidBody:maxLinearVelocity", Sdf.ValueTypeNames.Float, float(props.get("max_linear_velocity", 1000.0)))
        _set_attr(prim, "physxRigidBody:maxAngularVelocity", Sdf.ValueTypeNames.Float, float(props.get("max_angular_velocity", 1000.0)))
        _set_attr(prim, "physxRigidBody:maxDepenetrationVelocity", Sdf.ValueTypeNames.Float, float(props.get("max_depenetration_velocity", 1.0)))
    return len(rigid_prims)


def _apply_contact_report_api(robot_prim: Usd.Prim, cfg: dict[str, Any]) -> int:
    spawn_cfg = cfg.get("spawn", {})
    if not bool(spawn_cfg.get("activate_contact_sensors", False)):
        return 0
    if not _group_enabled(cfg, "contact_report_api", "apply_contact_report_api", default=False):
        return 0

    contact_api_cls = getattr(PhysxSchema, "PhysxContactReportAPI", None)
    if contact_api_cls is None:
        return 0

    count = 0
    for prim in _collect_rigid_body_prims(robot_prim):
        contact_api = _apply_api(contact_api_cls, prim)
        if contact_api:
            _set_schema_attr(contact_api, "GetThresholdAttr", "CreateThresholdAttr", 0.0)
            count += 1
    return count


def _apply_joint_state(joint_prim: Usd.Prim, drive_name: str, target_pos: float, target_vel: float) -> None:
    if not hasattr(UsdPhysics, "JointStateAPI"):
        _set_attr(joint_prim, f"state:{drive_name}:physics:position", Sdf.ValueTypeNames.Float, target_pos)
        _set_attr(joint_prim, f"state:{drive_name}:physics:velocity", Sdf.ValueTypeNames.Float, target_vel)
        return
    try:
        state_api = UsdPhysics.JointStateAPI.Apply(joint_prim, drive_name)
        _set_schema_attr(state_api, "GetPositionAttr", "CreatePositionAttr", target_pos)
        _set_schema_attr(state_api, "GetVelocityAttr", "CreateVelocityAttr", target_vel)
    except Exception:
        _set_attr(joint_prim, f"state:{drive_name}:physics:position", Sdf.ValueTypeNames.Float, target_pos)
        _set_attr(joint_prim, f"state:{drive_name}:physics:velocity", Sdf.ValueTypeNames.Float, target_vel)


def _apply_joint_drive(
    joint_prim: Usd.Prim,
    actuator_cfg: dict[str, Any],
    init_pos: Any,
    init_vel: Any,
    unit: str,
    angular_gain_input_unit: str,
    apply_drive_properties: bool,
    apply_target_position: bool,
    apply_joint_state: bool,
) -> None:
    joint_name = joint_prim.GetName()
    drive_name = _joint_drive_name(joint_prim)
    drive = None
    if apply_drive_properties or apply_target_position:
        drive = UsdPhysics.DriveAPI.Apply(joint_prim, drive_name)
        _apply_api(getattr(PhysxSchema, "PhysxJointAPI", None), joint_prim)

    stiffness = _resolve_pattern_value(actuator_cfg.get("stiffness"), joint_name)
    damping = _resolve_pattern_value(actuator_cfg.get("damping"), joint_name)
    effort = _resolve_pattern_value(actuator_cfg.get("effort_limit_sim", actuator_cfg.get("effort_limit")), joint_name)
    velocity = _resolve_pattern_value(actuator_cfg.get("velocity_limit_sim", actuator_cfg.get("velocity_limit")), joint_name)
    armature = _resolve_pattern_value(actuator_cfg.get("armature"), joint_name)
    friction = _resolve_pattern_value(actuator_cfg.get("friction"), joint_name)

    if apply_drive_properties and drive is not None:
        if stiffness is not None:
            usd_stiffness = _to_usd_drive_gain(drive_name, float(stiffness), angular_gain_input_unit)
            drive.CreateStiffnessAttr(usd_stiffness).Set(usd_stiffness)
        if damping is not None:
            usd_damping = _to_usd_drive_gain(drive_name, float(damping), angular_gain_input_unit)
            drive.CreateDampingAttr(usd_damping).Set(usd_damping)
        if effort is not None:
            drive.CreateMaxForceAttr(float(effort)).Set(float(effort))
        drive.CreateTypeAttr(str(actuator_cfg.get("drive_type", "force"))).Set(str(actuator_cfg.get("drive_type", "force")))
        if velocity is not None:
            _set_attr(joint_prim, "physxJoint:maxJointVelocity", Sdf.ValueTypeNames.Float, float(velocity))
        if armature is not None:
            _set_attr(joint_prim, "physxJoint:armature", Sdf.ValueTypeNames.Float, float(armature))
        if friction is not None:
            _set_attr(joint_prim, "physxJoint:jointFriction", Sdf.ValueTypeNames.Float, float(friction))

    if apply_target_position and drive is not None and init_pos is not None:
        target_pos = _to_usd_joint_position(joint_prim, float(init_pos), unit)
        drive.CreateTargetPositionAttr(target_pos).Set(target_pos)
    else:
        target_pos = 0.0
    target_vel = float(init_vel or 0.0)
    if apply_target_position and drive is not None:
        drive.CreateTargetVelocityAttr(target_vel).Set(target_vel)

    if apply_joint_state and init_pos is not None:
        _apply_joint_state(joint_prim, drive_name, target_pos, target_vel)


def _apply_actuators_and_init_state(robot_prim: Usd.Prim, cfg: dict[str, Any]) -> tuple[dict[str, int], set[str]]:
    joint_prims = _collect_joint_prims(robot_prim)
    init_state = cfg.get("init_state", {})
    joint_pos_cfg = init_state.get("joint_pos", {})
    joint_vel_cfg = init_state.get("joint_vel", {})
    joint_unit = init_state.get("joint_position_unit", "radians")
    angular_gain_input_unit = cfg.get("actuator_units", {}).get("angular_drive_gain_input_unit", "radians")
    apply_drive_properties = _group_enabled(cfg, "joint_drive_properties", default=False)
    apply_target_position = _group_enabled(cfg, "joint_drive_targets", "apply_joint_drive_targets_from_init_state", default=False)
    apply_joint_state = _group_enabled(cfg, "joint_state", "apply_joint_state_from_init_state", default=False)
    should_write_joint_config = apply_drive_properties or apply_target_position or apply_joint_state
    counts: dict[str, int] = {}
    actuated_joint_names: set[str] = set()

    for group_name, actuator_cfg in cfg.get("actuators", {}).items():
        patterns = actuator_cfg.get("joint_names_expr", [])
        if isinstance(patterns, str):
            patterns = [patterns]
        counts[group_name] = 0
        for joint_prim in joint_prims:
            joint_name = joint_prim.GetName()
            if not _matches_any(joint_name, patterns):
                continue
            init_pos = _find_joint_init_value(joint_name, joint_pos_cfg)
            if init_pos is None:
                init_pos = 0.0
            init_vel = _find_joint_init_value(joint_name, joint_vel_cfg)
            if should_write_joint_config:
                _apply_joint_drive(
                    joint_prim,
                    actuator_cfg,
                    init_pos,
                    init_vel,
                    joint_unit,
                    angular_gain_input_unit,
                    apply_drive_properties,
                    apply_target_position,
                    apply_joint_state,
                )
            counts[group_name] += 1
            actuated_joint_names.add(joint_name)
    return counts, actuated_joint_names


def _validate_robot_config(robot_prim: Usd.Prim, cfg: dict[str, Any], counts: dict[str, int], actuated_joint_names: set[str]) -> None:
    if not _group_enabled(cfg, "validation", default=False):
        return

    validation = cfg.get("validation", {})
    if bool(validation.get("fail_on_zero_actuator_matches", True)):
        zero_groups = [name for name, count in counts.items() if count == 0]
        if zero_groups:
            raise RuntimeError(f"These actuator groups matched zero joints: {zero_groups}")

    expected_count = validation.get("expected_joint_count")
    if expected_count is not None and len(actuated_joint_names) != int(expected_count):
        raise RuntimeError(
            f"Expected {expected_count} actuated joints, but matched {len(actuated_joint_names)}. "
            f"Counts: {counts}"
        )

    if bool(validation.get("fail_on_missing_configured_joints", True)):
        all_joint_names = {prim.GetName() for prim in _collect_joint_prims(robot_prim)}
        missing = []
        for pattern in cfg.get("init_state", {}).get("joint_pos", {}).keys():
            if not any(re.fullmatch(str(pattern), joint_name) for joint_name in all_joint_names):
                missing.append(str(pattern))
        if missing:
            raise RuntimeError(f"Configured joint names/patterns were not found in the robot USD: {missing}")


def _write_summary_attrs(
    robot_prim: Usd.Prim,
    cfg: dict[str, Any],
    counts: dict[str, int],
    rigid_body_count: int,
    contact_report_count: int,
) -> None:
    if not _group_enabled(cfg, "summary_attributes", "write_config_summary_attributes", default=True):
        return
    runtime_cfg = cfg.get("isaaclab_runtime", {})
    _set_attr(robot_prim, "unitree:robotConfigYaml", Sdf.ValueTypeNames.String, os.path.abspath(ROBOT_CONFIG_YAML_PATH))
    _set_attr(robot_prim, "unitree:robotUsdPath", Sdf.ValueTypeNames.String, os.path.abspath(ROBOT_USD_PATH))
    _set_attr(robot_prim, "unitree:sourceIsaacLabConfig", Sdf.ValueTypeNames.String, cfg.get("metadata", {}).get("source", {}).get("isaaclab_config", ""))
    _set_attr(robot_prim, "unitree:appliedMode", Sdf.ValueTypeNames.String, "existing_robot")
    _set_attr(robot_prim, "isaaclab:softJointPosLimitFactor", Sdf.ValueTypeNames.Float, float(runtime_cfg.get("soft_joint_pos_limit_factor", 1.0)))
    _set_attr(robot_prim, "unitree:appliedActuatorJointCounts", Sdf.ValueTypeNames.String, json.dumps(counts, sort_keys=True))
    _set_attr(robot_prim, "unitree:appliedRigidBodyCount", Sdf.ValueTypeNames.Int, int(rigid_body_count))
    _set_attr(robot_prim, "unitree:appliedContactReportCount", Sdf.ValueTypeNames.Int, int(contact_report_count))


def _select_prim(path: str) -> None:
    try:
        import omni.kit.commands

        omni.kit.commands.execute("SelectPrims", old_selected_paths=[], new_selected_paths=[path], expand_in_stage=True)
    except Exception:
        pass


def _save_stage_if_possible(stage: Usd.Stage) -> bool:
    root_layer = stage.GetRootLayer()
    if root_layer.anonymous:
        print(
            "[g1-existing-loader] Stage root layer is anonymous; skip Save(). "
            "Use File > Save As in Isaac Sim first if you want to persist these edits."
        )
        return False
    try:
        root_layer.Save()
        return True
    except Exception as exc:
        print(f"[g1-existing-loader] Warning: could not save stage root layer '{root_layer.identifier}': {exc}")
        return False


def apply_config_to_existing_robot() -> Usd.Prim:
    cfg = _load_yaml(ROBOT_CONFIG_YAML_PATH)
    stage = _current_stage()
    print(f"[g1-existing-loader] Debug apply groups: {json.dumps(_enabled_groups(cfg), sort_keys=True)}")
    robot_prim = _find_existing_robot_prim(stage, cfg)

    init_state = cfg.get("init_state", {})
    if _group_enabled(cfg, "root_transform", "apply_root_transform_from_yaml", default=False):
        _set_root_transform(
            robot_prim,
            init_state.get("pos", [0.0, 0.0, 0.8]),
            init_state.get("rot", [1.0, 0.0, 0.0, 0.0]),
        )

    _apply_articulation_props(robot_prim, cfg)
    rigid_body_count = _apply_rigid_body_props(robot_prim, cfg)
    contact_report_count = _apply_contact_report_api(robot_prim, cfg)
    actuator_counts, actuated_joint_names = _apply_actuators_and_init_state(robot_prim, cfg)
    _validate_robot_config(robot_prim, cfg, actuator_counts, actuated_joint_names)
    _write_summary_attrs(robot_prim, cfg, actuator_counts, rigid_body_count, contact_report_count)

    if _group_enabled(cfg, "select_robot_after_load", "select_robot_after_load", default=True):
        _select_prim(str(robot_prim.GetPath()))
    if _group_enabled(cfg, "save_stage_after_load", "save_stage_after_load", default=False):
        _save_stage_if_possible(stage)

    print("[g1-existing-loader] Applied robot config:")
    print(f"[g1-existing-loader]   prim: {robot_prim.GetPath()}")
    print(f"[g1-existing-loader]   usd:  {os.path.abspath(ROBOT_USD_PATH)}")
    print(f"[g1-existing-loader]   yaml: {os.path.abspath(ROBOT_CONFIG_YAML_PATH)}")
    print(f"[g1-existing-loader]   rigid bodies configured: {rigid_body_count}")
    print(f"[g1-existing-loader]   contact reports configured: {contact_report_count}")
    print(f"[g1-existing-loader]   actuator joint counts: {actuator_counts}")
    return robot_prim


try:
    apply_config_to_existing_robot()
except Exception as exc:
    print(f"[g1-existing-loader] ERROR: {exc}")
    raise
