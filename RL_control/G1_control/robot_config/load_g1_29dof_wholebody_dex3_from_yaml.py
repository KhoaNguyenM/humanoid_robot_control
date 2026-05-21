ROBOT_USD_PATH = "/home/khoa-ng/Job_Project/My_Code/humanoid_robot_control/unitree_sim_isaaclab/assets/robots/g1-29dof_wholebody_dex3/g1_29dof_with_dex3_rev_1_0.usd"
ROBOT_CONFIG_YAML_PATH = "/home/khoa-ng/Job_Project/My_Code/humanoid_robot_control/RL_control/G1_control/robot_config/g1_29dof_wholebody_dex3_robot.yaml"
ROBOT_PRIM_PATH = "/World/G1_29DOF_DEX3"

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


def _normalize_prim_path(path: str, robot_name: str) -> Sdf.Path:
    raw = str(path or "").strip()
    if not raw:
        raw = f"/World/{robot_name}"
    if not raw.startswith("/"):
        raw = "/" + raw
    if raw.endswith("/"):
        raw = raw.rstrip("/") + "/" + robot_name
    path_obj = Sdf.Path(raw)
    if not path_obj.IsAbsolutePath() or not path_obj.IsPrimPath():
        raise ValueError(f"Invalid robot prim path: {raw}")
    return path_obj


def _ensure_parent_xforms(stage: Usd.Stage, path: Sdf.Path) -> None:
    parent = path.GetParentPath()
    if parent == Sdf.Path.absoluteRootPath:
        return
    missing = []
    while parent != Sdf.Path.absoluteRootPath and not stage.GetPrimAtPath(parent):
        missing.append(parent)
        parent = parent.GetParentPath()
    for item in reversed(missing):
        UsdGeom.Xform.Define(stage, item)


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
    try:
        api = api_cls(prim)
        if not api:
            api = api_cls.Apply(prim)
        return api
    except Exception:
        return None


def _clear_xform_ops(prim: Usd.Prim) -> None:
    for attr in list(prim.GetAttributes()):
        if attr.GetName().startswith("xformOp:"):
            prim.RemoveProperty(attr.GetName())
    if prim.HasAttribute("xformOpOrder"):
        prim.RemoveProperty("xformOpOrder")


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

    if orient_op.GetPrecision() == UsdGeom.XformOp.PrecisionFloat:
        orient_op.Set(Gf.Quatf(float(rot[0]), Gf.Vec3f(float(rot[1]), float(rot[2]), float(rot[3]))))
    else:
        orient_op.Set(Gf.Quatd(float(rot[0]), Gf.Vec3d(float(rot[1]), float(rot[2]), float(rot[3]))))


def _add_robot_reference(stage: Usd.Stage, cfg: dict[str, Any]) -> Usd.Prim:
    robot_cfg = cfg["robot"]
    robot_name = robot_cfg.get("name", "G1_29DOF_DEX3")
    usd_path = os.path.abspath(os.path.expanduser(ROBOT_USD_PATH or robot_cfg["usd_path"]))
    if not os.path.isfile(usd_path):
        raise FileNotFoundError(f"Robot USD does not exist: {usd_path}")

    robot_path = _normalize_prim_path(ROBOT_PRIM_PATH or robot_cfg.get("default_prim_path"), robot_name)
    _ensure_parent_xforms(stage, robot_path)
    robot_xform = UsdGeom.Xform.Define(stage, robot_path)
    robot_prim = robot_xform.GetPrim()
    robot_prim.SetInstanceable(bool(robot_cfg.get("make_instanceable", False)))

    refs = robot_prim.GetReferences()
    if bool(robot_cfg.get("clear_existing_references", True)):
        refs.ClearReferences()

    reference_prim_path = robot_cfg.get("reference_prim_path")
    if reference_prim_path:
        refs.AddReference(assetPath=usd_path, primPath=Sdf.Path(reference_prim_path))
    else:
        refs.AddReference(assetPath=usd_path)

    init_state = cfg.get("init_state", {})
    _set_root_transform(
        robot_prim,
        init_state.get("pos", [0.0, 0.0, 0.8]),
        init_state.get("rot", [1.0, 0.0, 0.0, 0.0]),
    )
    return robot_prim


def _matches_any(name: str, patterns: list[str]) -> bool:
    return any(re.fullmatch(pattern, name) for pattern in patterns)


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
    try:
        prim.RemoveAPI(api_cls)
    except Exception as exc:
        print(f"[g1-loader] Warning: could not remove {api_cls} from {prim.GetPath()}: {exc}")


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


def _apply_articulation_props(robot_prim: Usd.Prim, cfg: dict[str, Any]) -> None:
    options = cfg.get("script_options", {})
    if not bool(options.get("apply_articulation_root_to_robot_prim", True)):
        return

    props = cfg.get("spawn", {}).get("articulation_props", {})
    for articulation_root in _collect_articulation_root_prims(robot_prim):
        if articulation_root != robot_prim:
            _remove_api_schema(articulation_root, UsdPhysics.ArticulationRootAPI)
            physx_articulation_api = getattr(PhysxSchema, "PhysxArticulationAPI", None)
            if physx_articulation_api is not None:
                _remove_api_schema(articulation_root, physx_articulation_api)

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
    if not bool(cfg.get("script_options", {}).get("apply_rigid_props_to_all_rigid_bodies", True)):
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
    apply_target_position: bool,
    apply_joint_state: bool,
) -> None:
    joint_name = joint_prim.GetName()
    drive_name = _joint_drive_name(joint_prim)
    drive = UsdPhysics.DriveAPI.Apply(joint_prim, drive_name)
    _apply_api(getattr(PhysxSchema, "PhysxJointAPI", None), joint_prim)

    stiffness = _resolve_pattern_value(actuator_cfg.get("stiffness"), joint_name)
    damping = _resolve_pattern_value(actuator_cfg.get("damping"), joint_name)
    effort = _resolve_pattern_value(actuator_cfg.get("effort_limit_sim", actuator_cfg.get("effort_limit")), joint_name)
    velocity = _resolve_pattern_value(actuator_cfg.get("velocity_limit_sim", actuator_cfg.get("velocity_limit")), joint_name)
    armature = _resolve_pattern_value(actuator_cfg.get("armature"), joint_name)
    friction = _resolve_pattern_value(actuator_cfg.get("friction"), joint_name)

    if stiffness is not None:
        drive.CreateStiffnessAttr(float(stiffness)).Set(float(stiffness))
    if damping is not None:
        drive.CreateDampingAttr(float(damping)).Set(float(damping))
    if effort is not None:
        drive.CreateMaxForceAttr(float(effort)).Set(float(effort))
    drive.CreateTypeAttr(str(actuator_cfg.get("drive_type", "force"))).Set(str(actuator_cfg.get("drive_type", "force")))

    if apply_target_position and init_pos is not None:
        target_pos = _to_usd_joint_position(joint_prim, float(init_pos), unit)
        drive.CreateTargetPositionAttr(target_pos).Set(target_pos)
    else:
        target_pos = 0.0
    target_vel = float(init_vel or 0.0)
    drive.CreateTargetVelocityAttr(target_vel).Set(target_vel)

    if velocity is not None:
        _set_attr(joint_prim, "physxJoint:maxJointVelocity", Sdf.ValueTypeNames.Float, float(velocity))
    if armature is not None:
        _set_attr(joint_prim, "physxJoint:armature", Sdf.ValueTypeNames.Float, float(armature))
    if friction is not None:
        _set_attr(joint_prim, "physxJoint:jointFriction", Sdf.ValueTypeNames.Float, float(friction))

    if apply_joint_state and init_pos is not None:
        _apply_joint_state(joint_prim, drive_name, target_pos, target_vel)


def _apply_actuators_and_init_state(robot_prim: Usd.Prim, cfg: dict[str, Any]) -> dict[str, int]:
    joint_prims = _collect_joint_prims(robot_prim)
    init_state = cfg.get("init_state", {})
    joint_pos_cfg = init_state.get("joint_pos", {})
    joint_vel_cfg = init_state.get("joint_vel", {})
    joint_unit = init_state.get("joint_position_unit", "radians")
    options = cfg.get("script_options", {})
    apply_target_position = bool(options.get("apply_joint_drive_targets_from_init_state", True))
    apply_joint_state = bool(options.get("apply_joint_state_from_init_state", True))
    counts: dict[str, int] = {}

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
            _apply_joint_drive(
                joint_prim,
                actuator_cfg,
                init_pos,
                init_vel,
                joint_unit,
                apply_target_position,
                apply_joint_state,
            )
            counts[group_name] += 1
    return counts


def _write_summary_attrs(robot_prim: Usd.Prim, cfg: dict[str, Any], counts: dict[str, int], rigid_body_count: int) -> None:
    if not bool(cfg.get("script_options", {}).get("write_config_summary_attributes", True)):
        return
    spawn_cfg = cfg.get("spawn", {})
    runtime_cfg = cfg.get("isaaclab_runtime", {})
    _set_attr(robot_prim, "unitree:robotConfigYaml", Sdf.ValueTypeNames.String, os.path.abspath(ROBOT_CONFIG_YAML_PATH))
    _set_attr(robot_prim, "unitree:robotUsdPath", Sdf.ValueTypeNames.String, os.path.abspath(ROBOT_USD_PATH))
    _set_attr(robot_prim, "unitree:sourceIsaacLabConfig", Sdf.ValueTypeNames.String, cfg.get("metadata", {}).get("source", {}).get("isaaclab_config", ""))
    _set_attr(robot_prim, "isaaclab:activateContactSensors", Sdf.ValueTypeNames.Bool, bool(spawn_cfg.get("activate_contact_sensors", False)))
    _set_attr(robot_prim, "isaaclab:softJointPosLimitFactor", Sdf.ValueTypeNames.Float, float(runtime_cfg.get("soft_joint_pos_limit_factor", 1.0)))
    _set_attr(robot_prim, "unitree:appliedActuatorJointCounts", Sdf.ValueTypeNames.String, json.dumps(counts, sort_keys=True))
    _set_attr(robot_prim, "unitree:appliedRigidBodyCount", Sdf.ValueTypeNames.Int, int(rigid_body_count))


def _select_prim(path: str) -> None:
    try:
        import omni.kit.commands

        omni.kit.commands.execute("SelectPrims", old_selected_paths=[], new_selected_paths=[path], expand_in_stage=True)
    except Exception:
        pass


def load_robot_from_yaml() -> Usd.Prim:
    cfg = _load_yaml(ROBOT_CONFIG_YAML_PATH)
    stage = _current_stage()
    robot_prim = _add_robot_reference(stage, cfg)
    _apply_articulation_props(robot_prim, cfg)
    rigid_body_count = _apply_rigid_body_props(robot_prim, cfg)
    actuator_counts = _apply_actuators_and_init_state(robot_prim, cfg)
    _write_summary_attrs(robot_prim, cfg, actuator_counts, rigid_body_count)

    if bool(cfg.get("script_options", {}).get("select_robot_after_load", True)):
        _select_prim(str(robot_prim.GetPath()))
    if bool(cfg.get("script_options", {}).get("save_stage_after_load", False)):
        stage.GetRootLayer().Save()

    print("[g1-loader] Loaded robot:")
    print(f"[g1-loader]   prim: {robot_prim.GetPath()}")
    print(f"[g1-loader]   usd:  {os.path.abspath(ROBOT_USD_PATH)}")
    print(f"[g1-loader]   yaml: {os.path.abspath(ROBOT_CONFIG_YAML_PATH)}")
    print(f"[g1-loader]   rigid bodies configured: {rigid_body_count}")
    print(f"[g1-loader]   actuator joint counts: {actuator_counts}")
    return robot_prim


load_robot_from_yaml()
