BASE_EXISTING_ROBOT_LOADER = "/home/khoa-ng/Job_Project/My_Code/humanoid_robot_control/RL_control/G1_control/robot_avai_config/apply_g1_29dof_wholebody_dex3_existing_robot_config.py"
ROBOT_GYM_CONFIG_YAML_PATH = "/home/khoa-ng/Job_Project/My_Code/humanoid_robot_control/RL_control/G1_control/robot_gym_config/g1_29dof_with_dex3_unitree_rl_gym_policy.yaml"

import os


def _load_base_loader_source() -> str:
    if not os.path.isfile(BASE_EXISTING_ROBOT_LOADER):
        raise FileNotFoundError(f"Base Isaac Sim robot loader does not exist: {BASE_EXISTING_ROBOT_LOADER}")
    with open(BASE_EXISTING_ROBOT_LOADER, "r", encoding="utf-8") as stream:
        return stream.read()


def _patch_loader_source(source: str) -> str:
    original = source
    source = source.replace(
        'ROBOT_CONFIG_YAML_PATH = "/home/khoa-ng/Job_Project/My_Code/humanoid_robot_control/RL_control/G1_control/robot_avai_config/g1_29dof_wholebody_dex3_existing_robot.yaml"',
        f'ROBOT_CONFIG_YAML_PATH = "{ROBOT_GYM_CONFIG_YAML_PATH}"',
    )
    if source == original:
        raise RuntimeError("Could not patch ROBOT_CONFIG_YAML_PATH in the base Isaac Sim loader.")
    source = source.replace("[g1-existing-loader]", "[g1-gym-policy-loader]")
    return source


exec(compile(_patch_loader_source(_load_base_loader_source()), BASE_EXISTING_ROBOT_LOADER, "exec"), globals())
