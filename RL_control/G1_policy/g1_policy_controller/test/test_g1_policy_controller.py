from pathlib import Path
import unittest

import numpy as np
import torch

from g1_policy_controller.core import (
    ACTION_SCALE,
    DEFAULT_JOINT_ANGLES,
    NUM_ACTIONS,
    NUM_OBSERVATIONS,
    POLICY_JOINT_NAMES,
    PolicyScheduler,
    build_full_joint_targets,
    build_observation,
    extract_policy_joint_state,
    projected_gravity,
)


ISAAC_SIM_JOINT_NAMES = [
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "waist_roll_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "waist_yaw_joint",
    "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint",
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "left_elbow_joint",
    "right_elbow_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "left_wrist_roll_joint",
    "right_wrist_roll_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "left_wrist_pitch_joint",
    "right_wrist_pitch_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_wrist_yaw_joint",
    "right_wrist_yaw_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
    "left_hand_index_0_joint",
    "left_hand_middle_0_joint",
    "left_hand_thumb_0_joint",
    "right_hand_index_0_joint",
    "right_hand_middle_0_joint",
    "right_hand_thumb_0_joint",
    "left_hand_index_1_joint",
    "left_hand_middle_1_joint",
    "left_hand_thumb_1_joint",
    "right_hand_index_1_joint",
    "right_hand_middle_1_joint",
    "right_hand_thumb_1_joint",
    "left_hand_thumb_2_joint",
    "right_hand_thumb_2_joint",
]


class G1PolicyCoreTest(unittest.TestCase):
    def test_projected_gravity_reference_poses(self):
        np.testing.assert_allclose(
            projected_gravity([1.0, 0.0, 0.0, 0.0]),
            [0.0, 0.0, -1.0],
            atol=1e-7,
        )

        half_angle = np.deg2rad(10.0) / 2.0
        roll_quaternion = [
            np.cos(half_angle),
            np.sin(half_angle),
            0.0,
            0.0,
        ]
        np.testing.assert_allclose(
            projected_gravity(roll_quaternion),
            [0.0, -0.17364818, -0.98480775],
            atol=1e-6,
        )

    def test_projected_gravity_rejects_zero_quaternion(self):
        with self.assertRaisesRegex(ValueError, "norm"):
            projected_gravity([0.0, 0.0, 0.0, 0.0])

    def test_extract_policy_joint_state_uses_names_not_message_order(self):
        positions = np.arange(len(ISAAC_SIM_JOINT_NAMES), dtype=np.float32)
        velocities = -positions

        joint_position, joint_velocity = extract_policy_joint_state(
            ISAAC_SIM_JOINT_NAMES,
            positions,
            velocities,
        )

        expected_indices = np.array(
            [9, 13, 17, 21, 25, 27, 10, 14, 18, 22, 26, 28],
            dtype=np.float32,
        )
        np.testing.assert_array_equal(joint_position, expected_indices)
        np.testing.assert_array_equal(joint_velocity, -expected_indices)

    def test_observation_matches_documented_numerical_example(self):
        joint_offset = np.array(
            [
                0.05,
                -0.02,
                0.03,
                -0.10,
                0.04,
                -0.01,
                -0.04,
                0.01,
                -0.02,
                0.08,
                -0.03,
                0.02,
            ],
            dtype=np.float32,
        )
        previous_action = np.array(
            [
                0.12,
                -0.08,
                0.03,
                0.25,
                -0.16,
                0.04,
                -0.10,
                0.06,
                -0.02,
                0.20,
                -0.12,
                0.01,
            ],
            dtype=np.float32,
        )

        observation = build_observation(
            angular_velocity=[0.20, -0.10, 0.40],
            quaternion_wxyz=[0.9961947, 0.0871557, 0.0, 0.0],
            command=[0.40, -0.20, 0.60],
            joint_position=DEFAULT_JOINT_ANGLES + joint_offset,
            joint_velocity=[
                0.40,
                -0.20,
                0.10,
                -1.00,
                0.60,
                -0.30,
                -0.50,
                0.20,
                -0.10,
                0.80,
                -0.40,
                0.30,
            ],
            previous_action=previous_action,
            policy_step=10,
        )

        expected = np.array(
            [
                0.05,
                -0.025,
                0.1,
                0.0,
                -0.173648,
                -0.984808,
                0.8,
                -0.4,
                0.15,
                0.05,
                -0.02,
                0.03,
                -0.1,
                0.04,
                -0.01,
                -0.04,
                0.01,
                -0.02,
                0.08,
                -0.03,
                0.02,
                0.02,
                -0.01,
                0.005,
                -0.05,
                0.03,
                -0.015,
                -0.025,
                0.01,
                -0.005,
                0.04,
                -0.02,
                0.015,
                0.12,
                -0.08,
                0.03,
                0.25,
                -0.16,
                0.04,
                -0.10,
                0.06,
                -0.02,
                0.20,
                -0.12,
                0.01,
                1.0,
                0.0,
            ],
            dtype=np.float32,
        )

        self.assertEqual(observation.shape, (NUM_OBSERVATIONS,))
        self.assertEqual(observation.dtype, np.float32)
        np.testing.assert_allclose(observation, expected, atol=1e-6)

    def test_first_policy_step_uses_python_deploy_phase(self):
        observation = build_observation(
            angular_velocity=np.zeros(3),
            quaternion_wxyz=[1.0, 0.0, 0.0, 0.0],
            command=np.zeros(3),
            joint_position=DEFAULT_JOINT_ANGLES,
            joint_velocity=np.zeros(NUM_ACTIONS),
            previous_action=np.zeros(NUM_ACTIONS),
            policy_step=1,
        )

        phase = 0.02 / 0.8
        np.testing.assert_allclose(
            observation[45:47],
            [np.sin(2.0 * np.pi * phase), np.cos(2.0 * np.pi * phase)],
            atol=1e-7,
        )

    def test_scheduler_runs_policy_every_four_callbacks(self):
        scheduler = PolicyScheduler()
        inference_steps = []

        for sample_index in range(200):
            decision = scheduler.update(sample_index * 5_000_000)
            if decision.infer:
                inference_steps.append(sample_index)

        self.assertEqual(inference_steps, list(range(0, 200, 4)))
        self.assertEqual(scheduler.policy_step, 50)

    def test_scheduler_uses_callback_count_not_timestamp_spacing(self):
        scheduler = PolicyScheduler()
        sensor_stamps = [
            0,
            5_000_000,
            50_000_000,
            55_000_000,
            200_000_000,
            205_000_000,
        ]

        decisions = [scheduler.update(stamp) for stamp in sensor_stamps]

        self.assertEqual(
            [index for index, decision in enumerate(decisions) if decision.infer],
            [0, 4],
        )
        self.assertEqual(scheduler.policy_step, 2)

    def test_scheduler_resets_when_simulation_time_moves_backwards(self):
        scheduler = PolicyScheduler()
        self.assertEqual(scheduler.update(100_000_000).policy_step, 1)
        self.assertFalse(scheduler.update(105_000_000).infer)
        self.assertFalse(scheduler.update(110_000_000).infer)
        self.assertFalse(scheduler.update(115_000_000).infer)
        self.assertEqual(scheduler.update(120_000_000).policy_step, 2)

        decision = scheduler.update(0)
        self.assertTrue(decision.reset)
        self.assertTrue(decision.infer)
        self.assertEqual(decision.policy_step, 1)
        self.assertEqual(scheduler.policy_step, 1)
        self.assertEqual(scheduler.callback_count, 1)

    def test_full_joint_targets_control_legs_and_hold_others_at_zero(self):
        action = np.linspace(-0.5, 0.5, NUM_ACTIONS, dtype=np.float32)
        targets = build_full_joint_targets(ISAAC_SIM_JOINT_NAMES, action)

        self.assertEqual(targets.shape, (43,))
        name_to_index = {
            name: index for index, name in enumerate(ISAAC_SIM_JOINT_NAMES)
        }
        expected_leg_targets = DEFAULT_JOINT_ANGLES + ACTION_SCALE * action
        for policy_index, name in enumerate(POLICY_JOINT_NAMES):
            self.assertAlmostEqual(
                targets[name_to_index[name]],
                float(expected_leg_targets[policy_index]),
                places=7,
            )

        policy_name_set = set(POLICY_JOINT_NAMES)
        for index, name in enumerate(ISAAC_SIM_JOINT_NAMES):
            if name not in policy_name_set:
                self.assertEqual(targets[index], 0.0)

    def test_packaged_torchscript_policy_shape_and_memory_reset(self):
        policy_path = Path(__file__).parents[1] / "policy" / "motion.pt"
        policy = torch.jit.load(str(policy_path), map_location="cpu")
        self.assertTrue(hasattr(policy, "reset_memory"))

        observation = torch.zeros((1, NUM_OBSERVATIONS), dtype=torch.float32)
        policy.reset_memory()
        with torch.inference_mode():
            first_output = policy(observation).detach().clone()
            second_output = policy(observation).detach().clone()
        policy.reset_memory()
        with torch.inference_mode():
            reset_output = policy(observation).detach().clone()

        self.assertEqual(tuple(first_output.shape), (1, NUM_ACTIONS))
        self.assertTrue(bool(torch.isfinite(first_output).all()))
        self.assertFalse(torch.allclose(first_output, second_output))
        torch.testing.assert_close(first_output, reset_output)


if __name__ == "__main__":
    unittest.main()
