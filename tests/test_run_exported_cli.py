from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "policy" / "run_exported_0711.py"
SPEC = importlib.util.spec_from_file_location("run_exported_0711_cli", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
run_exported = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = run_exported
SPEC.loader.exec_module(run_exported)


class RunExportedCLIColorControlTests(unittest.TestCase):
    def test_manual_color_control_arguments_are_parsed(self) -> None:
        args = run_exported.build_parser().parse_args(
            ["--exposure", "100", "--gain", "64"]
        )

        self.assertEqual(args.exposure, 100.0)
        self.assertEqual(args.gain, 64.0)
        self.assertFalse(args.auto_exposure)

    def test_auto_exposure_argument_is_parsed(self) -> None:
        args = run_exported.build_parser().parse_args(["--auto-exposure"])

        self.assertTrue(args.auto_exposure)
        self.assertIsNone(args.exposure)
        self.assertIsNone(args.gain)

    def test_manual_controls_disable_auto_exposure_in_config(self) -> None:
        args = run_exported.build_parser().parse_args(
            ["--exposure", "100", "--gain", "64"]
        )
        config = SimpleNamespace(
            camera=SimpleNamespace(auto_exposure=True, exposure=None, gain=None)
        )

        run_exported._apply_camera_control_overrides(config, args)

        self.assertFalse(config.camera.auto_exposure)
        self.assertEqual(config.camera.exposure, 100.0)
        self.assertEqual(config.camera.gain, 64.0)

    def test_auto_exposure_clears_configured_manual_values(self) -> None:
        args = run_exported.build_parser().parse_args(["--auto-exposure"])
        config = SimpleNamespace(
            camera=SimpleNamespace(auto_exposure=False, exposure=100.0, gain=64.0)
        )

        run_exported._apply_camera_control_overrides(config, args)

        self.assertTrue(config.camera.auto_exposure)
        self.assertIsNone(config.camera.exposure)
        self.assertIsNone(config.camera.gain)

    def test_auto_and_manual_controls_are_rejected(self) -> None:
        args = run_exported.build_parser().parse_args(
            ["--auto-exposure", "--exposure", "100"]
        )
        config = SimpleNamespace(camera=SimpleNamespace())

        with self.assertRaisesRegex(ValueError, "cannot be combined"):
            run_exported._apply_camera_control_overrides(config, args)


class RunExportedCLIHILTests(unittest.TestCase):
    @staticmethod
    def _config(*, control_mode="streaming", backend="server9_joint_position"):
        return SimpleNamespace(
            control_mode=control_mode,
            streaming=SimpleNamespace(backend=backend),
        )

    def test_hil_arguments_are_parsed_with_runtime_default(self) -> None:
        args = run_exported.build_parser().parse_args(["--hil"])

        settings = run_exported._build_hil_settings(self._config(), args)

        self.assertTrue(args.hil)
        self.assertIsNone(args.hil_speed_m_s)
        self.assertIsNotNone(settings)
        self.assertEqual(settings.speed_m_s, 0.05)

    def test_hil_custom_speed_is_parsed(self) -> None:
        args = run_exported.build_parser().parse_args(
            ["--hil", "--hil-speed-m-s", "0.025"]
        )

        settings = run_exported._build_hil_settings(self._config(), args)

        self.assertIsNotNone(settings)
        self.assertEqual(settings.speed_m_s, 0.025)

    def test_hil_speed_without_hil_is_rejected(self) -> None:
        args = run_exported.build_parser().parse_args(["--hil-speed-m-s", "0.05"])

        with self.assertRaisesRegex(ValueError, "requires --hil"):
            run_exported._build_hil_settings(self._config(), args)

    def test_hil_rejects_nonpositive_nonfinite_and_wrong_backend(self) -> None:
        for speed in ("0", "-0.1", "nan", "inf"):
            with self.subTest(speed=speed):
                args = run_exported.build_parser().parse_args(
                    ["--hil", "--hil-speed-m-s", speed]
                )
                with self.assertRaisesRegex(ValueError, "finite and positive"):
                    run_exported._build_hil_settings(self._config(), args)

        args = run_exported.build_parser().parse_args(["--hil"])
        with self.assertRaisesRegex(ValueError, "server9_joint_position"):
            run_exported._build_hil_settings(
                self._config(backend="async_position"), args
            )
        with self.assertRaisesRegex(ValueError, "streaming control"):
            run_exported._build_hil_settings(
                self._config(control_mode="blocking"), args
            )

    def test_hil_streaming_check_is_rejected_but_preview_and_validate_are_allowed(self) -> None:
        args = run_exported.build_parser().parse_args(["--hil", "--streaming-check"])
        with self.assertRaisesRegex(ValueError, "cannot be combined"):
            run_exported._build_hil_settings(self._config(), args)

        for mode in ("--preview-only", "--validate-only"):
            with self.subTest(mode=mode):
                args = run_exported.build_parser().parse_args(["--hil", mode])
                self.assertIsNotNone(
                    run_exported._build_hil_settings(self._config(), args)
                )


class RunExportedCLIResidualTests(unittest.TestCase):
    @staticmethod
    def _config():
        return SimpleNamespace(
            control_mode="streaming",
            streaming=SimpleNamespace(backend="server9_joint_position"),
            model=SimpleNamespace(
                optimize_for_inference=False,
                rma_position_source="vision",
                rma_contact_source="vision",
            ),
        )

    def test_preview_builds_conservative_runtime_settings(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = root / "residual.ts"
            metadata = root / "metadata.json"
            model.touch()
            metadata.touch()
            args = run_exported.build_parser().parse_args(
                ["--residual-model", str(model), "--preview-only"]
            )
            settings = run_exported._build_residual_settings(
                self._config(), args, None
            )
            self.assertIsNotNone(settings)
            self.assertEqual(settings.scale, 1.0)
            self.assertEqual(settings.max_abs, 0.1)

    def test_real_motion_requires_explicit_enable_and_rejects_yes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = root / "residual.ts"
            (root / "metadata.json").touch()
            model.touch()
            args = run_exported.build_parser().parse_args(
                ["--residual-model", str(model)]
            )
            with self.assertRaisesRegex(ValueError, "enable-residual-control"):
                run_exported._build_residual_settings(self._config(), args, None)

            args = run_exported.build_parser().parse_args(
                [
                    "--residual-model",
                    str(model),
                    "--enable-residual-control",
                    "--yes",
                ]
            )
            with self.assertRaisesRegex(ValueError, "cannot be combined with --yes"):
                run_exported._build_residual_settings(self._config(), args, None)

    def test_residual_rejects_hil_and_options_without_model(self) -> None:
        args = run_exported.build_parser().parse_args(["--residual-max-abs", "0.05"])
        with self.assertRaisesRegex(ValueError, "require --residual-model"):
            run_exported._build_residual_settings(self._config(), args, None)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = root / "residual.ts"
            (root / "metadata.json").touch()
            model.touch()
            args = run_exported.build_parser().parse_args(
                ["--residual-model", str(model), "--preview-only"]
            )
            with self.assertRaisesRegex(ValueError, "cannot be combined with --hil"):
                run_exported._build_residual_settings(
                    self._config(), args, SimpleNamespace(enabled=True)
                )


class RunExportedCLIGelSightTests(unittest.TestCase):
    def test_auto_gelsight_enables_runtime_discovery(self) -> None:
        args = run_exported.build_parser().parse_args(["--auto-gelsight"])
        config = SimpleNamespace(
            tactile_camera=SimpleNamespace(enabled=True, auto_discover=False)
        )

        run_exported._apply_gelsight_device_override(config, args)

        self.assertTrue(config.tactile_camera.auto_discover)

    def test_auto_gelsight_rejects_policy_without_tactile_camera(self) -> None:
        args = run_exported.build_parser().parse_args(["--auto-gelsight"])
        config = SimpleNamespace(
            tactile_camera=SimpleNamespace(enabled=False, auto_discover=False)
        )

        with self.assertRaisesRegex(ValueError, "tactile_camera.enabled=true"):
            run_exported._apply_gelsight_device_override(config, args)


class RunExportedCLIOperatorErrorTests(unittest.TestCase):
    def test_missing_pygame_has_install_hint(self) -> None:
        message = run_exported.format_deploy_error(RuntimeError(
            "HIL keyboard failed: Pygame is required for --hil"
        ))

        self.assertIn("HIL 键盘窗口依赖缺失", message)
        self.assertIn(".venv/bin/python -m pip install pygame", message)

    def test_latch_timeout_has_chinese_meaning_and_original_detail(self) -> None:
        message = run_exported.format_deploy_error(RuntimeError(
            "server9 worker did not latch policy action 238 within 50.0 ms "
            "(last latched 237; worker_status=running)"
        ))

        self.assertIn("原生控制线程未及时接收动作", message)
        self.assertIn("第 238 条策略动作", message)
        self.assertIn("第 237 条", message)
        self.assertIn("原始错误：server9 worker did not latch policy action 238", message)

    def test_initial_state_failure_preserves_safety_guidance(self) -> None:
        message = run_exported.format_deploy_error(RuntimeError(
            "Initial-state safety check failed; no policy motion command was sent."
        ))

        self.assertIn("初始状态安全门禁未通过", message)
        self.assertIn("不要放宽初始状态容差或关闭门禁", message)

    def test_missing_file_has_chinese_path_hint(self) -> None:
        message = run_exported.format_deploy_error(FileNotFoundError(
            2, "No such file or directory", "/tmp/missing.json"
        ))

        self.assertIn("部署所需文件不存在", message)
        self.assertIn("--config", message)

    def test_debug_flag_is_parsed(self) -> None:
        self.assertTrue(run_exported.build_parser().parse_args(["--debug"]).debug)

    def test_main_prints_chinese_hint_without_traceback_by_default(self) -> None:
        output = StringIO()
        with (
            patch.object(sys, "argv", ["run_exported_0711.py"]),
            patch.object(
                run_exported,
                "_run_deploy",
                side_effect=RuntimeError("server9 control worker aborted: FCI timeout"),
            ),
            redirect_stderr(output),
        ):
            exit_code = run_exported.main()

        self.assertEqual(exit_code, 2)
        self.assertIn("原生 server9 控制 worker 已中止", output.getvalue())
        self.assertNotIn("Traceback", output.getvalue())


if __name__ == "__main__":
    unittest.main()
