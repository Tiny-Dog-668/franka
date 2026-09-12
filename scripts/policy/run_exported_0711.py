#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from franka_sim2real.e2e_bundle import (
    load_bundle_config,
    run_bundle_deploy,
    validate_bundle_artifacts,
)
from franka_sim2real.hil import DEFAULT_HIL_SPEED_M_S, HILSettings
from franka_sim2real.residual_runtime import (
    DEFAULT_RESIDUAL_MAX_ABS,
    ResidualDeploySettings,
)

DEFAULT_CONFIG = REPO_ROOT / "configs" / "e2e_bundle_real_exported_0711.json"


def format_deploy_error(error: Exception) -> str:
    """将统一部署入口的异常转换为操作者可直接处理的中文提示。

    原始异常始终保留在输出末尾，避免翻译或归类错误掩盖底层诊断信息。
    """

    detail = str(error).strip() or repr(error)
    title = "部署未完成"
    meaning = "部署过程中发生未分类异常，程序已停止；若资源已启动，会按已有清理流程退出。"
    next_step = "请保存本段输出；使用 --debug 重跑一次以获取完整 traceback，再检查原始错误。"

    latch = re.search(
        r"server9 worker did not latch policy action (\d+).*?last latched (\d+)",
        detail,
    )
    if latch:
        expected, actual = latch.groups()
        title = "原生控制线程未及时接收动作"
        meaning = (
            f"Python 已提交第 {expected} 条策略动作，但 native server9 worker 最后锁存的仍是 "
            f"第 {actual} 条。在控制看门狗期限内没有完成动作切换，因此为安全起见停止。"
        )
        next_step = (
            "不要通过增大动作、放宽限幅或单纯调大 watchdog 来绕过。确认机械臂已恢复且环境安全后，"
            "先运行 `python scripts/policy/<入口>.py --streaming-check`；若仍失败，保留原始错误中的 "
            "worker 状态、控制周期和 IK tick 计数以检查 FCI/主机调度。"
        )
    elif "server9 control worker aborted" in detail:
        title = "原生 server9 控制 worker 已中止"
        meaning = "FCI 控制 worker 主动停止，后续策略动作不会再发送。"
        next_step = (
            "先查看原始错误中 worker 给出的具体原因，并按机器人控制柜状态完成必要的错误恢复。"
            "恢复后先执行 `--streaming-check`，不要直接重试多步轨迹。"
        )
    elif "server9 streaming worker did not become ready" in detail:
        title = "原生 server9 控制 worker 未就绪"
        meaning = "worker 在启动期限内没有进入可控制状态，尚未开始策略轨迹。"
        next_step = (
            "检查机器人网络、FCI 是否空闲、worker 是否已按当前源码重新构建；然后先执行 "
            "`./scripts/build_franka_server9_worker.sh` 和 `--streaming-check`。"
        )
    elif "Control watchdog exceeded" in detail or "Control tick stalled" in detail:
        title = "控制周期超时"
        meaning = "控制循环未在配置的安全时限内完成，程序已停止以避免继续发送过期指令。"
        next_step = (
            "检查主机实时性、CPU 负载和 FCI 网络；不要放宽控制 watchdog。恢复后先运行 "
            "`--streaming-check` 验证通信时序。"
        )
    elif "policy" in detail.lower() and (
        "deadline" in detail.lower() or "watchdog" in detail.lower()
    ):
        title = "策略推理未满足实时期限"
        meaning = "当前帧的策略推理或调度延迟超过了控制允许的时间窗口，动作没有被安全接受。"
        next_step = (
            "核对启动日志中的 inference device 和每步 policy 耗时；在不连接硬件的情况下先运行 "
            "`--validate-only`，并确认 CUDA 可用后再进行无运动预览。"
        )
    elif "Initial-state safety check failed" in detail:
        title = "初始状态安全门禁未通过"
        meaning = "机械臂关节、TCP、夹爪宽度或机器人模式与训练初始状态不一致；未发送 policy 运动命令。"
        next_step = (
            "确认工作空间、线缆和桌面安全后，按原始错误给出的 go_to_zero_pose.py 命令回到初始姿态；"
            "不要放宽初始状态容差或关闭门禁。"
        )
    elif "Robot reported errors" in detail or "server9 robot reported errors" in detail:
        title = "机器人报告了控制错误"
        meaning = "机器人状态中存在错误标志，部署已停止。"
        next_step = "查看控制柜和原始错误中的错误列表，完成机器人错误恢复后先进行 `--streaming-check`。"
    elif "unsafe mode" in detail.lower():
        title = "机器人进入非安全运行模式"
        meaning = "机器人不处于允许策略控制的 Move/Idle 模式，部署已停止。"
        next_step = "检查控制柜状态并完成错误恢复；不要在 Reflex、UserStopped 等模式下重试策略。"
    elif "CUDA was requested" in detail or "cuda" in detail.lower() and "available" in detail.lower():
        title = "CUDA 推理环境不可用"
        meaning = "配置要求使用 GPU，但当前 Python/PyTorch 环境无法使用所请求的 CUDA 设备。"
        next_step = (
            "运行 `.venv/bin/python -c \"import torch; print(torch.cuda.is_available())\"` 确认环境；"
            "在修复前可显式使用 `--device cpu` 做离线校验或预览。"
        )
    elif "rollout is capped at" in detail:
        title = "策略步数超过训练回合上限"
        meaning = "请求的真机策略步数超过导出模型的训练回合契约，部署在连接硬件前已停止。"
        next_step = "将 `--steps` 设为错误信息给出的上限或更小的正整数，再先运行 `--validate-only`。"
    elif isinstance(error, FileNotFoundError):
        title = "部署所需文件不存在"
        meaning = "配置、模型、metadata 或其引用的文件路径无法读取，部署尚未连接或控制机器人。"
        next_step = "核对命令中的 `--config` / `--model` / `--metadata` 路径，以及配置文件内的 checkpoint 路径。"
    elif isinstance(error, json.JSONDecodeError):
        title = "JSON 配置或 metadata 格式无效"
        meaning = "读取到的 JSON 不是有效格式，无法安全确认模型和部署契约。"
        next_step = "根据原始错误给出的行列位置修复 JSON 语法，再先运行 `--validate-only`。"
    elif "RealSense" in detail or "color frame" in detail:
        title = "RealSense 图像采集失败"
        meaning = "相机未能提供模型所需的彩色图像，策略未继续执行。"
        next_step = "检查相机 USB 连接、序列号与占用进程；先用相机预览脚本确认图像正常。"
    elif "GelSight" in detail:
        title = "GelSight 触觉图像采集失败"
        meaning = "触觉相机未能提供模型所需输入，策略未继续执行。"
        next_step = "检查左右设备号、USB 连接和 OpenCV 访问权限；确认两路触觉图像后再重试。"
    elif "workspace" in detail.lower():
        title = "目标超出安全工作空间"
        meaning = "计算出的末端目标不满足配置的 XYZ 工作空间约束，程序拒绝发送该动作。"
        next_step = "检查相机标定、初始姿态和 policy 输出；不要扩大 workspace 来绕过此保护。"
    elif "shared-memory ABI mismatch" in detail:
        title = "Python 与原生 worker 的共享内存版本不一致"
        meaning = "部署 Python 代码与已构建的 server9 worker 不使用同一 ABI，worker 不能安全启动。"
        next_step = "运行 `./scripts/build_franka_server9_worker.sh` 重新构建当前源码对应的 worker。"
    elif "Pygame is required for --hil" in detail:
        title = "HIL 键盘窗口依赖缺失"
        meaning = "已请求 HIL 模式，但当前 Python 环境没有可用的 Pygame；尚未开始策略控制。"
        next_step = "运行 `.venv/bin/python -m pip install pygame` 后重新启动 HIL。"
    elif isinstance(error, ValueError):
        title = "部署参数或模型契约不满足要求"
        meaning = "命令行参数、配置字段或模型输入输出契约不符合部署运行时要求，程序尚未继续执行。"
        next_step = "查看原始错误指出的字段；修复后先运行 `--validate-only`，不要通过删除校验来绕过。"

    return "\n".join((
        "=" * 72,
        f"部署失败：{title}",
        f"中文说明：{meaning}",
        f"建议处理：{next_step}",
        f"原始错误：{detail}",
        "调试模式：在原命令末尾加入 --debug 可保留完整 Python traceback。",
        "=" * 72,
    ))


def build_parser(default_config: str | Path = DEFAULT_CONFIG) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run an exported cube-grasp TorchScript policy on Franka."
    )
    parser.add_argument(
        "--config",
        default=str(default_config),
        help="Deployment config path.",
    )
    parser.add_argument(
        "--model",
        help="Override the TorchScript model path from the deployment config.",
    )
    parser.add_argument(
        "--metadata",
        help=(
            "Override the model metadata JSON path; when --model is supplied, "
            "the default is MODEL with a .json suffix."
        ),
    )
    parser.add_argument(
        "--device",
        help="Override the inference device from the config, for example cpu or cuda:0.",
    )
    parser.add_argument("--robot-ip", help="Override robot_ip from the config.")
    parser.add_argument("--steps", type=int, help="Override runner.steps from the config.")
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Run TorchScript policy inference on the first CUDA GPU (cuda:0).",
    )
    parser.add_argument("--image", help="Use a static RGB image instead of live RealSense input.")
    parser.add_argument(
        "--auto-gelsight",
        action="store_true",
        help=(
            "Discover exactly two primary GelSight V4L2 streams and use their "
            "current device order as left/right instead of configured device IDs."
        ),
    )
    parser.add_argument(
        "--exposure",
        type=float,
        help="Manual RealSense color exposure; disables auto exposure.",
    )
    parser.add_argument(
        "--gain",
        type=float,
        help="Manual RealSense color gain; disables auto exposure.",
    )
    parser.add_argument(
        "--auto-exposure",
        action="store_true",
        help="Enable RealSense color auto exposure and clear manual exposure/gain.",
    )
    parser.add_argument(
        "--preview-only",
        action="store_true",
        help="Run observation, image preprocessing, inference, and action mapping without motion.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help=(
            "Validate hashes, metadata contract, model loading, and dummy inference "
            "without hardware."
        ),
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Execute all requested real-robot steps without the first-step confirmation.",
    )
    parser.add_argument(
        "--confirm-each-step",
        action="store_true",
        help="Require confirmation before every real-robot policy step.",
    )
    parser.add_argument(
        "--control-mode",
        choices=("blocking", "streaming"),
        help="Override the Bundle control mode.",
    )
    parser.add_argument(
        "--allow-full-scale",
        action="store_true",
        help="Unlock streaming normalized actions from commissioning limit to +/-1.0.",
    )
    parser.add_argument(
        "--commissioning-limit",
        type=float,
        help=(
            "Temporarily lower the streaming normalized action limit for a hardware "
            "diagnostic run; it cannot raise the configured limit."
        ),
    )
    parser.add_argument(
        "--action-limit",
        type=float,
        help=(
            "Temporarily set the streaming normalized action limit in (0, 1]. "
            "Unlike --commissioning-limit, this may raise the configured limit; "
            "it cannot be combined with --yes."
        ),
    )
    parser.add_argument(
        "--streaming-check",
        action="store_true",
        help="Hold the current joints for two seconds and test 60 Hz communication only.",
    )
    parser.add_argument(
        "--hil",
        action="store_true",
        help=(
            "Enable focused-window Human-in-the-Loop XYZ takeover and Residual BC "
            "logging for the server9 streaming backend."
        ),
    )
    parser.add_argument(
        "--hil-speed-m-s",
        type=float,
        help=(
            "Cartesian speed while HIL direction keys are held; requires --hil "
            f"(default: {DEFAULT_HIL_SPEED_M_S:g} m/s)."
        ),
    )
    parser.add_argument(
        "--residual-model",
        type=Path,
        help="Load an independently trained XYZ Residual BC TorchScript head.",
    )
    parser.add_argument(
        "--residual-metadata",
        type=Path,
        help="Residual metadata JSON; defaults to metadata.json beside the model.",
    )
    parser.add_argument(
        "--residual-scale",
        type=float,
        help="Scale predicted XYZ residual before its safety cap (default: 1.0).",
    )
    parser.add_argument(
        "--residual-max-abs",
        type=float,
        help=(
            "Per-axis pre-limit normalized residual cap (default: "
            f"{DEFAULT_RESIDUAL_MAX_ABS:g})."
        ),
    )
    parser.add_argument(
        "--enable-residual-control",
        action="store_true",
        help=(
            "Explicitly authorize residual commands during real motion. Not required "
            "for --validate-only or --preview-only."
        ),
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Keep the full Python traceback instead of the operator-facing Chinese error hint.",
    )
    parser.add_argument(
        "--cube-position-root",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        help=(
            "Bypass RMA visual position prediction and use a fixed cube-center XYZ "
            "in robot_root metres for the whole session."
        ),
    )
    parser.add_argument(
        "--rma-position-source",
        choices=("vision", "oracle"),
        help="Select the RMA cube-position source.",
    )
    parser.add_argument(
        "--rma-contact-source",
        choices=("vision", "zero"),
        help="Select visual contact probabilities or constant [0,0] contact.",
    )
    return parser


def _apply_camera_control_overrides(config, args: argparse.Namespace) -> None:
    manual_controls = args.exposure is not None or args.gain is not None
    if args.auto_exposure and manual_controls:
        raise ValueError("--auto-exposure cannot be combined with --exposure or --gain")
    if args.image and (args.auto_exposure or manual_controls):
        raise ValueError("RealSense exposure controls cannot be combined with --image")
    if args.exposure is not None and not math.isfinite(args.exposure):
        raise ValueError("--exposure must be finite")
    if args.gain is not None and not math.isfinite(args.gain):
        raise ValueError("--gain must be finite")
    if manual_controls:
        config.camera.auto_exposure = False
        if args.exposure is not None:
            config.camera.exposure = args.exposure
        if args.gain is not None:
            config.camera.gain = args.gain
    elif args.auto_exposure:
        config.camera.auto_exposure = True
        config.camera.exposure = None
        config.camera.gain = None


def _apply_gelsight_device_override(config, args: argparse.Namespace) -> None:
    if not args.auto_gelsight:
        return
    if not config.tactile_camera.enabled:
        raise ValueError("--auto-gelsight requires tactile_camera.enabled=true")
    config.tactile_camera.auto_discover = True


def _build_hil_settings(config, args: argparse.Namespace) -> HILSettings | None:
    if args.hil_speed_m_s is not None and not args.hil:
        raise ValueError("--hil-speed-m-s requires --hil")
    settings = HILSettings(
        enabled=bool(args.hil),
        speed_m_s=(
            DEFAULT_HIL_SPEED_M_S
            if args.hil_speed_m_s is None
            else args.hil_speed_m_s
        ),
    )
    settings.validate()
    if not settings.enabled:
        return None
    if args.streaming_check:
        raise ValueError("--hil cannot be combined with --streaming-check")
    if config.control_mode != "streaming":
        raise ValueError("--hil is only valid with streaming control")
    if config.streaming.backend != "server9_joint_position":
        raise ValueError(
            "--hil requires streaming.backend='server9_joint_position', got "
            f"{config.streaming.backend!r}"
        )
    return settings


def _build_residual_settings(
    config,
    args: argparse.Namespace,
    hil_settings: HILSettings | None,
) -> ResidualDeploySettings | None:
    option_without_model = (
        args.residual_metadata is not None
        or args.residual_scale is not None
        or args.residual_max_abs is not None
        or args.enable_residual_control
    )
    if args.residual_model is None:
        if option_without_model:
            raise ValueError("Residual options require --residual-model")
        return None
    model_path = args.residual_model.expanduser().resolve()
    metadata_path = (
        args.residual_metadata.expanduser().resolve()
        if args.residual_metadata is not None
        else model_path.with_name("metadata.json")
    )
    settings = ResidualDeploySettings(
        model_path=model_path,
        metadata_path=metadata_path,
        scale=1.0 if args.residual_scale is None else args.residual_scale,
        max_abs=(
            DEFAULT_RESIDUAL_MAX_ABS
            if args.residual_max_abs is None
            else args.residual_max_abs
        ),
    )
    settings.validate()
    if config.control_mode != "streaming":
        raise ValueError("Residual BC is only valid with streaming control")
    if config.streaming.backend != "server9_joint_position":
        raise ValueError("Residual BC requires server9_joint_position streaming")
    if args.streaming_check:
        raise ValueError("--residual-model cannot be combined with --streaming-check")
    if hil_settings is not None:
        raise ValueError("--residual-model cannot be combined with --hil")
    if config.model.optimize_for_inference:
        raise ValueError("Residual BC requires model.optimize_for_inference=false")
    if (
        config.model.rma_position_source != "vision"
        or config.model.rma_contact_source != "vision"
    ):
        raise ValueError("Residual BC cannot be combined with RMA input overrides")
    if not args.validate_only and not args.preview_only:
        if not args.enable_residual_control:
            raise ValueError(
                "Real residual motion requires --enable-residual-control; run "
                "--preview-only first"
            )
        if args.yes:
            raise ValueError("Residual control cannot be combined with --yes")
    return settings


def main(default_config: str | Path = DEFAULT_CONFIG) -> int:
    args = build_parser(default_config).parse_args()
    try:
        return _run_deploy(args)
    except KeyboardInterrupt:
        print("\n部署已由操作者中断；不会继续发送新的策略动作。", file=sys.stderr)
        return 130
    except Exception as error:
        if args.debug:
            raise
        print(format_deploy_error(error), file=sys.stderr)
        return 2


def _run_deploy(args: argparse.Namespace) -> int:
    config = load_bundle_config(args.config)

    if args.gpu and args.device:
        raise ValueError("--gpu cannot be combined with --device")
    if args.model:
        model_path = Path(args.model).expanduser().resolve()
        config.model.model_path = str(model_path)
        if not args.metadata:
            config.model.metadata_path = str(model_path.with_suffix(".json"))
    if args.metadata:
        config.model.metadata_path = str(Path(args.metadata).expanduser().resolve())
    if args.device:
        config.model.device = args.device
    if args.gpu:
        config.model.device = "cuda:0"
    if args.robot_ip:
        config.robot_ip = args.robot_ip
    if args.steps is not None:
        if args.steps < 1:
            raise ValueError("--steps must be at least 1")
        config.runner.steps = args.steps
    if args.image:
        config.camera.source = "image"
        config.camera.image_path = args.image
    _apply_gelsight_device_override(config, args)
    _apply_camera_control_overrides(config, args)
    if args.control_mode:
        config.control_mode = args.control_mode
    if args.streaming_check:
        config.control_mode = "streaming"
    if args.rma_position_source:
        config.model.rma_position_source = args.rma_position_source
        if args.rma_position_source == "vision" and args.cube_position_root is None:
            config.model.rma_oracle_cube_position_root = None
    if args.cube_position_root is not None:
        config.model.rma_position_source = "oracle"
        config.model.rma_oracle_cube_position_root = list(args.cube_position_root)
    if args.rma_contact_source:
        config.model.rma_contact_source = args.rma_contact_source

    is_streaming = config.control_mode == "streaming"
    if args.action_limit is not None and args.commissioning_limit is not None:
        raise ValueError("--action-limit cannot be combined with --commissioning-limit")
    if args.action_limit is not None and args.allow_full_scale:
        raise ValueError("--action-limit cannot be combined with --allow-full-scale")
    if args.commissioning_limit is not None:
        if not is_streaming:
            raise ValueError("--commissioning-limit is only valid with streaming control")
        if not 0.0 < args.commissioning_limit <= config.streaming.commissioning_action_limit:
            raise ValueError(
                "--commissioning-limit must be positive and no greater than the "
                f"configured limit {config.streaming.commissioning_action_limit:g}"
            )
        config.streaming.commissioning_action_limit = args.commissioning_limit
    if args.action_limit is not None:
        if not is_streaming:
            raise ValueError("--action-limit is only valid with streaming control")
        if not 0.0 < args.action_limit <= 1.0:
            raise ValueError("--action-limit must be finite and in (0, 1]")
        config.streaming.commissioning_action_limit = args.action_limit
    if is_streaming and args.confirm_each_step:
        raise ValueError("Streaming control does not support --confirm-each-step")
    if args.allow_full_scale and not is_streaming:
        raise ValueError("--allow-full-scale is only valid with streaming control")
    if args.allow_full_scale and args.yes:
        raise ValueError("Full-scale streaming cannot be combined with --yes")
    if args.action_limit is not None and args.yes:
        raise ValueError("--action-limit cannot be combined with --yes")
    if args.streaming_check and args.preview_only:
        raise ValueError("--streaming-check cannot be combined with --preview-only")
    if args.streaming_check and args.validate_only:
        raise ValueError("--streaming-check cannot be combined with --validate-only")
    hil_settings = _build_hil_settings(config, args)
    residual_settings = _build_residual_settings(config, args, hil_settings)

    metadata = json.loads(Path(config.model.metadata_path).read_text(encoding="utf-8"))
    policy_name = metadata.get("task") or metadata.get("kind", "unknown")
    print(f"Policy task: {policy_name}")
    if metadata.get("deployment_variant") is not None:
        print(f"Deployment variant: {metadata['deployment_variant']}")
    print(f"Config: {args.config}")
    print(f"Model: {config.model.model_path}")
    print(f"Inference device: {config.model.device}")
    print(
        "TorchScript inference optimization: "
        f"{'enabled' if config.model.optimize_for_inference else 'disabled'}"
    )
    input_order = metadata.get("input_order", ["action_history", "proprio_obs", "wrist_rgb"])
    input_signature = metadata.get("input_signature", {})
    print(
        "Model inputs: "
        + ", ".join(
            (
                f"{name}[{','.join(str(value) for value in input_signature[name])}]"
                if name in input_signature
                else str(name)
            )
            for name in input_order
        )
    )
    output_signature = metadata.get("output_signature", {})
    if isinstance(output_signature, dict) and output_signature:
        print(
            "Model outputs: "
            + ", ".join(
                f"{name}[{','.join(str(value) for value in shape)}]"
                for name, shape in output_signature.items()
            )
        )
    else:
        print("Model outputs: [dx, dy, dz, gripper]")
    print(f"Control mode: {config.control_mode}")
    if config.tactile_camera.enabled:
        print(
            "GelSight devices: "
            + (
                "auto-discover exactly two primary streams at camera startup"
                if config.tactile_camera.auto_discover
                else (
                    f"configured left={config.tactile_camera.left_device!r}, "
                    f"right={config.tactile_camera.right_device!r}"
                )
            )
        )
    if hil_settings is not None:
        print(
            "HIL Residual BC capture: enabled "
            f"(keyboard XYZ speed={hil_settings.speed_m_s:g} m/s)"
        )
    if residual_settings is not None:
        print(
            "Residual BC: enabled "
            f"(model={residual_settings.model_path}, "
            f"scale={residual_settings.scale:g}, "
            f"per-axis cap=+/-{residual_settings.max_abs:g})"
        )
    print(
        "Camera color controls requested: "
        f"auto_exposure={config.camera.auto_exposure}, "
        f"exposure={config.camera.exposure}, gain={config.camera.gain}"
    )
    print(
        "History: "
        f"source={config.model.history_source}, "
        f"scale={config.model.history_scale}, "
        f"delay_steps={config.model.history_delay_steps}"
    )
    if metadata.get("kind") == "tacex_rma_xy_student_torchscript":
        print(
            "RMA XY contact input: "
            f"source={config.model.rma_contact_force_source} "
            "(gripper_is_grasped maps to [1,1] or [0,0])"
        )
    if (
        config.model.rma_position_source != "vision"
        or config.model.rma_contact_source != "vision"
    ):
        print(
            "RMA actor input override: "
            f"position_source={config.model.rma_position_source}, "
            f"cube_position_root={config.model.rma_oracle_cube_position_root}, "
            f"contact_source={config.model.rma_contact_source}"
        )
    if args.validate_only:
        report = validate_bundle_artifacts(config, residual_settings=residual_settings)
        print("Validation: PASS")
        print(json.dumps(report, indent=2))
        return 0
    if args.preview_only:
        print("Mode: preview only; arm, gripper, and automatic gripper homing are disabled.")
    else:
        print("Mode: REAL ROBOT MOTION")
        if is_streaming:
            action_limit = (
                1.0
                if args.allow_full_scale
                else config.streaming.commissioning_action_limit
            )
            print(
                "Streaming: "
                f"policy={config.streaming.policy_frequency_hz:g} Hz, "
                f"IK={config.streaming.ik_frequency_hz:g} Hz, "
                f"normalized action limit=+/-{action_limit:g}"
            )
            print(f"Streaming backend: {config.streaming.backend}")
            if config.streaming.impedance_mode == "cartesian":
                stiffness = config.streaming.cartesian_impedance
                assert stiffness is not None
                print(
                    "Impedance: Cartesian "
                    f"[Kx, Ky, Kz, Kroll, Kpitch, Kyaw]={stiffness} "
                    f"(Kz={stiffness[2]} N/m)"
                )
            else:
                print(
                    "Impedance: joint "
                    f"Kq={config.streaming.joint_impedance} Nm/rad"
                )
            collision_behavior = config.streaming.collision_behavior
            if collision_behavior is None:
                print(
                    "Collision behavior: unchanged; active controller thresholds "
                    "cannot be read back"
                )
            else:
                print(
                    "Collision behavior: explicitly configured before control "
                    "(Cartesian order: Fx, Fy, Fz, Mx, My, Mz)"
                )
                print(
                    "  joint contact/collision [Nm]: "
                    f"{collision_behavior.lower_torque_thresholds} / "
                    f"{collision_behavior.upper_torque_thresholds}"
                )
                print(
                    "  Cartesian contact/collision [N,N,N,Nm,Nm,Nm]: "
                    f"{collision_behavior.lower_force_thresholds} / "
                    f"{collision_behavior.upper_force_thresholds}"
                )

    confirmed_session = {"armed": bool(args.yes)}

    def confirm_first_step(step_index, robot_action, observation):
        if confirmed_session["armed"]:
            return True

        print(f"Step {step_index} proposed real-robot action:")
        print(
            json.dumps(
                {
                    "dx_m": robot_action.dx,
                    "dy_m": robot_action.dy,
                    "dz_m": robot_action.dz,
                    "gripper_width_m": robot_action.gripper_width,
                    "speed": robot_action.speed,
                    "current_tcp_m": observation.tcp_translation,
                    "current_gripper_width_m": observation.gripper_width,
                },
                indent=2,
            )
        )
        if is_streaming:
            prompt = "Type y/yes to start this streaming session: "
        else:
            prompt = (
                "Type y/yes to execute this step: "
                if args.confirm_each_step
                else "Type y/yes to execute this step and all remaining requested steps: "
            )
        confirm = input(prompt)
        confirmed = confirm.strip().lower() in {"y", "yes"}
        if confirmed and not args.confirm_each_step:
            confirmed_session["armed"] = True
        return confirmed

    summary = run_bundle_deploy(
        config,
        execute_motion=not args.preview_only,
        confirm_step_callback=(
            None if args.preview_only or args.yes else confirm_first_step
        ),
        allow_full_scale=args.allow_full_scale,
        streaming_check=args.streaming_check,
        save_step_data=hil_settings is not None or residual_settings is not None,
        hil_settings=hil_settings,
        residual_settings=residual_settings,
    )

    print(f"Run saved to: {summary['run_dir']}")
    print(f"Recorded steps: {summary.get('num_steps', 0)}")
    if summary.get("steps"):
        last_step = summary["steps"][-1]
        print(
            json.dumps(
                {
                    "raw_action": last_step.get("raw_action"),
                    "clipped_action": last_step.get("clipped_action"),
                    "robot_action": last_step.get("robot_action"),
                    "motion_executed": last_step.get("info", {}).get("motion_executed"),
                    "observation_after": last_step.get("observation_after"),
                },
                indent=2,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
