#!/usr/bin/env python3
"""模型离线导出：PyTorch 权重 → ONNX。

**本脚本只在联网的构建机上运行，绝不进入内网部署链路。**

为什么要单独写导出脚本而不是直接用第三方现成实现：
1. 校园监控的画质诉求与通用超分基准不同 —— 我们更关心压缩块效应、
   夜间高增益噪点和运动拖影，因此需要固化输入输出规格以便后续
   做通道裁剪与 TensorRT 引擎预编译；
2. 导出时必须固定 opset、关闭动态 shape（或用最小动态范围），
   否则内网首次推理会触发 GEMM 自动调优，延迟抖动可达数百毫秒；
3. 导出的 ONNX 会被冻结成静态图并做常量折叠，推理时更省显存。

用法示例：
    python tools/export_onnx.py --stage sr --ckpt weights/RealESRGAN_x2.pth \
        --out models/realesr_general_x2v3.onnx --scale 2

注意：不同论文实现的网络结构必须与本脚本中的封装对应；
若换用其他权重，请相应调整 ``_build_model`` 中的模型构造。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from campusclear.models.registry import sha256_of  # noqa: E402


def _build_model(stage: str, scale: int):
    """构建对应阶段的网络。

    这里给出的是**接口骨架**：内网项目不应把上游训练代码一并打包进来，
    而是由算法同学在此处挂接具体实现（import 自己的网络定义模块）。
    """
    try:
        import torch  # noqa: F401
    except ImportError:
        raise SystemExit("导出需要 PyTorch，请在构建机上安装：pip install torch onnx")

    raise SystemExit(
        f"请在 _build_model() 中挂接 {stage} 阶段的网络实现后再运行本脚本。"
        "该函数需要返回一个 torch.nn.Module 实例。"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="CampusClear 模型导出工具")
    parser.add_argument("--stage", required=True, choices=["lowlight", "denoise", "deblur", "sr"])
    parser.add_argument("--ckpt", required=True, help="PyTorch 权重路径")
    parser.add_argument("--out", required=True, help="ONNX 输出路径")
    parser.add_argument("--scale", type=int, default=1, help="放大倍率，非超分阶段填 1")
    parser.add_argument("--input-size", default="1080x1920", help="导出用的示例输入尺寸 HxW")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--fp16", action="store_true", help="同时产出 FP16 权重文件")
    args = parser.parse_args()

    import torch

    height, width = (int(v) for v in args.input_size.lower().split("x"))
    model = _build_model(args.stage, args.scale)

    checkpoint = torch.load(args.ckpt, map_location="cpu")
    state_dict = checkpoint.get("params_ema", checkpoint.get("params", checkpoint))
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    dummy = torch.randn(1, 3, height, width)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        dummy,
        str(out_path),
        input_names=["input"],
        output_names=["output"],
        opset_version=args.opset,
        do_constant_folding=True,
        dynamic_axes=None,  # 固定 shape：避免内网首次推理的调优抖动
    )

    if args.fp16:
        try:
            import onnx
            from onnxconverter_common import float16

            onnx_model = onnx.load(str(out_path))
            fp16_model = float16.convert_float_to_float16(onnx_model, keep_io_types=True)
            fp16_path = out_path.with_name(out_path.stem + "_fp16.onnx")
            onnx.save(fp16_model, str(fp16_path))
            print(f"FP16 权重已产出: {fp16_path}")
        except ImportError:
            print("未安装 onnxconverter-common，跳过 FP16 转换")

    digest = sha256_of(out_path)
    print(f"导出完成: {out_path}")
    print(f"sha256: {digest}")
    print("请把该哈希写入 src/campusclear/models/models_manifest.json 对应阶段的 sha256 字段。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
