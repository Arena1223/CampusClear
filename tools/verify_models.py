#!/usr/bin/env python3
"""模型权重完整性校验与清单维护。

两个用途：
1. 现场校验：``python tools/verify_models.py --models-dir ./models``
   逐个核对权重 SHA256，输出可粘贴进验收报告的结果表；
2. 打包时写清单：``--write-manifest`` 把实际哈希回填到 models_manifest.json。

这是「模型可信」这一安全诉求的落地手段：内网分发靠 U 盘或文件共享，
中间环节无法用 HTTPS 保护，哈希比对是唯一能证明权重未被替换的方法。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from campusclear.models.registry import MANIFEST_NAME, STAGES, sha256_of  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="CampusClear 模型校验工具")
    parser.add_argument("--models-dir", default="./models", help="权重所在目录")
    parser.add_argument("--manifest", default=None, help="清单路径，默认取 models-dir 下的模型清单")
    parser.add_argument("--write-manifest", action="store_true", help="把实际哈希回填到清单")
    parser.add_argument("--json", action="store_true", help="以 JSON 输出，便于程序消费")
    args = parser.parse_args()

    models_dir = Path(args.models_dir).resolve()
    manifest_path = Path(args.manifest).resolve() if args.manifest else models_dir / MANIFEST_NAME

    if not manifest_path.exists():
        print(f"[错误] 未找到清单文件 {manifest_path}", file=sys.stderr)
        return 2
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    models = manifest.get("models", {})

    rows: list[dict] = []
    all_ok = True
    updated = False

    for stage in STAGES:
        item = models.get(stage)
        if item is None:
            rows.append({"stage": stage, "file": "-", "state": "清单中未配置", "ok": False})
            all_ok = False
            continue
        path = models_dir / item["file"]
        if not path.exists():
            rows.append({"stage": stage, "file": item["file"], "state": "文件缺失", "ok": False})
            all_ok = False
            continue

        actual = sha256_of(path)
        expected = item.get("sha256", "")

        if args.write_manifest:
            if expected != actual:
                item["sha256"] = actual
                updated = True
            state = "已写入清单"
            ok = True
        elif not expected:
            state = "清单无哈希（视为未校验）"
            ok = False
            all_ok = False
        elif expected == actual:
            state = "一致"
            ok = True
        else:
            state = f"不一致（期望 {expected[:12]}… 实际 {actual[:12]}…）"
            ok = False
            all_ok = False

        size_mb = path.stat().st_size / (1024 * 1024)
        rows.append(
            {
                "stage": stage,
                "file": item["file"],
                "size_mb": round(size_mb, 2),
                "sha256": actual,
                "state": state,
                "ok": ok,
            }
        )

    if args.write_manifest and updated:
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    if args.json:
        print(json.dumps({"models_dir": str(models_dir), "results": rows}, ensure_ascii=False, indent=2))
    else:
        print(f"模型目录: {models_dir}")
        print(f"清单文件: {manifest_path}")
        print("-" * 78)
        print(f"{'阶段':<10}{'文件':<34}{'大小MB':>8}  状态")
        print("-" * 78)
        for row in rows:
            size = f"{row.get('size_mb', '-'):>8}" if "size_mb" in row else f"{'-':>8}"
            print(f"{row['stage']:<10}{row['file']:<34}{size}  {row['state']}")
        print("-" * 78)
        print("结论:", "全部校验通过" if all_ok else "存在未通过项，服务将拒绝加载对应阶段")

    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
