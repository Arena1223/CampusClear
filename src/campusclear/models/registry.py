"""离线模型注册表。

职责：
1. 读取 ``models_manifest.json``，把「增强阶段 → 权重文件」的映射固化下来；
2. 加载前逐文件校验 SHA256，防止内网拷贝/分发过程中权重被替换或损坏；
3. 在真正需要时才创建 ONNX Runtime 会话（惰性加载），缩短服务启动时间；
4. 显式记录每个模型的计算量、输入规格与离线来源，便于申报/审计追溯。

安全说明：本模块**不会**、也没有能力从网络获取任何权重。
权重必须由运维人员预先放入 ``CAMPUSCLEAR_MODELS_DIR``，与 manifest 一一对应。
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

MANIFEST_NAME = "models_manifest.json"

# 四个增强阶段。顺序即默认执行顺序（低光 → 降噪 → 去模糊 → 超分）。
STAGES = ("lowlight", "denoise", "deblur", "sr")


@dataclass
class ModelSpec:
    """单个权重文件的描述。"""

    stage: str
    file: str
    sha256: str = ""
    input_size: tuple[int, int] | None = None  # (H, W)，None 表示动态尺寸
    scale: int = 1
    fp16: bool = True
    description: str = ""
    source: str = ""

    @property
    def relative_file(self) -> str:
        return self.file


@dataclass
class LoadedModel:
    """已就绪的推理会话封装。"""

    stage: str
    session: object
    input_name: str
    output_name: str
    spec: ModelSpec
    provider: str
    input_shape: tuple = field(default_factory=tuple)


class ModelIntegrityError(RuntimeError):
    """权重缺失或哈希不匹配。此类错误必须让服务显式失败，不能静默降级。"""


def sha256_of(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


class ModelRegistry:
    """模型清单 + 惰性会话缓存。"""

    def __init__(self, models_dir: Path, *, verify_hash: bool = True) -> None:
        self.models_dir = Path(models_dir)
        self.verify_hash = verify_hash
        self.specs: dict[str, ModelSpec] = {}
        self._sessions: dict[str, LoadedModel] = {}
        self._manifest: dict = {}
        # 会话创建要读几十 MB 权重、校验 SHA256，TensorRT 还会编译 engine，
        # 可能持续几十秒。多路首帧并发时若 check-then-act 无锁，同一阶段会被
        # 建出多个会话（显存翻倍甚至 OOM），且后建的会覆盖先建的。
        self._load_lock = threading.Lock()
        self.load_manifest()

    # ------------------------------------------------------------------ #
    # 清单
    # ------------------------------------------------------------------ #
    def load_manifest(self) -> None:
        manifest_path = self.models_dir / MANIFEST_NAME
        if not manifest_path.exists():
            logger.warning("未找到模型清单 %s，所有增强阶段将退化为 CPU 经典算法", manifest_path)
            return
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
        self._manifest = raw
        for stage, item in raw.get("models", {}).items():
            if stage not in STAGES:
                logger.warning("清单中出现未知阶段 %r，已忽略", stage)
                continue
            input_size = item.get("input_size")
            self.specs[stage] = ModelSpec(
                stage=stage,
                file=item["file"],
                sha256=item.get("sha256", ""),
                input_size=tuple(input_size) if input_size else None,
                scale=int(item.get("scale", 1)),
                fp16=bool(item.get("fp16", True)),
                description=item.get("description", ""),
                source=item.get("source", ""),
            )

    def available_stages(self) -> list[str]:
        """真正**可用**的阶段：清单里有、权重文件在、onnxruntime 也装上了。

        只检查清单会把「有清单没权重」误判为可走 ONNX，随后每帧每阶段都
        尝试 get() → 权重缺失 → 回退经典算法，日志被刷爆且推理链路名不副实。
        """
        if importlib.util.find_spec("onnxruntime") is None:
            return []
        return [
            stage
            for stage in STAGES
            if stage in self.specs and (self.models_dir / self.specs[stage].file).is_file()
        ]

    # ------------------------------------------------------------------ #
    # 校验
    # ------------------------------------------------------------------ #
    def verify(self) -> dict[str, str]:
        """校验所有清单内权重。返回 {stage: 'ok' | 错误描述}。"""
        report: dict[str, str] = {}
        for stage, spec in self.specs.items():
            path = self.models_dir / spec.file
            if not path.exists():
                report[stage] = f"缺失: {spec.file}"
                continue
            if not self.verify_hash or not spec.sha256:
                report[stage] = "ok(未校验哈希)"
                continue
            actual = sha256_of(path)
            report[stage] = "ok" if actual == spec.sha256 else f"哈希不匹配: 期望 {spec.sha256[:12]}… 实际 {actual[:12]}…"
        return report

    def verify_or_raise(self) -> dict[str, str]:
        report = self.verify()
        broken = {k: v for k, v in report.items() if not v.startswith("ok")}
        if broken:
            detail = "; ".join(f"{k}->{v}" for k, v in broken.items())
            raise ModelIntegrityError(f"模型完整性校验失败: {detail}")
        return report

    # ------------------------------------------------------------------ #
    # 会话
    # ------------------------------------------------------------------ #
    def get(self, stage: str) -> LoadedModel | None:
        """取得（必要时创建）某阶段的推理会话。权重不可用时返回 None。

        整个「检查—创建—登记」过程在同一把锁内完成。不同阶段也会串行加载，
        这只发生在首次命中时；换来的确定性是：多路同时首帧不会创建重复会话。
        """
        with self._load_lock:
            if stage in self._sessions:
                return self._sessions[stage]
            spec = self.specs.get(stage)
            if spec is None:
                return None
            path = self.models_dir / spec.file
            if not path.exists():
                logger.warning("阶段 %s 权重缺失（%s），跳过该阶段", stage, path)
                return None
            if self.verify_hash and spec.sha256:
                if sha256_of(path) != spec.sha256:
                    # 这里刻意抛异常：内网安全项目里，"权重被换过" 属于必须人工介入的严重事件
                    raise ModelIntegrityError(f"阶段 {stage} 权重哈希校验失败: {spec.file}")

            loaded = self._load_session(spec, path)
            if loaded is not None:
                self._sessions[stage] = loaded
            return loaded

    def _load_session(self, spec: ModelSpec, path: Path) -> LoadedModel | None:
        """真正创建 ORT 会话（调用方必须持有 ``_load_lock``）。"""
        stage = spec.stage
        try:
            import onnxruntime as ort
        except ImportError:  # pragma: no cover - 取决于运行环境
            logger.warning("未安装 onnxruntime，阶段 %s 退化为经典算法", stage)
            return None

        providers = self._resolve_providers(ort)
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.intra_op_num_threads = 2  # 多路并发时避免线程争抢
        options.log_severity_level = 3

        session = ort.InferenceSession(str(path), sess_options=options, providers=providers)
        inputs = session.get_inputs()
        outputs = session.get_outputs()
        loaded = LoadedModel(
            stage=stage,
            session=session,
            input_name=inputs[0].name,
            output_name=outputs[0].name,
            spec=spec,
            provider=session.get_providers()[0] if session.get_providers() else "unknown",
            input_shape=tuple(inputs[0].shape),
        )
        logger.info(
            "阶段 %s 会话就绪: provider=%s input=%s",
            stage,
            loaded.provider,
            loaded.input_shape,
        )
        return loaded

    def _resolve_providers(self, ort) -> list[str]:
        available = set(ort.get_available_providers())
        chosen = [p for p in _PROVIDER_PRIORITY if p in available]
        return chosen or ["CPUExecutionProvider"]

    def describe(self) -> list[dict]:
        """给管理接口用的模型状态列表。"""
        report = self.verify()
        rows = []
        for stage in STAGES:
            spec = self.specs.get(stage)
            if spec is None:
                rows.append({"stage": stage, "state": "未配置", "provider": None})
                continue
            with self._load_lock:
                loaded = self._sessions.get(stage)
            rows.append(
                {
                    "stage": stage,
                    "file": spec.file,
                    "integrity": report.get(stage, "unknown"),
                    "provider": loaded.provider if loaded else None,
                    "scale": spec.scale,
                    "input_size": list(spec.input_size) if spec.input_size else None,
                    "description": spec.description,
                    "source": spec.source,
                }
            )
        return rows

    def close(self) -> None:
        with self._load_lock:
            self._sessions.clear()


_PROVIDER_PRIORITY = [
    "TensorrtExecutionProvider",
    "CUDAExecutionProvider",
    "CPUExecutionProvider",
]
