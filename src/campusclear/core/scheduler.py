"""多路调度监督器。

职责边界：
- 按数据库中的通道配置创建/销毁 ``StreamPipeline``；
- 周期性采集 GPU 利用率与队列压力，驱动全局负载档位；
- 通道级别的健康兜底（进程活着但画面冻结、回推断开等）。

刻意不做的两件事：
1. 不做跨通道的帧搬运（零拷贝只发生在单路内部）。跨路复用会引入复杂的
   锁竞争，收益不如直接做 ROI；
2. 不在监督器里做任何推理。监督器必须保持轻量，即使 GPU 卡死也能正常
   响应管理接口，否则运维会失去唯一的观测窗口。
"""

from __future__ import annotations

import logging
import threading
import time

from ..models.registry import ModelRegistry
from .camera import CameraConfig
from .enhancer import ClassicalEnhancer, OnnxEnhancer
from .metrics import read_gpu_util
from .pipeline import StreamPipeline
from .planner import LoadGovernor
from .store import CameraStore

logger = logging.getLogger(__name__)


def build_enhancer(settings, registry: ModelRegistry):
    """按可用性选择推理实现：有权重且装得上 onnxruntime 走 GPU，否则经典兜底。"""
    enhancer = OnnxEnhancer(
        registry,
        use_fp16=settings.use_fp16,
        tile_max_pixels=settings.tile_max_pixels,
        tile_overlap=settings.tile_overlap,
        device_id=settings.device_id,
    )
    if enhancer.available():
        logger.info("推理链路: ONNX（阶段 %s）", ",".join(registry.available_stages()))
        return enhancer
    logger.warning("未检测到可用离线权重，推理链路退化为经典 CPU 算法")
    return ClassicalEnhancer()


class StreamSupervisor:
    """所有视频通道的总控。"""

    def __init__(self, settings, store: CameraStore, registry: ModelRegistry | None = None) -> None:
        self.settings = settings
        self.store = store
        self.registry = registry or ModelRegistry(
            settings.models_dir, verify_hash=settings.verify_model_hash
        )
        self.enhancer = build_enhancer(settings, self.registry)
        self.governor = LoadGovernor(
            high_watermark=settings.gpu_high_watermark,
            low_watermark=settings.gpu_low_watermark,
        )

        self._pipelines: dict[str, StreamPipeline] = {}
        # ``_lock`` 只保护字典本身；probe / start / stop 这类慢操作一律在锁外。
        # ``_control_lock`` 串行化「改通道集合」的控制操作（reload/启停/重启），
        # 但不会阻塞 /system/status、/cameras 这类只读快照。
        self._lock = threading.RLock()
        self._control_lock = threading.RLock()
        self._stop_event = threading.Event()
        self._shutting_down = threading.Event()
        self._monitor: threading.Thread | None = None
        self._reload_thread: threading.Thread | None = None
        self.last_gpu: dict | None = None
        self.started_at = time.time()

    # ------------------------------------------------------------------ #
    # 生命周期
    # ------------------------------------------------------------------ #
    def start(self) -> None:
        self._shutting_down.clear()
        self._stop_event.clear()
        # 初始建流放到后台：几十路 probe+connect 可能要几分钟，不能把
        # uvicorn 的 lifespan 卡死，否则管理面连「正在启动」都看不到。
        self._reload_thread = threading.Thread(
            target=self._initial_reload, name="cc-reload", daemon=True
        )
        self._reload_thread.start()
        self._monitor = threading.Thread(target=self._monitor_loop, name="cc-monitor", daemon=True)
        self._monitor.start()

    def _initial_reload(self) -> None:
        try:
            self.reload()
        except Exception:  # noqa: BLE001 - 初始建流失败不能拖垮服务
            logger.exception("初始通道加载失败")

    def shutdown(self) -> None:
        self._shutting_down.set()
        self._stop_event.set()
        if self._monitor is not None:
            self._monitor.join(timeout=5)
        if self._reload_thread is not None:
            # 它每启动一路都会检查 _shutting_down，正常情况下很快退出
            self._reload_thread.join(timeout=20)
        with self._control_lock:
            with self._lock:
                pipelines = list(self._pipelines.values())
                self._pipelines.clear()
            for pipeline in pipelines:
                pipeline.stop()
        logger.info("调度器已关闭")

    # ------------------------------------------------------------------ #
    # 通道管理
    # ------------------------------------------------------------------ #
    def reload(self) -> dict:
        """按数据库配置**增量**对齐通道集合。

        只有新增、删除、停用或配置真正变化的通道会被重建；未变化的通道保持
        运行，避免一次 ``/system/reload`` 把几十路全部中断。

        probe/start/stop 都在 ``_lock`` 之外执行，``_lock`` 只保护字典本身，
        因此重载期间 ``/system/status`` 仍然能立即返回。
        """
        with self._control_lock:
            cameras = self.store.list_cameras()
            desired = {cam.id: cam for cam in cameras if cam.enabled}

            to_stop: list[StreamPipeline] = []
            with self._lock:
                for camera_id, pipeline in list(self._pipelines.items()):
                    camera = desired.get(camera_id)
                    if camera is None or pipeline.camera != camera:
                        to_stop.append(self._pipelines.pop(camera_id))
                unchanged = set(self._pipelines)

            for pipeline in to_stop:
                pipeline.stop()
                logger.info("通道 %s 已停止（删除/停用/配置变更）", pipeline.camera.id)

            for camera_id in sorted(desired):
                if self._shutting_down.is_set():
                    break
                if camera_id in unchanged:
                    continue
                camera = desired[camera_id]
                with self._lock:
                    at_limit = len(self._pipelines) >= self.settings.max_streams
                if at_limit:
                    logger.warning(
                        "已达最大并发通道数 %d，跳过 %s", self.settings.max_streams, camera_id
                    )
                    continue
                pipeline = StreamPipeline(
                    camera.resolve(self.settings), self.settings, self.enhancer, self.governor
                )
                try:
                    pipeline.start()
                except Exception as exc:  # noqa: BLE001 - 单路失败不影响其余通道
                    logger.error("通道 %s 启动失败: %s", camera_id, exc)
                    pipeline.stop()
                    continue

                rejected = False
                with self._lock:
                    if self._shutting_down.is_set():
                        rejected = True
                    else:
                        self._pipelines[camera_id] = pipeline
                if rejected:
                    pipeline.stop()
                    break

                # 错峰启动：几百路同时 probe+connect 会瞬间打满交换机/NVR 的
                # 并发会话表，表现为「所有路都打不开」。每路之间停一小会儿。
                stagger = getattr(self.settings, "start_stagger_seconds", 0.5)
                if stagger > 0 and not self._shutting_down.is_set():
                    self._stop_event.wait(stagger)

        return self.snapshot()

    def start_camera(self, camera: CameraConfig) -> bool:
        with self._control_lock:
            with self._lock:
                if (
                    len(self._pipelines) >= self.settings.max_streams
                    and camera.id not in self._pipelines
                ):
                    raise RuntimeError(f"已达最大并发通道数 {self.settings.max_streams}")
                previous = self._pipelines.pop(camera.id, None)
            if previous is not None:
                previous.stop()
            if self._shutting_down.is_set():
                raise RuntimeError("服务正在关闭，拒绝启动新通道")

            pipeline = StreamPipeline(
                camera.resolve(self.settings), self.settings, self.enhancer, self.governor
            )
            try:
                pipeline.start()
            except Exception as exc:  # noqa: BLE001 - 一律转成可读错误，配置已入库不回滚
                pipeline.stop()
                raise RuntimeError(f"通道 {camera.id} 启动失败: {exc}") from exc

            rejected = False
            with self._lock:
                if self._shutting_down.is_set():
                    rejected = True
                else:
                    self._pipelines[camera.id] = pipeline
            if rejected:
                pipeline.stop()
                raise RuntimeError("服务正在关闭，通道未启动")
        return True

    def stop_camera(self, camera_id: str) -> bool:
        with self._control_lock:
            with self._lock:
                pipeline = self._pipelines.pop(camera_id, None)
            if pipeline is None:
                return False
            pipeline.stop()
        return True

    def restart_camera(self, camera_id: str) -> bool:
        with self._control_lock:
            if self._shutting_down.is_set():
                return False
            with self._lock:
                pipeline = self._pipelines.get(camera_id)
            if pipeline is None:
                return False
            pipeline.restart()
        return True

    def get_pipeline(self, camera_id: str) -> StreamPipeline | None:
        with self._lock:
            return self._pipelines.get(camera_id)

    # ------------------------------------------------------------------ #
    # 监控循环
    # ------------------------------------------------------------------ #
    def _monitor_loop(self) -> None:
        interval = max(5, self.settings.health_interval_seconds)
        while not self._stop_event.wait(interval):
            try:
                self._tick()
            except Exception:  # noqa: BLE001 - 监控线程绝不能死
                logger.exception("监控循环异常")

    def _tick(self) -> None:
        gpu = read_gpu_util(self.settings.device_id)
        self.last_gpu = gpu

        with self._lock:
            pipelines = list(self._pipelines.values())

        # 队列压力：取所有通道中最深的队列占队列上限的比例
        pressure = 0.0
        for pipeline in pipelines:
            depth = pipeline._queue.qsize()  # noqa: SLF001 - 同包内访问，属内部协作
            pressure = max(pressure, depth / max(1, self.settings.queue_size))

        util = gpu["utilization"] if gpu else None
        previous = self.governor.level
        level = self.governor.observe(util, queue_pressure=pressure)
        if level is not previous:
            logger.warning("负载档位变更: %s → %s（%s）", previous.value, level.value, self.governor.last_reason)

        # 通道健康兜底：进程还在但指标长期不动，说明线程可能已死。
        # 用 started_at 兜底「从启动起就一帧没拿到」的通道——原实现会永远
        # 跳过这类通道，它们会一直挂在「在线但无画面」状态。
        for pipeline in pipelines:
            stats = pipeline.metrics
            last_seen = stats.last_frame_at or stats.started_at
            age = time.time() - last_seen
            if age <= 120:
                continue
            logger.warning(
                "通道 %s 已 %.0f 秒无新帧，触发重启", pipeline.camera.id, age
            )
            # 走 restart_camera 而不是直接 pipeline.restart()：它会确认该通道
            # 仍在 _pipelines 里，避免把刚被删除的通道又重启成一个孤儿进程。
            try:
                self.restart_camera(pipeline.camera.id)
            except Exception:  # noqa: BLE001
                logger.exception("通道 %s 重启失败", pipeline.camera.id)

    # ------------------------------------------------------------------ #
    # 快照
    # ------------------------------------------------------------------ #
    def snapshot(self) -> dict:
        with self._lock:
            rows = [pipeline.snapshot() for pipeline in self._pipelines.values()]
        return {
            "uptime_seconds": round(time.time() - self.started_at, 1),
            "stream_count": len(rows),
            "max_streams": self.settings.max_streams,
            "load": self.governor.snapshot(),
            "gpu": self.last_gpu,
            "enhancer": getattr(self.enhancer, "name", "unknown"),
            "streams": rows,
        }

    def metrics_snapshot(self) -> dict:
        with self._lock:
            pipelines = list(self._pipelines.values())
        return {
            "load": self.governor.snapshot(),
            "gpu": self.last_gpu,
            "streams": [pipeline.metrics.snapshot() for pipeline in pipelines],
            "stage_latency": (
                self.enhancer.stats_snapshot()
                if isinstance(self.enhancer, OnnxEnhancer)
                else {}
            ),
        }
