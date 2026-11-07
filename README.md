# CampusClear 明眸 · 校园监控离线增强客户端

面向学校用户的**单机客户端**：从监控服务器（摄像头 / NVR / 平台）取流，
在本机做画质增强，再显示出来。**完全离线、数据不出内网、可物理断网运行。**

- **实时看**：拉流 → 轻量增强 → 浏览器直接看（多协议自适应）。
- **回放查**：选一段录像 → 重量增强 → 播放（慢一点没关系）。
- **抓图留存**：点一下就抓当前帧，增强后自动落到右侧列表，不打断连续操作。
- **回推预览**：增强后的画面可以再推成一路本地 RTSP，用 VLC 或大屏播放器拉。
- **不用登录**：服务只绑 `127.0.0.1`，打开即用。

当前版本 `0.9.0`。

---

## 主要能力

| 能力 | 说明 |
| --- | --- |
| 免登录单机运行 | 只绑定本机，无账号体系；运维面板 `/api/v1` 的鉴权不受影响 |
| 多协议取流 | RTSP / rtsps / RTMP / HTTP-FLV / HLS / HTTP(MJPEG) / RTP·UDP / 本地文件 |
| 按厂商接入 | 海康 / 大华 / 宇视：实时（主/子码流）与按时间段回放地址自动构造 |
| 三级增强 | `realtime` 轻量（实时）/ `balanced`（2×）/ `forensic` 取证放大（慢、最优） |
| 离线 AI 权重 | 已接入 Real-ESRGAN general-x4v3（4× 超分）；**缺失阶段自动回退经典算法** |
| 异步抓帧 | 截图进单线程队列，连续点击不卡界面；列表里显示「排队中（第 N 位）/ 增强中」 |
| 回推预览 | 内置本地 RTSP 网关（MediaMTX），增强结果以 `rtsp://127.0.0.1:8554/cc_<通道>` 发布 |
| 对比与全屏 | 点画面全屏；每路「对比」看 *原图 vs 增强* |
| 系统内网安全 | Argon2 + JWT + RBAC + 审计（运维面板）；输出流鉴权 |
| 纯本地界面 | `static/*.html` 零 CDN、零外链，断网可用 |

## 环境要求

- **Python ≥ 3.10**（开发验证于 3.12）
- Linux / Windows / macOS
- 重量增强（`forensic`）需要 `onnxruntime`；实时档只用 OpenCV，无 GPU 也能跑
- 回推预览需要 `bin/mediamtx.exe`（Windows）或 `bin/mediamtx`（Linux/macOS），**没有该文件时回推自动关闭，其余功能不受影响**

## 快速开始

### 1. 取代码

```bash
git clone https://github.com/Logic922/CampusClear.git
cd CampusClear
```

### 2. 装依赖

**普通文件系统（推荐，用 venv）：**

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
```

**exfat / 不支持符号链接的分区**（venv 会创建 `lib64` 符号链接而失败），
改用 `pip --target` 装到一个普通目录：

```bash
pip install -r requirements.txt --target pylibs
```

`pylibs/` 是**本地生成**的目录，**不在本仓库里**（见下方「本仓库不包含什么」）。
只要它存在，`run_client.sh` / `start.sh` / `run_tests.sh` 会自动用 `PYTHONPATH` 指过去，
等价于一个「无符号链接的虚拟环境」。

> 没有 NVIDIA 显卡的机器：把 `requirements.txt` 里的 `onnxruntime-gpu`
> 换成 `onnxruntime`（两者互斥），或改用 `pip install -e ".[cpu]"`。
> Windows 上直接用 `packaging\requirements-win.txt` 更省事，见下方「Windows 打包」。

### 3. 启动

```bash
./run_client.sh                # 启动并自动打开界面
./run_client.sh --no-window    # 只起本地服务，不弹浏览器
./run_client.sh --port 9000
```

等价于：

```bash
PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}" python3 -m campusclear.app
```

客户端参数：

| 参数 | 说明 |
| --- | --- |
| `--host` | 默认 `127.0.0.1`。**勿改为对外地址** |
| `--port` | 默认 `8000` |
| `--data-dir` | 录像 / 输出 / 数据库目录 |
| `--models-dir` | ONNX 权重目录（重量增强用） |
| `--no-window` | 不自动打开界面 |
| `--url-only` | 只打印地址后退出（调试用） |

### 4. 跑测试

```bash
pip install -r requirements-dev.txt --target pylibs   # pytest + httpx
./run_tests.sh
```

21 个测试模块、216 个用例，覆盖客户端接口 / 取流生命周期 / 增强器 / 调度 / 指标 /
MJPEG 输出 / 多端口 / 网关 / 诊断工具 / 启动引导。

## 目录结构

```
src/campusclear/     核心代码
  app.py             桌面启动器（程序入口，也是打包入口）
  main.py            FastAPI 应用（/ 运维面板、/client 客户端、/api/v1 运维接口）
  config.py          配置（环境变量 CAMPUSCLEAR_*）
  compat.py          跨平台兼容层（子进程无窗口、路径、信号差异）
  api/               client.py（客户端接口）、routes.py + schemas.py（运维接口）
  auth/              Argon2 + JWT + RBAC
  core/              取流 / 分析 / 决策 / 增强 / 回放 / 来源 / 调度 / 快照 / 存储 / gateway.py（回推网关）
  models/            ONNX 权重注册与完整性校验
  static/            client.html（老师客户端）、dashboard.html（运维面板），零 CDN
tests/               21 个测试模块 + conftest
tools/               探测、试机、取证增强、权重校验、现场诊断（diagnose_server.py）、Windows 自检
models/              离线权重 + models_manifest.json
packaging/           PyInstaller 打包配置（spec）与一键脚本（build.sh / build_win.bat）
```

## 离线权重

权重放在 `models/`，清单记 `文件名 + SHA256 + 倍率`；启动时校验，不匹配拒绝启动。
当前已接入：

| 阶段 | 模型 | 说明 |
| --- | --- | --- |
| `sr` | Real-ESRGAN general-x4v3（4.65 MB） | 4× 超分，动态尺寸 |
| lowlight / denoise / deblur | — | 未接入，**自动回退经典算法** |

```bash
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
python tools/verify_models.py --models-dir models                  # 校验
python tools/verify_models.py --models-dir models --write-manifest # 回填哈希
```

## 取流与探测

`tools/` 下的脚本都 import `campusclear`，先设好 `PYTHONPATH`：

```bash
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"

python tools/probe_source.py --methods
python tools/probe_source.py --url "rtsp://user:pass@10.0.0.5:554/Streaming/Channels/101"
python tools/probe_source.py --brand hikvision --host 10.0.0.5 --user admin --password xxx --sub
python tools/offline_trial.py     # 离线试机：不接摄像头、不需 GPU，直接看增强效果
```

**现场上机诊断**：不知道对方服务器支持 RTSP 主/子码流、HTTP-FLV、HLS 还是厂商私有回放时：

```bash
bash run_diagnose.sh --host 10.0.0.9 --user admin --password xxx --brand auto \
     --start "2026-10-29 16:00:00" --end "2026-10-29 16:05:00"
```

Windows 上双击 `run_diagnose.bat`。它会逐个尝试实时/回放地址，记录 TCP、
RTSP OPTIONS/DESCRIBE、HTTP 头、ffprobe 完整 stderr、ffmpeg 实际解码帧数，
日志默认在 `logs/diagnose-*.log`。只看会试哪些地址可加 `--dry-run`。

RTSP 传输默认 TCP；老 NVR 或组播环境可切换：

```bash
CAMPUSCLEAR_RTSP_TRANSPORT=udp bash start.sh
```

## 回推预览（可选）

增强后的画面可以再作为一路 RTSP 发布出去，给大屏、VLC 或其他播放器拉取：

```
rtsp://127.0.0.1:8554/cc_<通道ID>
```

实现方式：程序启动时在 `bin/` 里寻找 `mediamtx`，找到就随程序拉起一个**只监听本机**的
RTSP 网关，退出时一起关掉（Windows 上用 Job Object 保证父进程被杀时子进程同步退出，
不会留下占端口的孤儿进程）。找不到就打印一行提示并自动关闭回推，其他功能照常。

## 打包

### Linux / macOS

```bash
bash packaging/build.sh
```

产物是 **PyInstaller onedir**：`dist/CampusClear/`，双击 `CampusClear` 即可启动。

### Windows

```bat
packaging\build_win.bat
```

脚本会自动做三件事：确认 PyInstaller 与依赖、把 `bin\ffmpeg.exe` 与 `bin\mediamtx.exe`
一起收进产物、生成 `dist\CampusClear\CampusClear.exe`。详细步骤与常见坑见
`packaging/README-Windows.md`。

> 体积主要由 `onnxruntime` + `opencv` 决定；不用重量 AI 时可从
> `packaging/campusclear.spec` 的 `collect_submodules` 里去掉 `onnxruntime` 显著瘦身。

> 体积较大的单文件分发物（AppImage / 打包好的 exe 目录）走 **GitHub Releases 附件**，不入库。

## 本仓库不包含什么

| 内容 | 大小 | 原因 |
| --- | --- | --- |
| `pylibs/`、`.venv/` | ~337 MB | 本地依赖，第三方二进制不随源码入库；按上面「装依赖」自行生成 |
| `bin/ffmpeg.exe`（87 MB）、`bin/mediamtx.exe`（57 MB） | ~144 MB | 第三方二进制，请自行下载放入 `bin/`；缺 ffmpeg 时程序会提示并给出下载地址 |
| `CampusClear-x86_64.AppImage` 等打包产物 | ~112 MB | 超过 GitHub 单文件 100 MB 硬上限；走 Releases 分发 |
| `data/` | — | 运行时目录（日志 / 快照 / 缓存 / 摄像头配置），含现场数据，不入库 |
| `.env` | — | 本地密钥与现场配置，请从 `.env.example` 复制 |

## 环境变量

所有配置项以 `CAMPUSCLEAR_` 开头，未设置的用代码内默认值。
完整列表见 [`.env.example`](.env.example)：服务监听、TLS 自签证书、
JWT/初始管理员、`data`/`models`/日志目录、执行提供器优先级与分块推理、
流与调度（最大路数、队列、推理/输出帧率、回推地址）等。

与回推相关的两个：

| 变量 | 默认 | 说明 |
| --- | --- | --- |
| `CAMPUSCLEAR_OUTPUT_ENABLED` | 有网关二进制时 `true` | 关闭后不推流，只保留本地预览 |
| `CAMPUSCLEAR_OUTPUT_URL` | `rtsp://127.0.0.1:8554` | 网关地址，一般不用改 |

## 说明与边界

- 只做**画质增强**，不做人脸识别、不采集生物特征。
- 不支持 RTSP/ONVIF、只有云端 SDK 的设备不适用。
- 厂商回放地址、通道编号各固件不同，**探测成功再写死**。
- CPU 上整帧重模型很慢——重量档本就是「离线按需」；可用 ROI/分块/降采样提速。
- 回推通道走本机 8554 端口，如与本机其他服务冲突可在 `bin/mediamtx.yml` 里改。
- 现场截图/录屏可能含摄像头 IP、账号与人脸车牌，**用于对外材料前必须脱敏**。
