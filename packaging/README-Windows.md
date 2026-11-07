# 明眸客户端 · Windows 免安装版

> 面向「无公网、无 GPU、只有一台办公电脑」的校园场景：拷走整个文件夹，双击即用。
> 全程离线，运行期不访问互联网，不下载任何东西。

---

## 一、构建

PyInstaller **不能跨平台**——Linux 上产不出 Windows 的 `.exe`。所以构建必须在 Windows 上做。

### 一键构建（推荐）

1. 把整个 `CampusClear` 文件夹拷到 Windows（U 盘即可）；
2. 装 **Python 3.10 ~ 3.13**，安装时勾选 “Add python.exe to PATH”；
3. 双击 **`packaging\build_win.bat`**。

脚本会依次做六件事，任何一步不过就直接中断（不会产出一个「看起来成功实际跑不起来」的包）：

| 步骤 | 做什么 | 不过关的后果 |
| --- | --- | --- |
| 1 | 检查 Python | 直接退出，提示装 Python |
| 2 | 建独立 `.venv` | 不污染机器上其它 Python 工程，删掉即干净 |
| 3 | 装依赖（先试清华镜像，失败自动回退官方 PyPI） | — |
| 4 | 准备 `bin\ffmpeg.exe`，并**实际执行一次**验证能用 | 缺 ffmpeg 就退出（运行时绝不允许联网下载） |
| 5 | 跑 `tools\win_selftest.py`（14 项）+ `pytest`（159 项） | 不过就不打包 |
| 6 | PyInstaller 打包 + 把权重放到 exe 旁边 | — |

### 手动构建

```bat
python -m venv .venv
.venv\Scripts\activate
pip install -r packaging\requirements-win.txt
pip install pyinstaller
python tools\prepare_ffmpeg.py
python tools\win_selftest.py && python -m pytest
python -m PyInstaller packaging\campusclear.spec --noconfirm --clean
```

---

## 二、产物与运行

```
dist\CampusClear\
├── CampusClear.exe          ← 双击这个
├── _internal\               ← 运行时依赖（勿删）
│   ├── bin\ffmpeg.exe       ← 内置 ffmpeg 7.1（83.6 MB，离线拉流靠它）
│   └── models\*.onnx        ← 增强权重（~4.9 MB）
└── models\                  ← 权重的另一份，放在这里是为了**现场可替换**
```

- **免安装**：整个 `dist\CampusClear\` 文件夹拷走即用，可放 U 盘、可放任意目录；
- 双击 `CampusClear.exe`，程序在 `127.0.0.1` 起服务并自动打开界面
  （有 pywebview 就开独立窗口，否则用系统默认浏览器）；
- 界面地址：<http://127.0.0.1:8000/client>；
- 端口被占用时会**自动换一个空闲端口**，控制台会打印实际地址；
- 数据（数据库、截图、日志）写在 **`%LOCALAPPDATA%\CampusClear\`**，
  不往程序目录写，所以文件夹可以放在只读位置。

### 首次登录（重要）

程序**不设默认弱口令**。首次启动会随机生成管理员密码，同时落在两个地方：

1. 启动时弹出的**控制台窗口**里；
2. 文件 **`%LOCALAPPDATA%\CampusClear\初始密码.txt`**（用户名 `admin`）。

之所以多写一份文件：库里存的是 Argon2 哈希，**不可逆**，密码丢了只能删库重来。
登录后在管理面板改掉密码，然后删除该文件即可。

> 批量部署时别让每台机器随机生成——统一设置：
> ```bat
> set CAMPUSCLEAR_BOOTSTRAP_PASSWORD=<你的口令>
> CampusClear.exe
> ```
> 设了这个环境变量后，程序不会写密码文件。
> 首次登录后同样建议尽快改掉。

---

## 三、性能预期（请先读这一段）

**这台开发机没有独立显卡**（只有 Intel HD Graphics 530），增强走 CPU 版 onnxruntime。
720p 实测数据：

| 场景 | 帧率 |
| --- | --- |
| 原始流（不增强，上限） | 14.1 fps |
| 增强后，合成测试图案 / 画面本已清晰的白天场景 | **约 9 fps** |
| 增强后，触发全部四步（暗光 + 高噪声 + 模糊） | 约 3~4 fps |

**为什么有两档**：程序的质量判据是**按帧**决定跑不跑某一步的——画面够亮就不做
低光增强，够干净就不做降噪。所以实际帧率取决于现场光照与噪声，不是一个固定值。
合成测试图案属于「最省」的情况，夜间真实监控画面会落在下面那一档。

4MP（2560×1440，多数海康筒机的默认主码流）像素数是 720p 的 4 倍，
按线性外推会掉到 1 fps 上下，实际不可用。三条可行的路：

**路线 A：降解码分辨率（不改硬件，立刻见效）**

```bat
set CAMPUSCLEAR_MAX_DECODE_WIDTH=1280
CampusClear.exe
```

画面会被等比缩到 1280 宽再增强，帧率能上到 3~4 fps。代价是增强结果的分辨率
也止步于 1280 宽——但「动得起来但没那么清晰」通常比「清晰但卡成幻灯片」有用。

**路线 B：只留最有效的增强项**

```bat
set CAMPUSCLEAR_ENABLE_DEBLUR=false
set CAMPUSCLEAR_ENABLE_SR=false
set CAMPUSCLEAR_INFER_FPS=12
CampusClear.exe
```

超分（SR）是四步里最贵的一步。只留低光增强 + 降噪，帧率能翻好几倍。

**路线 C：换一台带 NVIDIA 卡的机器**

装 `onnxruntime-gpu` + 对应 CUDA 运行库，配置里默认就会优先选
`TensorrtExecutionProvider` → `CUDAExecutionProvider`。这条路才是 1080p 实时
（25 fps）的正解。执行提供器由 `compat.resolve_providers()` 按本机实际能力
自动收敛，配置里写了 CUDA 但机器没有也不会报错，只会静默退回 CPU。

---

## 四、环境变量速查

所有参数都能用环境变量覆盖，**无需改代码、无需重新打包**。

### 部署时最常改的

| 环境变量 | 默认 | 说明 |
| --- | --- | --- |
| `CAMPUSCLEAR_PORT` | `8000` | 界面端口（被占用会自动换） |
| `CAMPUSCLEAR_MAX_DECODE_WIDTH` | `0` | 解码宽度上限，`0`=不限。CPU 机器建议 `1280` |
| `CAMPUSCLEAR_MAX_DECODE_HEIGHT` | `0` | 解码高度上限，`0`=不限。可与上一项配合 |
| `CAMPUSCLEAR_FFMPEG` | 自动探测 | 指定 ffmpeg 路径，覆盖内置那份 |
| `CAMPUSCLEAR_MODELS_DIR` | exe 旁 `models` | 权重目录 |
| `CAMPUSCLEAR_DATA_DIR` | `%LOCALAPPDATA%\CampusClear` | 数据目录 |
| `CAMPUSCLEAR_BOOTSTRAP_PASSWORD` | 随机 | 首次启动的 admin 密码，**生产环境务必设** |

### 增强开关与算力

| 环境变量 | 默认 | 说明 |
| --- | --- | --- |
| `CAMPUSCLEAR_ENABLE_LOWLIGHT` | `true` | 低光增强（便宜、收益明显） |
| `CAMPUSCLEAR_ENABLE_DENOISE` | `true` | 降噪 |
| `CAMPUSCLEAR_ENABLE_DEBLUR` | `true` | 去模糊（较贵） |
| `CAMPUSCLEAR_ENABLE_SR` | `true` | 超分（**最贵**，卡就关它） |
| `CAMPUSCLEAR_SR_SCALE` | `2` | 超分倍率 |
| `CAMPUSCLEAR_INFER_FPS` | `12` | 每秒实际增强多少帧，超过的帧复用上一帧结果 |
| `CAMPUSCLEAR_ENABLE_STATIC_REUSE` | `true` | 静止画面复用，省算力 |
| `CAMPUSCLEAR_PROVIDERS` | TensorRT→CUDA→CPU | 执行提供器优先级，按本机能力自动收敛 |
| `CAMPUSCLEAR_USE_FP16` | `true` | CPU-only 时被自动关闭（CPU 不支持 fp16） |
| `CAMPUSCLEAR_TILE_MAX_PIXELS` | `1036800` | 分块推理的单块像素上限，显存/内存不足时调小 |

### 回推与其它

| 环境变量 | 默认（客户端形态） | 说明 |
| --- | --- | --- |
| `CAMPUSCLEAR_OUTPUT_ENABLED` | 安装目录有 `bin\mediamtx.exe` 时为 `true`，否则 `false` | 是否把增强结果回推成 RTSP。包内自带网关时程序会**自动拉起**它，回推可直接用；没有网关时明确跳过，不影响观看/截图 |
| `CAMPUSCLEAR_RTSP_OUTPUT_BASE` | `rtsp://127.0.0.1:8554` | 回推目标。指向本机时网关由程序托管；指向校内已有 MediaMTX 时改这里即可 |
| `CAMPUSCLEAR_FORCE_DECODE_SIZE` | `true` | 解码端钉死尺寸，防「帧边界错位→画面雪花」 |
| `CAMPUSCLEAR_RTSP_TIMEOUT_US` | `5000000` | RTSP 接收超时（微秒），交换机瞬断时靠它触发重连 |
| `CAMPUSCLEAR_MAX_STREAMS` | `8` | 最多同时处理几路 |
| `CAMPUSCLEAR_LOGIN_MAX_FAILURES` | `5` | 登录失败几次锁定 |
| `CAMPUSCLEAR_VERIFY_MODEL_HASH` | `false` | 客户端形态放宽；正式部署建议开 `true` |

---

## 五、接摄像头

海康威视标准 RTSP 地址：

```
rtsp://<用户名>:<密码>@<摄像头IP>:554/Streaming/Channels/101     # 主码流
rtsp://<用户名>:<密码>@<摄像头IP>:554/Streaming/Channels/102     # 子码流（推荐）
```

**CPU 机器上强烈建议用子码流（102）**：分辨率低、解码便宜，配合
`CAMPUSCLEAR_MAX_DECODE_WIDTH` 效果更稳。

排查摄像头连不上时，先用源码里的探测工具确认地址本身可用（需在源码目录、
已激活 `.venv` 的环境下运行）：

```bat
python tools\probe_source.py --url "rtsp://admin:密码@192.168.1.64:554/Streaming/Channels/101"
python tools\probe_source.py --methods                                     REM 列出支持的连接方式
python tools\probe_source.py --brand hikvision --host 192.168.1.64 --user admin --password 密码 --sub
```

---

## 六、常见问题

**双击后窗口一闪而过**
用命令行运行，看报错：
```bat
dist\CampusClear\CampusClear.exe
```
最常见原因是缺 **Microsoft Visual C++ 2015-2022 Redistributable**（OpenCV 和
onnxruntime 都依赖它）。装上即可。

**提示拉流失败 / 画面全黑**
先确认内置 ffmpeg 在：
```bat
dist\CampusClear\_internal\bin\ffmpeg.exe -version
```
不在就重新构建（第 4 步会补齐）。也可以用 `CAMPUSCLEAR_FFMPEG` 指向别处的 ffmpeg。

**画面碎成雪花 / 撕裂**
典型的「rawvideo 帧边界错位」。确认 `CAMPUSCLEAR_FORCE_DECODE_SIZE` 没有被设成 `false`。
内核机制：rawvideo 管道按固定帧长 `H×W×3` 读取，只要 ffmpeg 实际输出的尺寸与预期不符，
后面每一帧都会从上一帧的中间开始拼——所以这里必须钉死尺寸。

**杀毒软件报毒 / 拦截**
PyInstaller 打包的程序被误报属于常见现象。给 `dist\CampusClear\` 整个目录加白名单。
另外内置的 `ffmpeg.exe` 也可能被杀软拦下，导致「文件在但执行不了」——
`tools\prepare_ffmpeg.py` 会实际跑一次 `-version` 来提前暴露这种情况。

**忘了管理员密码 / 登不进去**
看 `%LOCALAPPDATA%\CampusClear\初始密码.txt`（用户名 `admin`）。
如果那个文件已经被删、密码也改了却又忘了，只能删掉
`%LOCALAPPDATA%\CampusClear\campusclear.db` 让程序重建账号——**通道配置和
快照记录会一并丢失**，所以别在正式投用后才删。

**界面打不开，但进程还在**
端口没被占用时会自动换端口，控制台打印的才是真实地址。或者手动指定：
```bat
CampusClear.exe --port 9000
```

**想改成单个 exe 文件**
把 `packaging\campusclear.spec` 的 `COLLECT` 改成 `onefile`。代价是每次启动都要
解压约 200 MB，启动要慢十几秒——**校园场景不推荐**，文件夹形态体验更好。

---

## 七、为什么依赖里是 CPU 版 onnxruntime

`requirements.txt`（Linux/服务器用）装的是 `onnxruntime-gpu`；`packaging\requirements-win.txt`
刻意换成了 **`onnxruntime`（CPU 版）**：

- CPU 版任何 Windows 都能直接装，不需要 CUDA/cuDNN 那一整套（加起来好几个 GB）；
- GPU 版在没有 N 卡的机器上装得上但用不了，还会在每次建会话时刷一堆告警；
- 现场机器大概率是普通办公电脑（本例就是 Intel 核显），先保证「装得上、跑得起来」，
  真要上 GPU 再换包 + 配 CUDA 运行库即可，代码不用改。

## 八、体积构成

| 部分 | 大小 |
| --- | --- |
| `bin\ffmpeg.exe` | 83.6 MB |
| onnxruntime | ~47 MB |
| opencv-python-headless | ~37 MB |
| numpy | ~60 MB（部分在 `_internal` 里压缩存放） |
| 权重 + 其它 | ~10 MB |
| **合计** | **约 200 MB 量级** |

打包配置刻意**排除**了 `imageio_ffmpeg`：它自带的 ffmpeg 与第 4 步放进 `bin\` 的那份
完全重复，一起打进去会凭空多出 83.6 MB。
