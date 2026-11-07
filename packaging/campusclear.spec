# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller 打包配置：把「明眸客户端」打成可执行程序。

构建（在项目根目录执行）：
    pyinstaller packaging/campusclear.spec --noconfirm --clean
或直接双击 packaging\\build_win.bat（多了依赖安装与自检两道门）。

产物：
    dist/CampusClear/                    （onedir，双击 CampusClear.exe 启动）
    dist/CampusClear/CampusClear.exe

关于 ffmpeg 的取舍（Windows 版关键决定）
----------------------------------------
imageio-ffmpeg 带一个官方 hook，只要它被静态分析到，就会把 87 MB 的静态
ffmpeg 复制进 _internal/imageio_ffmpeg/binaries/。而我们**已经**在构建前把
ffmpeg 放到了项目 bin/ 并用 datas 显式打进 _internal/bin/。两处都塞等于白白
多出 87 MB（免安装包要塞 U 盘拷来拷去，这个体积不能不当回事）。

所以这里：
  * 把 imageio_ffmpeg 放进 excludes —— 它是**源码运行时**的兜底来源，
    打包形态下不需要；
  * 用 datas 把 bin/ffmpeg.exe 打到 _internal/bin/ —— 正好命中
    compat._candidate_dirs() 在 frozen 模式下的第一顺位，源码运行与打包运行
    走同一套解析逻辑；
  * 想换 ffmpeg 不必重新打包：设 CAMPUSCLEAR_FFMPEG 指向别处即可。

其它说明：
- uvicorn / onnxruntime 是**运行时才导入**的，PyInstaller 静态分析发现不了，
  因此显式收集它们的全部子模块；
- 界面静态文件（static/*.html）必须作为数据一并打包，否则打包后 /client 打不开；
- models/ 只有 ~5 MB，一并打进 _internal/models/，这样即便忘了在 exe 旁边放
  权重，增强功能也不会静默降级成经典插值。
"""
import os

from PyInstaller.utils.hooks import collect_submodules

ROOT = os.path.normpath(os.path.join(SPECPATH, ".."))
SRC = os.path.join(ROOT, "src")

hiddenimports = []
for _pkg in ("uvicorn", "onnxruntime"):
    hiddenimports += collect_submodules(_pkg)
hiddenimports += [
    "uvicorn", "onnxruntime", "cv2", "numpy",
    "fastapi", "starlette", "pydantic", "argon2", "jwt",
    "campusclear.app", "campusclear.main", "campusclear.compat",
]

datas = [
    (os.path.join(SRC, "campusclear", "static"), "campusclear/static"),
]

# ffmpeg：构建前由 build_win.bat（或手工）放到 bin/，这里打进 _internal/bin/
_ffmpeg = os.path.join(ROOT, "bin", "ffmpeg.exe")
if os.path.isfile(_ffmpeg):
    datas.append((_ffmpeg, "bin"))
else:  # pragma: no cover - 构建期提示，不阻断（会退化成 PATH 查找）
    print("[spec] 警告：未找到 bin\\ffmpeg.exe，打包产物将依赖 PATH 上的 ffmpeg")

# 本地 RTSP 网关（MediaMTX）：回推需要一个 RTSP 服务端接收。
# 放在 ffmpeg 同目录，core/gateway.py 会在这里查找并托管；
# 没有它时回推会被明确跳过（观看/截图不受影响），不会影响程序启动。
for _gw in ("mediamtx.exe", "mediamtx.yml"):
    _path = os.path.join(ROOT, "bin", _gw)
    if os.path.isfile(_path):
        datas.append((_path, "bin"))
    else:  # pragma: no cover
        print(f"[spec] 提示：未找到 bin\\{_gw}，打包产物将不带本地 RTSP 网关")

# 权重：打进 _internal/models/，app.py 会把它作为第二顺位候选目录
_models = os.path.join(ROOT, "models")
if os.path.isdir(_models):
    for _name in os.listdir(_models):
        if _name.endswith(".onnx") or _name == "models_manifest.json":
            datas.append((os.path.join(_models, _name), "models"))

a = Analysis(
    [os.path.join(SRC, "campusclear", "app.py")],
    pathex=[SRC],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    # imageio_ffmpeg 见文件头说明：避免 87 MB 的重复 ffmpeg
    excludes=["tkinter", "pytest", "matplotlib", "imageio_ffmpeg", "imageio"],
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="CampusClear",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="CampusClear",
)
