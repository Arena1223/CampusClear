@echo off
REM ============================================================
REM  明眸客户端 · Windows 一键构建（免安装目录）
REM  产出 dist\CampusClear\ ，双击 CampusClear.exe 即可运行
REM  用法：把整个 CampusClear 文件夹放到 Windows 上，双击本脚本
REM
REM  为什么必须在 Windows 上构建：PyInstaller 不能跨平台。
REM ============================================================
chcp 65001 >nul
setlocal enabledelayedexpansion
cd /d "%~dp0.."

echo ==== [1/6] 检查 Python ====
python --version >nul 2>&1
if errorlevel 1 (
  echo 未找到 python。请安装 Python 3.11 / 3.12 / 3.13，安装时勾选 "Add python.exe to PATH"。
  echo 下载地址： https://www.python.org/downloads/windows/
  pause & exit /b 1
)
python --version

echo.
echo ==== [2/6] 建立虚拟环境 .venv ====
REM 用独立 venv：不污染用户机器上的其他 Python 工程，删掉即干净。
if not exist .venv ( python -m venv .venv )
call .venv\Scripts\activate.bat
if errorlevel 1 ( echo 虚拟环境创建失败。 & pause & exit /b 1 )
python -c "import sys;print('python:',sys.executable)"

echo.
echo ==== [3/6] 安装依赖 ====
REM 先试国内镜像（校园网/内网通常更快），失败则回退官方源。
REM 之所以不写死一个源：镜像偶发缺包或返回空索引，写死会让构建直接卡死。
python -m pip install -q -U pip
pip install -q -r packaging\requirements-win.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
if errorlevel 1 (
  echo     镜像源失败或无响应，回退官方 PyPI ...
  pip install -q -r packaging\requirements-win.txt
)
if errorlevel 1 ( echo. & echo 依赖安装失败，请检查网络或代理设置。 & pause & exit /b 1 )
pip install -q pyinstaller
if errorlevel 1 ( echo. & echo PyInstaller 安装失败。 & pause & exit /b 1 )

echo.
echo ==== [4/6] 准备 ffmpeg ^(离线必需^) ====
REM 运行时绝不允许联网下载，所以 ffmpeg 必须随程序走。
REM 源取自 imageio-ffmpeg 自带的静态构建（见 tools\prepare_ffmpeg.py 注释）。
REM 脚本自带「能否执行」的验证，跑不通会返回非零码。
python tools\prepare_ffmpeg.py
if errorlevel 1 ( echo. & echo ffmpeg 未就位或无法执行，无法打包。 & pause & exit /b 1 )

echo.
echo ==== [5/6] 自检与测试 ====
REM 这两道门是打包前的硬门槛：不过就别打包，否则产出的 exe 一定有问题。
python tools\win_selftest.py
if errorlevel 1 ( echo. & echo ✗ Windows 兼容自检未通过，先修问题再打包。 & pause & exit /b 1 )
python -m pytest -q
if errorlevel 1 ( echo. & echo ✗ 测试未全部通过，先修问题再打包。 & pause & exit /b 1 )

echo.
echo ==== [6/6] 打包 ====
REM 先清掉上一次的产物，避免新旧文件混在一起（PyInstaller --clean 只管缓存）。
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist
python -m PyInstaller packaging\campusclear.spec --noconfirm --clean
if errorlevel 1 ( echo. & echo 打包失败，请把上面的报错发出来。 & pause & exit /b 1 )

REM 权重再放一份到 exe 同级：spec 已把 models 打进 _internal\models\，
REM 这里放 exe 旁边是为了「现场想换权重时看得见、随时可替换」——
REM app.py 的权重查找第一顺位就是这里。
if not exist dist\CampusClear\models mkdir dist\CampusClear\models
xcopy /y /q models\*.onnx dist\CampusClear\models\ >nul 2>&1
xcopy /y /q models\models_manifest.json dist\CampusClear\models\ >nul 2>&1

echo.
echo ==== 产物自检 ====
if not exist dist\CampusClear\CampusClear.exe (
  echo ✗ 未生成 CampusClear.exe。 & pause & exit /b 1
)
if not exist dist\CampusClear\_internal\bin\ffmpeg.exe (
  echo ! 警告：_internal\bin\ffmpeg.exe 缺失，拉流会失败。
)
echo 已确认 CampusClear.exe 与内置 ffmpeg 就位。

echo.
echo ============================================================
echo  完成！产出： %cd%\dist\CampusClear\CampusClear.exe
echo  免安装：整个 dist\CampusClear 文件夹拷走，双击 CampusClear.exe
echo  界面地址： http://127.0.0.1:8000/client
echo ============================================================
pause
endlocal
