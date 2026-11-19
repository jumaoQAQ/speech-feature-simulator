# -*- mode: python ; coding: utf-8 -*-
# 语音短时时域特征模拟系统 - PyInstaller 打包配置
# 在项目根目录（包含 main.py 与 vtsim/ 包）执行：pyinstaller main.spec
# 生成：dist/main.exe（单文件控制台程序，约 120 MB）

a = Analysis(
    ['main.py'],
    pathex=['.'],
    binaries=[],
    datas=[],
    hiddenimports=[
        'vtsim',
        'vtsim.preprocess',
        'vtsim.features',
        'vtsim.statistics',
        'vtsim.simulate',
        'vtsim.neural_sim',
        'vtsim.evaluate',
        'vtsim.report',
        'vtsim.pipeline',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='main',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)