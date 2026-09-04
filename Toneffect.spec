# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    ['Toneffect.py'],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='Toneffect',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file='entitlements.plist',
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='Toneffect',
)

app = BUNDLE(
    coll,
    name='Toneffect.app',
    icon='icon.icns',
    bundle_identifier='com.toneffect.app',
    info_plist={
        'NSMicrophoneUsageDescription': 'Toneffect 앱이 실시간 오디오 입력 및 톤 분석을 위해 마이크 권한을 필요로 합니다.',
        'NSHighResolutionCapable': 'True',
    },
)
