[app]

title = Laundry Scan
package.name = laundryscan
package.domain = org.stden

source.dir = .
source.include_exts = py,kv,ttf,txt,tflite
source.exclude_exts = spec,png,jpg
source.exclude_dirs = .buildozer,bin,.vscode,__pycache__

[app:requirements]
python3
kivy==2.0.0rc2
kivymd
opencv
numpy
https://dl.google.com/coral/python/tflite_runtime-1.14.0-cp37-cp37m-linux_armv7l.whl

version = 0.8
orientation = portrait
fullscreen = 0

android.permissions = CAMERA
android.api = 28
android.minapi = 21
android.sdk = 20
android.ndk = r19b
android.ndk_api = 21
android.private_storage = True
android.accept_sdk_license = True
android.arch = armeabi-v7a
p4a.branch = develop

[buildozer]

log_level = 2
warn_on_root = 1
