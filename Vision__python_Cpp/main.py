#!/usr/bin/env python3
import os
import time
import cv2
import numpy as np
from flask import Flask, Response

# 导入硬件库
try:
    from hobot_vio import libsrcampy as srcampy
except ImportError:
    from hobot_vio_rdkx5 import libsrcampy as srcampy

# 导入我们编译好的 C++ AI 模块
import rdk_ai_module 

app = Flask(__name__)
pipeline = None
cam = None

def init_system():
    global pipeline, cam
    
    # 1. 硬件初始化 (杀掉旧进程，打开 GPIO)
    os.system("sudo fuser -k /dev/video* > /dev/null 2>&1")
    os.system("echo 1 > /sys/class/gpio/gpio353/value 2>/dev/null")
    
    # 2. 打开摄像头 (Python 驱动)
    print("📷 正在启动摄像头...")
    cam = srcampy.Camera()
    ret = cam.open_cam(0, -1, 30, 640, 480)
    if ret == 0:
        print("✅ 摄像头启动成功")
    else:
        print(f"❌ 摄像头启动失败，错误码: {ret}")

    # 3. 启动 C++ AI 核心
    print("🚀 正在加载 AI 引擎 (C++)...")
    pipeline = rdk_ai_module.VideoPipeline(
        "models/haarcascade_frontalface_default.xml",
        "models/emotion.onnx",
        "models/v5n.onnx"
    )
    print("✅ AI 引擎加载完毕，系统就绪")

# 创建一张黑色背景的等待图片
def get_loading_frame():
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    # 写上文字
    cv2.putText(img, "SYSTEM INITIALIZING...", (120, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    return cv2.imencode(".jpg", img)[1].tobytes()

loading_bytes = get_loading_frame()

def generate():
    global pipeline, cam
    
    while True:
        # 1. Python 负责抓图 (搬运工)
        # 注意：这里只负责搬运原始数据，不做任何处理，速度极快
        img_bytes = cam.get_img(2, 640, 480)
        
        if img_bytes is None:
            time.sleep(0.01)
            continue
            
        # 2. C++ 负责所有脏活累活 (解码、缩放、AI、画框、压缩)
        # process_frame 会返回处理好的 JPEG 图片数据
        jpeg_bytes = pipeline.process_frame(img_bytes)
        
        # 3. 如果 C++ 还没准备好 (比如正在初始化)，发一张等待图，防止网页转圈
        if not jpeg_bytes:
            yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + loading_bytes + b'\r\n')
            time.sleep(0.05)
            continue
            
        # 4. 正常推流
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + jpeg_bytes + b'\r\n')
        
        # 控制帧率 (30fps 左右)
        time.sleep(0.03)

@app.route("/")
@app.route("/video")
def index():
    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")

if __name__ == '__main__':
    # 初始化
    init_system()
    
    # 获取本机 IP 用于打印
    try:
        ip = os.popen("hostname -I | cut -d' ' -f1").read().strip()
    except:
        ip = "0.0.0.0"
        
    print(f"\n✨========================================✨")
    print(f"   AI 互动系统已启动")
    print(f"   访问地址: http://{ip}:5000")
    print(f"✨========================================✨\n")
    
    # 启动 Web 服务器
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)