#!/usr/bin/env python3
import sys
import os
import time
import threading
import asyncio
import random
import serial 
import subprocess
import aiohttp
import re
import numpy as np 
import traceback 
import logging
from concurrent.futures import ThreadPoolExecutor
from collections import Counter, deque 
import requests

# === Web 支持库 ===
import cv2
from flask import Flask, Response

# 1. 系统路径配置
sys.path.append("/usr/lib/python3/dist-packages")
sys.path.append("/usr/local/lib/python3.10/dist-packages")
sys.path.append("/usr/lib/python3.10/dist-packages")

# ================= 1. 配置区域 =================
SERIAL_PORT = "/dev/ttyS1" 
BAUD_RATE = 115200
WHISPER_SERVER_URL = "YOUR_WHISPER_SERVER_URL_HERE"

# Qwen-VL 配置
DASH_API_KEY = "YOUR_DASHSCOPE_API_KEY_HERE"
QWEN_VL_MODEL = "qwen3-vl-flash"
import dashscope
from dashscope import MultiModalConversation
dashscope.api_key = DASH_API_KEY

# 腾讯云 TTS 配置
SECRET_ID = "YOUR_TENCENT_SECRET_ID_HERE"
SECRET_KEY = "YOUR_TENCENT_SECRET_KEY_HERE"

# 路径与设备
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
VISION_DIR = os.path.join(ROOT_DIR, "Vision__python_Cpp")
VOICE_CODE_DIR = os.path.join(ROOT_DIR, "Voice_Subsystem", "code")
sys.path.append(VISION_DIR)
sys.path.append(VOICE_CODE_DIR)

AUDIO_SAVE_PATH = os.path.join(VOICE_CODE_DIR, "audio", "live.flac")
PCM_PATH = "/tmp/tts_out.pcm"
MIC_DEVICE = "plughw:1,0" 
SPK_DEVICE = "plughw:0,0" 

# 模型路径
FACE_XML = os.path.join(VISION_DIR, "models", "haarcascade_frontalface_default.xml")
EMOTION_ONNX = os.path.join(VISION_DIR, "models", "emotion.onnx")
HAND_ONNX = os.path.join(VISION_DIR, "models", "v5n.onnx")

# 🔥 自检
def check_models():
    print("\n🔍 [系统] 正在检查模型文件...")
    missing = []
    if not os.path.exists(FACE_XML): missing.append(f"人脸模型丢失: {FACE_XML}")
    if not os.path.exists(EMOTION_ONNX): missing.append(f"表情模型丢失: {EMOTION_ONNX}")
    if not os.path.exists(HAND_ONNX): missing.append(f"手势模型丢失: {HAND_ONNX}")
    if missing:
        print("❌ [严重错误] 找不到以下文件，AI 无法启动！")
        sys.exit(1)
    print("✅ [系统] 模型文件就绪！\n")

# 硬件库
try:
    from hobot_vio import libsrcampy as srcampy
except ImportError:
    try:
        from hobot_vio_rdkx5 import libsrcampy as srcampy
    except ImportError:
        pass

import rdk_ai_module
from tencentcloud.common import credential
from tencentcloud.tts.v20190823 import tts_client, models

executor = ThreadPoolExecutor(max_workers=4)
conversation_history = []
MAX_HISTORY = 6

# ================= 2. 全局共享数据 =================
shared_data = {
    "frame_nv12": None,      
    "ai_face": "None",       
    "ai_hand": "None",       
    "ai_box": [],            
    "lock": threading.Lock() 
}

# ================= 3. 硬件控制 =================
class SerialDisplayManager:
    def __init__(self):
        self.ser = None
        self.lock = threading.Lock()
        self.connect_serial()
    def connect_serial(self):
        try: self.ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        except: pass
    def send_cmd(self, char_cmd):
        with self.lock:
            if self.ser and self.ser.is_open:
                try: self.ser.write(char_cmd.encode('utf-8'))
                except: pass
    def set_expression(self, expr_id): self.send_cmd(str(expr_id))
    def set_hand(self, hand_type): self.send_cmd(hand_type)

display = SerialDisplayManager()

class SmartController:
    def __init__(self):
        self.mode = "IDLE" 
        self.last_emotion_check = time.time()
        self.face = "None"
        self.hand = "None"
    def set_mode(self, mode): 
        print(f"🔄 [系统] 模式切换: {self.mode} -> {mode}")
        self.mode = mode

controller = SmartController()

# ================= 4. 三大线程 =================
# --- 线程 A: 摄像头 ---
def thread_camera_capture():
    os.system("sudo fuser -k /dev/video* > /dev/null 2>&1")
    cam = srcampy.Camera()
    cam.open_cam(0, -1, 30, 640, 480)
    print("📷 [Cam] 摄像头启动")
    while True:
        img = cam.get_img(2, 640, 480)
        if img is not None:
            with shared_data["lock"]:
                shared_data["frame_nv12"] = img
        time.sleep(0.01)

# --- 线程 B: AI 推理 (无框流畅版) ---
def thread_ai_inference():
    try: pipeline = rdk_ai_module.VideoPipeline(FACE_XML, EMOTION_ONNX, HAND_ONNX); print("🧠 [AI-C++] 引擎就绪")
    except: pass

    face_cascade = cv2.CascadeClassifier(FACE_XML)
    emotion_net = None
    try: emotion_net = cv2.dnn.readNetFromONNX(EMOTION_ONNX); print("🧠 [AI-Py] 引擎就绪")
    except: pass

    # AI 输出的 7 种标签
    emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    emotion_history = deque(maxlen=10)

    while True:
        # 如果是 HOLD 或 IDLE，强制清空框数据
        if controller.mode in ["IDLE", "HOLD"]:
            with shared_data["lock"]:
                shared_data["ai_face"] = "None"
                shared_data["ai_hand"] = "None"
                shared_data["ai_box"] = [] 
            emotion_history.clear()
            
            if controller.mode == "HOLD":
                time.sleep(0.5) 
            else:
                time.sleep(0.1) 
            continue

        curr_nv12 = None
        with shared_data["lock"]:
            if shared_data["frame_nv12"] is not None:
                curr_nv12 = shared_data["frame_nv12"]
        
        if curr_nv12 is None: time.sleep(0.1); continue
            
        try:
            res_face = "None"
            res_hand = "None"
            res_box = []

            if controller.mode == "GAME":
                c_res = pipeline.process_frame(curr_nv12, 2) 
                res_hand = c_res[2]
                if len(c_res) > 3: res_box = c_res[3]

            elif controller.mode == "EMOTION":
                img_np = np.frombuffer(curr_nv12, dtype=np.uint8)
                img_bgr = cv2.cvtColor(img_np.reshape((720, 640)), cv2.COLOR_YUV2BGR_NV12)
                img_bgr = cv2.flip(img_bgr, -1) 
                img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
                
                faces = face_cascade.detectMultiScale(img_gray, 1.1, 4)
                
                if len(faces) > 0:
                    (x, y, w, h) = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)[0]
                    res_box = [int(x), int(y), int(w), int(h)]
                    
                    if emotion_net:
                        try:
                            roi = img_gray[y:y+h, x:x+w]
                            roi = cv2.resize(roi, (48, 48))
                            blob = cv2.dnn.blobFromImage(roi, 1.0/255.0, (48, 48), (0,0,0), swapRB=False, crop=False)
                            emotion_net.setInput(blob)
                            preds = emotion_net.forward()
                            raw_label = emotion_labels[preds[0].argmax()]
                            emotion_history.append(raw_label)
                            res_face = Counter(emotion_history).most_common(1)[0][0]
                        except: pass
                else:
                    emotion_history.clear()

            with shared_data["lock"]:
                shared_data["ai_face"] = res_face
                shared_data["ai_hand"] = res_hand
                shared_data["ai_box"] = res_box
                
            controller.face = res_face
            controller.hand = res_hand

        except Exception as e:
            pass
        
        time.sleep(0.08) 

# --- 线程 C: 网页展示 (30FPS 流畅版) ---
app = Flask(__name__)
import logging
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

def generate_frames():
    while True:
        time.sleep(0.033) 
        
        img_raw = None
        hand_res = ""; face_res = ""; box = []
        
        with shared_data["lock"]:
            img_raw = shared_data["frame_nv12"]
            hand_res = shared_data["ai_hand"]
            face_res = shared_data["ai_face"]
            box = shared_data["ai_box"]
        
        if img_raw is None: continue
        try:
            img_np = np.frombuffer(img_raw, dtype=np.uint8)
            img_yuv = img_np.reshape((720, 640))
            img_bgr = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR_NV12)
            img_bgr = cv2.flip(img_bgr, -1)
            
            if len(box) == 4:
                x, y, w, h = box
                cv2.rectangle(img_bgr, (x, y), (x+w, y+h), (0, 255, 0), 2)
                label = ""
                if controller.mode == "EMOTION": label = face_res if face_res != "None" else ""
                else: label = hand_res if hand_res != "None" else ""
                if label: cv2.putText(img_bgr, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            (flag, encodedImage) = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if flag: yield(b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n')
        except: continue

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>RDK X5 AI View</title>
        <style>
            body { margin: 0; background: #000; display: flex; justify-content: center; align-items: center; height: 100vh; overflow: hidden;}
            img { max-width: 100%; max-height: 100vh; object-fit: contain; border: 2px solid #333;}
        </style>
    </head>
    <body><img src='/video_feed'></body>
    </html>
    """

def run_flask():
    print("🌍 Web 服务: http://0.0.0.0:5000")
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False, threaded=True)

# ================= 5. 语音模块 =================
def kill_alsa_hogs(): 
    for dev in os.listdir("/dev/snd"): 
        try: subprocess.run(["fuser", "-k", f"/dev/snd/{dev}"], stderr=subprocess.DEVNULL)
        except: pass
    time.sleep(0.1)

async def record_audio(silence_duration: float = 0.8, threshold_percent: float = 1.0, max_time: int = None) -> None:
    time_msg = f" (限时{max_time}s)" if max_time else " (等待说话...)"
    print(f"\n🎤 [系统] 开始监听{time_msg}", flush=True)
    await asyncio.get_running_loop().run_in_executor(executor, kill_alsa_hogs)
    
    base_cmd = (f"arecord -q -D {MIC_DEVICE} -f S16_LE -c 1 -r 48000 -t raw | "
                f"sox -t raw -r 48000 -e signed -b 16 -c 1 - -r 16000 -t flac {AUDIO_SAVE_PATH} "
                f"silence 1 0.3 {threshold_percent}% 1 {silence_duration} {threshold_percent}%")
    final_cmd = f"timeout {max_time}s {base_cmd}" if max_time else base_cmd
    
    try:
        proc = await asyncio.create_subprocess_shell(final_cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        await proc.communicate()
    except Exception as e:
        print(f"❌ [Error] 录音出错: {e}", flush=True)

async def transcribe_audio_async(audio_path: str) -> str:
    if not os.path.exists(audio_path): return ""
    print(f"☁️ [系统] 正在识别...", end="\r", flush=True)
    async with aiohttp.ClientSession() as session:
        with open(audio_path, "rb") as f:
            try:
                async with session.post(WHISPER_SERVER_URL, data={"file": f}, timeout=120) as resp:
                    if resp.status == 200:
                        res = await resp.json()
                        text = res.get("text", "").strip()
                        detected = res.get("detected_lang", "unknown")
                        print(f"\n----- Whisper: {detected} -> {text}")
                        if text: return text
                    else:
                        print(f"\n❌ [Whisper] 错误: {resp.status}")
            except Exception as e: 
                print(f"\n❌ [网络] 连接失败: {e}")
    return ""

def remove_emojis(text: str) -> str:
    return re.sub(r"[^\w\s,.?!\u4e00-\u9fa5]+", "", text).strip()

async def call_qwen_async(user_text: str) -> str:
    global conversation_history
    conversation_history.append({"role": "user", "content": [{"text": user_text}]})
    if len(conversation_history) > MAX_HISTORY: conversation_history = conversation_history[-MAX_HISTORY:]
    messages = conversation_history + [{"role": "system", "content": [{"text": "回答简洁，50字以内。"}]}]
    
    start_t = time.time()
    try:
        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(executor, lambda: MultiModalConversation.call(model=QWEN_VL_MODEL, messages=messages))
        print(f"⏱️ [性能] Qwen 耗时: {time.time() - start_t:.2f}s")
        if response.status_code == 200:
            reply = remove_emojis(response.output.choices[0].message.content[0]["text"])
            conversation_history.append({"role": "assistant", "content": [{"text": reply}]})
            print(f"🤖 [Qwen]: {reply}")
            return reply
    except: pass
    return "脑子有点短路了。"

# 🔥 TTS 修复版
async def play_tts(text):
    if not text: return
    try:
        cred = credential.Credential(SECRET_ID, SECRET_KEY)
        client = tts_client.TtsClient(cred, "ap-shanghai")
        req = models.CreateTtsTaskRequest()
        req.Text = text; req.VoiceType = 501001; req.Codec = "pcm"; req.SampleRate = 16000; req.ModelType = 1
        
        def _get_url():
            r = client.CreateTtsTask(req); tid = r.Data.TaskId
            status_req = models.DescribeTtsTaskStatusRequest()
            status_req.TaskId = tid
            
            for _ in range(30): 
                s = client.DescribeTtsTaskStatus(status_req)
                if s.Data.Status == 2: return s.Data.ResultUrl
                elif s.Data.Status == 3: return None
                time.sleep(0.2)
            return None

        start_t = time.time()
        url = await asyncio.get_event_loop().run_in_executor(None, _get_url)
        if url:
            pcm_content = await asyncio.get_event_loop().run_in_executor(None, lambda: requests.get(url, timeout=30).content)
            with open(PCM_PATH, "wb") as f: f.write(pcm_content)
            print(f"⏱️ [性能] TTS 耗时: {time.time() - start_t:.2f}s")
            os.system(f"amixer -c 0 set Speaker 100% unmute >/dev/null 2>&1")
            subprocess.run(["aplay", "-q", "-D", SPK_DEVICE, "-f", "S16_LE", "-r", "16000", "-c", "1", PCM_PATH], stderr=subprocess.DEVNULL)
    except Exception as e: print(f"❌ TTS Error: {e}")

# ================= 6. 游戏逻辑 =================
async def play_game_logic():
    controller.set_mode("GAME")
    print("🎮 [Game] 猜拳开始")
    while True:
        display.set_expression(1)
        await play_tts("石头剪刀布,出拳吧！")
        user_choice = None
        timeout = time.time(); stable = time.time(); last_h = "None"
        while time.time() - timeout < 15.0:
            cur = controller.hand
            if cur in ["Paper", "Scissors", "Rock"]:
                if cur == last_h:
                    if time.time() - stable >= 1.5: user_choice = cur; break
                else: last_h = cur; stable = time.time()
            else: last_h = "None"; stable = time.time()
            await asyncio.sleep(0.1)
        
        if not user_choice: display.set_expression(8); await play_tts("没看清喔")
        else:
            ai = random.choice(['a', 'b', 'c'])
            display.set_hand(ai)
            cn = {"Paper":"布", "Scissors":"剪刀", "Rock":"石头"}.get(user_choice); await play_tts(f"你出{cn}"); await asyncio.sleep(2)
            u_code = {"Paper":"a", "Scissors":"b", "Rock":"c"}.get(user_choice)
            if u_code == ai: display.set_expression(4); await play_tts("平局，势均力敌呀！")
            elif (u_code=='a' and ai=='c') or (u_code=='b' and ai=='a') or (u_code=='c' and ai=='b'): display.set_expression(9); await play_tts("你赢了，真厉害呢！")
            else: display.set_expression(6); await play_tts("我赢了，你要继续加油呀！")

        await asyncio.sleep(1); await play_tts("还玩吗"); await record_audio(max_time=15)
        reply = await transcribe_audio_async(AUDIO_SAVE_PATH)
        if not reply or any(k in reply for k in ["不", "退", "停"]): await play_tts("拜拜"); break
        elif any(k in reply for k in ["好", "来", "ok"]): await play_tts("再来"); continue
        else: await play_tts("没听懂"); break
    controller.last_emotion_check = time.time(); controller.set_mode("IDLE")

# ================= 7. 🔥 主循环 (1分钟触发，30秒跟随) =================
async def emotion_task():
    await asyncio.sleep(5)
    print("\n⚡ [DEBUG] 首测启动...")
    controller.last_emotion_check = time.time() - 100

    while True:
        # 每60秒触发一次
        if controller.mode == "IDLE" and (time.time() - controller.last_emotion_check > 60):
            print("\n👀 [Emotion] 自动触发 (30秒实时跟随)...")
            controller.set_mode("EMOTION")
            
            last_face_sent = None
            
            # 🔥 映射表：确保 9 个表情都覆盖到
            # 1-兴奋, 2-右看, 3-左看, 4-眨眼, 5-愤怒, 6-不屑, 7-恐惧, 8-难过, 9-哭泣
            emo_map = {
                "Happy": 1,    # 兴奋
                "Angry": 5,    # 愤怒
                "Disgust": 6,  # 不屑
                "Fear": 7,     # 恐惧
                "Sad": 8,      # 难过
                "Surprise": 7, # 惊讶 -> 兴奋
                "Neutral": 4   # 中性 -> 眨眼
            }

            # 持续 30 秒 (300次 * 0.1s)
            for i in range(300): 
                if i % 10 == 0: print(".", end="", flush=True)
                
                current_face = controller.face
                
                # 如果检测到了人脸，并且表情变了，就立刻模仿
                if current_face not in ["None", "Face"] and current_face != last_face_sent:
                    eid = emo_map.get(current_face, 4) # 默认眨眼
                    display.set_expression(eid)
                    last_face_sent = current_face
                
                await asyncio.sleep(0.1)
            
            print("\n⏰ 跟随结束，恢复待机")
            controller.last_emotion_check = time.time()
            controller.set_mode("IDLE")
            
        await asyncio.sleep(1)

async def idle_task():
    while True:
        if controller.mode == "IDLE":
            await asyncio.sleep(random.randint(8, 15))
            # 待机时只用 2(右), 3(左), 4(眨眼)
            if controller.mode == "IDLE": display.set_expression(random.choice([2, 3, 4]))
        await asyncio.sleep(1)

async def main_loop():
    check_models()
    kill_alsa_hogs()
    asyncio.create_task(emotion_task())
    asyncio.create_task(idle_task())
    print("🚀 系统启动")
    os.system("amixer -c 0 set Speaker 100% unmute >/dev/null 2>&1")
    os.system("amixer -c 0 set Master 100% unmute >/dev/null 2>&1")

    while True:
        try:
            t1 = time.time(); await record_audio(max_time=None); t2 = time.time()
            if os.path.exists(AUDIO_SAVE_PATH) and os.path.getsize(AUDIO_SAVE_PATH) > 1024:
                print(f"⏱️ [性能] 录音耗时: {t2 - t1:.2f}s")
            
            t3 = time.time(); text = await transcribe_audio_async(AUDIO_SAVE_PATH); t4 = time.time()
            print(f"⏱️ [性能] Whisper耗时: {t4 - t3:.2f}s")

            if not text: await asyncio.sleep(1); continue
            
            if "猜拳" in text: await play_game_logic()
            elif "退出" in text: await play_tts("拜拜"); break
            else:
                reply = await call_qwen_async(text)
                display.set_expression(2); await play_tts(reply)
            print(f"⏱️ [性能] 总耗时: {time.time() - t1:.2f}s\n")
            
        except Exception as e:
            print(f"❌ Error: {e}"); await asyncio.sleep(1)

if __name__ == "__main__":
    if __name__ == "__main__": time.sleep(10) 
    threading.Thread(target=thread_camera_capture, daemon=True).start()
    threading.Thread(target=thread_ai_inference, daemon=True).start()
    threading.Thread(target=run_flask, daemon=True).start()
    asyncio.run(main_loop())