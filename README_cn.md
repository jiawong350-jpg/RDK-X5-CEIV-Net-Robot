# RDK-X5-CEIV-Net-Robot
Edge control &amp; multi-threading scheduling code for 'Zhiban Tongxin' on RDK X5. Features: 1. Asyncio &amp; ThreadPoolExecutor concurrent architecture; 2. Real-time face, emotion &amp; gesture tracking (C++ &amp; ONNX); 3. Edge-cloud integration (Qwen-VL, Whisper, Tencent TTS); 4. Serial adaptive expression mirroring &amp; Rock-Paper-Scissors game logic.

# 智伴童心：基于 RDK X5 的具身陪伴机器人与多模态双系统情感计算优化

本项目为全国嵌入式芯片与系统设计竞赛作品——**「智伴童心」具身陪伴机器人**的边缘端智能计算主控调度中枢与完整测试套件开源仓库。系统局限于家庭全景陪伴与基层心理监护场景，基于地平线 RDK X5 边缘计算大脑与多核异构硬件平台，实现了集“高效视觉感知、端云协同双语语音、大模型认知与物理串口具身控制”于一体的系统级流控闭环。

---

## 🚀 核心技术特性

本边缘端核心调度中枢（以 `run_system.py` 为骨干）具备以下四大工程级设计亮点：

1. **多线程异步高并发调度架构 (Concurrent Execution)**
   * 底层采用 Python `asyncio` 协程异步事件循环与 `ThreadPoolExecutor` 线程池相结合的混合并发控制流。
   * 物理隔离并平行调度 Camera 原始帧高频捕获（30FPS 吞吐）、本地 AI 引擎高并发推理、Flask 实时看护推流服务器以及主交互监听循环，在整机多维数据高速无阻塞流转的状态下实现了线程零死锁。

2. **端侧轻量化多模态感知追踪 (Edge ML Inference)**
   * 端侧深度集成 OpenCV DNN 与 C++ 推理网络封装（对接本地 `rdk_ai_module` 动态计算引擎）。
   * 采用模式热切换机制，支持单目高召回率人脸边界框捕获、7 种面部表态情感实时解算（结合防抖队列提纯概率真值）以及手势目标实时状态机提取。

3. **端云协同高阶语义认知闭环 (Edge-Cloud Collaboration)**
   * **听 (VAD & ASR)**：联动 ALSA 架构，通过 `arecord` 与 `sox` 过滤居家噪声环境并进行轻量化语音激活捕获，流式对接云端异步 **Whisper 语义识别服务器**进行毫秒级文字解算。
   * **思 (LLM)**：基于 DashScope 异步并发调用 **Qwen-VL-Flash 多模态大模型**，内置高密度上下文历史管理，控制全链路响应端到端延迟低至 1.45 秒。
   * **说 (TTS)**：无缝挂载**腾讯云在线语音合成 (TTS)** 异步长任务，动态拉取 PCM 原始音频流并通过底层音频总线实现流畅的拟人化语言对答。

4. **串口具身动作随动与博弈逻辑 (Embodied Interaction)**
   * 抽象出 `SerialDisplayManager` 总线控制器，通过 `/dev/ttyS1` 串口链路（波特率 115200）实时驱动视听外设表现层屏幕。
   * **主动跟随镜像任务 (`emotion_task`)**：每隔 60 秒自动触发一次长达 30 秒的面部微表情随动追踪，将用户的真实显性情绪瞬时映射为表现层屏幕的 9 种拟人眼神与微表情，实现深度具身共情。
   * **人机猜拳具身博弈 (`play_game_logic`)**：内置了完整的语音引导、手势视觉意图 Token 化识别、随机算法对决及串口动作反馈的“石头剪刀布”独立小游戏闭环。
  
---

## 🛠️ 系统自检与单模块硬件诊断工具

为保障多核异构平台在全负载复杂场景下的稳定运行，本仓库在根目录下部署了完备的硬件外设冒烟测试与系统级诊断矩阵，请在主控拉起前分步执行：

### 1. 麦克风硬件与云端识别诊断 (`test_audio.py`)

**功能定位**：专门排查本地 Linux 音频死锁占用、麦克风 ADC 采样率缺陷以及内网穿透隧道稳定性。

**诊断机理**：

**【进程清理】**：自动调用 `fuser -k /dev/snd/*` 强行拦截并清理 ALSA 声卡由于死锁抢占导致的僵尸进程。

**【底层录音】**：绕过流式 VAD（静音检测），强制拉起 `arecord` 在底层捕获 5 秒原始音频并进行文件边界保护校验（防物理输入空载）。

**【网络穿透】**：采用同步阻塞链发起 `requests.post` 直连远程 Whisper API 服务器，精确透传测试 ngrok 内网隧道的连通性、吞吐量和在线文字转译回执耗时。

---

### 2. 本地表情计算网络与串口显示联动诊断 (`test_face.py`)

**功能定位**：验证本地 ONNX 表情网络在边缘算力下的推理时延，联调测试物理串口屏的动态控制帧。

**诊断机理**：

**【核心模式】**：多线程驱动 `srcampy.Camera` 捕获原生图像，加载 `rdk_ai_module.VideoPipeline` 并强制锁定为 Mode 1（面部感知模式）。

**【串口联动】**：将解算出的 7 种英文情绪极性映射为表现层屏幕所需控制字符（如 `Happy -> "1"`, `Sad -> "8"`），高频透传给 `/dev/ttyS1` 串口总线，并附带表情状态相异性判定（防串口刷屏死锁）。

**【推流可视化】**：本地挂载 Flask 服务器，在远程画面中为不同情绪绘制彩色动态框（Angry 绘制红框、Happy 绘制黄框），直观监测防抖平均时延 (ms)。

---

### 3. 本地手势目标跟踪与防抖博弈诊断 (`test_hand.py`)

**功能定位**：强制锁定手势目标推理流，联调验证“人机猜拳”具身小游戏的物理执行稳定性。

**诊断机理**：

**【手势检测】**：强制系统加载 C++ 引擎的 Mode 2（手势状态检测模式），调用本地高性能 `v5n.onnx` 进行目标检测。

**【指令映射】**：映射 `Paper -> "a"`, `Scissors -> "b"`, `Rock -> "c"` 猜拳控制字符并直连串口驱动外设。

**【总线防抖】**：硬件级状态防抖：内置 `last_sent_label` 状态锁，只有当在变暗或大电流高冲击环境下捕捉到有效且与上一帧完全不同的稳定手势时，才允许向串口发送数据，彻底解决了由于抢占总线导致的系统死锁崩溃。

---

### 4. 边缘算力极限吞吐推流压力测试 (`test_vision_only.py`)

**功能定位**：High Performance / High Recall 极端环境测试。在不连接任何物理表现层串口屏的环境下，极限测试系统多线程异步 I/O 的最高并发吞吐上限。

**诊断机理**：

**【零拷贝同步】**：基于 `data_lock` 线程锁建立高安全的物理内存分配通道，实现高频 NV12 图像帧在 Camera 线程与 AI 工作线程间的零拷贝同步流转。

**【算力反哺】**：优化了 Flask 视频推流线程的微秒级休眠参数 (`time.sleep(0.03)`)，在保障远程 PC 端监护上位机 30FPS 流畅视轨的前提下，将绝大部分 CPU 算力反哺给 C++ 深度学习硬件流水线，防止内核发生因资源抢占而造成的系统崩溃。

---

## 🛠️ 部署与运行指南

### 1. 环境配置与安全脱敏说明

为了保障个人云端算力资产安全，本开源仓库中的核心代码已进行了全链路的**安全脱敏处理（De-sensitization）**。若要在本地实际复现或拉起完整系统，请在运行前使用编辑器打开 `run_system.py`，并将以下变量的占位符手动修改为您个人的有效凭证：

```python
# 1. 本地 Whisper 语音识别服务端通信接口
WHISPER_SERVER_URL = "填写您本地部署的 Whisper 服务器公网穿透/局域网真实地址"

# 2. 地平线/阿里云 DashScope 大模型鉴权配置
DASH_API_KEY = "填写您个人的阿里云 DashScope 真实密钥"

# 3. 腾讯云在线语音合成 (TTS) 鉴权配置
SECRET_ID = "填写您个人的腾讯云 SECRET_ID"
SECRET_KEY = "填写您个人的腾讯云 SECRET_KEY"
```
---

### 2. 本地依赖与模型就绪校验

系统内置了严格的模型完整性哨兵检测机制（`check_models`）。启动前请确保在 `Vision__python_Cpp/models/` 目录下嵌套放置了相应的量化编译权重文件：

```bash
python3 -c "from run_system import check_models; check_models()"
```
---

### 3. 拉起主系统闭环

对于语音交互系统的调试，可直接激活已经完成环境配置的隔离 Python 沙盒环境单独测试语音交互内核：

```bash
cd Voise_Subsystem
source bin/activate
python3 code/bot\ test.py
```

当单模块硬件诊断均通过后，退出虚拟环境，使用 sudo 权限在全局异构高频时钟域下直接拉起多线程核心主控中枢：

```bash
sudo python3 run_system.py
```

## 📂 项目多级级联嵌套目录树 (Directory Tree)

项目的模块化设计遵循机电一体化与软硬件解耦工程规范，各文件夹嵌套与依赖树状结构如下：

```text
MULTIMODAL_INTERACTION_SYSTEM/         # 具身智能多模态交互系统根目录
│
├── run_system.py                     # 🚀 核心：边缘端多线程主控调度中枢（总体流控闭环）
├── test_audio.py                     # 诊断：硬件音频录制、声卡清理与远程 Whisper 网络连通性测试 
├── test_face.py                      # 诊断：边缘端面部表情本地推理与串口随动镜像联调测试
├── test_hand.py                      # 诊断：边缘端手势目标识别与猜拳防抖串口执行链测试 
├── test_vision_only.py               # 压力测试：高性能/高召回率纯视觉算法流多线程无阻塞推流调测
│
├── 📂 Vision__python_Cpp/             # 1. Python-C++ 混合视觉推理封装包
│   ├── build/                        # C++ 编译生成的本地二进制动态库目标文件
│   ├── include/                      # 视觉推理核心 C++ 头文件
│   ├── models/                       # 本地深度学习权重（包含面部检测、情感网络与手势 ONNX/XML 权重）
│   │   ├── haarcascade_frontalface_default.xml
│   │   ├── emotion.onnx
│   │   └── v5n.onnx
│   ├── src/                          # 核心视觉计算与特征提纯 C++ 源码 
│   ├── CMakeLists.txt                # 本地 C++ 推理引擎 CMake 构建配置文件
│   └── main.py                       # 视觉混合封装包本地单体功能调试入口
│
├── 📂 Vision_Subsystem/               # 2. 纯 C++ 底层视觉处理子系统
│   ├── build/                        # 底层 C++ 编译构建缓存
│   ├── include/                      # 基础图像处理与格式转换头文件
│   ├── src/                          # 底层视觉基础处理源码
│   ├── CMakeLists.txt                # 底层子系统构建管理配置文件
│   └── main.cpp                      # 纯 C++ 视觉子系统主程序入口 
│
└── 📂 Voice_Subsystem/                # 3. 语音系统主程序运行期流数据中转站 
    └── code/
        └── audio/ 
            └── live.flac              # 主循环调度运行期间的实时流式录音高频暂存音轨文件
```

