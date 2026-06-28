#include <pybind11/pybind11.h>
#include <pybind11/stl.h> 
#include <opencv2/opencv.hpp>
#include <vector>
#include <deque>
#include <map>
#include <iostream>
#include "../include/face_engine.h"
#include "../include/hand_engine.h"

namespace py = pybind11;
using namespace cv;
using namespace std;

// 投票算法：从队列中选出出现次数最多的标签
std::string get_stable_result(const std::deque<std::string>& queue, int min_valid_frames) {
    if (queue.empty()) return "None";
    
    std::map<std::string, int> counts;
    int valid_count = 0;

    for (const auto& label : queue) {
        if (label != "None") {
            counts[label]++;
            valid_count++;
        }
    }

    // 如果有效帧数太少（比如全是 None），则认为没识别到
    if (valid_count < min_valid_frames) return "None";

    std::string best_label = "None";
    int max_count = 0;
    
    for (const auto& pair : counts) {
        if (pair.second > max_count) {
            max_count = pair.second;
            best_label = pair.first;
        }
    }
    return best_label;
}

class VideoPipeline {
private:
    unique_ptr<FaceEngine> face_engine;
    unique_ptr<HandEngine> hand_engine;

    // === 两个独立的防抖队列 ===
    std::deque<std::string> hand_queue; // 手势队列
    std::deque<std::string> face_queue; // 表情队列

    const int HAND_QUEUE_SIZE = 5;      // 手势需要更稳，设为5帧
    const int FACE_QUEUE_SIZE = 3;      // 表情需要反应快一点，设为3帧

    // 缓存上一次的框，防止闪烁时框消失
    std::vector<int> last_hand_box;
    std::vector<int> last_face_box;

public:
    VideoPipeline(const std::string& face_xml, const std::string& emotion_model, const std::string& hand_model) {
        face_engine = std::unique_ptr<FaceEngine>(new FaceEngine(face_xml, emotion_model));
        hand_engine = std::unique_ptr<HandEngine>(new HandEngine(hand_model));
    }

    py::tuple process_frame(py::bytes input_data, int mode) {
        std::string s_data = input_data;
        string final_face = "None";
        string final_hand = "None";
        std::vector<int> box_out; // 输出给 Python 画图用的框
        
        vector<uchar> buf; 

        {
            py::gil_scoped_release release; // 释放 GIL 锁，允许 Python 多线程并发
            
            size_t len = s_data.size();
            const uchar* ptr = (const uchar*)s_data.data();
            Mat frame_bgr;

            // NV12 解码
            if (len == 460800) { 
                Mat raw(720, 640, CV_8UC1, (void*)ptr); 
                cvtColor(raw, frame_bgr, COLOR_YUV2BGR_NV12); 
            } else {
                 return py::make_tuple(py::bytes(), "Err", "Err", box_out);
            }

            if (!frame_bgr.empty()) {
                // 底层翻转：保证 AI 视角与屏幕显示一致
                cv::flip(frame_bgr, frame_bgr, -1); 

                // ========================================================
                // 模式 2: 手势识别 (Hand Mode) - 保持原有高稳定性逻辑
                // ========================================================
                if (mode == 2) { 
                    face_queue.clear(); // 清空表情队列
                    
                    auto hands = hand_engine->detect(frame_bgr);
                    string current_label = "None";
                    
                    if (!hands.empty()) {
                        current_label = hands[0].label;
                        // 只要检测到，就更新坐标缓存
                        last_hand_box = {hands[0].box.x, hands[0].box.y, hands[0].box.width, hands[0].box.height};
                    }
                    
                    // 入队防抖
                    hand_queue.push_back(current_label);
                    if (hand_queue.size() > HAND_QUEUE_SIZE) hand_queue.pop_front();
                    
                    // 获取稳定结果 (要求5帧里至少有2帧有效)
                    final_hand = get_stable_result(hand_queue, 2);
                    
                    // 如果稳定结果有效，且有坐标缓存，就输出框
                    if (final_hand != "None" && !last_hand_box.empty()) {
                        box_out = last_hand_box;
                    }
                } 
                
                // ========================================================
                // 模式 1: 面部识别 (Face Mode) - 优化版：实时框 + 标签平滑
                // ========================================================
                else if (mode == 1) { 
                    hand_queue.clear(); // 清空手势队列
                    
                    auto faces = face_engine->detect(frame_bgr);
                    string current_label = "None";

                    if (!faces.empty()) {
                        current_label = faces[0].label;
                        
                        // 🔥 重点优化：只要检测到脸，立马更新坐标！
                        // 这样保证框是实时的，不会滞后
                        last_face_box = {faces[0].box.x, faces[0].box.y, faces[0].box.width, faces[0].box.height};
                        
                        // 🔥 重点优化：只要检测到脸，哪怕标签还在计算防抖，也先把框画出来
                        box_out = last_face_box; 
                    }

                    // 标签依然走防抖流程，防止表情乱跳
                    face_queue.push_back(current_label);
                    if (face_queue.size() > FACE_QUEUE_SIZE) face_queue.pop_front();
                    
                    // 获取稳定表情 (要求3帧里至少有1帧有效，反应更快)
                    final_face = get_stable_result(face_queue, 1);
                    
                    // 如果当前没检测到脸，但刚才有脸，且防抖结果还没消失
                    // 这种情况可以把旧的框发出去，防止框闪烁（可选，这里我保留了最实时的逻辑）
                    if (box_out.empty() && final_face != "None" && !last_face_box.empty()) {
                        // box_out = last_face_box; // 如果你想让框也粘滞一点，可以取消这行注释
                    }
                }
            }
        } 

        return py::make_tuple(
            py::bytes((char*)buf.data(), buf.size()), 
            final_face, 
            final_hand,
            box_out 
        );
    }
};

PYBIND11_MODULE(rdk_ai_module, m) {
    py::class_<VideoPipeline>(m, "VideoPipeline")
        .def(py::init<const std::string&, const std::string&, const std::string&>())
        .def("process_frame", &VideoPipeline::process_frame);
}