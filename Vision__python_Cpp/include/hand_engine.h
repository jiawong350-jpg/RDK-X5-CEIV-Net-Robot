#ifndef HAND_ENGINE_H
#define HAND_ENGINE_H

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <vector>
#include <string>

struct HandResult {
    cv::Rect box;
    std::string label;
    cv::Scalar color;
};

class HandEngine {
public:
    HandEngine(const std::string& onnxPath);
    std::vector<HandResult> detect(const cv::Mat& frame);

private:
    void letterbox(const cv::Mat& src, cv::Mat& dst, float& ratio, float& dw, float& dh);

    cv::dnn::Net handNet;
    std::vector<std::string> classNames;
    std::vector<cv::Scalar> colors;

    // 🔥🔥 关键修改：必须改成 320！🔥🔥
    // 因为你现在的 v5n.onnx 是 320 版本
    const int INPUT_SIZE = 320; 
    
    // 保持低阈值，方便识别
    const float CONF_THRES = 0.10f; 
    const float NMS_THRES = 0.45f;
};

#endif // HAND_ENGINE_H