#ifndef FACE_ENGINE_H
#define FACE_ENGINE_H

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <vector>
#include <string>

// 定义面部识别结果结构体
struct FaceResult {
    cv::Rect box;
    std::string label;
    float confidence; // 之前报错缺这个，现在加上
    cv::Scalar color;
};

class FaceEngine {
public:
    FaceEngine(const std::string& faceXml, const std::string& emotionOnnx);
    std::vector<FaceResult> detect(const cv::Mat& frame);

private:
    cv::CascadeClassifier faceCascade;
    cv::dnn::Net emotionNet;
    
    // 统一变量名，防止报错
    std::vector<std::string> emotionLabels; 
};

#endif // FACE_ENGINE_H