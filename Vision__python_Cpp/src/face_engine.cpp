#include "../include/face_engine.h"
#include <iostream>
#include <vector>

using namespace cv;
using namespace std;

// 辅助函数：Gamma 矫正 (定义在类外部，作为全局函数)
void adjustGamma(Mat& image, double gamma) {
    Mat lookUpTable(1, 256, CV_8U);
    uchar* p = lookUpTable.ptr();
    for( int i = 0; i < 256; ++i)
        p[i] = saturate_cast<uchar>(pow(i / 255.0, gamma) * 255.0);
    LUT(image, image, lookUpTable);
}

FaceEngine::FaceEngine(const string& faceXml, const string& emotionOnnx) {
    // 加载 Haar
    if (!faceCascade.load(faceXml)) {
        cerr << "❌ [FaceEngine] Load Error: " << faceXml << endl;
    }

    // 加载 ONNX
    try {
        emotionNet = dnn::readNetFromONNX(emotionOnnx);
        emotionNet.setPreferableBackend(dnn::DNN_BACKEND_OPENCV);
        emotionNet.setPreferableTarget(dnn::DNN_TARGET_CPU);
    } catch (const cv::Exception& e) {
        cerr << "❌ [FaceEngine] ONNX Error: " << e.what() << endl;
    }

    // 初始化标签 (与头文件变量名一致)
    emotionLabels = {"Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"};
}

std::vector<FaceResult> FaceEngine::detect(const cv::Mat& frame) {
    std::vector<FaceResult> results;
    if (frame.empty() || faceCascade.empty()) return results;

    Mat gray;
    cvtColor(frame, gray, COLOR_BGR2GRAY);
    equalizeHist(gray, gray); 
    
    // 调用辅助函数
    adjustGamma(gray, 1.5); 

    std::vector<Rect> faces;
    // 🔥 灵敏度优化：minNeighbors = 3
    faceCascade.detectMultiScale(gray, faces, 1.1, 3, 0, Size(30, 30));

    if (faces.empty()) return results;

    // 找最大的脸
    Rect largest;
    int maxArea = 0;
    for (const auto& r : faces) {
        if (r.area() > maxArea) {
            maxArea = r.area();
            largest = r;
        }
    }

    // 比例过滤
    float ratio = (float)largest.height / (float)largest.width;
    if (ratio < 0.6 || ratio > 2.0) return results;

    // 扩大框 (ROI)
    int pad_w = largest.width * 0.15;
    int pad_h = largest.height * 0.15;
    int x1 = max(0, largest.x - pad_w);
    int y1 = max(0, largest.y - pad_h);
    int x2 = min(frame.cols, largest.x + largest.width + pad_w);
    int y2 = min(frame.rows, largest.y + largest.height + pad_h);
    
    Rect enlarged_box(x1, y1, x2 - x1, y2 - y1);
    Mat faceROI = gray(enlarged_box);

    // 推理
    Mat blob;
    dnn::blobFromImage(faceROI, blob, 1.0, Size(48, 48), Scalar(), false, false);
    emotionNet.setInput(blob);
    Mat prob = emotionNet.forward();

    Point classIdPoint;
    double confidence;
    minMaxLoc(prob, 0, &confidence, 0, &classIdPoint);
    int classId = classIdPoint.x;

    FaceResult res;
    res.box = largest; // 返回原始框
    
    if (classId >= 0 && classId < emotionLabels.size()) {
        res.label = emotionLabels[classId];
        res.confidence = (float)confidence; // 赋值
    } else {
        res.label = "Unknown";
    }
    
    results.push_back(res);
    return results;
}