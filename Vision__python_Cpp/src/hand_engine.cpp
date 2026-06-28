#include "../include/hand_engine.h"
#include <iostream>
#include <cmath>

using namespace cv;
using namespace cv::dnn;
using namespace std;

HandEngine::HandEngine(const string& onnxPath) {
    // 1. 开启多线程加速 (RDK X5 核心优化)
    cv::setNumThreads(4);

    handNet = readNetFromONNX(onnxPath);
    if (handNet.empty()) {
        cerr << "❌ HandEngine: 模型加载失败 " << onnxPath << endl;
    } else {
        handNet.setPreferableBackend(DNN_BACKEND_OPENCV);
        handNet.setPreferableTarget(DNN_TARGET_CPU);
    }
    classNames = {"Paper", "Rock", "Scissors"};
    colors = {Scalar(0, 255, 0), Scalar(0, 0, 255), Scalar(255, 0, 0)};
}

void HandEngine::letterbox(const Mat& src, Mat& dst, float& ratio, float& dw, float& dh) {
    int shape_w = src.cols;
    int shape_h = src.rows;
    float r = min((float)INPUT_SIZE / shape_h, (float)INPUT_SIZE / shape_w);
    
    int new_unpad_w = int(round(shape_w * r));
    int new_unpad_h = int(round(shape_h * r));
    
    dw = (INPUT_SIZE - new_unpad_w) / 2.0f;
    dh = (INPUT_SIZE - new_unpad_h) / 2.0f;

    if (shape_w != new_unpad_w || shape_h != new_unpad_h)
        // 使用线性插值，保证手势特征清晰
        resize(src, dst, Size(new_unpad_w, new_unpad_h), 0, 0, INTER_LINEAR);
    else
        dst = src.clone();

    int top = int(round(dh - 0.1f));
    int bottom = int(round(dh + 0.1f));
    int left = int(round(dw - 0.1f));
    int right = int(round(dw + 0.1f));
    
    copyMakeBorder(dst, dst, top, bottom, left, right, BORDER_CONSTANT, Scalar(114, 114, 114));
    ratio = r;
}

vector<HandResult> HandEngine::detect(const Mat& frame) {
    vector<HandResult> results;
    if (frame.empty()) return results;

    Mat model_input;
    float ratio, dw, dh;
    letterbox(frame, model_input, ratio, dw, dh);

    Mat blob;
    blobFromImage(model_input, blob, 1.0/255.0, Size(), Scalar(), true, false);
    handNet.setInput(blob);

    vector<String> outNames = handNet.getUnconnectedOutLayersNames();
    Mat outputs = handNet.forward(outNames[0]);

    float* data = (float*)outputs.data;
    int rows = outputs.size[1]; 
    int dimensions = outputs.size[2];

    vector<int> class_ids;
    vector<float> confidences;
    vector<Rect> boxes;
    
    class_ids.reserve(20);
    confidences.reserve(20);
    boxes.reserve(20);

    for (int i = 0; i < rows; ++i) {
        float* row_data = data + (i * dimensions);
        float confidence = row_data[4]; // Objectness (是物体的概率)

        // 🔥 还原你提供的“流畅版”逻辑：
        // 先看是不是物体，再看是什么分类，不使用乘积
        if (confidence >= CONF_THRES) {
            float* classes_scores = row_data + 5;
            Mat scores(1, (int)classNames.size(), CV_32FC1, classes_scores);
            Point class_id;
            double maxClassScore;
            minMaxLoc(scores, 0, &maxClassScore, 0, &class_id);

            // 只有当分类置信度也达标时才通过
            if (maxClassScore > CONF_THRES) {
                float cx = row_data[0];
                float cy = row_data[1];
                float w = row_data[2];
                float h = row_data[3];

                float x = (cx - dw) / ratio;
                float y = (cy - dh) / ratio;
                float width = w / ratio;
                float height = h / ratio;

                int left = int(x - width / 2);
                int top = int(y - height / 2);

                class_ids.push_back(class_id.x);
                // 存入 confidence 而不是 score，这也还原了你的代码逻辑
                confidences.push_back(confidence); 
                boxes.push_back(Rect(left, top, (int)width, (int)height));
            }
        }
    }

    vector<int> nms_result;
    NMSBoxes(boxes, confidences, CONF_THRES, NMS_THRES, nms_result);

    for (int idx : nms_result) {
        results.push_back({boxes[idx], classNames[class_ids[idx]], colors[class_ids[idx]]});
    }
    return results;
}