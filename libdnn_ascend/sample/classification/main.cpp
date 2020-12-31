/**
* File main.cpp
* Description: 
*/
#include <iostream>
#include <vector>
#include <string>

#include "opencv2/opencv.hpp"
#include "opencv2/imgcodecs/legacy/constants_c.h"
#include "opencv2/imgproc/types_c.h"


#include <iostream>
#include <stdlib.h>
#include <dirent.h>

#include "dnn_ascend.h"
#include "imageNetClasses.h"

#define INFO_LOG(fmt, args...) fprintf(stdout, "[INFO]  " fmt "\n", ##args)
#define WARN_LOG(fmt, args...) fprintf(stdout, "[WARN]  " fmt "\n", ##args)
#define ERROR_LOG(fmt, args...) fprintf(stdout, "[ERROR]  " fmt "\n", ##args)

using namespace std;

namespace {
uint32_t kModelWidth = 224;
uint32_t kModelHeight = 224;
const char* kModelPath = "./model/googlenet.om";
}


void LabelClassToImage(int classIdx, const string& origImagePath) {
    cv::Mat resultImage = cv::imread(origImagePath, CV_LOAD_IMAGE_COLOR);

    // generate colorized image
    int pos = origImagePath.find_last_of("/");
    string filename(origImagePath.substr(pos + 1));
    stringstream sstream;
    sstream.str("");
    sstream << "./output/out_"  << filename;

    string outputPath = sstream.str();
    string text;

    if (classIdx < 0 || classIdx >= IMAGE_NET_CLASSES_NUM) {
        text = "none";
    } else {
        text = kStrImageNetClasses[classIdx];
    }

    int fontFace = 0;
    double fontScale = 1;
    int thickness = 2;
    int baseline;
    cv::Point origin;
    origin.x = 10;
    origin.y = 50;
    cv::putText(resultImage, text, origin, fontFace, fontScale, cv::Scalar(0, 255, 255), thickness, 4, 0);
    cv::imwrite(outputPath, resultImage);
}

int main() {

    const string imageFile = "./data/dog1_1024_683.jpg";

    INFO_LOG("Read image %s", imageFile.c_str());
    cv::Mat origMat = cv::imread(imageFile, CV_LOAD_IMAGE_COLOR);
    if (origMat.empty()) {
        ERROR_LOG("Read image failed");
        return 1;
    }

    INFO_LOG("Resize image %s", imageFile.c_str());
    //resize
    cv::Mat reiszeMat;
    cv::resize(origMat, reiszeMat, cv::Size(kModelWidth, kModelHeight));
    if (reiszeMat.empty()) {
        ERROR_LOG("Resize image failed");
        return 1;
    }

    unsigned char *in_data = reiszeMat.ptr<uint8_t>();

    // for (unsigned int c = 0; c < 3; c++)
    // {
    //     for (unsigned int h = 0; h < kModelHeight; h++)
    //     {
    //         for (unsigned int w = 0; w < kModelWidth; w++)
    //         {
    //             in_data[c * kModelHeight * kModelWidth + h * 3 + w] = reiszeMat.at<cv::Vec3b>(h, w)[c];
    //         }
    //     }
    // }
    
   // #printf("%f, %f, %f", in_data[0], in_data[1], in_data[2]);
    float *data_out = (float*)malloc(1 * 1000 * sizeof(float));
    //Instantiate the classification reasoning object, the parameter is the classification model path, the width and height of the model input requirements
    aclRegisterModel(kModelPath);
    //Initialize the acl resources, models and memory for classification inference

    aclForward(&in_data, &data_out);

    float *temp_dataout = data_out;
    map<float, unsigned int, greater<float> > resultMap;
    for (uint32_t j = 0; j < 1000; ++j) {
        resultMap[*data_out] = j;
        data_out++;
    }

    int maxScoreCls = -1;
    float maxScore = 0;
    int cnt = 0;
    for (auto it = resultMap.begin(); it != resultMap.end(); ++it) {
        // print top 5
        if (++cnt > 5) {
            break;
        }
        INFO_LOG("top %d: index[%d] value[%lf]", cnt, it->second, it->first);

        if (it->first > maxScore) {
            maxScore = it->first;
            maxScoreCls = it->second;
        }
    }

    LabelClassToImage(maxScoreCls, imageFile);

 
    free(temp_dataout);

    aclUnregisterModel();
    INFO_LOG("Execute sample success");

    return 0;

}
