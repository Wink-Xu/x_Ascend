/**
*
* File main.cpp
* Description: Sample of body_genderAge using lib dnn_ascend
* Input: One body image
* Output: Gender and Age of this body
*
*/

#include <iostream>
#include <stdlib.h>
#include <dirent.h>

#include "classify_process.h"
#include "utils.h"
using namespace std;

namespace {
uint32_t kModelWidth = 224;
uint32_t kModelHeight = 224;
const char* kModelPath = "../model/googlenet.om";
}

int main(int argc, char *argv[]) {
    //Check the input when the application is executed, the program execution requires the input of picture directory parameters
    if((argc < 2) || (argv[1] == nullptr)){
        ERROR_LOG("Please input: ./main <image_dir>");
        return FAILED;
    }
    //Instantiate the classification reasoning object, the parameter is the classification model path, the width and height of the model input requirements
    ClassifyProcess classify(kModelPath, kModelWidth, kModelHeight);
    //Initialize the acl resources, models and memory for classification inference
    Result ret = classify.Init();
    if (ret != SUCCESS) {
        ERROR_LOG("Classification Init resource failed");
        return FAILED;
    }
    //Get all the image file names in the image directory
    string inputImageDir = string(argv[1]);
    vector<string> fileVec;
    Utils::GetAllFiles(inputImageDir, fileVec);
    if (fileVec.empty()) {
        ERROR_LOG("Failed to deal all empty path=%s.", inputImageDir.c_str());
        return FAILED;
    }
    //Reasoning picture by picture
    for (string imageFile : fileVec) {
        //Preprocess the picture: read the picture and zoom the picture to the size required by the model input
        Result ret = classify.Preprocess(imageFile);
        if (ret != SUCCESS) {
            ERROR_LOG("Read file %s failed, continue to read next",
                      imageFile.c_str());                
            continue;
        }
        //Send the preprocessed pictures to the model for inference and get the inference results
        aclmdlDataset* inferenceOutput = nullptr;
        ret = classify.Inference(inferenceOutput);
        if ((ret != SUCCESS) || (inferenceOutput == nullptr)) {
            ERROR_LOG("Inference model inference output data failed");
            return FAILED;
        }
        //Analyze the inference output and mark the object category obtained by the inference on the picture
        ret = classify.Postprocess(imageFile, inferenceOutput);
        if (ret != SUCCESS) {
            ERROR_LOG("Process model inference output data failed");
            return FAILED;
        }
    }

    INFO_LOG("Execute sample success");
    return SUCCESS;
}
