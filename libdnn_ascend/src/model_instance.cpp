/**
* File model_instance.cpp
* Description: handle acl resource
*/
#include "model_instance.h"
#include <iostream>

#include "acl/acl.h"
#include "model_manager.h"
#include "utils.h"

using namespace std;


ModelInstance::ModelInstance()
:deviceId_(0), context_(nullptr), stream_(nullptr), isInited_(false){
}

ModelInstance::~ModelInstance() {
  //  DestroyResource();
}

Result ModelInstance::loadModel(const char* model_path)
{
    if (isInited_) {
    INFO_LOG("Classify instance is initied already!");
    return SUCCESS;
    }

    Result ret = InitResource();
    if (ret != SUCCESS) {
        ERROR_LOG("Init acl resource failed");
        return FAILED;
    }

    ret = InitModel(model_path);
    if (ret != SUCCESS) {
        ERROR_LOG("Init model failed");
        return FAILED;
    }

    isInited_ = true;
    return SUCCESS;
}

Result ModelInstance::unloadModel()
{
    model_.Unload();
    model_.DestroyDesc();
    model_.DestroyInput();
    model_.DestroyOutput();
    DestroyResource();
    return SUCCESS;
}

Result ModelInstance::inference(unsigned char **data_in, float **data_out)
{
    uint32_t inputDataSize_;
    
    void *inputBuf_ = nullptr;

    model_.GetModelInputInfo(inputDataSize_, &inputBuf_);
// googlenet.om
    if(inputDataSize_ == 1505280)
        inputDataSize_ = 150528;

    if (runMode_ == ACL_HOST) {     
        //When running on AI1, you need to copy the image data to the device side   
        aclError ret_ = aclrtMemcpy(inputBuf_, inputDataSize_, 
                                   (*data_in), inputDataSize_,
                                   ACL_MEMCPY_HOST_TO_DEVICE);
        if (ret_ != ACL_ERROR_NONE) {
            ERROR_LOG("Copy resized image data to device failed.");
            return FAILED;
        }
    } else {
        //When running on Atals200DK, the data can be copied locally.
        //reiszeMat is a local variable, the data cannot be transferred out of the function, it needs to be copied
        memcpy(inputBuf_, (*data_in), inputDataSize_);
    }   

    Result ret = model_.Execute();
    if (ret != SUCCESS) {
        ERROR_LOG("Execute model inference failed");
        return FAILED;
    }
    aclmdlDataset* inferenceOutput = model_.GetModelOutputData();

    uint32_t dataSize = 0;
    void* data = GetInferenceOutputItem(dataSize, inferenceOutput);
    if (data == nullptr) return FAILED;

    
    (*data_out) = reinterpret_cast<float*>(data);


    return SUCCESS;

}


Result ModelInstance::InitResource() {
    // ACL init
    
    aclError ret = aclInit(NULL); //  is able to pass acl.json
    if (ret != ACL_ERROR_NONE) {
        ERROR_LOG("Acl init failed");
        return FAILED;
    }
    INFO_LOG("Acl init success");

    // open device
    ret = aclrtSetDevice(deviceId_);
    if (ret != ACL_ERROR_NONE) {
        ERROR_LOG("Acl open device %d failed", deviceId_);
        return FAILED;
    }
    INFO_LOG("Open device %d success", deviceId_);

    ret = aclrtCreateContext(&context_, deviceId_);
    if (ret != ACL_ERROR_NONE) {
        ERROR_LOG("Acl open context failed");
        return FAILED;
    }
    ret = aclrtCreateStream(&stream_);
    INFO_LOG("Create device success" );

    if (ret != ACL_ERROR_NONE) {
        ERROR_LOG("Acl open stream failed");
        return FAILED;
    }
    INFO_LOG("Create stream success" );

    ret = aclrtGetRunMode(&runMode_);
    if (ret != ACL_ERROR_NONE) {
        ERROR_LOG("acl get run mode failed");
        return FAILED;
    }

    return SUCCESS;
}

Result ModelInstance::InitModel(const char* omModelPath) {
    Result ret = model_.LoadModelFromFileWithMem(omModelPath);
    if (ret != SUCCESS) {
        ERROR_LOG("execute LoadModelFromFileWithMem failed");
        return FAILED;
    }

    ret = model_.CreateDesc();
    if (ret != SUCCESS) {
        ERROR_LOG("execute CreateDesc failed");
        return FAILED;
    }

    ret = model_.CreateOutput();
    if (ret != SUCCESS) {
        ERROR_LOG("execute CreateOutput failed");
        return FAILED;
    }

    ret = model_.CreateInput();
    if (ret != SUCCESS) {
        ERROR_LOG("Create mode input dataset failed");
        return FAILED;
    }

    return SUCCESS;
}


void* ModelInstance::GetInferenceOutputItem(uint32_t& itemDataSize,
                                              aclmdlDataset* inferenceOutput) {
    aclDataBuffer* dataBuffer = aclmdlGetDatasetBuffer(inferenceOutput, 0);
    if (dataBuffer == nullptr) {
        ERROR_LOG("Get the dataset buffer from model "
                  "inference output failed");
        return nullptr;
    }

    void* dataBufferDev = aclGetDataBufferAddr(dataBuffer);
    if (dataBufferDev == nullptr) {
        ERROR_LOG("Get the dataset buffer address "
                  "from model inference output failed");
        return nullptr;
    }

    size_t bufferSize = aclGetDataBufferSize(dataBuffer);
    if (bufferSize == 0) {
        ERROR_LOG("The dataset buffer size of "
                  "model inference output is 0");
        return nullptr;
    }

    void* data = nullptr;
    if (runMode_ == ACL_HOST) {
        data = Utils::CopyDataDeviceToLocal(dataBufferDev, bufferSize);
        if (data == nullptr) {
            ERROR_LOG("Copy inference output to host failed");
            return nullptr;
        }
    }
    else {
        data = dataBufferDev;
    }

    itemDataSize = bufferSize;
    return data;
}


void ModelInstance::DestroyResource()
{   

    aclError ret;
    if (stream_ != nullptr) {
        ret = aclrtDestroyStream(stream_);
        if (ret != ACL_ERROR_NONE) {
            ERROR_LOG("destroy stream failed");
        }
        stream_ = nullptr;
    }
    INFO_LOG("end to destroy stream");

    if (context_ != nullptr) {
        ret = aclrtDestroyContext(context_);
        if (ret != ACL_ERROR_NONE) {
            ERROR_LOG("destroy context failed");
        }
        context_ = nullptr;
    }
    INFO_LOG("end to destroy context");

    ret = aclrtResetDevice(deviceId_);
    if (ret != ACL_ERROR_NONE) {
        ERROR_LOG("reset device failed");
    }
    INFO_LOG("end to reset device is %d", deviceId_);

    ret = aclFinalize();
    if (ret != ACL_ERROR_NONE) {
        ERROR_LOG("finalize acl failed");
    }
    INFO_LOG("end to finalize acl");
}
