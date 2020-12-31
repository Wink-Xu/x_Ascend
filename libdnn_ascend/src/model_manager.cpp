/**
* File model_manager.cpp
* Description: handle model process
*/

#include "model_manager.h"
#include <iostream> 
#include "utils.h"
using namespace std;

ModelManager::ModelManager():loadFlag_(false), modelId_(0), modelMemPtr_(nullptr), modelMemSize_(0),
modelWeightPtr_(nullptr),modelWeightSize_(0), modelDesc_(nullptr), input_(nullptr), output_(nullptr) {

}

ModelManager::~ModelManager() {
    
}

Result ModelManager::LoadModelFromFileWithMem(const char *modelPath) {
    if (loadFlag_) {
        ERROR_LOG("has already loaded a model");
        return FAILED;
    }

    aclError ret = aclmdlQuerySize(modelPath, &modelMemSize_, &modelWeightSize_);
    if (ret != ACL_ERROR_NONE) {
        ERROR_LOG("query model failed, model file is %s", modelPath);
        return FAILED;
    }

    ret = aclrtMalloc(&modelMemPtr_, modelMemSize_, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_ERROR_NONE) {
        ERROR_LOG("malloc buffer for mem failed, require size is %zu", modelMemSize_);
        return FAILED;
    }

    ret = aclrtMalloc(&modelWeightPtr_, modelWeightSize_, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_ERROR_NONE) {
        ERROR_LOG("malloc buffer for weight failed, require size is %zu", modelWeightSize_);
        return FAILED;
    }

    ret = aclmdlLoadFromFileWithMem(modelPath, &modelId_, modelMemPtr_,
        modelMemSize_, modelWeightPtr_, modelWeightSize_);
    if (ret != ACL_ERROR_NONE) {
        ERROR_LOG("load model from file failed, model file is %s", modelPath);
        return FAILED;
    }

    loadFlag_ = true;
    INFO_LOG("load model %s success", modelPath);
    return SUCCESS;
}

Result ModelManager::CreateDesc() {
    modelDesc_ = aclmdlCreateDesc();
    if (modelDesc_ == nullptr) {
        ERROR_LOG("create model description failed");
        return FAILED;
    }

    aclError ret = aclmdlGetDesc(modelDesc_, modelId_);
    if (ret != ACL_ERROR_NONE) {
        ERROR_LOG("get model description failed");
        return FAILED;
    }

    INFO_LOG("create model description success");
    return SUCCESS;
}

void ModelManager::DestroyDesc() {
    if (modelDesc_ != nullptr) {
        (void)aclmdlDestroyDesc(modelDesc_);
        modelDesc_ = nullptr;
    }
}

Result ModelManager::CreateInput() 
{
    if (modelDesc_ == nullptr) {
        ERROR_LOG("no model description, create input failed");
        return FAILED;
    }

    input_ = aclmdlCreateDataset();
    if (input_ == nullptr) {
        ERROR_LOG("can't create dataset, create input failed");
        return FAILED;
    }

    size_t inputSize = aclmdlGetNumInputs(modelDesc_);
    for (size_t i = 0; i < inputSize; ++i) {
        size_t buffer_size = aclmdlGetInputSizeByIndex(modelDesc_, i);
        if(buffer_size == 224 * 224 * 3 * 10)
            buffer_size = buffer_size/10;
        void *inputBuffer = nullptr;
        aclError ret = aclrtMalloc(&inputBuffer, buffer_size, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_ERROR_NONE) {
            ERROR_LOG("can't malloc buffer, size is %zu, create input failed", buffer_size);
            return FAILED;
        }

        aclDataBuffer* inputData = aclCreateDataBuffer(inputBuffer, buffer_size);

        if (ret != ACL_ERROR_NONE) {
            ERROR_LOG("can't create data buffer, create input failed");
            aclrtFree(inputBuffer);
            return FAILED;
        }

        ret = aclmdlAddDatasetBuffer(input_, inputData);
        if (ret != ACL_ERROR_NONE) {
            ERROR_LOG("can't add data buffer, create input failed");
            aclrtFree(inputBuffer);
            aclDestroyDataBuffer(inputData);
            return FAILED;
        }
    }

    INFO_LOG("create model input success");
    return SUCCESS;
}

void ModelManager::DestroyInput() {
    if (input_ == nullptr) {
        return;
    }

    for (size_t i = 0; i < aclmdlGetDatasetNumBuffers(input_); ++i) {
        aclDataBuffer* dataBuffer = aclmdlGetDatasetBuffer(input_, i);
        void* data = aclGetDataBufferAddr(dataBuffer);
        (void)aclrtFree(data);
        (void)aclDestroyDataBuffer(dataBuffer);
    }
    aclmdlDestroyDataset(input_);
    input_ = nullptr;
}

Result ModelManager::CreateOutput() {
    if (modelDesc_ == nullptr) {
        ERROR_LOG("no model description, create ouput failed");
        return FAILED;
    }

    output_ = aclmdlCreateDataset();
    if (output_ == nullptr) {
        ERROR_LOG("can't create dataset, create output failed");
        return FAILED;
    }

    size_t outputSize = aclmdlGetNumOutputs(modelDesc_);
    for (size_t i = 0; i < outputSize; ++i) {
        size_t buffer_size = aclmdlGetOutputSizeByIndex(modelDesc_, i);

        void *outputBuffer = nullptr;
        aclError ret = aclrtMalloc(&outputBuffer, buffer_size, ACL_MEM_MALLOC_NORMAL_ONLY);
        if (ret != ACL_ERROR_NONE) {
            ERROR_LOG("can't malloc buffer, size is %zu, create output failed", buffer_size);
            return FAILED;
        }

        aclDataBuffer* outputData = aclCreateDataBuffer(outputBuffer, buffer_size);
        if (ret != ACL_ERROR_NONE) {
            ERROR_LOG("can't create data buffer, create output failed");
            aclrtFree(outputBuffer);
            return FAILED;
        }

        ret = aclmdlAddDatasetBuffer(output_, outputData);
        if (ret != ACL_ERROR_NONE) {
            ERROR_LOG("can't add data buffer, create output failed");
            aclrtFree(outputBuffer);
            aclDestroyDataBuffer(outputData);
            return FAILED;
        }
    }

    INFO_LOG("create model output success");
    return SUCCESS;
}

void ModelManager::DestroyOutput() {
    if (output_ == nullptr) {
        return;
    }

    for (size_t i = 0; i < aclmdlGetDatasetNumBuffers(output_); ++i) {
        aclDataBuffer* dataBuffer = aclmdlGetDatasetBuffer(output_, i);
        void* data = aclGetDataBufferAddr(dataBuffer);
        (void)aclrtFree(data);
        (void)aclDestroyDataBuffer(dataBuffer);
    }

    (void)aclmdlDestroyDataset(output_);
    output_ = nullptr;
}

Result ModelManager::Execute() {

    aclError ret = aclmdlExecute(modelId_, input_, output_);

    if (ret != ACL_ERROR_NONE) {
        ERROR_LOG("execute model failed, modelId is %u", modelId_);
        return FAILED;
    }

    INFO_LOG("model execute success");
    return SUCCESS;
}

void ModelManager::Unload() {
    if (!loadFlag_) {
        WARN_LOG("no model had been loaded, unload failed");
        return;
    }

    aclError ret = aclmdlUnload(modelId_);
    if (ret != ACL_ERROR_NONE) {
        ERROR_LOG("unload model failed, modelId is %u", modelId_);
    }

    if (modelDesc_ != nullptr) {
        (void)aclmdlDestroyDesc(modelDesc_);
        modelDesc_ = nullptr;
    }

    if (modelMemPtr_ != nullptr) {
        aclrtFree(modelMemPtr_);
        modelMemPtr_ = nullptr;
        modelMemSize_ = 0;
    }

    if (modelWeightPtr_ != nullptr) {
        aclrtFree(modelWeightPtr_);
        modelWeightPtr_ = nullptr;
        modelWeightSize_ = 0;
    }

    loadFlag_ = false;
    INFO_LOG("unload model success, modelId is %u", modelId_);
}

aclmdlDataset *ModelManager::GetModelOutputData() {
    return output_;
}

void* ModelManager::GetModelInputInfo(uint32_t &inputDatasize_, void **inputbuf_) {

    aclDataBuffer* inputbuffer = aclmdlGetDatasetBuffer(input_, 0);
    *inputbuf_  = aclGetDataBufferAddr(inputbuffer);

    if (inputbuf_ == nullptr) {
        ERROR_LOG("Get the dataset buffer from model "
                  "inference output failed");
        return nullptr;
    } 

    inputDatasize_ = aclGetDataBufferSize(inputbuffer);
    if (inputDatasize_ == 0) {
        ERROR_LOG("The dataset buffer size of "
                  "model inference input is 0");
        return nullptr;
    }

}

