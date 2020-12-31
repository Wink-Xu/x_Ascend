#ifndef MODEL_MANAGER_H
#define MODEL_MANAGER_H

#include <iostream>
#include "utils.h"
#include "acl/acl.h"


/**
* ModelManager
*/
class ModelManager {
public:

    ModelManager();
    ~ModelManager();

    /**
    * @brief load model from file with mem
    * @param [in] modelPath: model path
    * @return result
    */
    Result LoadModelFromFileWithMem(const char *modelPath);
    void Unload();


    Result CreateDesc();
    void DestroyDesc();
   
  
   
    Result CreateInput();
    void DestroyInput();


    Result CreateOutput();
    void DestroyOutput();

    Result Execute();

    aclmdlDataset *GetModelOutputData();
    void* GetModelInputInfo(uint32_t &inputDatasize_, void **inputbuf_);

private:
    bool loadFlag_;  // model load flag
    uint32_t modelId_;
    void *modelMemPtr_;
    size_t modelMemSize_;
    void *modelWeightPtr_;
    size_t modelWeightSize_;
    aclmdlDesc *modelDesc_;
    aclmdlDataset *input_;
    aclmdlDataset *output_;

};

#endif