#ifndef MODEL_INSTANCE_H
#define MODEL_INSTANCE_H


#include "utils.h"
#include "acl/acl.h"
#include "model_instance.h"
#include "model_manager.h"


using namespace std;

/**
* ModelInstance
*/
class ModelInstance {
public:
    ModelInstance();
    ~ModelInstance();
    
    Result loadModel(const char* model_path);
    Result unloadModel();
    Result inference(unsigned char **data_in, float **data_out);

private:
    Result InitResource();
    Result InitModel(const char* omModelPath);

    void* GetInferenceOutputItem(uint32_t& itemDataSize,
                                 aclmdlDataset* inferenceOutput);
    void DestroyResource();

private:
    int32_t deviceId_;
    aclrtContext context_;
    aclrtStream stream_;
    ModelManager model_;
    aclrtRunMode runMode_;
    
    bool isInited_;
};

#endif