
#include "dnn_ascend.h"
#include "model_manager.h"
#include "model_instance.h"


static ModelInstance gModelInstance;

int aclRegisterModel(const char* model_path) {
  gModelInstance.loadModel(model_path);
  return 0;
}

int aclUnregisterModel() {
    gModelInstance.unloadModel();
    return 0;
}

int aclForward(unsigned char **data_in, float **data_out) {
    gModelInstance.inference(data_in, data_out);
    return 0;
}



