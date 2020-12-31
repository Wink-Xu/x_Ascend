#ifndef DNN_ASCEND_H_
#define DNN_ASCEND_H_



#ifdef __cplusplus
extern "C" {
#endif

  int  aclRegisterModel(const char* model_path);
  int  aclUnregisterModel();
  int  aclForward(unsigned char **data_in, float **data_out);

#ifdef __cplusplus
}
#endif

#endif

