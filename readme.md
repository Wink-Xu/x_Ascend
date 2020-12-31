# Huawei Ascend
 CANN 
## TO DO before Spring Festival

1. Run through one sample
2. Finish the libdnn_ascend  (20.1)
3. Run through the body keypoints model and detect model


## Timeline

* Development Environment and MindStudio        
refer to  https://blog.csdn.net/hello_yes112/article/details/107560186                ----- 2020-12-07

* ATC  model transform                                                                  
refer to https://support.huaweicloud.com/ti-atc-A500_3000_3010/altasatc_16_004.html   ----- 2020-12-09

* Succeed to compile the classification demo                                          ----- 2020-12-10

* Succeed to run in Ai1                                                               ----- 2020-12-17
------------  Finish the whole process in https://blog.csdn.net/hello_yes112/article/details/107560186

* Finish the libdnn_ascend version 1.0 (20.0)                                           ----- 2020-12-31  
   (Succeed to use libdnn_ascend.so to execute the sample code in HuaWeiYun AI1)  
   you can use "aclRegisterModel(const char* model_path)" to load model,and use "aclForward(unsigned char **data_in, float **data_out)" to inference,At last, use "aclUnregisterModel()" to free the resource.In libdnn_ascend/sample, there is a whole process to use the library.


