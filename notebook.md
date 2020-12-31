# Notebook

## ATC

Pay much attention to the VERSION. Here we us 20.0.0

download the Ascend-Toolkit, and configure the environment variables.


```
sudo -s
echo "Configure environment variables"
install_path=/home/wink/Ascend/ascend-toolkit/20.0
export PATH=/usr/local/python3.7.5/bin:${install_path}/atc/ccec_compiler/bin:${install_path}/atc/bin:$PATH
export PYTHONPATH=${install_path}/atc/python/site-packages/te:${install_path}/atc/python/site-packages/topi:$PYTHONPATH
export LD_LIBRARY_PATH=${install_path}/atc/lib64:$LD_LIBRARY_PATH
export ASCEND_OPP_PATH=${install_path}/opp
```

execute the model transform, if dont use sudo,  permission error may happen

```
./atc --model=/home/wink/x_Ascend/sample_demo/classification/model/googlenet.prototxt --weight=/home/wink/x_Ascend/sample_demo/classification/model/googlenet.caffemodel --framework=0 --output=/home/wink/x_Ascend/sample_demo/classification/model/googlenet --soc_version=Ascend310
```

if you want use AIPP(data pre-process in hardware)
```
./atc --model=/home/wink/x_Ascend/sample_demo/classification/model/googlenet.prototxt --weight=/home/wink/x_Ascend/sample_demo/classification/model/googlenet.caffemodel --framework=0 --output=/home/wink/x_Ascend/sample_demo/classification/model/googlenet --soc_version=Ascend310 --insert_op_conf=/home/wink/x_Ascend/sample_demo/classification/model/aipp.cfg
```
aipp.cfg needs set up correctly.

Other information, refer to https://support.huaweicloud.com/tg-Inference-cann/atlasatc_16_0002.html



## Compile
Pay attention to changing the path of inc and lib


## Execute
1. model transform, now use Mindstudio is okay to trans.
2. acl.json


##  libdnn_ascend  20.0.0

1. Library Interface 
   * int  aclRegisterModel(const char* model_path);
   * int  aclUnregisterModel();
   * int  aclForward(unsigned char **data_in, float **data_out);
2. Code Composition
   * dnn_ascend.h   ---  main interface
   * model_instance.h  --- model execution
   * model_manager.h   --- model manager
Reference to the classification sample, abstract the interface for all the model.

3. Include and lib
   * opencv
   * acl
 