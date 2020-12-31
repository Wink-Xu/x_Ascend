DDK_PATH := /home/wink/Ascend
NPU_HOST_LIB := /home/wink/Ascend/ascend-toolkit/20.0/acllib_centos7.6.x86_64/acllib/lib64/stub
ATC_HOST_LIB := /home/wink/Ascend/ascend-toolkit/20.0/acllib_centos7.6.x86_64/atc/lib64

CXX = g++

SDKDIR := /home/wink/Ascend/ascend-toolkit/20.0/acllib_centos7.6.x86_64/acllib
WORKDIR := /home/wink/x_Ascend/libdnn_ascend_x
OPENCVDIR := /home/wink/ascend_ddk/x86

CXXFLAGS += -std=c++11 -O2 -g  -Wall -pthread -fopenmp -fPIC -shared
LDFLAGS  += -L$(SDKDIR)/lib64/stub -L$(NPU_HOST_LIB) -lascendcl  -Wl,-rpath=/home/wink/Ascend/ascend-toolkit/20.0/acllib_centos7.6.x86_64/acllib/lib64/stub
INCS += -I$(SDKDIR)/include -I$(WORKDIR)/include -I$(OPENCVDIR)/include/opencv4


TARGET := $(WORKDIR)/lib/libdnn_ascend.so


SRCS := $(wildcard src/*.cpp)
OBJS := $(foreach n, $(SRCS), $(addsuffix .o, $(basename $(n))))

.PHONY: dnn_ascend clean

dnn_ascend: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) -o $(TARGET) $(OBJS) $(CXXFLAGS) $(LDFLAGS)
	cp ./lib/libdnn_ascend.so ./sample/classification

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCS)  -c $< -o $@

clean:
	rm -rf $(TARGET) $(OBJS)
