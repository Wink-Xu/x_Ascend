OPENCVDIR := /home/wink/ascend_ddk/x86

INCLUDE=-I$(OPENCVDIR)/include/opencv4  -I/home/wink/x_Ascend/libdnn_ascend_x/include 
LIBRARY=-L/home/wink/x_Ascend/libdnn_ascend_x/lib -ldnn_ascend  -L$(OPENCVDIR)/lib -lopencv_imgcodecs -lopencv_imgproc -lopencv_core -Wl,-rpath=../../lib


CXXFLAGS += -std=c++11 -O2 -g  -Wall -fPIC 
OBJ=main


all:
	g++ main.cpp  -o ${OBJ} ${CXXFLAGS} ${INCLUDE} ${LIBRARY} 
clean:
	rm -rf ${OBJ}

