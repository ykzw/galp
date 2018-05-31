INCLUDE = -I include -I ../moderngpu/include -I ../cub
CCFLAGS = $(INCLUDE) -O3 -std=c++0x
GPUARCH = -gencode arch=compute_60,code=sm_60
CUFLAGS = $(INCLUDE) -O3 -std=c++11 -use_fast_math $(GPUARCH) -Xptxas -v -Xcompiler -fopenmp $(LIBRARY) -lnccl -lineinfo
TARGET = bin/label_propagation

SOURCE = src/main.cu src/common/*.cuh src/dpp/*.cuh src/lfht/*.cuh include/*.h Makefile


all: $(TARGET)


$(TARGET): $(SOURCE) obj/mgpucontext.o obj/mgpuutil.o
	mkdir -p bin
	nvcc $(CUFLAGS) -o $@ $< obj/mgpucontext.o obj/mgpuutil.o

obj/mgpucontext.o: ../moderngpu/src/mgpucontext.cu
	mkdir -p obj
	nvcc $(CUFLAGS) -c -o $@ $<

obj/mgpuutil.o: ../moderngpu/src/mgpuutil.cpp
	mkdir -p obj
	nvcc $(CUFLAGS) -c -o $@ $<


clean:
	rm $(TARGET) obj/mgpucontext.o obj/mgpuutil.o
