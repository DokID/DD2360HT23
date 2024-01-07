include ../../common/make.config

# C compiler
CC = gcc
CC_FLAGS = -g -O0 -fopenmp

# CUDA compiler
NVCC = $(CUDA_DIR)/bin/nvcc
NVCC_FLAGS = -I$(CUDA_DIR)/include -g -G

# 'make emu=1' compiles the CUDA kernels for emulation
ifeq ($(emu),1)
	NVCC_FLAGS += -deviceemu
endif

all: kmeans

kmeans: alloc.o cluster.o kmeans.o kmeans_clustering.o kmeans_cuda.o kmeans_cuda_kernel.o rmse.o
	$(CC) $(CC_FLAGS) $^ -o kmeans -L$(CUDA_LIB_DIR) -lcuda -lcudart -lm -lstdc++

%.o: %.c
	$(CC) $(CC_FLAGS) $< -c

alloc.o: alloc.cu
	$(NVCC) $(NVCC_FLAGS) alloc.cu -c

kmeans_cuda.o: kmeans_cuda.cu
	$(NVCC) $(NVCC_FLAGS) kmeans_cuda.cu -c

kmeans_cuda_kernel.o: kmeans_cuda_kernel.cu
	$(NVCC) $(NVCC_FLAGS) kmeans_cuda_kernel.cu -c

clean:
	rm -f *.o *~ kmeans kmeans_cuda.linkinfo
