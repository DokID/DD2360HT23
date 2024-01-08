//
// Created by gaborn on 2024-01-07.
//
#include <cuda.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "kmeans_cuda_kernel.cuh"
#include "alloc.h"

#ifdef PREFETCH_ENABLED
int concurrentAccess = 0;

extern "C"
void initPrefetch() {
    gpuCheck(cudaDeviceGetAttribute(&concurrentAccess, cudaDevAttrConcurrentManagedAccess, 0));
}

void _prefetch(void *devPtr, int size, int dstDevice) {
    if (concurrentAccess) {
        gpuCheck(cudaMemPrefetchAsync(devPtr, size, dstDevice));
        if (dstDevice == cudaCpuDeviceId) {
            // fetching data to host must be awaited explicitly
            gpuCheck(cudaDeviceSynchronize());
        }
    }
}
#endif // PREFETCH_ENABLED

extern "C"
void allocateFeatures(float **features, int npoints, int nfeatures) {
    gpuCheck(cudaMallocManaged(features, npoints * nfeatures * sizeof(float)));
#ifdef PREFETCH_ENABLED
    _prefetch(features[0], npoints * nfeatures * sizeof(float), cudaCpuDeviceId);
#endif // PREFETCH_ENABLED
}

#ifdef PREFETCH_ENABLED
extern "C"
void prefetchFeaturesToHost(float *features, int npoints, int nfeatures) {
    _prefetch(features, npoints * nfeatures * sizeof(float), cudaCpuDeviceId);
}

extern "C"
void prefetchFeaturesToDevice(float *features, int npoints, int nfeatures) {
    _prefetch(features, npoints * nfeatures * sizeof(float), 0);
}
#endif // PREFETCH_ENABLED

extern "C"
void deallocateFeatures(float *features) {
    gpuCheck(cudaFree(features));
}

extern "C"
void allocateClusters(float **clusters, int nclusters, int nfeatures) {
    gpuCheck(cudaMallocManaged(clusters, nclusters * nfeatures * sizeof(float)));
#ifdef PREFETCH_ENABLED
    _prefetch(clusters[0], nclusters * nfeatures * sizeof(float), cudaCpuDeviceId);
#endif // PREFETCH_ENABLED
}

#ifdef PREFETCH_ENABLED
extern "C"
void prefetchClustersToHost(float *clusters, int nclusters, int nfeatures) {
    _prefetch(clusters, nclusters * nfeatures * sizeof(float), cudaCpuDeviceId);
}

extern "C"
void prefetchClustersToDevice(float *clusters, int nclusters, int nfeatures) {
    _prefetch(clusters, nclusters * nfeatures * sizeof(float), 0);
}
#endif // PREFETCH_ENABLED

extern "C"
void deallocateClusters(float *clusters) {
    gpuCheck(cudaFree(clusters));
}

extern "C"
void allocateMembership(int **membership, int npoints) {
    gpuCheck(cudaMallocManaged(membership, npoints * sizeof(int)));
#ifdef PREFETCH_ENABLED
    _prefetch(membership[0], npoints * sizeof(int), cudaCpuDeviceId);
#endif // PREFETCH_ENABLED
}

#ifdef PREFETCH_ENABLED
extern "C"
void prefetchMembershipToHost(int *memberhsip, int npoints) {
    _prefetch(memberhsip, npoints * sizeof(int), cudaCpuDeviceId);
}

extern "C"
void prefetchMembershipToDevice(int *memberhsip, int npoints) {
    _prefetch(memberhsip, npoints * sizeof(int), 0);
}
#endif // PREFETCH_ENABLED

extern "C"
void deallocateMembership(int *membership) {
    gpuCheck(cudaFree(membership));
}
