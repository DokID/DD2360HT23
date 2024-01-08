//
// Created by gaborn on 2024-01-07.
//

#ifndef GPUPROG_HT23_KMEANS_CUDA_CUH
#define GPUPROG_HT23_KMEANS_CUDA_CUH

#ifdef __cplusplus
extern "C" {
#endif
void initializeKernelLayout(int);
void invertFeatures(float **, int, int);
void freeInvertedFeatures();
int kmeansCuda(float **, int, int, int, int *, float **, int *, float **);
#ifdef __cplusplus
}
#endif

#endif //GPUPROG_HT23_KMEANS_CUDA_CUH
