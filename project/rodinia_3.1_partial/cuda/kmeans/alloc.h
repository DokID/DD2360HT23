//
// Created by gaborn on 2024-01-07.
//

#ifndef GPUPROG_HT23_ALLOC_CUH
#define GPUPROG_HT23_ALLOC_CUH

#define PREFETCH_ENABLED

#ifdef __cplusplus
extern "C" {
#endif
void allocateFeatures(float **, int, int);
void deallocateFeatures(float *);
void allocateClusters(float **, int, int);
void deallocateClusters(float *);
void allocateMembership(int **, int);
void deallocateMembership(int *);

#ifdef PREFETCH_ENABLED /* remove all prefetch related functions when it's turned off */
void initPrefetch();
void prefetchFeaturesToHost(float *, int, int);
void prefetchFeaturesToDevice(float *, int, int);
void prefetchClustersToHost(float *, int, int);
void prefetchClustersToDevice(float *, int, int);
void prefetchMembershipToHost(int *, int);
void prefetchMembershipToDevice(int *, int);
#endif // PREFETCH_ENABLED

#ifdef __cplusplus
}
#endif

#endif //GPUPROG_HT23_ALLOC_CUH
