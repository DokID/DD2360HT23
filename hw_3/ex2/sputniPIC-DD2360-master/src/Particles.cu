#include "Particles.h"
#include "Alloc.h"
#include <cuda.h>
#include <cuda_runtime.h>

/** allocate particle arrays */
void particle_allocate(struct parameters* param, struct particles* part, int is)
{
    
    // set species ID
    part->species_ID = is;
    // number of particles
    part->nop = param->np[is];
    // maximum number of particles
    part->npmax = param->npMax[is];
    
    // choose a different number of mover iterations for ions and electrons
    if (param->qom[is] < 0){  //electrons
        part->NiterMover = param->NiterMover;
        part->n_sub_cycles = param->n_sub_cycles;
    } else {                  // ions: only one iteration
        part->NiterMover = 1;
        part->n_sub_cycles = 1;
    }
    
    // particles per cell
    part->npcelx = param->npcelx[is];
    part->npcely = param->npcely[is];
    part->npcelz = param->npcelz[is];
    part->npcel = part->npcelx*part->npcely*part->npcelz;
    
    // cast it to required precision
    part->qom = (FPpart) param->qom[is];
    
    long npmax = part->npmax;
    
    // initialize drift and thermal velocities
    // drift
    part->u0 = (FPpart) param->u0[is];
    part->v0 = (FPpart) param->v0[is];
    part->w0 = (FPpart) param->w0[is];
    // thermal
    part->uth = (FPpart) param->uth[is];
    part->vth = (FPpart) param->vth[is];
    part->wth = (FPpart) param->wth[is];
    
    
    //////////////////////////////
    /// ALLOCATION PARTICLE ARRAYS
    //////////////////////////////
    part->x = new FPpart[npmax];
    part->y = new FPpart[npmax];
    part->z = new FPpart[npmax];
    // allocate velocity
    part->u = new FPpart[npmax];
    part->v = new FPpart[npmax];
    part->w = new FPpart[npmax];
    // allocate charge = q * statistical weight
    part->q = new FPinterp[npmax];
    
}
/** deallocate */
void particle_deallocate(struct particles* part)
{
    // deallocate particle variables
    delete[] part->x;
    delete[] part->y;
    delete[] part->z;
    delete[] part->u;
    delete[] part->v;
    delete[] part->w;
    delete[] part->q;
}

/**
 * Clones the content of a Particles object in host memory to a
 * Particles object in device memory.
*/
void device_particle_deepcopy(struct particles* hostPart, struct particles* devicePart) {
    // set species ID
    devicePart->species_ID = hostPart->species_ID;
    // number of particles
    devicePart->nop = hostPart->nop;
    // maximum number of particles
    devicePart->npmax = hostPart->npmax;
    
    devicePart->NiterMover = hostPart->NiterMover;
    devicePart->n_sub_cycles = hostPart->n_sub_cycles;
    
    // particles per cell
    devicePart->npcelx = hostPart->npcelx;
    devicePart->npcely = hostPart->npcely;
    devicePart->npcelz = hostPart->npcelz;
    devicePart->npcel = hostPart->npcel;
    
    // cast it to required precision
    devicePart->qom = (FPpart) hostPart->qom;
    
    long npmax = hostPart->npmax;
    
    // initialize drift and thermal velocities
    // drift
    devicePart->u0 = hostPart->u0;
    devicePart->v0 = hostPart->v0;
    devicePart->w0 = hostPart->w0;
    // thermal
    devicePart->uth = hostPart->uth;
    devicePart->vth = hostPart->vth;
    devicePart->wth = hostPart->wth;
    
    
    ///////////////////////////////////////////
    /// ALLOCATION PARTICLE ARRAYS ON DEVICE///
    ///////////////////////////////////////////
    cudaMalloc(&devicePart->x, npmax * sizeof(FPpart));
    cudaMemcpy(devicePart->x, hostPart->x, npmax * sizeof(FPpart), cudaMemcpyHostToDevice);

    cudaMalloc(&devicePart->y, npmax * sizeof(FPpart));
    cudaMemcpy(devicePart->y, hostPart->y, npmax * sizeof(FPpart), cudaMemcpyHostToDevice);

    cudaMalloc(&devicePart->z, npmax * sizeof(FPpart));
    cudaMemcpy(devicePart->z, hostPart->z, npmax * sizeof(FPpart), cudaMemcpyHostToDevice);

    //// allocate velocity
    cudaMalloc(&devicePart->u, npmax * sizeof(FPpart));
    cudaMemcpy(devicePart->u, hostPart->u, npmax * sizeof(FPpart), cudaMemcpyHostToDevice);

    cudaMalloc(&devicePart->v, npmax * sizeof(FPpart));
    cudaMemcpy(devicePart->v, hostPart->v, npmax * sizeof(FPpart), cudaMemcpyHostToDevice);

    cudaMalloc(&devicePart->w, npmax * sizeof(FPpart));
    cudaMemcpy(devicePart->w, hostPart->w, npmax * sizeof(FPpart), cudaMemcpyHostToDevice);
    
    //// allocate charge = q * statistical weight
    cudaMalloc(&devicePart->q, npmax * sizeof(FPpart));
    cudaMemcpy(devicePart->q, hostPart->q, npmax * sizeof(FPpart), cudaMemcpyHostToDevice);

    // copy devicePart to device
    particles* deviceMemSegment;
    particles* tmp;
    cudaMalloc(&deviceMemSegment, sizeof(particles));
    cudaMemcpy(deviceMemSegment, devicePart, sizeof(particles), cudaMemcpyHostToDevice);

    // do a classic three way switcheroo
    tmp = devicePart;
    devicePart = deviceMemSegment;
    free(tmp);
}

__global__ void mover_PC_kernel(
    struct particles* part, 
    struct EMfield* field, 
    struct grid* grd, 
    struct parameters* param,
    FPpart dt_sub_cycling, 
    FPpart dto2,
    FPpart qomdt2,
    int n_sub_cycles
    ) {
    
    // auxiliary variables
    //variable
    FPpart omdtsq, denom, ut, vt, wt, udotb;
    
    //variable
    // local (to the particle) electric and magnetic field
    FPfield Exl=0.0, Eyl=0.0, Ezl=0.0, Bxl=0.0, Byl=0.0, Bzl=0.0;
    
    //variable
    // interpolation densities
    int ix,iy,iz;
    FPfield weight[2][2][2];
    FPfield xi[2], eta[2], zeta[2];
    
    //variable
    // intermediate particle position and velocity
    FPpart xptilde, yptilde, zptilde, uptilde, vptilde, wptilde;
    
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i >= part->nop) {

        // start subcycling
        for (int i_sub=0; i_sub <  n_sub_cycles; i_sub++){
        // move each particle with new fields
            xptilde = part->x[i];
            yptilde = part->y[i];
            zptilde = part->z[i];
            // calculate the average velocity iteratively
            for(int innter=0; innter < part->NiterMover; innter++){
                // interpolation G-->P
                ix = 2 +  int((part->x[i] - grd->xStart)*grd->invdx);
                iy = 2 +  int((part->y[i] - grd->yStart)*grd->invdy);
                iz = 2 +  int((part->z[i] - grd->zStart)*grd->invdz);
                
                // calculate weights
                xi[0]   = part->x[i] - grd->XN[ix - 1][iy][iz];
                eta[0]  = part->y[i] - grd->YN[ix][iy - 1][iz];
                zeta[0] = part->z[i] - grd->ZN[ix][iy][iz - 1];
                xi[1]   = grd->XN[ix][iy][iz] - part->x[i];
                eta[1]  = grd->YN[ix][iy][iz] - part->y[i];
                zeta[1] = grd->ZN[ix][iy][iz] - part->z[i];
                for (int ii = 0; ii < 2; ii++)
                    for (int jj = 0; jj < 2; jj++)
                        for (int kk = 0; kk < 2; kk++)
                            weight[ii][jj][kk] = xi[ii] * eta[jj] * zeta[kk] * grd->invVOL;
                
                // set to zero local electric and magnetic field
                Exl=0.0, Eyl = 0.0, Ezl = 0.0, Bxl = 0.0, Byl = 0.0, Bzl = 0.0;
                
                for (int ii=0; ii < 2; ii++)
                    for (int jj=0; jj < 2; jj++)
                        for(int kk=0; kk < 2; kk++){
                            Exl += weight[ii][jj][kk]*field->Ex[ix- ii][iy -jj][iz- kk ];
                            Eyl += weight[ii][jj][kk]*field->Ey[ix- ii][iy -jj][iz- kk ];
                            Ezl += weight[ii][jj][kk]*field->Ez[ix- ii][iy -jj][iz -kk ];
                            Bxl += weight[ii][jj][kk]*field->Bxn[ix- ii][iy -jj][iz -kk ];
                            Byl += weight[ii][jj][kk]*field->Byn[ix- ii][iy -jj][iz -kk ];
                            Bzl += weight[ii][jj][kk]*field->Bzn[ix- ii][iy -jj][iz -kk ];
                        }
                
                // end interpolation
                omdtsq = qomdt2*qomdt2*(Bxl*Bxl+Byl*Byl+Bzl*Bzl);
                denom = 1.0/(1.0 + omdtsq);
                // solve the position equation
                ut= part->u[i] + qomdt2*Exl;
                vt= part->v[i] + qomdt2*Eyl;
                wt= part->w[i] + qomdt2*Ezl;
                udotb = ut*Bxl + vt*Byl + wt*Bzl;
                // solve the velocity equation
                uptilde = (ut+qomdt2*(vt*Bzl -wt*Byl + qomdt2*udotb*Bxl))*denom;
                vptilde = (vt+qomdt2*(wt*Bxl -ut*Bzl + qomdt2*udotb*Byl))*denom;
                wptilde = (wt+qomdt2*(ut*Byl -vt*Bxl + qomdt2*udotb*Bzl))*denom;
                // update position
                part->x[i] = xptilde + uptilde*dto2;
                part->y[i] = yptilde + vptilde*dto2;
                part->z[i] = zptilde + wptilde*dto2;
                
                
            } // end of iteration
            // update the final position and velocity
            part->u[i]= 2.0*uptilde - part->u[i];
            part->v[i]= 2.0*vptilde - part->v[i];
            part->w[i]= 2.0*wptilde - part->w[i];
            part->x[i] = xptilde + uptilde*dt_sub_cycling;
            part->y[i] = yptilde + vptilde*dt_sub_cycling;
            part->z[i] = zptilde + wptilde*dt_sub_cycling;
            
            
            //////////
            //////////
            ////////// BC

            // X-DIRECTION: BC particles
            if (part->x[i] > grd->Lx){
                if (param->PERIODICX==true){ // PERIODIC
                    part->x[i] = part->x[i] - grd->Lx;
                } else { // REFLECTING BC
                    part->u[i] = -part->u[i];
                    part->x[i] = 2*grd->Lx - part->x[i];
                }
            }

            if (part->x[i] < 0){
                if (param->PERIODICX==true){ // PERIODIC
                    part->x[i] = part->x[i] + grd->Lx;
                } else { // REFLECTING BC
                    part->u[i] = -part->u[i];
                    part->x[i] = -part->x[i];
                }
            }
            
            // Y-DIRECTION: BC particles
            if (part->y[i] > grd->Ly){
                if (param->PERIODICY==true){ // PERIODIC
                    part->y[i] = part->y[i] - grd->Ly;
                } else { // REFLECTING BC
                    part->v[i] = -part->v[i];
                    part->y[i] = 2*grd->Ly - part->y[i];
                }
            }

            if (part->y[i] < 0){
                if (param->PERIODICY==true){ // PERIODIC
                    part->y[i] = part->y[i] + grd->Ly;
                } else { // REFLECTING BC
                    part->v[i] = -part->v[i];
                    part->y[i] = -part->y[i];
                }
            }

            // Z-DIRECTION: BC particles
            if (part->z[i] > grd->Lz){
                if (param->PERIODICZ==true){ // PERIODIC
                    part->z[i] = part->z[i] - grd->Lz;
                } else { // REFLECTING BC
                    part->w[i] = -part->w[i];
                    part->z[i] = 2*grd->Lz - part->z[i];
                }
            }

            if (part->z[i] < 0){
                if (param->PERIODICZ==true){ // PERIODIC
                    part->z[i] = part->z[i] + grd->Lz;
                } else { // REFLECTING BC
                    part->w[i] = -part->w[i];
                    part->z[i] = -part->z[i];
                }
            }
        } // end of subcycling
    }
}

/** particle mover */
int mover_PC_GPU(struct particles* part, struct EMfield* field, struct grid* grd, struct parameters* param)
{
    
    // print species and subcycling
    std::cout << "***  MOVER with SUBCYCLYING "<< param->n_sub_cycles << " - species " << part->species_ID << " ***" << std::endl;
 
    //constant
    FPpart dt_sub_cycling = (FPpart) param->dt/((double) part->n_sub_cycles);
    FPpart dto2 = .5*dt_sub_cycling;
    FPpart qomdt2 = part->qom*dto2/param->c;

    //long nop = part->nop;
    
    // allocate gpu memory

    /**
     * 1. Allocate separate memory segment for each array on device
     * 2. Copy content of each array from host to device's allocation from 1
     * 3. Construct top-level struct holding all device specific pointers
     * 4. Pass struct from 3 to kernel as single parameter
    */

    //particles* devicePart = new particles;
    //device_particle_deepcopy(part, devicePart);
    
    //FPpart *deviceField_Ex;
    //FPpart *deviceField_Ey;
    //FPpart *deviceField_Ez;
    //FPpart *deviceField_Bxn;
    //FPpart *deviceField_Byn;
    //FPpart *deviceField_Bzn;

    FPpart *device_part_x;
    FPpart *device_part_y;
    FPpart *device_part_z;
    FPpart *device_part_u;
    FPpart *device_part_v;
    FPpart *device_part_w;
    FPfield *device_field_flattened_Ex;
    FPfield *device_field_flattened_Ey;
    FPfield *device_field_flattened_Ez;
    FPfield *device_field_flattened_Bxn;
    FPfield *device_field_flattened_Byn;
    FPfield *device_field_flattened_Bzn;
    FPfield *device_grid_flattened_XN;
    FPfield *device_grid_flattened_YN;
    FPfield *device_grid_flattened_ZN;

    cudaMalloc(&device_part_x, part->nop * sizeof(FPpart));
    cudaMalloc(&device_part_y, part->nop * sizeof(FPpart));
    cudaMalloc(&device_part_z, part->nop * sizeof(FPpart));
    cudaMalloc(&device_part_u, part->nop * sizeof(FPpart));
    cudaMalloc(&device_part_v, part->nop * sizeof(FPpart));
    cudaMalloc(&device_part_w, part->nop * sizeof(FPpart));
    cudaMalloc(&device_field_flattened_Ex, grd->nxn*grd->nyn*grd->nzn * sizeof(FPfield));
    cudaMalloc(&device_field_flattened_Ey, grd->nxn*grd->nyn*grd->nzn * sizeof(FPfield));
    cudaMalloc(&device_field_flattened_Ez, grd->nxn*grd->nyn*grd->nzn * sizeof(FPfield));
    cudaMalloc(&device_field_flattened_Bxn, grd->nxn*grd->nyn*grd->nzn * sizeof(FPfield));
    cudaMalloc(&device_field_flattened_Byn, grd->nxn*grd->nyn*grd->nzn * sizeof(FPfield));
    cudaMalloc(&device_field_flattened_Bzn, grd->nxn*grd->nyn*grd->nzn * sizeof(FPfield));
    cudaMalloc(&device_grid_flattened_XN, grd->nxn*grd->nyn*grd->nzn * sizeof(FPfield));
    cudaMalloc(&device_grid_flattened_YN, grd->nxn*grd->nyn*grd->nzn * sizeof(FPfield));
    cudaMalloc(&device_grid_flattened_ZN, grd->nxn*grd->nyn*grd->nzn * sizeof(FPfield));

    //cudaMalloc(&deviceField, sizeof(EMfield));
    // allocate field -> Ex Ey Ez Bxn Byn Bzn

    //cudaMalloc(&deviceGrd, sizeof(grid));
    //cudaMalloc(&deviceParam, sizeof(parameters));

    cudaMemcpy(device_part_x, part->x, (part->nop) * sizeof(FPpart), cudaMemcpyHostToDevice);
    cudaMemcpy(device_part_y, part->y, (part->nop) * sizeof(FPpart), cudaMemcpyHostToDevice);
    cudaMemcpy(device_part_z, part->z, (part->nop) * sizeof(FPpart), cudaMemcpyHostToDevice);
    cudaMemcpy(device_part_u, part->u, (part->nop) * sizeof(FPpart), cudaMemcpyHostToDevice);
    cudaMemcpy(device_part_v, part->v, (part->nop) * sizeof(FPpart), cudaMemcpyHostToDevice);
    cudaMemcpy(device_part_w, part->w, (part->nop) * sizeof(FPpart), cudaMemcpyHostToDevice);
    cudaMemcpy(device_field_flattened_Ex, (field->Ex_flat), (grd->nxn)*(grd->nyn)*(grd->nzn) * sizeof(FPfield), cudaMemcpyHostToDevice);
    cudaMemcpy(device_field_flattened_Ey, (field->Ey_flat), (grd->nxn)*(grd->nyn)*(grd->nzn) * sizeof(FPfield), cudaMemcpyHostToDevice);
    cudaMemcpy(device_field_flattened_Ez, (field->Ez_flat), (grd->nxn)*(grd->nyn)*(grd->nzn) * sizeof(FPfield), cudaMemcpyHostToDevice);
    cudaMemcpy(device_field_flattened_Bxn, (field->Bxn_flat), (grd->nxn)*(grd->nyn)*(grd->nzn) * sizeof(FPfield), cudaMemcpyHostToDevice);
    cudaMemcpy(device_field_flattened_Byn, (field->Byn_flat), (grd->nxn)*(grd->nyn)*(grd->nzn) * sizeof(FPfield), cudaMemcpyHostToDevice);
    cudaMemcpy(device_field_flattened_Bzn, (field->Bzn_flat), (grd->nxn)*(grd->nyn)*(grd->nzn) * sizeof(FPfield), cudaMemcpyHostToDevice);
    cudaMemcpy(device_grid_flattened_XN, (grd->XN_flat), (grd->nxn)*(grd->nyn)*(grd->nzn) * sizeof(FPfield), cudaMemcpyHostToDevice);
    cudaMemcpy(device_grid_flattened_YN, (grd->YN_flat), (grd->nxn)*(grd->nyn)*(grd->nzn) * sizeof(FPfield), cudaMemcpyHostToDevice);
    cudaMemcpy(device_grid_flattened_ZN, (grd->ZN_flat), (grd->nxn)*(grd->nyn)*(grd->nzn) * sizeof(FPfield), cudaMemcpyHostToDevice);

    cudaDeviceProp *prop = (cudaDeviceProp *) malloc (sizeof(cudaDeviceProp));
    cudaGetDeviceProperties(prop, 0);
    int threads_per_block = prop->maxThreadsPerBlock;
    dim3 block(threads_per_block);
    dim3 grid((int) ceil ((double) part->nop/ (double) block.x));

    //mover_PC_kernel<<<grid, block>>>(device_part, device_field, device_grd, device_param, dt_sub_cycling, dto2, qomdt2, part->n_sub_cycles);
    mover_PC_SIMPLE<<<grid, block>>>(device_part_x, device_part_z, device_part_y, 
                    device_part_u, device_part_v, device_part_w,
                    part->n_sub_cycles, part->NiterMover, part->nop, part->species_ID,
                    device_field_flattened_Ex, device_field_flattened_Ey, device_field_flattened_Ez, 
                    device_field_flattened_Bxn, device_field_flattened_Byn, device_field_flattened_Bzn, 
                    device_grid_flattened_XN, device_grid_flattened_YN, device_grid_flattened_ZN,
                    grd->nxn, grd->nyn, grd->nzn,
                    grd->xStart, grd->yStart, grd->zStart, 
                    grd->invdx, grd->invdy, grd->invdz,
                    grd->Lx, grd->Ly, grd->Lz, grd->invVOL,
                    grd->PERIODICX, grd->PERIODICY, grd->PERIODICZ, 
                    dt_sub_cycling,  dto2,  qomdt2);
    cudaDeviceSynchronize();

    // read from device
    //copy_particles_from_device(devicePart, partDeepCopy)
    cudaError_t err = cudaMemcpy(part->x, device_part_x, (part->nop) * sizeof(FPpart), cudaMemcpyDeviceToHost);
    if (err) printf("device to host: %s: %s\n", cudaGetErrorName(err), cudaGetErrorString(err));

    cudaMemcpy(part->x, device_part_x, (part->nop) * sizeof(FPpart), cudaMemcpyDeviceToHost);
    cudaMemcpy(part->y, device_part_y, (part->nop) * sizeof(FPpart), cudaMemcpyDeviceToHost);
    cudaMemcpy(part->z, device_part_z, (part->nop) * sizeof(FPpart), cudaMemcpyDeviceToHost);
    cudaMemcpy(part->u, device_part_u, (part->nop) * sizeof(FPpart), cudaMemcpyDeviceToHost);
    cudaMemcpy(part->v, device_part_v, (part->nop) * sizeof(FPpart), cudaMemcpyDeviceToHost);
    cudaMemcpy(part->w, device_part_w, (part->nop) * sizeof(FPpart), cudaMemcpyDeviceToHost);
    //cudaMemcpy(field->Ex_flat, device_field_flattened_Ex, (grd->nxn)*(grd->nyn)*(grd->nzn) * sizeof(FPfield), cudaMemcpyDeviceToHost);
    //cudaMemcpy(field->Ey_flat, device_field_flattened_Ey, (grd->nxn)*(grd->nyn)*(grd->nzn) * sizeof(FPfield), cudaMemcpyDeviceToHost);
    //cudaMemcpy(field->Ez_flat, device_field_flattened_Ez, (grd->nxn)*(grd->nyn)*(grd->nzn) * sizeof(FPfield), cudaMemcpyDeviceToHost);
    //cudaMemcpy(field->Bxn_flat, device_field_flattened_Bxn, (grd->nxn)*(grd->nyn)*(grd->nzn) * sizeof(FPfield), cudaMemcpyDeviceToHost);
    //cudaMemcpy(field->Byn_flat, device_field_flattened_Byn, (grd->nxn)*(grd->nyn)*(grd->nzn) * sizeof(FPfield), cudaMemcpyDeviceToHost);
    //cudaMemcpy(field->Bzn_flat, device_field_flattened_Bzn, (grd->nxn)*(grd->nyn)*(grd->nzn) * sizeof(FPfield), cudaMemcpyDeviceToHost);
    //cudaMemcpy(grd->XN_flat, device_grid_flattened_XN, (grd->nxn)*(grd->nyn)*(grd->nzn) * sizeof(FPfield), cudaMemcpyDeviceToHost);
    //cudaMemcpy(grd->YN_flat, device_grid_flattened_YN, (grd->nxn)*(grd->nyn)*(grd->nzn) * sizeof(FPfield), cudaMemcpyDeviceToHost);
    //cudaMemcpy(grd->ZN_flat, device_grid_flattened_ZN, (grd->nxn)*(grd->nyn)*(grd->nzn) * sizeof(FPfield), cudaMemcpyDeviceToHost);
    
    // deallocate gpu memory
    cudaFree(device_part_x);
    cudaFree(device_part_y);
    cudaFree(device_part_z);
    cudaFree(device_part_u);
    cudaFree(device_part_v);
    cudaFree(device_part_w);
    cudaFree(device_field_flattened_Ex);
    cudaFree(device_field_flattened_Ey);
    cudaFree(device_field_flattened_Ez);
    cudaFree(device_field_flattened_Bxn);
    cudaFree(device_field_flattened_Byn);
    cudaFree(device_field_flattened_Bzn);
    cudaFree(device_grid_flattened_XN);
    cudaFree(device_grid_flattened_YN);
    cudaFree(device_grid_flattened_ZN);

    return(0); // exit succcesfully
} // end of the mover

//int mover_PC_SIMPLE(struct particles* part, struct EMfield* field, struct grid* grd, struct parameters* param,
__global__ void mover_PC_SIMPLE(FPpart* device_part_x, FPpart* device_part_y, FPpart* device_part_z, 
                                FPpart* device_part_u, FPpart* device_part_v, FPpart* device_part_w,
                                int n_sub_cycles, int NiterMover, long nop, int species_ID,
                                FPfield* device_field_flattened_Ex, FPfield* device_field_flattened_Ey, FPfield* device_field_flattened_Ez, 
                                FPfield* device_field_flattened_Bxn, FPfield* device_field_flattened_Byn, FPfield* device_field_flattened_Bzn, 
                                FPfield* device_grid_flattened_XN, FPfield* device_grid_flattened_YN, FPfield* device_grid_flattened_ZN,
                                int grd_nxn, int grd_nyn, int grd_nzn,
                                double grd_xStart, double grd_yStart, double grd_zStart, 
                                double grd_invdx, double grd_invdy, double grd_invdz,
                                double grd_Lx, double grd_Ly, double grd_Lz, double grd_invVOL,
                                bool PERIODICX, bool PERIODICY, bool PERIODICZ,
                                FPpart dt_sub_cycling, FPpart dto2, FPpart qomdt2)
{
    // print species and subcycling
    // std::cout << "*** ## GPU ## MOVER with SUBCYCLYING "<< n_sub_cycles << " - species " << species_ID << " ***" << std::endl;
 
    //consts
    // auxiliary variables
    //FPpart dt_sub_cycling = (FPpart) param->dt/((double) part->n_sub_cycles);
    //FPpart dto2 = .5*dt_sub_cycling;
    //FPpart qomdt2 = part->qom*dto2/param->c;
    
    FPpart omdtsq, denom, ut, vt, wt, udotb;
    
    // local (to the particle) electric and magnetic field
    FPfield Exl = 0.0;
    FPfield Eyl = 0.0;
    FPfield Ezl = 0.0;
    FPfield Bxl = 0.0;
    FPfield Byl = 0.0;
    FPfield Bzl = 0.0;
    
    // interpolation densities
    int ix,iy,iz;
    FPfield weight[2][2][2];
    FPfield xi[2], eta[2], zeta[2];
    
    // intermediate particle position and velocity
    FPpart xptilde, yptilde, zptilde, uptilde, vptilde, wptilde;

    // IDX
    const int i = threadIdx.x + blockDim.x * blockIdx.x;
    if ( i < nop ) {
        // start subcycling
        for (int i_sub=0; i_sub <  n_sub_cycles; i_sub++){
            // move each particle with new fields
            //for (int i=0; i <  part->nop; i++){
                xptilde = device_part_x[i];
                yptilde = device_part_y[i];
                zptilde = device_part_z[i];
                // calculate the average velocity iteratively
                for(int innter=0; innter < NiterMover; innter++){
                    // interpolation G-->P
                    ix = 2 +  int((device_part_x[i] - grd_xStart)*grd_invdx);
                    iy = 2 +  int((device_part_y[i] - grd_yStart)*grd_invdy);
                    iz = 2 +  int((device_part_z[i] - grd_zStart)*grd_invdz);
                    
                    // calculate weights
                    xi[0]   = device_part_x[i] - device_grid_flattened_XN[(ix - 1)*grd_nyn*grd_nzn + iy*grd_nzn + iz];
                    eta[0]  = device_part_y[i] - device_grid_flattened_YN[ix*grd_nyn*grd_nzn + (iy - 1)*grd_nzn + iz];
                    zeta[0] = device_part_z[i] - device_grid_flattened_ZN[ix*grd_nyn*grd_nzn + iy*grd_nzn + (iz - 1)];
                    xi[1]   = device_grid_flattened_XN[ix*grd_nyn*grd_nzn + iy*grd_nzn + iz] - device_part_x[i];
                    eta[1]  = device_grid_flattened_YN[ix*grd_nyn*grd_nzn + iy*grd_nzn + iz] - device_part_y[i];
                    zeta[1] = device_grid_flattened_ZN[ix*grd_nyn*grd_nzn + iy*grd_nzn + iz] - device_part_z[i];
                    for (int ii = 0; ii < 2; ii++)
                        for (int jj = 0; jj < 2; jj++)
                            for (int kk = 0; kk < 2; kk++)
                                weight[ii][jj][kk] = xi[ii] * eta[jj] * zeta[kk] * grd_invVOL;
                    
                    // set to zero local electric and magnetic field
                    Exl = 0.0;
                    Eyl = 0.0;
                    Ezl = 0.0;
                    Bxl = 0.0;
                    Byl = 0.0;
                    Bzl = 0.0;
                    
                    for (int ii=0; ii < 2; ii++)
                        for (int jj=0; jj < 2; jj++)
                            for(int kk=0; kk < 2; kk++){
                                Exl += weight[ii][jj][kk]*device_field_flattened_Ex[(ix - ii)*grd_nyn*grd_nzn + (iy - jj)*grd_nzn + (iz - kk)];
                                Eyl += weight[ii][jj][kk]*device_field_flattened_Ey[(ix - ii)*grd_nyn*grd_nzn + (iy - jj)*grd_nzn + (iz - kk)];
                                Ezl += weight[ii][jj][kk]*device_field_flattened_Ez[(ix - ii)*grd_nyn*grd_nzn + (iy - jj)*grd_nzn + (iz - kk)];
                                Bxl += weight[ii][jj][kk]*device_field_flattened_Bxn[(ix - ii)*grd_nyn*grd_nzn + (iy - jj)*grd_nzn + (iz - kk)];
                                Byl += weight[ii][jj][kk]*device_field_flattened_Byn[(ix - ii)*grd_nyn*grd_nzn + (iy - jj)*grd_nzn + (iz - kk)];
                                Bzl += weight[ii][jj][kk]*device_field_flattened_Bzn[(ix - ii)*grd_nyn*grd_nzn + (iy - jj)*grd_nzn + (iz - kk)];
                            }
                    
                    // end interpolation
                    omdtsq = qomdt2*qomdt2*(Bxl*Bxl+Byl*Byl+Bzl*Bzl);
                    denom = 1.0/(1.0 + omdtsq);
                    // solve the position equation
                    ut= device_part_u[i] + qomdt2*Exl;
                    vt= device_part_v[i] + qomdt2*Eyl;
                    wt= device_part_w[i] + qomdt2*Ezl;
                    udotb = ut*Bxl + vt*Byl + wt*Bzl;
                    // solve the velocity equation
                    uptilde = (ut + qomdt2*(vt*Bzl -wt*Byl + qomdt2*udotb*Bxl))*denom;
                    vptilde = (vt + qomdt2*(wt*Bxl -ut*Bzl + qomdt2*udotb*Byl))*denom;
                    wptilde = (wt + qomdt2*(ut*Byl -vt*Bxl + qomdt2*udotb*Bzl))*denom;
                    // update position
                    device_part_x[i] = xptilde + uptilde*dto2;
                    device_part_y[i] = yptilde + vptilde*dto2;
                    device_part_z[i] = zptilde + wptilde*dto2;
                    
                    
                } // end of iteration
                __syncthreads();
                // update the final position and velocity
                device_part_u[i]= 2.0*uptilde - device_part_u[i];
                device_part_v[i]= 2.0*vptilde - device_part_v[i];
                device_part_w[i]= 2.0*wptilde - device_part_w[i];
                device_part_x[i] = xptilde + uptilde*dt_sub_cycling;
                device_part_y[i] = yptilde + vptilde*dt_sub_cycling;
                device_part_z[i] = zptilde + wptilde*dt_sub_cycling;
                
                
                //////////
                //////////
                ////////// BC
                                            
                // X-DIRECTION: BC particles
                if (device_part_x[i] > grd_Lx){
                    if (PERIODICX==true){ // PERIODIC
                        device_part_x[i] = device_part_x[i] - grd_Lx;
                    } else { // REFLECTING BC
                        device_part_u[i] = -device_part_u[i];
                        device_part_x[i] = 2*grd_Lx - device_part_x[i];
                    }
                }
                                                                            
                if (device_part_x[i] < 0){
                    if (PERIODICX==true){ // PERIODIC
                    device_part_x[i] = device_part_x[i] + grd_Lx;
                    } else { // REFLECTING BC
                        device_part_u[i] = -device_part_u[i];
                        device_part_x[i] = -device_part_x[i];
                    }
                }
                    
                
                // Y-DIRECTION: BC particles
                if (device_part_y[i] > grd_Ly){
                    if (PERIODICY==true){ // PERIODIC
                        device_part_y[i] = device_part_y[i] - grd_Ly;
                    } else { // REFLECTING BC
                        device_part_v[i] = -device_part_v[i];
                        device_part_y[i] = 2*grd_Ly - device_part_y[i];
                    }
                }
                                                                            
                if (device_part_y[i] < 0){
                    if (PERIODICY==true){ // PERIODIC
                        device_part_y[i] = device_part_y[i] + grd_Ly;
                    } else { // REFLECTING BC
                        device_part_v[i] = -device_part_v[i];
                        device_part_y[i] = -device_part_y[i];
                    }
                }
                                                                            
                // Z-DIRECTION: BC particles
                if (device_part_z[i] > grd_Lz){
                    if (PERIODICZ==true){ // PERIODIC
                        device_part_z[i] = device_part_z[i] - grd_Lz;
                    } else { // REFLECTING BC
                        device_part_w[i] = -device_part_w[i];
                        device_part_z[i] = 2*grd_Lz - device_part_z[i];
                    }
                }
                                                                            
                if (device_part_z[i] < 0){
                    if (PERIODICZ==true){ // PERIODIC
                        device_part_z[i] = device_part_z[i] + grd_Lz;
                    } else { // REFLECTING BC
                        device_part_w[i] = -device_part_w[i];
                        device_part_z[i] = -device_part_z[i];
                    }
                }
                                                                            
                
                
            //} // end of one particle
        } // end of subcycling
        __syncthreads();
        //return(0); // exit succcesfully
    }
} // end of the mover


/** particle mover */
int mover_PC(struct particles* part, struct EMfield* field, struct grid* grd, struct parameters* param)
{
    // print species and subcycling
    std::cout << "***  MOVER with SUBCYCLYING "<< param->n_sub_cycles << " - species " << part->species_ID << " ***" << std::endl;
 
    // auxiliary variables
    FPpart dt_sub_cycling = (FPpart) param->dt/((double) part->n_sub_cycles);
    FPpart dto2 = .5*dt_sub_cycling, qomdt2 = part->qom*dto2/param->c;
    FPpart omdtsq, denom, ut, vt, wt, udotb;
    
    // local (to the particle) electric and magnetic field
    FPfield Exl=0.0, Eyl=0.0, Ezl=0.0, Bxl=0.0, Byl=0.0, Bzl=0.0;
    
    // interpolation densities
    int ix,iy,iz;
    FPfield weight[2][2][2];
    FPfield xi[2], eta[2], zeta[2];
    
    // intermediate particle position and velocity
    FPpart xptilde, yptilde, zptilde, uptilde, vptilde, wptilde;
    
    // start subcycling
    for (int i_sub=0; i_sub <  part->n_sub_cycles; i_sub++){
        // move each particle with new fields
        for (int i=0; i <  part->nop; i++){
            xptilde = part->x[i];
            yptilde = part->y[i];
            zptilde = part->z[i];
            // calculate the average velocity iteratively
            for(int innter=0; innter < part->NiterMover; innter++){
                // interpolation G-->P
                ix = 2 +  int((part->x[i] - grd->xStart)*grd->invdx);
                iy = 2 +  int((part->y[i] - grd->yStart)*grd->invdy);
                iz = 2 +  int((part->z[i] - grd->zStart)*grd->invdz);
                
                // calculate weights
                xi[0]   = part->x[i] - grd->XN[ix - 1][iy][iz];
                eta[0]  = part->y[i] - grd->YN[ix][iy - 1][iz];
                zeta[0] = part->z[i] - grd->ZN[ix][iy][iz - 1];
                xi[1]   = grd->XN[ix][iy][iz] - part->x[i];
                eta[1]  = grd->YN[ix][iy][iz] - part->y[i];
                zeta[1] = grd->ZN[ix][iy][iz] - part->z[i];
                for (int ii = 0; ii < 2; ii++)
                    for (int jj = 0; jj < 2; jj++)
                        for (int kk = 0; kk < 2; kk++)
                            weight[ii][jj][kk] = xi[ii] * eta[jj] * zeta[kk] * grd->invVOL;
                
                // set to zero local electric and magnetic field
                Exl=0.0, Eyl = 0.0, Ezl = 0.0, Bxl = 0.0, Byl = 0.0, Bzl = 0.0;
                
                for (int ii=0; ii < 2; ii++)
                    for (int jj=0; jj < 2; jj++)
                        for(int kk=0; kk < 2; kk++){
                            Exl += weight[ii][jj][kk]*field->Ex[ix- ii][iy -jj][iz- kk ];
                            Eyl += weight[ii][jj][kk]*field->Ey[ix- ii][iy -jj][iz- kk ];
                            Ezl += weight[ii][jj][kk]*field->Ez[ix- ii][iy -jj][iz -kk ];
                            Bxl += weight[ii][jj][kk]*field->Bxn[ix- ii][iy -jj][iz -kk ];
                            Byl += weight[ii][jj][kk]*field->Byn[ix- ii][iy -jj][iz -kk ];
                            Bzl += weight[ii][jj][kk]*field->Bzn[ix- ii][iy -jj][iz -kk ];
                        }
                
                // end interpolation
                omdtsq = qomdt2*qomdt2*(Bxl*Bxl+Byl*Byl+Bzl*Bzl);
                denom = 1.0/(1.0 + omdtsq);
                // solve the position equation
                ut= part->u[i] + qomdt2*Exl;
                vt= part->v[i] + qomdt2*Eyl;
                wt= part->w[i] + qomdt2*Ezl;
                udotb = ut*Bxl + vt*Byl + wt*Bzl;
                // solve the velocity equation
                uptilde = (ut+qomdt2*(vt*Bzl -wt*Byl + qomdt2*udotb*Bxl))*denom;
                vptilde = (vt+qomdt2*(wt*Bxl -ut*Bzl + qomdt2*udotb*Byl))*denom;
                wptilde = (wt+qomdt2*(ut*Byl -vt*Bxl + qomdt2*udotb*Bzl))*denom;
                // update position
                part->x[i] = xptilde + uptilde*dto2;
                part->y[i] = yptilde + vptilde*dto2;
                part->z[i] = zptilde + wptilde*dto2;
                
                
            } // end of iteration
            // update the final position and velocity
            part->u[i]= 2.0*uptilde - part->u[i];
            part->v[i]= 2.0*vptilde - part->v[i];
            part->w[i]= 2.0*wptilde - part->w[i];
            part->x[i] = xptilde + uptilde*dt_sub_cycling;
            part->y[i] = yptilde + vptilde*dt_sub_cycling;
            part->z[i] = zptilde + wptilde*dt_sub_cycling;
            
            
            //////////
            //////////
            ////////// BC
                                        
            // X-DIRECTION: BC particles
            if (part->x[i] > grd->Lx){
                if (param->PERIODICX==true){ // PERIODIC
                    part->x[i] = part->x[i] - grd->Lx;
                } else { // REFLECTING BC
                    part->u[i] = -part->u[i];
                    part->x[i] = 2*grd->Lx - part->x[i];
                }
            }
                                                                        
            if (part->x[i] < 0){
                if (param->PERIODICX==true){ // PERIODIC
                   part->x[i] = part->x[i] + grd->Lx;
                } else { // REFLECTING BC
                    part->u[i] = -part->u[i];
                    part->x[i] = -part->x[i];
                }
            }
                
            
            // Y-DIRECTION: BC particles
            if (part->y[i] > grd->Ly){
                if (param->PERIODICY==true){ // PERIODIC
                    part->y[i] = part->y[i] - grd->Ly;
                } else { // REFLECTING BC
                    part->v[i] = -part->v[i];
                    part->y[i] = 2*grd->Ly - part->y[i];
                }
            }
                                                                        
            if (part->y[i] < 0){
                if (param->PERIODICY==true){ // PERIODIC
                    part->y[i] = part->y[i] + grd->Ly;
                } else { // REFLECTING BC
                    part->v[i] = -part->v[i];
                    part->y[i] = -part->y[i];
                }
            }
                                                                        
            // Z-DIRECTION: BC particles
            if (part->z[i] > grd->Lz){
                if (param->PERIODICZ==true){ // PERIODIC
                    part->z[i] = part->z[i] - grd->Lz;
                } else { // REFLECTING BC
                    part->w[i] = -part->w[i];
                    part->z[i] = 2*grd->Lz - part->z[i];
                }
            }
                                                                        
            if (part->z[i] < 0){
                if (param->PERIODICZ==true){ // PERIODIC
                    part->z[i] = part->z[i] + grd->Lz;
                } else { // REFLECTING BC
                    part->w[i] = -part->w[i];
                    part->z[i] = -part->z[i];
                }
            }
                                                                        
            
            
        } // end of one particle
    } // end of subcycling
                                                                        
    return(0); // exit succcesfully
} // end of the mover



/** Interpolation Particle --> Grid: This is for species */
void interpP2G(struct particles* part, struct interpDensSpecies* ids, struct grid* grd)
{
    
    // arrays needed for interpolation
    FPpart weight[2][2][2];
    FPpart temp[2][2][2];
    FPpart xi[2], eta[2], zeta[2];
    
    // index of the cell
    int ix, iy, iz;
    
    
    for (register long long i = 0; i < part->nop; i++) {
        
        // determine cell: can we change to int()? is it faster?
        ix = 2 + int (floor((part->x[i] - grd->xStart) * grd->invdx));
        iy = 2 + int (floor((part->y[i] - grd->yStart) * grd->invdy));
        iz = 2 + int (floor((part->z[i] - grd->zStart) * grd->invdz));
        
        // distances from node
        xi[0]   = part->x[i] - grd->XN[ix - 1][iy][iz];
        eta[0]  = part->y[i] - grd->YN[ix][iy - 1][iz];
        zeta[0] = part->z[i] - grd->ZN[ix][iy][iz - 1];
        xi[1]   = grd->XN[ix][iy][iz] - part->x[i];
        eta[1]  = grd->YN[ix][iy][iz] - part->y[i];
        zeta[1] = grd->ZN[ix][iy][iz] - part->z[i];
        
        // calculate the weights for different nodes
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    weight[ii][jj][kk] = part->q[i] * xi[ii] * eta[jj] * zeta[kk] * grd->invVOL;
        
        //////////////////////////
        // add charge density
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->rhon[ix - ii][iy - jj][iz - kk] += weight[ii][jj][kk] * grd->invVOL;
        
        
        ////////////////////////////
        // add current density - Jx
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->u[i] * weight[ii][jj][kk];
        
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->Jx[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
        
        
        ////////////////////////////
        // add current density - Jy
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->v[i] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->Jy[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
        
        
        
        ////////////////////////////
        // add current density - Jz
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->w[i] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->Jz[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
        
        
        ////////////////////////////
        // add pressure pxx
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->u[i] * part->u[i] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->pxx[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
        
        
        ////////////////////////////
        // add pressure pxy
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->u[i] * part->v[i] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->pxy[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
        
        
        
        /////////////////////////////
        // add pressure pxz
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->u[i] * part->w[i] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->pxz[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
        
        
        /////////////////////////////
        // add pressure pyy
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->v[i] * part->v[i] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->pyy[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
        
        
        /////////////////////////////
        // add pressure pyz
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->v[i] * part->w[i] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->pyz[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
        
        
        /////////////////////////////
        // add pressure pzz
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->w[i] * part->w[i] * weight[ii][jj][kk];
        for (int ii=0; ii < 2; ii++)
            for (int jj=0; jj < 2; jj++)
                for(int kk=0; kk < 2; kk++)
                    ids->pzz[ix -ii][iy -jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
    
    }
   
}
