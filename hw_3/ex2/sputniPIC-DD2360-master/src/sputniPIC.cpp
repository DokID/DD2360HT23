/** A mixed-precision implicit Particle-in-Cell simulator for heterogeneous systems **/

// Allocator for 2D, 3D and 4D array: chain of pointers
#include "Alloc.h"

// Precision: fix precision for different quantities
#include "PrecisionTypes.h"
// Simulation Parameter - structure
#include "Parameters.h"
// Grid structure
#include "Grid.h"
// Interpolated Quantities Structures
#include "InterpDensSpecies.h"
#include "InterpDensNet.h"

// Field structure
#include "EMfield.h" // Just E and Bn
#include "EMfield_aux.h" // Bc, Phi, Eth, D

// Particles structure
#include "Particles.h"
#include "Particles_aux.h" // Needed only if dointerpolation on GPU - avoid reduction on GPU

// Initial Condition
#include "IC.h"
// Boundary Conditions
#include "BC.h"
// timing
#include "Timing.h"
// Read and output operations
#include "RW_IO.h"


int main(int argc, char **argv){
    
    // Read the inputfile and fill the param structure
    parameters param;
    // Read the input file name from command line
    int conf = readInputFile(&param,argc,argv);
    bool use_gpu = conf;
    if (conf > 1) {
        string files[] = {"B_10.vtk", "E_10.vtk", "rho_net_10.vtk", "rhoe_10.vtk", "rhoi_10.vtk", "sputniPICparameters.txt"};
        string new_filename, old_filename, tmp1, tmp2;
        int line;
        bool match;
        for (int i = 0; i < 6; i++) {
            new_filename = "./data/";
            new_filename += files[i];
            old_filename = "./old/";
            old_filename += files[i];
            printf("Comparing \'%s\' to \'%s\'\n", new_filename.c_str(), old_filename.c_str());

            std::ifstream newFile;
            std::ifstream oldFile;

            newFile.open(new_filename);
            oldFile.open(old_filename);
            if (newFile.fail() || oldFile.fail()) break;

            line = 0;
            match = true;
            while(getline(newFile, tmp1)) {
                line++;
                getline(oldFile, tmp2);
                if (tmp1 != tmp2) {
                    printf("File mismatch on line %d!\n", line);
                    printf("%s\nvs\n%s\n", tmp1.c_str(), tmp2.c_str());
                    match = false;
                    break;
                }
            }
            if (match) {
                printf("Files match!\n");
            }
        }
        return 0;
    }
    printParameters(&param);
    saveParameters(&param);
    
    // Timing variables
    double iStart = cpuSecond();
    double iMover, iInterp, eMover = 0.0, eInterp= 0.0;
    
    // Set-up the grid information
    grid grd;
    setGrid(&param, &grd);
    
    // Allocate Fields
    EMfield field;
    field_allocate(&grd,&field);
    EMfield_aux field_aux;
    field_aux_allocate(&grd,&field_aux);
    
    
    // Allocate Interpolated Quantities
    // per species
    interpDensSpecies *ids = new interpDensSpecies[param.ns];
    for (int is=0; is < param.ns; is++)
        interp_dens_species_allocate(&grd,&ids[is],is);
    // Net densities
    interpDensNet idn;
    interp_dens_net_allocate(&grd,&idn);
    
    // Allocate Particles
    particles *part = new particles[param.ns];
    // allocation
    for (int is=0; is < param.ns; is++){
        particle_allocate(&param,&part[is],is);
    }
    
    // Initialization
    initGEM(&param,&grd,&field,&field_aux,part,ids);
    
    //bool use_gpu = false;
    if (use_gpu) {
        printf("\nUsing GPU\n");
    }
    
    // **********************************************************//
    // **** Start the Simulation!  Cycle index start from 1  *** //
    // **********************************************************//
    for (int cycle = param.first_cycle_n; cycle < (param.first_cycle_n + param.ncycles); cycle++) {
        
        std::cout << std::endl;
        std::cout << "***********************" << std::endl;
        std::cout << "   cycle = " << cycle << std::endl;
        std::cout << "***********************" << std::endl;
    
        // set to zero the densities - needed for interpolation
        setZeroDensities(&idn,ids,&grd,param.ns);
        
        //printf("test\n");
        //printf("index 0 0 0: %f\n", (float) grd.XN[2][2][2]);
        //grd.XN_flat[2*grd.nyn*grd.nzn + 2*grd.nzn + 2] += grd.XN_flat[2*grd.nyn*grd.nzn + 2*grd.//nzn + 2];
        //printf("index 0 0 0: %f\n", (float) grd.XN[2][2][2]);
        
        // implicit mover
        iMover = cpuSecond(); // start timer for mover
        for (int is=0; is < param.ns; is++) {
            if (use_gpu) {
                //printf("use gpu lol\n");
                mover_PC_GPU(&part[is],&field,&grd,&param);
                //mover_PC(&part[is],&field,&grd,&param);
            } else {
                mover_PC(&part[is],&field,&grd,&param);
            }
        }
        eMover += (cpuSecond() - iMover); // stop timer for mover
        
        
        
        
        // interpolation particle to grid
        iInterp = cpuSecond(); // start timer for the interpolation step
        // interpolate species
        for (int is=0; is < param.ns; is++)
            interpP2G(&part[is],&ids[is],&grd);
        // apply BC to interpolated densities
        for (int is=0; is < param.ns; is++)
            applyBCids(&ids[is],&grd,&param);
        // sum over species
        sumOverSpecies(&idn,ids,&grd,param.ns);
        // interpolate charge density from center to node
        applyBCscalarDensN(idn.rhon,&grd,&param);
        
        
        
        // write E, B, rho to disk
        if (cycle%param.FieldOutputCycle==0){
            VTK_Write_Vectors(cycle, &grd,&field);
            VTK_Write_Scalars(cycle, &grd,ids,&idn);
        }
        
        eInterp += (cpuSecond() - iInterp); // stop timer for interpolation
        
        
    
    }  // end of one PIC cycle
    
    /// Release the resources
    // deallocate field
    grid_deallocate(&grd);
    field_deallocate(&grd,&field);
    // interp
    interp_dens_net_deallocate(&grd,&idn);
    
    // Deallocate interpolated densities and particles
    for (int is=0; is < param.ns; is++){
        interp_dens_species_deallocate(&grd,&ids[is]);
        particle_deallocate(&part[is]);
    }
    
    
    // stop timer
    double iElaps = cpuSecond() - iStart;
    
    // Print timing of simulation
    std::cout << std::endl;
    std::cout << "**************************************" << std::endl;
    std::cout << "   Tot. Simulation Time (s) = " << iElaps << std::endl;
    std::cout << "   Mover Time / Cycle   (s) = " << eMover/param.ncycles << std::endl;
    std::cout << "   Interp. Time / Cycle (s) = " << eInterp/param.ncycles  << std::endl;
    std::cout << "**************************************" << std::endl;

    
    // exit
    return 0;
}
