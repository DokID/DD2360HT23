
#include <stdio.h>
#include <sys/time.h>

#define DataType double

// Compute C = A * B
__global__ void gemm(DataType *A, DataType *B, DataType *C, int numARows,
                      int numAColumns, int numBRows, int numBColumns){
  //@@ Insert code to implement matrix multiplication here

}

int main(int argc, char **argv) {
  
  DataType *hostA; // The A matrix
  DataType *hostB; // The B matrix
  DataType *hostC; // The output C matrix
  DataType *resultRef; // The reference result
  DataType *deviceA;
  DataType *deviceB;
  DataType *deviceC;
  int numARows;    // number of rows in the matrix A
  int numAColumns; // number of columns in the matrix A
  int numBRows;    // number of rows in the matrix B
  int numBColumns; // number of columns in the matrix B
  int numCRows;
  int numCColumns;

  //@@ Insert code below to read in numARows, numAColumns, 
  numARows = std::stoi(argv[1], nullptr);   
  numAColumns = std::stoi(argv[2], nullptr);
  numBRows = std::stoi(argv[3], nullptr);   
  numBColumns = std::stoi(argv[4], nullptr);
  numCRows = std::stoi(argv[5], nullptr);
  numCColumns = std::stoi(argv[6], nullptr);

  printf("Input matrix dim (%d x %d) (%d x %d) (%d x %d)\n", numARows, numAColumns, numBRows, numBColumns, numCRows, numCColumns);

  // check dims match
  if !((numARows == numCRows) && (numBColumns == numCColumns) && (numAColumns == numBRows)) {
    fprintf(stderr, "Input dim mismatch.\n");
    return -1;
  }
  
  //@@ Insert code below to allocate Host memory for input and output
  // index by offset = i + numAColumns * j
  DataType *hostA = (DataType *) malloc(numARows * numAColumns * sizeof(DataType));
  // index by 
  DataType **hostA = (DataType **) malloc(numARows * sizeof(DataType *));
  for(int i = 0; i < numARows; i++) hostA[i] = (DataType *)malloc(numAColumns * sizeof(DataType));
  
  //@@ Insert code below to initialize hostA and hostB to random numbers, and create reference result in CPU


  //@@ Insert code below to allocate GPU memory here


  //@@ Insert code to below to Copy memory to the GPU here


  //@@ Initialize the grid and block dimensions here


  //@@ Launch the GPU Kernel here


  //@@ Copy the GPU memory back to the CPU here


  //@@ Insert code below to compare the output with the reference


  //@@ Free the GPU memory here


  //@@ Free the CPU memory here


  return 0;
}
