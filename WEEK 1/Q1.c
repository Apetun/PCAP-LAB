#include "mpi.h"
#include <stdio.h>
#include <math.h>

int main(int argc, char** argv) {
    int rank, size;
    const int x = 2;  // Integer constant

    // Initialize the MPI environment
    MPI_Init(&argc, &argv);

    // Get the rank of the process
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Calculate pow(x, rank)
    double result = pow(x, rank);

    // Print the result from each process
    printf("Process %d: %d^%d = %.2f\n", rank, x, rank, result);

    // Finalize the MPI environment
    MPI_Finalize();

    return 0;
}
