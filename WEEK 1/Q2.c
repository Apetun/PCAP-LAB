#include <mpi.h>
#include <stdio.h>

int main(int argc, char** argv) {
    int rank, size;

    // Initialize the MPI environment
    MPI_Init(&argc, &argv);

    // Get the rank of the process
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Check if the rank is even or odd and print accordingly
    if (rank % 2 == 0) {
        printf("Process %d: Hello\n", rank);
    } else {
        printf("Process %d: World\n", rank);
    }

    // Finalize the MPI environment
    MPI_Finalize();

    return 0;
}
