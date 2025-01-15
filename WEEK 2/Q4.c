#include <mpi.h>
#include <stdio.h>
#include <string.h>

int main(int argc, char *argv[]) {
    int rank, size, n;
    MPI_Status status;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        // Root process reads the integer value
        printf("Enter number: ");
        scanf("%d", &n);

        // Increment the value
        n = n + 1;

        // Send the value to process 1
        MPI_Send(&n, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);

        // Receive the final incremented value from the last process (size-1)
        MPI_Recv(&n, 1, MPI_INT, size - 1, 0, MPI_COMM_WORLD, &status);
        printf("Rank %d received final value: %d\n", rank, n);
    } else if (rank == size - 1) {
        // Last process (size-1)
        MPI_Recv(&n, 1, MPI_INT, rank - 1, 0, MPI_COMM_WORLD, &status);
        printf("Rank %d received: %d\n", rank, n);

        // Increment the value
        n = n + 1;

        // Send the value back to the root process (rank 0)
        MPI_Send(&n, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
    } else {
        // Intermediate processes
        MPI_Recv(&n, 1, MPI_INT, rank - 1, 0, MPI_COMM_WORLD, &status);
        printf("Rank %d received: %d\n", rank, n);

        // Increment the value
        n = n + 1;

        // Send the incremented value to the next process
        MPI_Send(&n, 1, MPI_INT, rank + 1, 0, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}
