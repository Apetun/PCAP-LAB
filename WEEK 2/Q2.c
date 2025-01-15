#include <mpi.h>
#include <stdio.h>

int main(int argc, char *argv[]) {
    int rank, size, n;
    MPI_Status status;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        // Master process
        printf("Enter a number: ");
        scanf("%d", &n);

        // Send the number to each slave process (rank 1 to size-1)
        for (int i = 1; i < size; i++) {
            MPI_Send(&n, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
            printf("Process 0 sent number %d to process %d\n", n, i);
        }
    } else {
        // Slave processes (rank 1 to size-1)
        MPI_Recv(&n, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
        printf("Rank %d received: %d\n", rank, n);
    }

    MPI_Finalize();
    return 0;
}
