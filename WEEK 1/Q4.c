#include <mpi.h>
#include <stdio.h>
#include <ctype.h>
#include <string.h>

int main(int argc, char** argv) {
    int rank, size;
    char str[] = "HELLO";  // Input string

    // Initialize MPI environment
    MPI_Init(&argc, &argv);

    // Get the rank and size of the processes
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int length = strlen(str);

    // Ensure there are enough processes for the string length
    if (size < length) {
        if (rank == 0) {
            printf("Please run the program with at least %d processes.\n", length);
        }
        MPI_Finalize();
        return 0;
    }

    // Each process toggles the character based on its rank
    if (rank < length) {
        if (isupper(str[rank])) {
            str[rank] = tolower(str[rank]);
        } else if (islower(str[rank])) {
            str[rank] = toupper(str[rank]);
        }
        printf("Process %d toggled: %c\n", rank, str[rank]);
    }

    // Finalize MPI environment
    MPI_Finalize();
    return 0;
}
