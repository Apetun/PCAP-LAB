#include <mpi.h>
#include <stdio.h>

int main(int argc, char** argv) {
    int rank, size;
    const int a = 10, b = 5;  // Initializing the two variables

    // Initialize the MPI environment
    MPI_Init(&argc, &argv);

    // Get the rank and size of the processes
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Ensure there are at least 4 processes
    if (size < 4) {
        if (rank == 0) {
            printf("Please run the program with at least 4 processes.\n");
        }
        MPI_Finalize();
        return 0;
    }

    // Each process performs a different operation
    if (rank == 0) {
        int sum = a + b;
        printf("Process %d: Addition: %d + %d = %d\n", rank, a, b, sum);
    } else if (rank == 1) {
        int difference = a - b;
        printf("Process %d: Subtraction: %d - %d = %d\n", rank, a, b, difference);
    } else if (rank == 2) {
        int product = a * b;
        printf("Process %d: Multiplication: %d * %d = %d\n", rank, a, b, product);
    } else if (rank == 3) {
        if (b != 0) {
            double quotient = (double)a / b;
            printf("Process %d: Division: %d / %d = %.2f\n", rank, a, b, quotient);
        } else {
            printf("Process %d: Division by zero error.\n", rank);
        }
    }

    // Finalize the MPI environment
    MPI_Finalize();
    return 0;
}
