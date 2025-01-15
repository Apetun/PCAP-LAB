#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

// Function to reverse the digits of a number
int reverse_digits(int num) {
    int reversed = 0;
    while (num > 0) {
        reversed = reversed * 10 + (num % 10);
        num /= 10;
    }
    return reversed;
}

int main(int argc, char** argv) {
    int rank, size;
    int input_array[9] = {18, 523, 301, 1234, 2, 14, 108, 150, 1928};
    int output_array[9];

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size != 9) {
        if (rank == 0) {
            printf("Error: This program requires exactly 9 processes.\n");
        }
        MPI_Finalize();
        return 1;
    }

    // Each process reverses the element at its index
    int reversed_number = reverse_digits(input_array[rank]);

    // Gather results from all processes
    MPI_Gather(&reversed_number, 1, MPI_INT, output_array, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // The root process prints the result
    if (rank == 0) {
        printf("Reversed array: ");
        for (int i = 0; i < 9; i++) {
            printf("%d ", output_array[i]);
        }
        printf("\n");
    }

    MPI_Finalize();
    return 0;
}
