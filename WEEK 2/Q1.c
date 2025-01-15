#include <mpi.h>
#include <stdio.h>
#include <string.h>

int main(int argc, char *argv[]) {
    char s[100];
    int len, rank;
    MPI_Status status;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0) {
        // Sender process (rank 0)
        printf("Enter word: \n");
        scanf("%99s", s);  // Read input safely, limit to 99 chars
        len = strlen(s);

        // Send the length of the string to process 1
        MPI_Ssend(&len, 1, MPI_INT, 1, 1, MPI_COMM_WORLD);
        printf("Process 0 sent word length: %d\n", len);

        // Send the string to process 1
        MPI_Ssend(s, len, MPI_CHAR, 1, 2, MPI_COMM_WORLD);
        printf("Process 0 sent word: %s\n", s);

        // Receive the toggled string from process 1
        MPI_Recv(s, len, MPI_CHAR, 1, 3, MPI_COMM_WORLD, &status);
        printf("Process 0 received modified word: %s\n", s);
    }
    else {
        // Receiver process (rank 1)
        // Receive the length of the string
        MPI_Recv(&len, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, &status);
        printf("Process 1 received word length: %d\n", len);

        // Receive the string from process 0
        MPI_Recv(s, len, MPI_CHAR, 0, 2, MPI_COMM_WORLD, &status);
        printf("Process 1 received word: %s\n", s);

        // Toggle the case of each character in the string
        for (int i = 0; i < len; i++) {
            if (s[i] >= 'a' && s[i] <= 'z') {
                s[i] = s[i] - 32;  // Convert lowercase to uppercase
            } else if (s[i] >= 'A' && s[i] <= 'Z') {
                s[i] = s[i] + 32;  // Convert uppercase to lowercase
            }
        }

        // Send the toggled string back to process 0
        MPI_Ssend(s, len, MPI_CHAR, 0, 3, MPI_COMM_WORLD);
        printf("Process 1 sent modified word: %s\n", s);
    }

    MPI_Finalize();
    return 0;
}
