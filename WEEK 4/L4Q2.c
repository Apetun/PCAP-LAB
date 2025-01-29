#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char **argv) {

  int rank, size;
  int err;
  char estr[50];
  int len = 50;  // Change len to int instead of size_t

  MPI_Init(&argc, &argv);
  err = MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (err != MPI_SUCCESS) {
    MPI_Error_string(err, estr, &len);
    printf("Error in MPI_Comm_rank: %s\n", estr);
    MPI_Abort(MPI_COMM_WORLD, err);
    return -1;
  }

  err = MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (err != MPI_SUCCESS) {
    MPI_Error_string(err, estr, &len);
    printf("Error in MPI_Comm_size: %s\n", estr);
    MPI_Abort(MPI_COMM_WORLD, err);
    return -1;
  }

  MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN);

  int matrix[3][3];
  int element;

  if (rank == 0) {
    printf("Enter the matrix:\n");
    for (int t = 0; t < 9; ++t)
      scanf("%d", &matrix[t/3][t%3]);
    printf("Enter the element: ");
    scanf("%d", &element);
  }

  MPI_Bcast(&element, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(matrix, 9, MPI_INT, 0, MPI_COMM_WORLD);

  int row[3];
  MPI_Scatter(matrix, 3, MPI_INT, row, 3, MPI_INT, 0, MPI_COMM_WORLD);

  int count = 0;
  for (int t = 0; t < 3; ++t) {
    if (row[t] == element) {
      count++;
    }
  }

  int total;
  MPI_Reduce(&count, &total, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
  if (rank == 0) {
    printf("Total number of occurrences: %d\n", total);
  }

  MPI_Finalize();

  return 0;
}
