#include "mpi.h"
#include <stdio.h>

void ErrorHandler(int ecode)
{
    if (ecode != MPI_SUCCESS)
    {
        char err_str[BUFSIZ];
        int strlen, err_class;

        MPI_Error_class(ecode, &err_class);       
        MPI_Error_string(err_class, err_str, &strlen); 

        
        printf("MPI Error Code: %d\n", ecode); 
        printf("MPI Error Class: %d\n", err_class); 
        printf("MPI Error Message: %s\n", err_str);  
}

int main(int argc, char *argv[])
{
    int rank, size, fact = 1, factsum = 0, i;
    int ecode;

    // Error 1: Calling MPI_Init multiple times (Uncomment to test)
    ecode = MPI_Init(&argc, &argv);
    if (ecode != MPI_SUCCESS)
    {
        ErrorHandler(ecode);
        MPI_Finalize();
        return -1;
    }

    // Uncommenting below will cause an error for calling MPI_Init twice
    // ecode = MPI_Init(&argc, &argv); // Second call (ERROR)
    // ErrorHandler(ecode);

    // Set the error handler to return errors instead of aborting the program
    ecode = MPI_Errhandler_set(MPI_COMM_WORLD, MPI_ERRORS_RETURN);
    if (ecode != MPI_SUCCESS)
    {
        ErrorHandler(ecode);
        MPI_Finalize();
        return -1;
    }

    // Error 2: Invalid Communicator (MPI_COMM_NULL) – Uncomment to test
    ecode = MPI_Comm_rank(MPI_COMM_NULL, &rank); // Invalid communicator
    if (ecode != MPI_SUCCESS)
    {
        printf("Error in MPI_Comm_rank with MPI_COMM_NULL\n");
        ErrorHandler(ecode);
    }

    // Error 3: Mismatched types in MPI_Send and MPI_Recv
    int send_data = 42;
    float recv_data; // Mismatched type (float instead of int)
    ecode = MPI_Send(&send_data, 1, MPI_INT, 1, 0, MPI_COMM_WORLD); // This will work
    if (ecode != MPI_SUCCESS)
    {
        ErrorHandler(ecode);
        MPI_Finalize();
        return -1;
    }

    ecode = MPI_Recv(&recv_data, 1, MPI_FLOAT, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE); // Mismatched type
    if (ecode != MPI_SUCCESS)
    {
        printf("Error in MPI_Recv with mismatched type\n");
        ErrorHandler(ecode);
    }

    // Get the rank and size of the communicator
    ecode = MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (ecode != MPI_SUCCESS)
    {
        ErrorHandler(ecode);
        MPI_Finalize();
        return -1;
    }

    ecode = MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (ecode != MPI_SUCCESS)
    {
        ErrorHandler(ecode);
        MPI_Finalize();
        return -1;
    }

    // Calculate the factorial for this rank
    for (i = 1; i <= rank + 1; i++)
    {
        fact *= i;
    }

    // Perform a scan operation to calculate the cumulative sum of factorials
    ecode = MPI_Scan(&fact, &factsum, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    if (ecode != MPI_SUCCESS)
    {
        ErrorHandler(ecode);
        MPI_Finalize();
        return -1;
    }

    // Print the result
    printf("Rank %d: Factorial = %d, Sum of all factorials so far: %d\n", rank, fact, factsum);

    // Finalize the MPI environment
    ecode = MPI_Finalize();
    if (ecode != MPI_SUCCESS)
    {
        ErrorHandler(ecode);
        return -1;
    }

    return 0;
}
