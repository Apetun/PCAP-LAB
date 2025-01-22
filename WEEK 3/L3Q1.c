#include <stdio.h>
#include <stdlib.h>
#include "mpi.h"

int main(int argc,char *argv[])
{
	int rank,size;
	MPI_Init(&argc,&argv);
	MPI_Comm_rank(MPI_COMM_WORLD,&rank);
	MPI_Comm_size(MPI_COMM_WORLD,&size);

	int a[size];
	if (rank==0)
	{
		fprintf(stdout,"Enter %d values:\n",size);
		fflush(stdout);
		for(int t=0;t<size;t++)
			scanf("%d",&a[t]);
	}
	int out_a[size];
	int c;
	MPI_Scatter(a,1,MPI_INT,&c,1,MPI_INT,0,MPI_COMM_WORLD);
	fprintf(stdout,"I have received %d in process %d\n",c,rank);
	fflush(stdout);
	int f=1;
	for(int r=c;r>1;r--)
		f=f*r;

	MPI_Gather(&f,1,MPI_INT,out_a,1,MPI_INT,0,MPI_COMM_WORLD);
	if (rank==0){
		fprintf(stdout,"The sum of all factorials in the root: \n");
		fflush(stdout);
		int sum=0;
		for (int t=0;t<size;t++)
			sum+=out_a[t];
		printf("Sum = %d\n",sum);
	}
	MPI_Finalize();
	return 0;
}