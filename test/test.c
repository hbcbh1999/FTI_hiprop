/*!
 * \file test.c
 * \brief Test file of hi-prop library
 *
 * \author Yijie Zhou
 * \date 2012.08.23
 */

#include "stdafx.h"

#include "util.h"
#include "hiprop.h"

int main (int argc, char *argv[])
{

    int i,j,k;
    k = 1;
    int num_proc, rank;
    int tag = 1;

    MPI_Init(&argc, &argv);

    MPI_Comm_size(MPI_COMM_WORLD, &num_proc);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);


    if (rank == 0)
	printf("\n Welcome to the test of Hi-Prop Library \n");


    if (rank == 0)
    {
	emxArray_int32_T *test = emxCreate_int32_T(4, 3);
	for (i = 1; i <= 4; i++)
	    for (j = 1; j <=3; j++)
		test->data[I2dm(i,j,test->size)] = k++;

	printf("\nIn proc 0, before sending, the array is:\n");
    	printArray_int32_T(test);
    	addColumnToArray_int32_T(test, 2);
    	addRowToArray_int32_T(test, 4);
	send2D_int32_T(test, 1, tag, MPI_COMM_WORLD);
    	emxFree_int32_T(&test);
    }
    else if (rank == 1)
    {
	emxArray_int32_T *test_recv;

	recv2D_int32_T(&test_recv, 0, tag, MPI_COMM_WORLD);

	printf("\nIn proc 1, after receiving, the array is:\n");

	printArray_int32_T(test_recv);
    }


    MPI_Finalize();

    return 0;
}
