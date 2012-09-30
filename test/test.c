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
    int num_proc, rank;

    MPI_Init(&argc, &argv);

    MPI_Comm_size(MPI_COMM_WORLD, &num_proc);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    printf("\n Welcome to the test of Hi-Prop Library from proc %d\n", rank);

    hiPropMesh *mesh;
    hpInitMesh(&mesh);

    char in_filename[200];
    char rank_str[5];
    right_flush(rank,4,rank_str);
    sprintf(in_filename, "data/parallel/%s-p%s.vtk", argv[1], rank_str);
    if (!hpReadUnstrMeshVtk3d(in_filename, mesh))
    {
	printf("Reading fail!\n");
	return 0;
    }

    hpGetNbProcListAuto(mesh);


    hpFreeMesh(&mesh);
    printf("Success processor %d\n", rank);
    MPI_Finalize();

    return 1;
}
