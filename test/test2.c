/*
 *	This file is to test input and output utilities in VTK format
 *
 */

#include "stdafx.h"
#include "util.h"
#include "hiprop.h"

int main(int argc, char* argv[])
{
    int i;
    int num_proc, rank;
    MPI_Init(&argc, &argv);

    MPI_Comm_size(MPI_COMM_WORLD, &num_proc);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    char rank_str[5];

    char in_filename[250];
    char out_filename[250];

    right_flush(rank,4,rank_str);

    sprintf(in_filename, "%s-p%s.vtk", argv[1], rank_str);
    sprintf(out_filename, "output-p%s.vtk", rank_str);

    hiPropMesh *mesh;

    hpInitMesh(&mesh);

    if(!hpReadUnstrMeshVtk3d(in_filename, mesh))
    {
	printf("Mesh Read fail\n");
	return 0;
    }

    int num_ps_all = mesh->ps->allocatedSize;

    if (rank == 0)
    {
	for (i = 1; i <= num_ps_all; i++)
	    mesh->ps->data[i-1] = mesh->ps->data[i-1] + 1.0;
    }
    else if (rank == 1)
    {
	for (i = 1; i <= num_ps_all; i++)
	    mesh->ps->data[i-1] = mesh->ps->data[i-1] + 5.0;
    }

    if(!hpWriteUnstrMeshVtk3d(out_filename, mesh))
    {
	printf("Mesh Write fail\n");
	return 0;
    }

    hpDeleteMesh(&mesh);

    if (mesh == (hiPropMesh *)NULL)
	printf("Success processor %d\n", rank);

    MPI_Finalize();

    return 1;
}


