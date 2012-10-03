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
    int i;
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
    printf("\n GetNbProcInfo passed, proc %d \n", rank);

    hpInitPInfo(mesh);
    printf("\n InitPInfo passed, proc %d \n", rank);

    hpBuildPInfoNoOverlappingTris(mesh);
    printf("\n BuildPInfo passed, proc %d \n", rank);

    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0)
    {
	printf("ps pinfo of rank 0:\n");
	for (i = 1; i <= mesh->ps->size[0]; i++)
	{
	    printf("point %d: ", i);
	    int next = mesh->ps_pinfo->head[I1dm(i)];
	    while (next != -1)
	    {
		int cur_node = next;
		printf("%d/%d-->", mesh->ps_pinfo->pdata[I1dm(cur_node)].proc, mesh->ps_pinfo->pdata[I1dm(cur_node)].lindex);
		next = mesh->ps_pinfo->pdata[I1dm(cur_node)].next;
	    }
	    printf("\n");
	}
    }

    hpFreeMesh(&mesh);
    printf("Success processor %d\n", rank);
    MPI_Finalize();

    return 1;
}
