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
    int tag = 1;
    int root = 0;


    MPI_Init(&argc, &argv);

    MPI_Comm_size(MPI_COMM_WORLD, &num_proc);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    char runlog_filename[200];
    char rank_str[5];
    right_flush(rank,4,rank_str);
    sprintf(runlog_filename, "run-log.%s",rank_str);
    FILE *runlog_stream;
    if((runlog_stream = freopen(runlog_filename, "w", stdout)) == NULL)
	exit(-1);

    printf("\n Welcome to the test of Hi-Prop Library from proc %d\n", rank);

    hiPropMesh *mesh;
    hpInitMesh(&mesh);

    char in_filename[200];
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

    hpBuildPUpdateInfo(mesh);
    printf("\n BuildPUpdateInfo passed, proc %d \n", rank);

    hpBuildOppositeHalfEdge(mesh);
    printf("\n BuildOppHalfEdge passed, proc %d \n", rank);

    hpBuildIncidentHalfEdge(mesh);
    printf("\n BuildIncidentHalfEdge passed, proc %d \n", rank);

    printf("\nBefore build nring ghost \n");
    for (i = 1; i <= mesh->ps->size[0]; i++)
    {
	printf("%d-th point: ", i);
	int cur_node = mesh->ps_pinfo->head[I1dm(i)];
	while (cur_node != -1)
	{
	    printf("%d/%d-->", mesh->ps_pinfo->pdata[I1dm(cur_node)].proc, mesh->ps_pinfo->pdata[I1dm(cur_node)].lindex);
	    cur_node = mesh->ps_pinfo->pdata[I1dm(cur_node)].next;
	}
	printf("\n");
    }

    hpBuildNRingGhost(mesh, 2);

    printf("\n BuildNRingGhost passed, proc %d \n", rank);
    char debug_filename[200];
    sprintf(debug_filename, "debugout-p%s.vtk", rank_str);
    hpWriteUnstrMeshWithPInfo(debug_filename, mesh);

    printf("\nAfter build nring ghost \n");
    for (i = 1; i <= mesh->ps->size[0]; i++)
    {
	printf("%d-th point: ", i);
	int cur_node = mesh->ps_pinfo->head[I1dm(i)];
	while (cur_node != -1)
	{
	    printf("%d/%d-->", mesh->ps_pinfo->pdata[I1dm(cur_node)].proc, mesh->ps_pinfo->pdata[I1dm(cur_node)].lindex);
	    cur_node = mesh->ps_pinfo->pdata[I1dm(cur_node)].next;
	}
	printf("\n");
    }


    hpBuildPUpdateInfo(mesh);
//    printf("\n BuildPUpdateInfo passed, proc %d \n", rank);

//    hpBuildOppositeHalfEdge(mesh);
//    printf("\n BuildOppHalfEdge passed, proc %d \n", rank);
//
//    hpBuildIncidentHalfEdge(mesh);
//    printf("\n BuildIncidentHalfEdge passed, proc %d \n", rank);
//    hpBuildNRingGhost(mesh, 2);
//    printf("\n BuildNRingGhost passed, proc %d \n", rank);
//    char debug_filename2[200];
//    sprintf(debug_filename2, "debugout2-p%s.vtk", rank_str);
//    hpWriteUnstrMeshWithPInfo(debug_filename2, mesh);

    hpDeleteMesh(&mesh);


    printf("Success processor %d\n", rank);

    MPI_Finalize();

    return 1;
}


