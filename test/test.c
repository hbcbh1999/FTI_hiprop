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




    hiPropMesh *mesh;	//mesh after partition
    hpInitMesh(&mesh);
    emxArray_int32_T* ps_globalid;
    emxArray_int32_T* tri_globalid;

    if (rank == 0)
    {
	char in_filename[200];
    	sprintf(in_filename, "data/serial/%s.vtk",argv[1]);

    	hiPropMesh *in_mesh;	//input mesh
    	hpInitMesh(&in_mesh);

    	if(!hpReadUnstrMeshVtk3d(in_filename, in_mesh))
    	{
	    printf("Mesh Read fail\n");
	    return 0;
    	}

    	int* tri_part; 
    	int* pt_part;


    	hpMetisPartMesh(in_mesh, num_proc, &tri_part, &pt_part);
	hpDistMesh(root, in_mesh, mesh, tri_part, tag, ps_globalid, tri_globalid);
	hpDeleteMesh(&in_mesh);
    }
    else
	hpDistMesh(root, NULL, mesh, NULL, tag, ps_globalid, tri_globalid);

    hpBuildPUpdateInfo(mesh);

/*
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
*/
    MPI_Barrier(MPI_COMM_WORLD);


    //if (rank == 0)
    //{
	printf("ps pinfo of rank 0:\n");
	for (i = 1; i <= mesh->ps->size[0]; i++)
	{
	    printf("point %d: ", i);
	    int next = mesh->ps_pinfo->head[I1dm(i)];
	    while (next != -1)
	    {
		int cur_node = next;
		printf("%d/%d-->", mesh->ps_pinfo->pdata[I1dm(cur_node)].proc, mesh->ps_pinfo->pdata[I1dm(cur_node)].lindex);
		fflush(runlog_stream);
		next = mesh->ps_pinfo->pdata[I1dm(cur_node)].next;
	    }
	    printf("\n");
	}

/*	
	printf("tris pinfo of rank 0:\n");
	for (i = 1; i <= mesh->tris->size[0]; i++)
	{
	    printf("triangle %d: ", i);
	    int next = mesh->tris_pinfo->head[I1dm(i)];
	    while (next != -1)
	    {
		int cur_node = next;
		printf("%d/%d-->", mesh->tris_pinfo->pdata[I1dm(cur_node)].proc, mesh->tris_pinfo->pdata[I1dm(cur_node)].lindex);
		next = mesh->tris_pinfo->pdata[I1dm(cur_node)].next;
	    }
	    printf("\n");
	}
    //}
*/

	int cur_proc, j;
    for(i = 1; i<=mesh->nb_proc->size[0]; i++)
    {
	cur_proc = mesh->nb_proc->data[I1dm(i)];
	if(cur_proc<rank)
	{
	    printf("point recv index with proc %d\n", cur_proc);
	    for(j = 1; j<=mesh->ps_recv_index[cur_proc]->size[0]; j++)
		printf("%d\n", mesh->ps_recv_index[cur_proc]->data[I1dm(j)]);
	}
	else
	{
	    printf("point send index with proc %d\n", cur_proc);
	    for(j = 1; j<=mesh->ps_send_index[cur_proc]->size[0]; j++)
		printf("%d\n", mesh->ps_send_index[cur_proc]->data[I1dm(j)]);
	}
    }

    hpDeleteMesh(&mesh);


    printf("Success processor %d\n", rank);
    MPI_Finalize();

    return 1;
}
