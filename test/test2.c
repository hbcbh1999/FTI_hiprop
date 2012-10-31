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

    hpCleanMeshByPinfo(mesh);
    printf("\n hpCleanMeshByPinfo, proc %d \n", rank);

    char debug_out_name[250];
    sprintf(debug_out_name, "debugout-p%s.vtk", rank_str);

    hpWriteUnstrMeshVtk3d(debug_out_name, mesh);

/*
    hpBuildOppositeHalfEdge(mesh);
    printf("\n BuildOppHalfEdge passed, proc %d \n", rank);

    hpBuildIncidentHalfEdge(mesh);
    printf("\n BuildIncidentHalfEdge passed, proc %d \n", rank);

    printf("\nBefore building n-ring, pinfo for ps \n");
    for (i = 1; i <= mesh->ps->size[0]; i++)
    {
	int cur_node = mesh->ps_pinfo->head[I1dm(i)];
	while(cur_node != -1)
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
    hpWriteUnstrMeshVtk3d(debug_filename, mesh);
    hpWriteUnstrMeshWithPInfo(debug_filename, mesh);

    int j;


    for (i = 1; i <= mesh->ps->size[0]-1; i++)
    {
	for (j = i+1; j <= mesh->ps->size[0]; j++)
	{
	    double x1 = mesh->ps->data[I2dm(i,1,mesh->ps->size)];
	    double y1 = mesh->ps->data[I2dm(i,2,mesh->ps->size)];
	    double z1 = mesh->ps->data[I2dm(i,3,mesh->ps->size)];

	    double x2 = mesh->ps->data[I2dm(j,1,mesh->ps->size)];
	    double y2 = mesh->ps->data[I2dm(j,2,mesh->ps->size)];
	    double z2 = mesh->ps->data[I2dm(j,3,mesh->ps->size)];

	    if ( (x1 == x2) && (y1 == y2) && (z1 == z2) )
		printf("Same point for index %d and %d\n",i, j);
	}
    }

    printf("\npinfo for ps \n");
    for (i = 1; i <= mesh->ps->size[0]; i++)
    {
	int cur_node = mesh->ps_pinfo->head[I1dm(i)];
	while(cur_node != -1)
	{
	    printf("%d/%d-->", mesh->ps_pinfo->pdata[I1dm(cur_node)].proc, mesh->ps_pinfo->pdata[I1dm(cur_node)].lindex);
	    cur_node = mesh->ps_pinfo->pdata[I1dm(cur_node)].next;
	}
	printf("\n");
    }


    hpBuildOppositeHalfEdge(mesh);
    printf("\n BuildOppHalfEdge passed, proc %d \n", rank);

    hpBuildIncidentHalfEdge(mesh);
    printf("\n BuildIncidentHalfEdge passed, proc %d \n", rank);

    hpBuildNRingGhost(mesh, 2);
    printf("\n BuildNRingGhost passed, proc %d \n", rank);

    char debug_filename2[200];
    sprintf(debug_filename2, "debugout2-p%s.vtk", rank_str);
    hpWriteUnstrMeshWithPInfo(debug_filename2, mesh);
    */

    /*
    emxArray_int32_T *ring_ps, *ring_tris;
    emxArray_boolean_T *tag_ps, *tag_tris;
    int num_ring_ps, num_ring_tris;
    int center;

    for (i = 1; i <= mesh->ps->size[0]; i++)
    {
	int j = 0;
	int next = mesh->ps_pinfo->head[I1dm(i)];
	while (next != -1)
	{
	    int cur_node = next;
	    next = mesh->ps_pinfo->pdata[I1dm(cur_node)].next;
	    j++;
	}
	if (j == 3)
	    center = i;
    }

    hpObtainNRingTris(mesh, center, 2, 0, 128, 256, (&ring_ps), (&ring_tris), (&tag_ps),
	    (&tag_tris), &num_ring_ps, &num_ring_tris);


    char debug_out_name[250];
    sprintf(debug_out_name, "debugout-p%s.vtk", rank_str);
    FILE* file = fopen(debug_out_name, "w");

    fprintf(file, "# vtk DataFile Version 3.0\n");
    fprintf(file, "Debug output by hiProp\n");
    fprintf(file, "ASCII\n");
    fprintf(file, "DATASET UNSTRUCTURED_GRID\n");

    fprintf(file, "POINTS %d double\n", mesh->ps->size[0]);
    for (i = 1; i <= mesh->ps->size[0]; i++)
    {
	fprintf(file, "%lf %lf %lf\n",
		mesh->ps->data[I2dm(i,1,mesh->ps->size)],
		mesh->ps->data[I2dm(i,2,mesh->ps->size)],
		mesh->ps->data[I2dm(i,3,mesh->ps->size)]);
    }
    fprintf(file, "CELLS %d %d\n", num_ring_tris, 4*num_ring_tris);
    for (i = 1; i <= num_ring_tris; i++)
    {
	int tri_index = ring_tris->data[I1dm(i)];
	fprintf(file, "3 %d %d %d\n",
		mesh->tris->data[I2dm(tri_index,1,mesh->tris->size)]-1,
		mesh->tris->data[I2dm(tri_index,2,mesh->tris->size)]-1,
		mesh->tris->data[I2dm(tri_index,3,mesh->tris->size)]-1);

    }
    fprintf(file, "CELL_TYPES %d\n", num_ring_tris);
    for (i = 1; i <= num_ring_tris; i++)
	fprintf(file, "5\n");
    fclose(file);

    printf("\n Num ps in ring = %d\n", num_ring_ps);
    printf("\n ObtainNRingTris passed, proc %d \n", rank);
    emxFree_int32_T((&ring_ps));
    emxFree_int32_T((&ring_tris));
    emxFree_boolean_T((&tag_ps));
    emxFree_boolean_T((&tag_tris));

    */
    /*
    printf("\n Neighbor processors : \n");
    for (i = 1; i <= mesh->nb_proc->size[0]; i++)
	printf("%d ", mesh->nb_proc->data[I1dm(i)]);
    printf("\n");

    printf("\n ps_send_index: \n");
    for (i = 0; i < num_proc; i++)
    {
	if (mesh->ps_send_index[i] == NULL)
	    printf("proc %d, ps_send_index = NULL\n", i);
	else
	    printf("proc %d, ps_send_index != NULL\n", i);
    }

    printf("\n ps_send_buffer: \n");
    for (i = 0; i < num_proc; i++)
    {
	if (mesh->ps_send_buffer[i] == NULL)
	    printf("proc %d, ps_send_buffer = NULL\n", i);
	else
	    printf("proc %d, ps_send_buffer != NULL\n", i);
    }

    printf("\n ps_recv_index: \n");
    for (i = 0; i < num_proc; i++)
    {
	if (mesh->ps_recv_index[i] == NULL)
	    printf("proc %d, ps_recv_index = NULL\n", i);
	else
	    printf("proc %d, ps_recv_index != NULL\n", i);
    }

    printf("\n ps_recv_buffer: \n");
    for (i = 0; i < num_proc; i++)
    {
	if (mesh->ps_recv_buffer[i] == NULL)
	    printf("proc %d, ps_recv_buffer = NULL\n", i);
	else
	    printf("proc %d, ps_recv_buffer != NULL\n", i);
    }

    */

    hpDeleteMesh(&mesh);


    printf("Success processor %d\n", rank);

    MPI_Finalize();

    return 1;
}


