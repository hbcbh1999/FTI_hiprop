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

    hpBuildOppositeHalfEdge(mesh);
    printf("\n BuildOppHalfEdge passed, proc %d \n", rank);

    hpBuildIncidentHalfEdge(mesh);
    printf("\n BuildIncidentHalfEdge passed, proc %d \n", rank);

    printf("\nbefore build nring ghost, num tris = %d\n", mesh->tris->size[0]);
    hpBuildNRingGhost(mesh, 2);
    printf("\nafter build nring ghost, num tris = %d\n", mesh->tris->size[0]);

    printf("\n BuildNRingGhost passed, proc %d \n", rank);
    char debug_filename[200];
    sprintf(debug_filename, "debugout-p%s.vtk", rank_str);
    hpWriteUnstrMeshWithPInfo(debug_filename, mesh);

    hpBuildOppositeHalfEdge(mesh);
    printf("\n BuildOppHalfEdge passed, proc %d \n", rank);

    hpBuildIncidentHalfEdge(mesh);
    printf("\n BuildIncidentHalfEdge passed, proc %d \n", rank);

    if (rank == 0)
    {
    emxArray_int32_T *ngbvs, *ngbfs;
    emxArray_boolean_T *vtags, *ftags;
    int32_T num_ring_ps, num_ring_tris;

    hpObtainNRingTris(mesh, 11, 2.0, 0, 128, 256, &ngbvs, &ngbfs, &vtags, &ftags, &num_ring_ps, &num_ring_tris);

    printf("num of ps = %d, num of tris = %d\n", num_ring_ps, num_ring_tris);

    for (i = 1; i <= ngbvs->size[0]; i++)
	printf("point %d\n", ngbvs->data[I1dm(i)]);

    for (i = 1; i <= ngbfs->size[0]; i++)
    {
	int tri_index = ngbfs->data[I1dm(i)];
	printf("tris %d: (%d, %d, %d)\n", tri_index, mesh->tris->data[I2dm(tri_index,1,mesh->tris->size)], 
		 mesh->tris->data[I2dm(tri_index,2,mesh->tris->size)], mesh->tris->data[I2dm(tri_index,3,mesh->tris->size)]);  
    }

    num_ring_ps++;

    emxArray_int32_T *ps_vis = emxCreateND_int32_T(1, &num_ring_ps);
    emxArray_int32_T *tris_vis = emxCreateND_int32_T(1, &num_ring_tris);

    for (i = 1; i < num_ring_ps; i++)
	ps_vis->data[I1dm(i)] = ngbvs->data[I1dm(i)];
    ps_vis->data[I1dm(num_ring_ps)] = 11;

    for (i = 1; i <= num_ring_tris; i++)
	tris_vis->data[I1dm(i)] = ngbfs->data[I1dm(i)];



    emxFree_int32_T(&(ngbvs));
    emxFree_int32_T(&(ngbfs));
    emxFree_boolean_T(&(vtags));
    emxFree_boolean_T(&(ftags));

    char dfilename[200];
    sprintf(dfilename, "debout.vtk");
    hpDebugOutput(mesh, ps_vis, tris_vis, dfilename);

    emxFree_int32_T(&(ps_vis));
    emxFree_int32_T(&(tris_vis));

    }

    hpComputeDiffops(mesh, 2);

    char nor_curv_filename[200];
    sprintf(nor_curv_filename, "diffquant-p%s.out", rank_str);
    FILE *diff_outfile = fopen(nor_curv_filename, "w");

    for (i = 1; i <= mesh->ps->size[0]; i++)
    {
	int head = mesh->ps_pinfo->head[I1dm(i)];
	if (mesh->ps_pinfo->pdata[I1dm(head)].proc == rank)
	{
	    fprintf(diff_outfile, "%22.16g %22.16g %22.16g %22.16g %22.16g %22.16g %22.16g %22.16g\n",
		    mesh->ps->data[I2dm(i,1,mesh->ps->size)], mesh->ps->data[I2dm(i,2,mesh->ps->size)], mesh->ps->data[I2dm(i,3,mesh->ps->size)],
		    mesh->nor->data[I2dm(i,1,mesh->nor->size)], mesh->nor->data[I2dm(i,2,mesh->nor->size)], mesh->nor->data[I2dm(i,3,mesh->nor->size)],
		    mesh->curv->data[I2dm(i,1,mesh->curv->size)], mesh->curv->data[I2dm(i,2,mesh->curv->size)]);

	}
    }

    hpDeleteMesh(&mesh);

    printf("Success processor %d\n", rank);

    MPI_Finalize();

    return 1;
/*
    hpBuildNRingGhost(mesh, 4);
    double *bounding_box = (double *) calloc(6, sizeof(double));

    switch(rank)
    {
	case 0:
	    bounding_box[0] = 0.4; bounding_box[1] = 1.0;
	    bounding_box[2] = 0; bounding_box[3] = 0.6;
	    bounding_box[4] = -0.1; bounding_box[5] = 0.1;
	    break;
	case 1:
	    bounding_box[0] = 0.4; bounding_box[1] = 1.0;
	    bounding_box[2] = 0.4; bounding_box[3] = 1.0;
	    bounding_box[4] = -0.1; bounding_box[5] = 0.1;
	    break;
	case 2:
	    bounding_box[0] = 0; bounding_box[1] = 0.6;
	    bounding_box[2] = 0.4; bounding_box[3] = 1.0;
	    bounding_box[4] = -0.1; bounding_box[5] = 0.1;
	    break;
	case 3:
	    bounding_box[0] = 0; bounding_box[1] = 0.6;
	    bounding_box[2] = 0; bounding_box[3] = 0.6;
	    bounding_box[4] = -0.1; bounding_box[5] = 0.1;
	    break;
	default:
	    break;
    }
    hpBuildBoundingBoxGhost(mesh, bounding_box);
    printf("\n BuildNRingGhost passed, proc %d \n", rank);
    char debug_filename[200];
    sprintf(debug_filename, "debugout-p%s.vtk", rank_str);
    hpWriteUnstrMeshWithPInfo(debug_filename, mesh);

    hpBuildOppositeHalfEdge(mesh);
    printf("\n BuildOppHalfEdge passed, proc %d \n", rank);

    hpBuildIncidentHalfEdge(mesh);
    printf("\n BuildIncidentHalfEdge passed, proc %d \n", rank);
*/
}


