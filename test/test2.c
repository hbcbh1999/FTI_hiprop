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

    hpBuildNRingGhost(mesh, 2);

    printf("\n BuildNRingGhost passed, proc %d \n", rank);
    char debug_filename[200];
    sprintf(debug_filename, "debugout-p%s.vtk", rank_str);
    hpWriteUnstrMeshWithPInfo(debug_filename, mesh);

    hpBuildPUpdateInfo(mesh);


/*
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
    free(bounding_box);
*/

    hpDeleteMesh(&mesh);


    printf("Success processor %d\n", rank);

    MPI_Finalize();

    return 1;
}


