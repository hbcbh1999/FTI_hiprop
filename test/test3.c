/*
 *	test file for mesh partition and distribution
 */


#include "stdafx.h"
#include "util.h"
#include "hiprop.h"
#include "metis.h"

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);

    int num_proc, rank;
    int tag = 1;
    int root = 0;

    MPI_Comm_size(MPI_COMM_WORLD, &num_proc);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);


    hiPropMesh *mesh;	//mesh after partition
    hpInitMesh(&mesh);

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
	hpDistMesh(root, in_mesh, mesh, tri_part, tag);
	hpDeleteMesh(&in_mesh);
    }
    else
	hpDistMesh(root, NULL, mesh, NULL, tag);

    // output
    char rank_str[5];
    char out_name[200];

    right_flush(rank,4,rank_str);
    sprintf(out_name, "data/parallel/%s-p%s.vtk",argv[1], rank_str);

    if(!hpWriteUnstrMeshVtk3d(out_name, mesh))
    {
	printf("Write fail\n");
	return 0;
    }
    hpDeleteMesh(&mesh);
    printf("Success for proc %d\n", rank);

    MPI_Finalize();
    return 1;
}


