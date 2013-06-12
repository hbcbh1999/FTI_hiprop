/*
 *	test file for mesh partition and distribution
 */


#include "stdafx.h"
#include "util.h"
#include "hiprop.h"
/* #include "metis.h" */

int main(int argc, char* argv[])
{
    /*
    MPI_Init(&argc, &argv);

    int num_proc, rank;
    int tag = 1;
    int root = 0;

    MPI_Comm_size(MPI_COMM_WORLD, &num_proc);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);


    hiPropMesh *mesh;
    hpInitMesh(&mesh);
    emxArray_int32_T* ps_globalid, *tri_globalid;

    if (rank == 0)
    {
	char in_filename[200];
    	sprintf(in_filename, "data/serial/%s.vtk",argv[1]);

    	hiPropMesh *in_mesh;	
	hpInitMesh(&in_mesh);

    	if(!hpReadUnstrMeshVtk3d(in_filename, in_mesh))
    	{
	    printf("Mesh Read fail\n");
	    return 0;
    	}

    	int* tri_part; 
    	int* pt_part;

    	hpMetisPartMesh(in_mesh, num_proc, &tri_part, &pt_part);
	hpDistMesh(root, in_mesh, mesh, tri_part, tag, &ps_globalid, &tri_globalid);
	hpDeleteMesh(&in_mesh);
    }
    else
	hpDistMesh(root, NULL, mesh, NULL, tag, &ps_globalid, &tri_globalid);

    char rank_str[5];
    char out_name[200];
    char id_out_name[200];

    numIntoString(rank,4,rank_str);
    sprintf(out_name, "data/parallel/%s-p%s.vtk",argv[1], rank_str);
    sprintf(id_out_name, "trisid-p%s.data",rank_str);

    FILE *id_out_file = fopen(id_out_name, "w");


    int i;

    for (i = 1; i <= tri_globalid->size[0]; i++)
	fprintf(id_out_file, "%d\n", tri_globalid->data[i-1]);

    fclose(id_out_file);

    if(!hpWriteUnstrMeshVtk3d(out_name, mesh))
    {
	printf("Write fail\n");
	return 0;
    }

    hpDeleteMesh(&mesh);
    printf("Success for proc %d\n", rank);

    MPI_Finalize();
    */
    return 1;
}


