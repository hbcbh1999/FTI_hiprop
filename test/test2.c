/*
 *	This file is to test input and output utilities in VTK format
 *
 */

#include "stdafx.h"
#include "memutil.h"
#include "io.h"

int main(int argc, char* argv[])
{
    hiPropMesh mesh;
    FILE* file = fopen(argv[1], "r");
    if(!ReadMeshVtk3d(file, &mesh))
    {
	printf("Read fail\n");
	return 0;
    }


    if(!WriteMeshVtk3d("output.vtk", &mesh))
    {
	printf("Write fail\n");
	return 0;
    }
    printf("Success processor %d\n");
    fclose(file);

    return 1;
}


