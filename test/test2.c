/*
 *	This file is to test input and output utilities in VTK format
 *
 */

#include "stdafx.h"
#include "memutil.h"
#include "io.h"

int main(int argc, char* argv[])
{
    hiPropMesh *mesh;

    hpInitMesh(&mesh);

    if(!ReadUnstrMeshVtk3d(argv[1], mesh))
    {
	printf("Mesh Read fail\n");
	return 0;
    }

    int num_ps_all = mesh->ps->allocatedSize;

    int i;
    for (i = 1; i <= num_ps_all; i++)
	mesh->ps->data[i-1] = mesh->ps->data[i-1] + 1.0;

    if(!WriteUnstrMeshVtk3d("output.vtk", mesh))
    {
	printf("Mesh Write fail\n");
	return 0;
    }

    hpFreeMesh(&mesh);

    if (mesh == (hiPropMesh *)NULL)
	printf("Success processor\n");

    return 1;
}


