/*!
 * \file hiprop.c
 * \brief Implementation of functions in hiprop.h
 *
 *
 *
 * \author Yijie Zhou
 * \data 2012.09.18
 *
 */

#include "stdafx.h"
#include "util.h"
#include "hiprop.h"
#include "metis.h"

void hpInitMesh(hiPropMesh **pmesh)
{
    hiPropMesh *mesh;
    *pmesh = (hiPropMesh*) malloc(sizeof(hiPropMesh));
    mesh = *pmesh;
    mesh->ps = (emxArray_real_T *) NULL;
    mesh->tris = (emxArray_int32_T *) NULL;
    mesh->nor = (emxArray_real_T *) NULL;
}

void hpFreeMesh(hiPropMesh **pmesh)
{
    if( (*pmesh)->ps != ((emxArray_real_T *) NULL) )
	emxFree_real_T(&((*pmesh)->ps));
    if( (*pmesh)->tris != ((emxArray_int32_T *) NULL) )
	emxFree_int32_T(&((*pmesh)->tris));
    if( (*pmesh)->nor != ((emxArray_real_T *) NULL) )
	emxFree_real_T(&((*pmesh)->nor));

    (*pmesh) = (hiPropMesh *)NULL;
}

int hpReadPolyMeshVtk3d(
	const char *name,
	hiPropMesh *mesh)
{
    FILE* file = fopen(name, "r");
    int i, j;
    int num_points, num_tris, size;
    double* pt_coord;
    int* tri_index;



    if (!findString(file, "ASCII"))
    {
	printf("Unknown format\n");
	return 0;
    }

    if(!findString(file, "POINTS"))
    {
	printf("Cannot find points info\n");
	return 0;
    }

    fscanf(file, "%d", &num_points);
    if (!findString(file, "double"))
    {
	printf("points data type is not double\n");
	return 0;
    }

    pt_coord = (double*) malloc(3 * num_points * sizeof(double));
    for (i = 0; i< (3*num_points); i++)
	fscanf(file, "%lf", &pt_coord[i]);

    if(!findString(file, "POLYGONS"))
	return 0;

    fscanf(file, "%d", &num_tris);
    fscanf(file, "%d", &size);

    tri_index = (int*) malloc(size * sizeof(int));
    for (i = 0; i< size; i++)
	fscanf(file, "%d", &tri_index[i]);

    /*	store the info into the hiPropMesh structure */
    /* points */
    (mesh->ps) = emxCreate_real_T(num_points, 3);
    for (i = 0; i<num_points; i++)
	for (j = 0; j<3; j++)
	    mesh->ps->data[j*num_points+i] = pt_coord[i*3+j];
    mesh->ps->canFreeData = 1;

    /* triangles */
    (mesh->tris) = emxCreate_int32_T(num_tris, 3);
    for (i = 0; i< num_tris; i++)
	for (j = 0; j<3; j++)
	    mesh->tris->data[j*num_tris+i] = tri_index[i*4+(j+1)];
    mesh->tris->canFreeData = 1;

    free(pt_coord);
    free(tri_index);

    fclose(file);
    return 1;

}

int hpWritePolyMeshVtk3d(const char* name, 
	hiPropMesh *mesh)
{
    FILE* file;
    int i;
    emxArray_real_T* points = mesh->ps;
    emxArray_int32_T* tris = mesh->tris;

    file = fopen(name, "w");

    fprintf(file, "# vtk DataFile Version 3.0\n");
    fprintf(file, "Mesh output by hiProp\n");
    fprintf(file, "ASCII\n");
    fprintf(file, "DATASET POLYDATA\n");

    int num_points = mesh->ps->size[0];
    int num_tris = mesh->tris->size[0];

    fprintf(file, "POINTS %d double\n", num_points);
    for (i = 0; i<num_points; i++)
	fprintf(file, "%lf %lf %lf\n", points->data[i], points->data[num_points+i], points->data[2*num_points+i]);

    fprintf(file, "POLYGONS %d %d\n", num_tris, 4*num_tris);
    for (i = 0; i<num_tris; i++)
	fprintf(file, "3 %d %d %d\n", tris->data[i], tris->data[num_tris+i], tris->data[2*num_tris+i]);

    fclose(file);
    return 1;

}

int hpReadUnstrMeshVtk3d(
	const char *name,
	hiPropMesh* mesh)
{
    FILE* file;
    if ( !(file = fopen(name, "r")) )
    {
	printf("Cannot read file!\n");
	return 0;
    }

    int i, j;
    int num_points, num_tris, size;
    double* pt_coord;
    int* tri_index;



    if (!findString(file, "ASCII"))
    {
	printf("Unknown format\n");
	return 0;
    }

    if(!findString(file, "POINTS"))
    {
	printf("Cannot find points info\n");
	return 0;
    }

    fscanf(file, "%d", &num_points);
    if (!findString(file, "double"))
    {
	printf("points data type is not double\n");
	return 0;
    }

    pt_coord = (double*) malloc(3 * num_points * sizeof(double));
    for (i = 0; i< (3*num_points); i++)
	fscanf(file, "%lf", &pt_coord[i]);

    if(!findString(file, "CELLS"))
	return 0;

    fscanf(file, "%d", &num_tris);
    fscanf(file, "%d", &size);

    tri_index = (int*) malloc(size * sizeof(int));
    for (i = 0; i< size; i++)
	fscanf(file, "%d", &tri_index[i]);

    /*	store the info into the hiPropMesh structure */
    /* points */
    (mesh->ps) = emxCreate_real_T(num_points, 3);
    for (i = 0; i<num_points; i++)
	for (j = 0; j<3; j++)
	    mesh->ps->data[j*num_points+i] = pt_coord[i*3+j];

    /* triangles */
    (mesh->tris) = emxCreate_int32_T(num_tris, 3);
    for (i = 0; i< num_tris; i++)
	for (j = 0; j<3; j++)
	    mesh->tris->data[j*num_tris+i] = tri_index[i*4+(j+1)];
    free(pt_coord);
    free(tri_index);

    fclose(file);
    return 1;

}

int hpWriteUnstrMeshVtk3d(const char* name, 
	hiPropMesh* mesh)
{
    FILE* file;
    int i;
    emxArray_real_T* points = mesh->ps;
    emxArray_int32_T* tris = mesh->tris;

    file = fopen(name, "w");

    fprintf(file, "# vtk DataFile Version 3.0\n");
    fprintf(file, "Mesh output by hiProp\n");
    fprintf(file, "ASCII\n");
    fprintf(file, "DATASET UNSTRUCTURED_GRID\n");

    int num_points = mesh->ps->size[0];
    int num_tris = mesh->tris->size[0];

    fprintf(file, "POINTS %d double\n", num_points);
    for (i = 0; i<num_points; i++)
	fprintf(file, "%lf %lf %lf\n", points->data[i], points->data[num_points+i], points->data[2*num_points+i]);

    fprintf(file, "CELLS %d %d\n", num_tris, 4*num_tris);
    for (i = 0; i<num_tris; i++)
	fprintf(file, "3 %d %d %d\n", tris->data[i], tris->data[num_tris+i], tris->data[2*num_tris+i]);

    fprintf(file, "CELL_TYPES %d\n", num_tris);
    for (i = 0; i<num_tris; i++)
	fprintf(file, "5\n");
    fclose(file);
    return 1;

}

int hpMetisDistMesh(hiPropMesh* mesh, const int nparts, 
	int** tri_part, int** pt_part)
{

    //to be consistent with Metis, idx_t denote integer numbers, real_t denote floating point numbers

    printf("entered hpMetisDistMesh\n");
    int i, flag;
    idx_t np = nparts;

    idx_t ne = mesh->tris->size[0];	// number of triangles
    idx_t nn = mesh->ps->size[0];	// number of points
 
    idx_t *eptr = (idx_t*) calloc(ne+1, sizeof(idx_t));
    idx_t *eind = (idx_t*) calloc(3*ne, sizeof(idx_t));

    printf("num_tri to be partitioned = %d\n", ne);
    printf("num_pt to be partitioned = %d\n", nn);

    for(i = 0; i<ne; i++)
    {
	eptr[i] = 3*i;
	eind[eptr[i]] = mesh->tris->data[I2dm(i+1,1,mesh->tris->size)];
	eind[eptr[i]+1] = mesh->tris->data[I2dm(i+1,2,mesh->tris->size)];
	eind[eptr[i]+2] = mesh->tris->data[I2dm(i+1,3,mesh->tris->size)];
    }
    eptr[ne] = 3*i;

    idx_t* vwgt = NULL;
    idx_t* vsize = NULL;
    idx_t ncommonnodes = 2;
    real_t* tpwgts = NULL;
    idx_t* options = NULL;
    idx_t objval;


    idx_t* epart = (idx_t*) calloc(ne, sizeof(idx_t));
    (*tri_part) = epart;
    idx_t* npart = (idx_t*) calloc(nn, sizeof(idx_t));
    (*pt_part) = npart;

    flag = METIS_PartMeshDual(&ne, &nn, eptr, eind, vwgt, vsize,
	    &ncommonnodes, &np, tpwgts, options, &objval, 
	    epart, npart);
    free(eptr);
    free(eind);

    if (flag == METIS_OK)
    {
    	printf("passed hpMetisDistMesh\n");
	return 1;
    }
    else
    {	printf("Metis Error!\n");
	return 0;
    }
}
