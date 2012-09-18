/*
 *	io.c
 * basic functions to read meshes and output meshes
 *
 */

/* find the string from the beginning of the file,
 * put the cursor next to the string. 
 */
#include "datatypes.h"
#include "memutil.h"

int FindString(
	FILE *file,
	const char *in_string)
{
    const char *s;
    int ch;
    long current;

    if (!file)
    {
	printf("Cannot find file.\n");
	return 0;
    }

    current = ftell(file);
    s = in_string;
    rewind(file);

    while((ch = getc(file)) != EOF)
    {
	if (ch != *s)
	    s = in_string;
	else if (!*++s)
	    break;
    }
    if (ch == EOF)
    {
	fseek(file,current,SEEK_SET);
	printf("Cannot find the string %s\n", in_string);
	return 0;
    }
    else
	return 1;
}

/* read the vtk format of the mesh info 
 * points should be saved in double
 * mesh info should be in POLYDATA
 * we use POLYGONS to store the triangles
 */
extern int ReadPolyMeshVtk3d(
	const char *name,
	hiPropMesh *mesh)
{
    FILE* file = fopen(name, "r");
    int i, j;
    int num_points, num_tris, size;
    double* pt_coord;
    int* tri_index;



    if (!FindString(file, "ASCII"))
    {
	printf("Unknown format\n");
	return 0;
    }

    if(!FindString(file, "POINTS"))
    {
	printf("Cannot find points info\n");
	return 0;
    }

    fscanf(file, "%d", &num_points);
    if (!FindString(file, "double"))
    {
	printf("points data type is not double\n");
	return 0;
    }

    pt_coord = (double*) malloc(3 * num_points * sizeof(double));
    for (i = 0; i< (3*num_points); i++)
	fscanf(file, "%lf", &pt_coord[i]);

    if(!FindString(file, "POLYGONS"))
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

extern int WritePolyMeshVtk3d(const char* name, 
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

extern int ReadUnstrMeshVtk3d(
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



    if (!FindString(file, "ASCII"))
    {
	printf("Unknown format\n");
	return 0;
    }

    if(!FindString(file, "POINTS"))
    {
	printf("Cannot find points info\n");
	return 0;
    }

    fscanf(file, "%d", &num_points);
    if (!FindString(file, "double"))
    {
	printf("points data type is not double\n");
	return 0;
    }

    pt_coord = (double*) malloc(3 * num_points * sizeof(double));
    for (i = 0; i< (3*num_points); i++)
	fscanf(file, "%lf", &pt_coord[i]);

    if(!FindString(file, "CELLS"))
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

extern int WriteUnstrMeshVtk3d(const char* name, 
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

