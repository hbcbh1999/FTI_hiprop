/*!
 * \file hiprop.c
 * \brief Implementation of functions in hiprop.h
 *
 * \author Yijie Zhou
 * \data 2012.09.18
 *
 */

#include "stdafx.h"
#include "util.h"
#include "hiprop.h"
#include "metis.h"

/*!
 * \brief Determine whether 2 triangles are the same base on floating point
 * comparison
 * \param ps1 points list 1
 * \param tri1 triangle list 1
 * \param tri_index1 triangle index in tri1 for comparison
 * \param ps2 points list 2
 * \param tri2 triangle list 2
 * \param tri_index2 triangle index in tri2 for comparison
 */

static boolean_T sameTriangle(const emxArray_real_T* ps1,
			      const emxArray_int32_T* tri1,
			      const int tri_index1,
			      const emxArray_real_T* ps2,
			      const emxArray_int32_T* tri2,
			      const int tri_index2,
			      const double eps);




static boolean_T sameTriangle(const emxArray_real_T* ps1,
			      const emxArray_int32_T* tri1,
			      const int tri_index1,
			      const emxArray_real_T* ps2,
			      const emxArray_int32_T* tri2, 
			      const int tri_index2,
			      const double eps)
{
    int p11 = tri1->data[I2dm(tri_index1,1,tri1->size)];
    int p12 = tri1->data[I2dm(tri_index1,2,tri1->size)];
    int p13 = tri1->data[I2dm(tri_index1,3,tri1->size)];

    int p21 = tri2->data[I2dm(tri_index2,1,tri2->size)];
    int p22 = tri2->data[I2dm(tri_index2,2,tri2->size)];
    int p23 = tri2->data[I2dm(tri_index2,3,tri2->size)];


    if ( (fabs(ps1->data[I2dm(p11,1,ps1->size)] - ps2->data[I2dm(p21,1,ps2->size)]) < eps) &&
	 (fabs(ps1->data[I2dm(p11,2,ps1->size)] - ps2->data[I2dm(p21,2,ps2->size)]) < eps) &&
	 (fabs(ps1->data[I2dm(p11,3,ps1->size)] - ps2->data[I2dm(p21,3,ps2->size)]) < eps) &&
	 (fabs(ps1->data[I2dm(p12,1,ps1->size)] - ps2->data[I2dm(p22,1,ps2->size)]) < eps) &&
	 (fabs(ps1->data[I2dm(p12,2,ps1->size)] - ps2->data[I2dm(p22,2,ps2->size)]) < eps) &&
	 (fabs(ps1->data[I2dm(p12,3,ps1->size)] - ps2->data[I2dm(p22,3,ps2->size)]) < eps) &&
	 (fabs(ps1->data[I2dm(p13,1,ps1->size)] - ps2->data[I2dm(p23,1,ps2->size)]) < eps) &&
	 (fabs(ps1->data[I2dm(p13,2,ps1->size)] - ps2->data[I2dm(p23,2,ps2->size)]) < eps) &&
	 (fabs(ps1->data[I2dm(p13,3,ps1->size)] - ps2->data[I2dm(p23,3,ps2->size)]) < eps)
       )
	return 1;
    else
	return 0;
}


void hpInitMesh(hiPropMesh **pmesh)
{
    hiPropMesh *mesh;
    *pmesh = (hiPropMesh*) calloc(1, sizeof(hiPropMesh));
    mesh = *pmesh;
    mesh->ps = (emxArray_real_T *) NULL;
    mesh->tris = (emxArray_int32_T *) NULL;
    mesh->nor = (emxArray_real_T *) NULL;
    mesh->curv = (emxArray_real_T *) NULL;

    mesh->nb_proc = (emxArray_int32_T *) NULL;
    mesh->part_bdry = (emxArray_int32_T *) NULL;
    mesh->ps_type = (emxArray_int32_T *) NULL;
    mesh->ps_pinfo = (hpPInfoList *) NULL;
    mesh->tris_pinfo = (hpPInfoList *) NULL;

    mesh->ps_send_index = (emxArray_int32_T **) NULL;
    mesh->ps_recv_index = (emxArray_int32_T **) NULL;
    mesh->ps_send_buffer = (emxArray_real_T **) NULL;
    mesh->ps_recv_buffer = (emxArray_real_T **) NULL;

    mesh->opphe = (emxArray_int32_T *) NULL;
    mesh->inhe = (emxArray_int32_T *) NULL;
    mesh->est_nor = (emxArray_real_T *) NULL;

    mesh->num_int_ps = 0;
    mesh->num_int_tris = 0;
    mesh->num_int_pspinfo = 0;
    mesh->is_clean = 1;
}

void hpFreeMeshAugmentInfo(hiPropMesh *pmesh)
{
    if (pmesh->opphe != ((emxArray_int32_T *) NULL))
	emxFree_int32_T(&(pmesh->opphe));
    if (pmesh->inhe != ((emxArray_int32_T *) NULL))
	emxFree_int32_T(&(pmesh->inhe));
    if (pmesh->est_nor != ((emxArray_real_T *) NULL))
	emxFree_real_T(&(pmesh->est_nor));
}

void hpFreeMeshUpdateInfo(hiPropMesh *pmesh)
{
    int rank, num_proc;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_proc);

    int i;
    if (pmesh->ps_send_index != ((emxArray_int32_T **) NULL) )
    {
	for (i = 0; i < num_proc; i++)
	{
	    if (pmesh->ps_send_index[i] != ((emxArray_int32_T *) NULL) )
	    {
		emxFree_int32_T(&(pmesh->ps_send_index[i]));
		emxFree_real_T(&(pmesh->ps_send_buffer[i]));
	    }
	}
	free(pmesh->ps_send_index);
	free(pmesh->ps_send_buffer);
	pmesh->ps_send_index = NULL;
	pmesh->ps_send_buffer = NULL;
    }
    if (pmesh->ps_recv_index != ((emxArray_int32_T **) NULL) )
    {
	for (i = 0; i < num_proc; i++)
	{
	    if (pmesh->ps_recv_index[i] != ((emxArray_int32_T *) NULL) )
	    {
		emxFree_int32_T(&(pmesh->ps_recv_index[i]));
		emxFree_real_T(&(pmesh->ps_recv_buffer[i]));
	    }
	}
	free(pmesh->ps_recv_index);
	free(pmesh->ps_recv_buffer);
	pmesh->ps_recv_index = NULL;
	pmesh->ps_recv_buffer = NULL;
    }
    
}

void hpFreeMeshParallelInfo(hiPropMesh *pmesh)
{
    hpDeletePInfoList(&(pmesh->ps_pinfo));
    hpDeletePInfoList(&(pmesh->tris_pinfo));

    if (pmesh->nb_proc != ((emxArray_int32_T *) NULL) )
	emxFree_int32_T(&(pmesh->nb_proc));

    if (pmesh->part_bdry != ((emxArray_int32_T *) NULL) )
	emxFree_int32_T(&(pmesh->part_bdry));

    if (pmesh->ps_type != ((emxArray_int32_T *) NULL) )
	emxFree_int32_T(&(pmesh->ps_type));
}

void hpFreeMeshBasicInfo(hiPropMesh *pmesh)
{
    if( pmesh->ps != ((emxArray_real_T *) NULL) )
	emxFree_real_T(&(pmesh->ps));
    if( pmesh->tris != ((emxArray_int32_T *) NULL) )
	emxFree_int32_T(&(pmesh->tris));
    if( pmesh->nor != ((emxArray_real_T *) NULL) )
	emxFree_real_T(&(pmesh->nor));
    if( pmesh->curv != ((emxArray_real_T *) NULL) )
	emxFree_real_T(&(pmesh->curv));
}

void hpFreeMesh(hiPropMesh *pmesh)
{
    hpFreeMeshUpdateInfo(pmesh);
    hpFreeMeshParallelInfo(pmesh);
    hpFreeMeshAugmentInfo(pmesh);
    hpFreeMeshBasicInfo(pmesh);
    pmesh->num_int_ps = 0;
    pmesh->num_int_tris = 0;
    pmesh->num_int_pspinfo = 0;
    pmesh->is_clean = 1;
}

void hpDeleteMesh(hiPropMesh **pmesh)
{
    hpFreeMeshUpdateInfo((*pmesh));
    hpFreeMeshParallelInfo((*pmesh));
    hpFreeMeshAugmentInfo((*pmesh));
    hpFreeMeshBasicInfo((*pmesh));

    free((*pmesh));

    (*pmesh) = (hiPropMesh *)NULL;
}

void hpDeletePInfoList(hpPInfoList **plist)
{
    hpPInfoList *list = (*plist);
    if (list != ((hpPInfoList *) NULL))
    {
	if (list->pdata != ((hpPInfoNode *) NULL))
	    free(list->pdata);
	if (list->head != ((int*) NULL))
	    free(list->head);
	if (list->tail != ((int*) NULL))
	    free(list->tail);
	free(list);
	(*plist) = (hpPInfoList *) NULL;
    }
}

int hpReadPolyMeshVtk3d(
	const char *name,
	hiPropMesh *mesh)
{
    hpFreeMesh(mesh);
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
	fscanf(file, "%lg", &pt_coord[i]);

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
	    mesh->tris->data[j*num_tris+i] = tri_index[i*4+(j+1)] + 1;
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
    for (i = 1; i <= num_points; i++)
	fprintf(file, "%22.16lg %22.16lg %22.16lg\n", 
		points->data[I2dm(i,1,points->size)],
		points->data[I2dm(i,2,points->size)], 
		points->data[I2dm(i,3,points->size)]);

    fprintf(file, "POLYGONS %d %d\n", num_tris, 4*num_tris);
    for (i = 1; i <= num_tris; i++)
	fprintf(file, "3 %d %d %d\n",
		tris->data[I2dm(i,1,tris->size)]-1, 
		tris->data[I2dm(i,2,tris->size)]-1, 
		tris->data[I2dm(i,3,tris->size)]-1);

    fclose(file);
    return 1;

}

int hpReadUnstrMeshVtk3d(
	const char *name,
	hiPropMesh* mesh)
{
    hpFreeMesh(mesh);
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
	fscanf(file, "%lg", &pt_coord[i]);

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
	    mesh->tris->data[j*num_tris+i] = tri_index[i*4+(j+1)] + 1;
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
    for (i = 1; i <= num_points; i++)
	fprintf(file, "%22.16lg %22.16lg %22.16lg\n",
		points->data[I2dm(i,1,points->size)],
		points->data[I2dm(i,2,points->size)],
		points->data[I2dm(i,3,points->size)]);

    fprintf(file, "CELLS %d %d\n", num_tris, 4*num_tris);
    for (i = 1; i <= num_tris; i++)
	fprintf(file, "3 %d %d %d\n",
		tris->data[I2dm(i,1,tris->size)]-1,
		tris->data[I2dm(i,2,tris->size)]-1,
		tris->data[I2dm(i,3,tris->size)]-1);

    fprintf(file, "CELL_TYPES %d\n", num_tris);
    for (i = 0; i<num_tris; i++)
	fprintf(file, "5\n");
    fclose(file);
    return 1;

}

int hpMetisPartMesh(hiPropMesh* mesh, const int nparts, 
	int** tri_part, int** pt_part)
{

    /*
    to be consistent with Metis, idx_t denote integer numbers, 
    real_t denote floating point numbers in Metis, tri and points 
    arrays all start from index 0, which is different from HiProp,
    so we need to convert to Metis convention, the output tri_part 
    and pt_part are in Metis convention
    */

    printf("entered hpMetisPartMesh\n");
    int i, flag;
    idx_t np = nparts;

    idx_t ne = mesh->tris->size[0];	/* number of triangles */
    idx_t nn = mesh->ps->size[0];	/* number of points */
 
    idx_t *eptr = (idx_t*) calloc(ne+1, sizeof(idx_t));
    idx_t *eind = (idx_t*) calloc(3*ne, sizeof(idx_t));

    printf("num_tri to be partitioned = %d\n", ne);
    printf("num_pt to be partitioned = %d\n", nn);

    for(i = 0; i<ne; i++)
    {
	eptr[i] = 3*i;
	eind[eptr[i]] = mesh->tris->data[I2dm(i+1,1,mesh->tris->size)] - 1;
	eind[eptr[i]+1] = mesh->tris->data[I2dm(i+1,2,mesh->tris->size)] - 1;
	eind[eptr[i]+2] = mesh->tris->data[I2dm(i+1,3,mesh->tris->size)] - 1;
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
    	printf("passed hpMetisPartMesh\n");
	return 1;
    }
    else
    {	printf("Metis Error!\n");
	return 0;
    }
}

int hpDistMesh(int root, hiPropMesh *in_mesh,
	hiPropMesh *mesh, int *tri_part,
	int tag,
	emxArray_int32_T **ps_globalid,
	emxArray_int32_T **tri_globalid)
{
    hpFreeMesh(mesh);
    int i,j,k;
    int rank, num_proc;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_proc);
    printf("Entered hpDistMesh proc %d, root = %d\n", rank, root);

    /* calculate the partitioned mesh on the root, then send to other processors */
    if (rank == root)
    {
	if(in_mesh==NULL)
	{
	    printf("No mesh to be distributed\n");
	    return 0;
	}

	/* initialize an array of pointers to the partitioned meshes to be sent */
	hiPropMesh** p_mesh = (hiPropMesh**)malloc(num_proc*sizeof(hiPropMesh*));
	for(i = 0; i<num_proc; i++)
	    hpInitMesh(&p_mesh[i]);

	/* an array to store the number of triangles on each processor */
	int* num_tri = (int*)malloc(num_proc*sizeof(int));	
	for(i = 0; i<num_proc; i++)
	    num_tri[i] = 0;
	/* an array to store the number of points on each processor */
	int* num_pt = (int*)malloc(num_proc*sizeof(int));
	for(i = 0; i<num_proc; i++)
	    num_pt[i] = 0;

	int total_num_tri = in_mesh->tris->size[0];
	int total_num_pt = in_mesh->ps->size[0];

	/* calculate the number of triangles on each proc */
	for(i = 0; i < total_num_tri; i++)
	    num_tri[tri_part[i]]++;



	for(i = 0; i< num_proc; i++)
	{
	    printf("num_tri[%d] = %d\n", i, num_tri[i]);
	    (p_mesh[i]->tris) = emxCreate_int32_T(num_tri[i], 3);
	}


	/* calculate the list of global index of triangles existing on each proc
	 * tri_index[rank][i-1] is the global index of the i-th tri on the ranked proc
	 */
	int** tri_index = (int**) calloc(num_proc,sizeof(int*));
	for(i = 0; i<num_proc; i++)
	{
	    tri_index[i] = (int*) calloc(num_tri[i],sizeof(int));

	}

	/* fill tri_index by looping over all tris */
	int* p = (int*)malloc(num_proc*sizeof(int));/* pointer to the end of the list */
	for(i = 0; i< num_proc; i++)
	    p[i] = 0;
	int tri_rk;	/* the proc rank of the current tri */
	for(i = 1; i<=total_num_tri; i++)
	{
	    tri_rk = tri_part[i-1];	/* convert because Metis 
					   convention use index starts from 0 */
	    tri_index[tri_rk][p[tri_rk]] = i;
	    p[tri_rk]++;
	}


	/* construct an index table to store the local index of every point 
	 * (global to local)
	 * if pt_local[i][j-1] = -1, point[j] is not on proc[i], 
	 * if pt_local[i][j-1] = m >= 0, the local index of point[j] on proc[i] is m.
	 * looks space and time consuming, however easy to convert 
	 * between globle and local index of points
	 */
	int** pt_local = (int**)calloc(num_proc,sizeof(int*));
	for(i = 0; i<num_proc; i++)
	{
	    pt_local[i] = (int*) calloc(total_num_pt , sizeof(int));
	    for(j = 0; j<total_num_pt; j++)
		pt_local[i][j] = -1;	/*initialize to -1 */
	}

	/* fill in pt_local table, calculate num_pt[] on each proc at the same time
	 * in this situation, the point local index is sorted as global index 
	 */
	for (i = 1; i<=total_num_pt; i++)
	{
	    for(j = 0; j<num_proc; j++)
		for(k = 1; k<=num_tri[j]; k++)
		    if((in_mesh->tris->data[I2dm(tri_index[j][k-1],1,in_mesh->tris->size)]==i)||
		       (in_mesh->tris->data[I2dm(tri_index[j][k-1],2,in_mesh->tris->size)]==i)||
		       (in_mesh->tris->data[I2dm(tri_index[j][k-1],3,in_mesh->tris->size)]==i))
		    {
			pt_local[j][i-1] = num_pt[j]+1;
			num_pt[j]++;
			break;
		    }
	}
	for(i = 0; i<num_proc; i++)
	    printf("num_pt[%d] = %d\n", i, num_pt[i]);

	/* fill in p_mesh[]->tris->data[] according to pt_local table */
	int global_index;
	for( i = 0; i<num_proc; i++)
	{
	    for(j = 1; j<=num_tri[i]; j++)
	    {
		global_index = in_mesh->tris->data[I2dm(tri_index[i][j-1],1,in_mesh->tris->size)];
		p_mesh[i]->tris->data[I2dm(j,1,p_mesh[i]->tris->size)] = pt_local[i][I1dm(global_index)];

		global_index = in_mesh->tris->data[I2dm(tri_index[i][j-1],2,in_mesh->tris->size)];
		p_mesh[i]->tris->data[I2dm(j,2,p_mesh[i]->tris->size)] = pt_local[i][I1dm(global_index)];

		global_index = in_mesh->tris->data[I2dm(tri_index[i][j-1],3,in_mesh->tris->size)];
		p_mesh[i]->tris->data[I2dm(j,3,p_mesh[i]->tris->size)] = pt_local[i][I1dm(global_index)];
	    }
	}

	/* pt_index is similar to tri_index
	 * pt_index[rank][i-1] is the global index of the i-th point on the ranked proc
	 * constructed using pt_local
	 */
	int** pt_index = (int**) malloc(num_proc*sizeof(int*));
	for(i = 0; i<num_proc; i++)
	{
	    pt_index[i] = (int*) malloc(num_pt[i]*sizeof(int));
	    for(j = 1; j<=num_pt[i]; j++)
	    {
		for(k = 1; k<=total_num_pt; k++)
		{
		    if(pt_local[i][k-1] == j)
			break;
		}
		if(k>total_num_pt)
		{
		    printf("Cannot find the point global index error!\n");
		    return 0;
		}
		else
		    pt_index[i][j-1] = k;
	    }
	}


	/* finally, get in p_mesh[]->ps, :) */
	for(i = 0; i< num_proc; i++)
	    (p_mesh[i]->ps) = emxCreate_real_T(num_pt[i], 3);
	/* fill in p_mesh[]->ps->data with pt_index */
	for (i = 0; i<num_proc; i++)
	{
	    for(j=1; j<=num_pt[i]; j++)
	    {
		p_mesh[i]->ps->data[I2dm(j,1,p_mesh[i]->ps->size)] 
		    = in_mesh->ps->data[I2dm(pt_index[i][j-1],1,in_mesh->ps->size)];
		p_mesh[i]->ps->data[I2dm(j,2,p_mesh[i]->ps->size)] 
		    = in_mesh->ps->data[I2dm(pt_index[i][j-1],2,in_mesh->ps->size)];
		p_mesh[i]->ps->data[I2dm(j,3,p_mesh[i]->ps->size)] 
		    = in_mesh->ps->data[I2dm(pt_index[i][j-1],3,in_mesh->ps->size)];
	    }
	}


	/* communication of basic mesh info */
	for(i = 0; i<num_proc; i++)
	{
	    if(i==rank)
	    {
		mesh->ps = p_mesh[i]->ps;
		mesh->tris = p_mesh[i]->tris;	

		int* l2gindex;
		int** g2lindex;
		l2gindex = pt_index[i];
		g2lindex = pt_local;
    		hpConstrPInfoFromGlobalLocalInfo(mesh, g2lindex, l2gindex, rank);

		/* Create a wrapper for points and triangles global index arrays.
		 * This wrapper is an output of the function 
		 */
		int array_size[1];
		array_size[0] = num_pt[i];
		*ps_globalid = emxCreateWrapperND_int32_T(pt_index[i],1,array_size);
		array_size[0] = num_tri[i];
		*tri_globalid = emxCreateWrapperND_int32_T(tri_index[i], 1, array_size);

	    }
	    else
	    {
	    	send2D_int32_T(p_mesh[i]->tris, i, tag, MPI_COMM_WORLD);
	    	send2D_real_T(p_mesh[i]->ps, i, tag+5, MPI_COMM_WORLD);
		
		MPI_Send(pt_index[i], p_mesh[i]->ps->size[0], MPI_INT, i, tag+10, MPI_COMM_WORLD);
		MPI_Send(tri_index[i], p_mesh[i]->tris->size[0], MPI_INT, i, tag+11, MPI_COMM_WORLD);

		MPI_Send(&total_num_pt, 1, MPI_INT, i, tag+12, MPI_COMM_WORLD);
		for (j = 0; j<num_proc; j++)
		    MPI_Send(pt_local[j], total_num_pt, MPI_INT, i, tag+13+j, MPI_COMM_WORLD);

	    	hpDeleteMesh(&p_mesh[i]);
		free(tri_index[i]);
		free(pt_index[i]);
	    }
	}

	/* free pointers */
	for (i = 0; i<num_proc; i++)
	    free(pt_local[i]);
	free(p_mesh);
	free(pt_index);
	free(tri_index);
	free(pt_local);
	free(num_tri);
	free(num_pt);
	free(p);
    }

    else	/*for other proc, receive the mesh info */
    {
	recv2D_int32_T(&(mesh->tris),root, tag, MPI_COMM_WORLD);
	recv2D_real_T(&(mesh->ps),root, tag+5, MPI_COMM_WORLD);

	MPI_Status recv_stat;
	int total_num_pt;
	int* l2gindex = (int*) calloc(mesh->ps->size[0], sizeof(int));
	int** g2lindex = (int**) calloc(num_proc, sizeof(int*));
	int* tri_index = (int*) calloc(mesh->tris->size[0], sizeof(int));
	
	MPI_Recv(l2gindex, mesh->ps->size[0], MPI_INT, root, tag+10, MPI_COMM_WORLD, &recv_stat);
	MPI_Recv(tri_index, mesh->tris->size[0], MPI_INT, root, tag+11, MPI_COMM_WORLD, &recv_stat);
	MPI_Recv(&total_num_pt, 1, MPI_INT, root, tag+12, MPI_COMM_WORLD, &recv_stat);
	for(i = 0; i<num_proc; i++)
	{
	    g2lindex[i] = (int*) calloc(total_num_pt,sizeof(int));
	    MPI_Recv(g2lindex[i], total_num_pt, MPI_INT, root, tag+13+i, MPI_COMM_WORLD, &recv_stat);
	}

    	hpConstrPInfoFromGlobalLocalInfo(mesh, g2lindex, l2gindex, rank);

	int array_size[1];
	array_size[0] = mesh->ps->size[0];
	*ps_globalid = emxCreateWrapperND_int32_T(l2gindex,1,array_size);
	array_size[0] = mesh->tris->size[0];
	*tri_globalid = emxCreateWrapperND_int32_T(tri_index, 1, array_size);

	for(i = 0; i<num_proc; i++)
	    free(g2lindex[i]);
	free(g2lindex);
    }


    printf("Leaving hpDistMesh proc %d\n", rank);
    return 1;
}

void hpConstrPInfoFromGlobalLocalInfo(hiPropMesh *mesh,
	int** g2lindex, int* l2gindex, int rank)
{
    int num_proc;
    MPI_Comm_size(MPI_COMM_WORLD, &num_proc);
    int ps_estimate = 2*mesh->ps->size[0];
    int num_ps = mesh->ps->size[0];
    int num_tris = mesh->tris->size[0];
    int tris_estimate = 2*num_tris;
    int nb_proc_size[1];
    int i,j,k;
    int cur_head, cur_tail;

    int* nb_proc_bool = (int*) malloc(num_proc*sizeof(int));
    for(j = 0; j<num_proc; j++)
	nb_proc_bool[j] = 0;

    mesh->ps_pinfo = (hpPInfoList *) calloc(1, sizeof(hpPInfoList));
    mesh->ps_pinfo->pdata = (hpPInfoNode *) calloc(ps_estimate, sizeof(hpPInfoNode));
    mesh->ps_pinfo->max_len = ps_estimate;
    mesh->ps_pinfo->allocated_len = num_ps;
    mesh->ps_pinfo->head = (int *) calloc(num_ps, sizeof(int));
    mesh->ps_pinfo->tail = (int *) calloc(num_ps, sizeof(int));


    mesh->tris_pinfo = (hpPInfoList *) calloc(1, sizeof(hpPInfoList));
    mesh->tris_pinfo->pdata = (hpPInfoNode *) calloc(tris_estimate, sizeof(hpPInfoNode));
    mesh->tris_pinfo->head = (int *) calloc(num_tris, sizeof(int));
    mesh->tris_pinfo->tail = (int *) calloc(num_tris, sizeof(int));
    mesh->tris_pinfo->max_len = tris_estimate;
    mesh->tris_pinfo->allocated_len = num_tris;

    for (i = 1; i <= num_tris; i++)
    {
	(mesh->tris_pinfo->pdata[I1dm(i)]).proc = rank;
	(mesh->tris_pinfo->pdata[I1dm(i)]).lindex = i;
	(mesh->tris_pinfo->pdata[I1dm(i)]).next = -1;
	mesh->tris_pinfo->head[I1dm(i)] = i;
	mesh->tris_pinfo->tail[I1dm(i)] = i;
    }


    for (j = 1; j <= num_ps; j++)
    {
	mesh->ps_pinfo->head[I1dm(j)] = j;
	mesh->ps_pinfo->tail[I1dm(j)] = -1;	/* the list is empty */
    }

    for(j = 1; j<=num_ps; j++)
    {
	for(k = 0; k<num_proc; k++)
	{
	    if(g2lindex[k][l2gindex[j-1]-1]!=-1)
	    {
		nb_proc_bool[k] = 1;
		if(mesh->ps_pinfo->max_len == mesh->ps_pinfo->allocated_len)
		    hpEnsurePInfoCapacity(mesh->ps_pinfo);

		if(mesh->ps_pinfo->tail[I1dm(j)]!=-1)	/* the list is nonempty for this point */
		{
		    cur_tail = mesh->ps_pinfo->tail[I1dm(j)];
		    (mesh->ps_pinfo->allocated_len)++;
		    (mesh->ps_pinfo->pdata[I1dm(cur_tail)]).next = mesh->ps_pinfo->allocated_len;
		    cur_tail = mesh->ps_pinfo->allocated_len;
		    mesh->ps_pinfo->tail[I1dm(j)] = cur_tail;
		    (mesh->ps_pinfo->pdata[I1dm(cur_tail)]).proc = k;
		    (mesh->ps_pinfo->pdata[I1dm(cur_tail)]).lindex = g2lindex[k][l2gindex[j-1]-1];
		    (mesh->ps_pinfo->pdata[I1dm(cur_tail)]).next = -1;
		}
		else	/* the list is empty for this point */
		{
		    cur_head = mesh->ps_pinfo->head[I1dm(j)];
		    (mesh->ps_pinfo->pdata[I1dm(cur_head)]).proc = k;
		    (mesh->ps_pinfo->pdata[I1dm(cur_head)]).lindex = g2lindex[k][l2gindex[j-1]-1];
		    (mesh->ps_pinfo->pdata[I1dm(cur_head)]).next = -1;
		    mesh->ps_pinfo->tail[I1dm(j)] = cur_head;
		}
	    }
	}
    }

    nb_proc_size[0] = 0;
    for(j = 0; j<num_proc; j++)
	nb_proc_size[0]+=nb_proc_bool[j];
    nb_proc_size[0]--;		/* to exclude itself */
    mesh->nb_proc = emxCreateND_int32_T(1,nb_proc_size);

    k=0;
    for (j = 0; j<num_proc; j++)
	if((j!=rank)&&(nb_proc_bool[j]==1))
	    mesh->nb_proc->data[k++] = j;

    printf("After hpConstrPInfoFromGlobalLocalInfo\n");
}


void hpGetNbProcListFromInput(hiPropMesh *mesh, const int num_nb_proc,
			      const int *in_nb_proc)
{
    if (mesh->nb_proc != ((emxArray_int32_T *) NULL))
	emxFree_int32_T(&(mesh->nb_proc));

    int i;
    int num_nb[1];
    num_nb[0] = num_nb_proc;

    mesh->nb_proc = emxCreateND_int32_T(1, num_nb);
    for (i = 1; i <= num_nb_proc; i++)
	mesh->nb_proc->data[I1dm(i)] = in_nb_proc[i-1];

}

void hpGetNbProcListAuto(hiPropMesh *mesh)
{
    if (mesh->nb_proc != ((emxArray_int32_T *) NULL))
	emxFree_int32_T(&(mesh->nb_proc));

    int i, tag_send, tag_recv, j, k;
    int num_proc, rank;
    double eps = 1e-14;

    MPI_Comm_size(MPI_COMM_WORLD, &num_proc);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int *nb_ptemp = (int *) calloc (num_proc-1, sizeof(int));
    int num_nbp = 0;

    MPI_Request *send_req_list = (MPI_Request *) malloc ( 2*(num_proc-1)*sizeof(MPI_Request) );
    MPI_Status *send_status_list = (MPI_Status *) malloc( 2*(num_proc-1)*sizeof(MPI_Status) );

    MPI_Request *recv_req_list = (MPI_Request *) malloc( (num_proc-1)*sizeof(MPI_Request) );

    /* Stores the received array size */
    int *common_info = (int *) calloc( (num_proc)*2, sizeof(int));

    j = 0;
    for (i = 0; i < num_proc; i++)
    {
	tag_send = i;
	if (rank != i)
	{
	    isend2D_real_T(mesh->ps, i, tag_send, MPI_COMM_WORLD,
		    	   &(send_req_list[j]), &(send_req_list[num_proc-1+j]));
	    j++;
	}
    }

    j = 0;
    for (i = 0; i < num_proc; i++)
    {
	tag_recv = rank;

	if (rank != i)
	{
	    MPI_Irecv(&(common_info[2*i]), 2, MPI_INT, i, tag_recv+1, MPI_COMM_WORLD, &(recv_req_list[j]));
	    j++;
	}
    }

    for (i = 0; i < num_proc-1; i++)
    {
	emxArray_real_T *ps_recv;
	tag_recv = rank;
	MPI_Status recv_status1;
	MPI_Status recv_status2;
	int recv_index;
	int source_id;

	MPI_Waitany(num_proc-1, recv_req_list, &recv_index, &recv_status1);

	source_id = recv_status1.MPI_SOURCE;

	ps_recv = emxCreate_real_T(common_info[2*source_id], common_info[2*source_id+1]);

	MPI_Recv(ps_recv->data, common_info[2*source_id]*common_info[2*source_id+1], MPI_DOUBLE, source_id, tag_recv+2, MPI_COMM_WORLD, &recv_status2);

	for (j = 1; j <= mesh->ps->size[0]; j++)
	{
	    double current_x = mesh->ps->data[I2dm(j,1,mesh->ps->size)];
	    double current_y = mesh->ps->data[I2dm(j,2,mesh->ps->size)];
	    double current_z = mesh->ps->data[I2dm(j,3,mesh->ps->size)];

	    for (k = 1; k <= ps_recv->size[0]; k++)
	    {
		if ( (fabs(current_x - ps_recv->data[I2dm(k,1,ps_recv->size)]) < eps) && 
			(fabs(current_y - ps_recv->data[I2dm(k,2,ps_recv->size)]) < eps) &&
			(fabs(current_z - ps_recv->data[I2dm(k,3,ps_recv->size)]) < eps)
			)
		{
		    nb_ptemp[num_nbp++] = source_id;
		    break;
		}
	    }
	    if (k <= ps_recv->size[0])
		break;
	}
	emxFree_real_T(&ps_recv);
    }

    
    int num_nb[1];
    num_nb[0] = num_nbp;

    mesh->nb_proc = emxCreateND_int32_T(1, num_nb);
    for (i = 1; i <= num_nbp; i++)
	mesh->nb_proc->data[I1dm(i)] = nb_ptemp[i-1];

    free(nb_ptemp);
    free(recv_req_list);
    free(common_info);

    MPI_Waitall(2*(num_proc-1), send_req_list, send_status_list);

    free(send_req_list);
    free(send_status_list);


}

void hpInitPInfo(hiPropMesh *mesh)
{
    hpDeletePInfoList(&(mesh->ps_pinfo));
    hpDeletePInfoList(&(mesh->tris_pinfo));

    int i;
    int num_ps = mesh->ps->size[0];
    int num_tris = mesh->tris->size[0];

    int ps_estimate = 2*num_ps;
    int tris_estimate = 2*num_tris;

    int my_rank;

    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);


    mesh->ps_pinfo = (hpPInfoList *) calloc(1, sizeof(hpPInfoList));
    mesh->tris_pinfo = (hpPInfoList *) calloc(1, sizeof(hpPInfoList));

    mesh->ps_pinfo->pdata = (hpPInfoNode *) calloc(ps_estimate, sizeof(hpPInfoNode));
    mesh->tris_pinfo->pdata = (hpPInfoNode *) calloc(tris_estimate, sizeof(hpPInfoNode));

    mesh->ps_pinfo->head = (int *) calloc(num_ps, sizeof(int));
    mesh->tris_pinfo->head = (int *) calloc(num_tris, sizeof(int));

    mesh->ps_pinfo->tail = (int *) calloc(num_ps, sizeof(int));
    mesh->tris_pinfo->tail = (int *) calloc(num_tris, sizeof(int));

    mesh->ps_pinfo->max_len = ps_estimate;
    mesh->tris_pinfo->max_len = tris_estimate;

    for (i = 1; i <= num_ps; i++)
    {
	(mesh->ps_pinfo->pdata[I1dm(i)]).proc = my_rank;
	(mesh->ps_pinfo->pdata[I1dm(i)]).lindex = i;
	(mesh->ps_pinfo->pdata[I1dm(i)]).next = -1;
	mesh->ps_pinfo->head[I1dm(i)] = i;
	mesh->ps_pinfo->tail[I1dm(i)] = i;
    }
    for (i = 1; i <= num_tris; i++)
    {
	(mesh->tris_pinfo->pdata[I1dm(i)]).proc = my_rank;
	(mesh->tris_pinfo->pdata[I1dm(i)]).lindex = i;
	(mesh->tris_pinfo->pdata[I1dm(i)]).next = -1;
	mesh->tris_pinfo->head[I1dm(i)] = i;
	mesh->tris_pinfo->tail[I1dm(i)] = i;
    }
    mesh->ps_pinfo->allocated_len = num_ps;
    mesh->tris_pinfo->allocated_len = num_tris;

}

void hpEnsurePInfoCapacity(hpPInfoList *pinfo)
{
    if (pinfo->allocated_len >= pinfo->max_len)
    {
	double len_temp = pinfo->max_len * 1.1;
	int new_max_len = (int) (len_temp); /* Increase 10% */
	hpPInfoNode *new_pdata = (hpPInfoNode *) calloc(new_max_len, sizeof(hpPInfoNode));
	memcpy(new_pdata, pinfo->pdata, pinfo->allocated_len*sizeof(hpPInfoNode));

	free(pinfo->pdata);
	pinfo->pdata = new_pdata;
	pinfo->max_len = new_max_len;
    }
}

void hpBuildPInfoNoOverlappingTris(hiPropMesh *mesh)
{
    int i, tag_send, tag_recv, j, k;
    int proc_send, proc_recv;
    int num_proc, rank;
    double eps = 1e-14;

    emxArray_real_T* ps = mesh->ps;
    emxArray_int32_T *nb_proc = mesh->nb_proc;

    hpPInfoList *ps_pinfo = mesh->ps_pinfo;


    MPI_Comm_size(MPI_COMM_WORLD, &num_proc);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int num_nbp = nb_proc->size[0];

    MPI_Request *send_req_list = (MPI_Request *) malloc( 2*num_nbp*sizeof(MPI_Request) );

    MPI_Status *send_status_list = (MPI_Status *) malloc( 2*num_nbp*sizeof(MPI_Status) );

    MPI_Request *recv_req_list = (MPI_Request *) malloc ( num_nbp*sizeof(MPI_Request) );

    /* Stores the received array size */

    int *recv_size = (int *) calloc ( 2*num_nbp, sizeof(int));
    
    for (i = 1; i <= num_nbp; i++)
    {
	proc_send = nb_proc->data[I1dm(i)];
	tag_send = proc_send;
	isend2D_real_T(ps, proc_send, tag_send, MPI_COMM_WORLD,
		       &(send_req_list[I1dm(i)]), &(send_req_list[I1dm(i)+num_nbp]));
    }

    for (i = 1; i <= num_nbp; i++)
    {
	proc_recv = nb_proc->data[I1dm(i)];
	tag_recv = rank;
	MPI_Irecv(&(recv_size[2*I1dm(i)]), 2, MPI_INT, proc_recv, tag_recv+1, MPI_COMM_WORLD, &(recv_req_list[I1dm(i)]));
    }

    for (i = 1; i <= num_nbp; i++)
    {
	emxArray_real_T *ps_recv;
	tag_recv = rank;
	MPI_Status recv_status1;
	MPI_Status recv_status2;
	int recv_index;
	
	MPI_Waitany(num_nbp, recv_req_list, &recv_index, &recv_status1);

	proc_recv = recv_status1.MPI_SOURCE;

	ps_recv = emxCreate_real_T(recv_size[2*recv_index], recv_size[2*recv_index+1]);

	MPI_Recv(ps_recv->data, recv_size[2*recv_index]*recv_size[2*recv_index+1], MPI_DOUBLE, proc_recv, tag_recv+2, MPI_COMM_WORLD, &recv_status2);

	for (j = 1; j <= ps->size[0]; j++)
	{
	    double current_x = ps->data[I2dm(j,1,ps->size)];
	    double current_y = ps->data[I2dm(j,2,ps->size)];
	    double current_z = ps->data[I2dm(j,3,ps->size)];

	    for (k = 1; k <= ps_recv->size[0]; k++)
	    {
		if ( (fabs(current_x - ps_recv->data[I2dm(k,1,ps_recv->size)]) < eps) && 
			(fabs(current_y - ps_recv->data[I2dm(k,2,ps_recv->size)]) < eps) &&
			(fabs(current_z - ps_recv->data[I2dm(k,3,ps_recv->size)]) < eps)
		   )
		{
		    hpEnsurePInfoCapacity(ps_pinfo); /* first ensure 
							list has enough space */
		    ps_pinfo->allocated_len++; /* new node */
		    int cur_head = ps_pinfo->head[I1dm(j)];
		    int cur_tail = ps_pinfo->tail[I1dm(j)];
		    int cur_master_proc = ps_pinfo->pdata[I1dm(cur_head)].proc;
		    if (proc_recv < cur_master_proc)
		    {
			ps_pinfo->head[I1dm(j)] = ps_pinfo->allocated_len;
			ps_pinfo->pdata[I1dm(ps_pinfo->allocated_len)].proc = proc_recv;
			ps_pinfo->pdata[I1dm(ps_pinfo->allocated_len)].lindex = k;
			ps_pinfo->pdata[I1dm(ps_pinfo->allocated_len)].next = cur_head;
		    }
		    else if (proc_recv > cur_master_proc)
		    {
			ps_pinfo->tail[I1dm(j)] = ps_pinfo->allocated_len;
			ps_pinfo->pdata[I1dm(ps_pinfo->allocated_len)].proc = proc_recv;
			ps_pinfo->pdata[I1dm(ps_pinfo->allocated_len)].lindex = k;
			ps_pinfo->pdata[I1dm(ps_pinfo->allocated_len)].next = -1;
			ps_pinfo->pdata[I1dm(cur_tail)].next = ps_pinfo->allocated_len;
		    }
		    else
		    {
			printf("\n Receiving processor ID already in the PInfo list!\n");
			exit(0);
		    }
		    break;
		}
	    }
	}
	emxFree_real_T(&ps_recv);
    }

    free(recv_size);
    free(recv_req_list);

    MPI_Waitall(2*num_nbp, send_req_list, send_status_list);

    free(send_req_list);
    free(send_status_list);

    mesh->num_int_ps = mesh->ps->size[0];
    mesh->num_int_tris = mesh->tris->size[0];
    mesh->num_int_pspinfo = mesh->ps_pinfo->allocated_len;
    mesh->is_clean = 1;

}

void hpBuildPInfoWithOverlappingTris(hiPropMesh *mesh)
{
    int i, j, k;
    int ps_tag_send, ps_tag_recv, tris_tag_send, tris_tag_recv;
    int proc_send, proc_recv;
    int num_proc, rank;
    double eps = 1e-14;

    emxArray_real_T* ps = mesh->ps;
    emxArray_int32_T* tris = mesh->tris;
    emxArray_int32_T *nb_proc = mesh->nb_proc;

    hpPInfoList *ps_pinfo = mesh->ps_pinfo;
    hpPInfoList *tris_pinfo = mesh->tris_pinfo;

    MPI_Comm_size(MPI_COMM_WORLD, &num_proc);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int num_nbp = nb_proc->size[0];


    MPI_Request *send_req_list = (MPI_Request *) malloc( 4*num_nbp*sizeof(MPI_Request) );
    MPI_Status *send_status_list = (MPI_Status *) malloc( 4*num_nbp*sizeof(MPI_Status) );

    MPI_Request *ps_recv_req_list = (MPI_Request *) malloc ( num_nbp*sizeof(MPI_Request));

    int *ps_recv_size = (int *) calloc (2*num_nbp, sizeof(int));
    
    unsigned char is_overlapping_tri = 0;

    for (i = 1; i <= num_nbp; i++)
    {
	proc_send = nb_proc->data[I1dm(i)];
	ps_tag_send = proc_send;
	tris_tag_send = proc_send + 10;
	isend2D_real_T(ps, proc_send, ps_tag_send, MPI_COMM_WORLD,
		       &(send_req_list[I1dm(i)]), &(send_req_list[I1dm(i)+num_nbp]));
	isend2D_int32_T(tris, proc_send, tris_tag_send, MPI_COMM_WORLD, 
			&(send_req_list[I1dm(i)+2*num_nbp]), &(send_req_list[I1dm(i)+3*num_nbp]));
    }

    for (i = 1; i <= num_nbp; i++)
    {
	proc_recv = nb_proc->data[I1dm(i)];
	ps_tag_recv = rank;
	tris_tag_recv = rank + 10;
	MPI_Irecv(&(ps_recv_size[2*I1dm(i)]), 2, MPI_INT, proc_recv, ps_tag_recv+1, MPI_COMM_WORLD, &(ps_recv_req_list[I1dm(i)]));
    }

    for (i = 1; i <= num_nbp; i++)
    {
	emxArray_real_T *ps_recv;
	emxArray_int32_T *tris_recv;
	ps_tag_recv = rank;
	tris_tag_recv = rank + 10;

	MPI_Status ps_recv_status1, ps_recv_status2;
	int recv_index;

	MPI_Waitany(num_nbp, ps_recv_req_list, &recv_index, &ps_recv_status1);
	proc_recv = ps_recv_status1.MPI_SOURCE;

	ps_recv = emxCreate_real_T(ps_recv_size[2*recv_index], ps_recv_size[2*recv_index+1]);

	MPI_Recv(ps_recv->data, ps_recv_size[2*recv_index]*ps_recv_size[2*recv_index+1], MPI_DOUBLE, proc_recv, ps_tag_recv+2, MPI_COMM_WORLD, &ps_recv_status2);

	recv2D_int32_T(&tris_recv, proc_recv, tris_tag_recv, MPI_COMM_WORLD);

	/* Build the pinfo for points */
	for (j = 1; j <= ps->size[0]; j++)
	{
	    double current_x = ps->data[I2dm(j,1,ps->size)];
	    double current_y = ps->data[I2dm(j,2,ps->size)];
	    double current_z = ps->data[I2dm(j,3,ps->size)];

	    for (k = 1; k <= ps_recv->size[0]; k++)
	    {
		if ( (fabs(current_x - ps_recv->data[I2dm(k,1,ps_recv->size)]) < eps) && 
		     (fabs(current_y - ps_recv->data[I2dm(k,2,ps_recv->size)]) < eps) &&
		     (fabs(current_z - ps_recv->data[I2dm(k,3,ps_recv->size)]) < eps)
		   )
		{
		    hpEnsurePInfoCapacity(ps_pinfo); /* first ensure 
							list has enough space */
		    ps_pinfo->allocated_len++; /* new node */
		    int cur_head = ps_pinfo->head[I1dm(j)];
		    int cur_tail = ps_pinfo->tail[I1dm(j)];
		    int cur_master_proc = ps_pinfo->pdata[I1dm(cur_head)].proc;
		    if (proc_recv < cur_master_proc)
		    {
			ps_pinfo->head[I1dm(j)] = ps_pinfo->allocated_len;
			ps_pinfo->pdata[I1dm(ps_pinfo->allocated_len)].proc = proc_recv;
			ps_pinfo->pdata[I1dm(ps_pinfo->allocated_len)].lindex = k;
			ps_pinfo->pdata[I1dm(ps_pinfo->allocated_len)].next = cur_head;
		    }
		    else if (proc_recv > cur_master_proc)
		    {
			ps_pinfo->tail[I1dm(j)] = ps_pinfo->allocated_len;
			ps_pinfo->pdata[I1dm(ps_pinfo->allocated_len)].proc = proc_recv;
			ps_pinfo->pdata[I1dm(ps_pinfo->allocated_len)].lindex = k;
			ps_pinfo->pdata[I1dm(ps_pinfo->allocated_len)].next = -1;
			ps_pinfo->pdata[I1dm(cur_tail)].next = ps_pinfo->allocated_len;
		    }
		    else
		    {
			printf("\n Receiving processor ID already in the ps PInfo list!\n");
			exit(0);
		    }
		    break;
		}
	    }
	}


	/* Build the tris pinfo */
	for (j = 1; j <= tris->size[0]; j++)
	{
	    for (k = 1; k <= tris_recv->size[0]; k++)
	    {
		if ( sameTriangle(ps, tris, j, ps_recv, tris_recv, k, eps) )
		{
		    is_overlapping_tri = 1;
		    hpEnsurePInfoCapacity(tris_pinfo); /* first ensure 
							  list has enough space */
		    tris_pinfo->allocated_len++; /* new node */
		    int cur_head = tris_pinfo->head[I1dm(j)];
		    int cur_tail = tris_pinfo->tail[I1dm(j)];
		    int cur_master_proc = tris_pinfo->pdata[I1dm(cur_head)].proc;
		    if (proc_recv < cur_master_proc)
		    {
			tris_pinfo->head[I1dm(j)] = tris_pinfo->allocated_len;
			tris_pinfo->pdata[I1dm(tris_pinfo->allocated_len)].proc = proc_recv;
			tris_pinfo->pdata[I1dm(tris_pinfo->allocated_len)].lindex = k;
			tris_pinfo->pdata[I1dm(tris_pinfo->allocated_len)].next = cur_head;
		    }
		    else if (proc_recv > cur_master_proc)
		    {
			tris_pinfo->tail[I1dm(j)] = tris_pinfo->allocated_len;
			tris_pinfo->pdata[I1dm(tris_pinfo->allocated_len)].proc = proc_recv;
			tris_pinfo->pdata[I1dm(tris_pinfo->allocated_len)].lindex = k;
			tris_pinfo->pdata[I1dm(tris_pinfo->allocated_len)].next = -1;
			tris_pinfo->pdata[I1dm(cur_tail)].next = tris_pinfo->allocated_len;
		    }
		    else
		    {
			printf("\n Receiving processor ID already in the tris PInfo list!\n");
			exit(0);
		    }
		    break;
		}
	    }
	}

	emxFree_real_T(&ps_recv);
	emxFree_int32_T(&tris_recv);
    }

    free(ps_recv_size);
    free(ps_recv_req_list);

    MPI_Waitall(4*num_nbp, send_req_list, send_status_list);

    free(send_req_list);
    free(send_status_list);

    if (is_overlapping_tri == 1)
	mesh->is_clean = 0;
    else
    {
	mesh->is_clean = 1;
	mesh->num_int_ps = mesh->ps->size[0];
	mesh->num_int_tris = mesh->tris->size[0];
	mesh->num_int_pspinfo = mesh->ps_pinfo->allocated_len;
    }

    boolean_T out_clean;

    MPI_Allreduce(&(mesh->is_clean), &(out_clean), 1, MPI_UNSIGNED_CHAR, MPI_MIN, MPI_COMM_WORLD);

    mesh->is_clean = out_clean;
}

void hpBuildOppositeHalfEdge(hiPropMesh *mesh)
{
    if (mesh->opphe != ((emxArray_int32_T *) NULL))
	emxFree_int32_T(&(mesh->opphe));

    int num_ps = mesh->ps->size[0];
    int num_tris = mesh->tris->size[0];

    mesh->opphe = emxCreate_int32_T(num_tris, 3);
    
    determine_opposite_halfedge_tri(num_ps, mesh->tris ,mesh->opphe);
}

void hpBuildIncidentHalfEdge(hiPropMesh *mesh)
{
    if (mesh->inhe != ((emxArray_int32_T *) NULL))
	emxFree_int32_T(&(mesh->inhe));

    int num_ps = mesh->ps->size[0];

    int temp[1];
    temp[0] = num_ps;

    mesh->inhe = emxCreateND_int32_T(1, temp);

    determine_incident_halfedges(mesh->tris, mesh->opphe, mesh->inhe);
}

void hpObtainNRingTris(const hiPropMesh *mesh, 
		       const int32_T in_vid,
		       const real_T in_ring,
		       const int32_T in_minpnts,
		       const int32_T max_numps,
		       const int32_T max_numtris,
		       emxArray_int32_T **in_ngbvs,
		       emxArray_int32_T **in_ngbfs,
		       emxArray_boolean_T **in_vtags, 
		       emxArray_boolean_T **in_ftags,
		       int32_T *in_nverts, int32_T *in_nfaces)
{
    int i;

    int max_b_numps[1]; max_b_numps[0] = max_numps;
    int max_b_numtris[1]; max_b_numtris[0] = max_numtris;
    int num_ps[1]; num_ps[0] = mesh->ps->size[0];
    int num_tris[1]; num_tris[0] = mesh->tris->size[0];

    (*in_ngbvs) = emxCreateND_int32_T(1, max_b_numps);
    (*in_ngbfs) = emxCreateND_int32_T(1, max_b_numtris);

    (*in_vtags) = emxCreateND_boolean_T(1, num_ps);
    (*in_ftags) = emxCreateND_boolean_T(1, num_tris);

    for (i = 1; i <= num_ps[0]; i++)
	(*in_vtags)->data[I1dm(i)] = false;
    for (i = 1; i <= num_tris[0]; i++)
	(*in_ftags)->data[I1dm(i)] = false;

    obtain_nring_surf(in_vid, in_ring, in_minpnts, mesh->tris, mesh->opphe, mesh->inhe, (*in_ngbvs), (*in_vtags), (*in_ftags), (*in_ngbfs), in_nverts, in_nfaces);

}

void hpBuildPUpdateInfo(hiPropMesh *mesh)
{
    hpFreeMeshUpdateInfo(mesh);

    int num_nb_proc = mesh->nb_proc->size[0];
    int num_pt = mesh->ps->size[0];
    int cur_head, cur_node, cur_proc;
    int master;
    int rank, i, j, num_proc;
    int buffer_size[1];

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_proc);

    /* initialization of pointers */
    mesh->ps_send_index = (emxArray_int32_T **) calloc(num_proc, sizeof(emxArray_int32_T*));
    mesh->ps_recv_index = (emxArray_int32_T **) calloc(num_proc, sizeof(emxArray_int32_T*));
    mesh->ps_send_buffer = (emxArray_real_T **) calloc(num_proc, sizeof(emxArray_real_T*));
    mesh->ps_recv_buffer = (emxArray_real_T **) calloc(num_proc, sizeof(emxArray_real_T*));
    for (i = 0; i<num_proc; i++)
    {
	mesh->ps_send_index[i] = (emxArray_int32_T *) NULL;
	mesh->ps_recv_index[i] = (emxArray_int32_T *) NULL;
	mesh->ps_send_buffer[i] = (emxArray_real_T *) NULL;
	mesh->ps_recv_buffer[i] = (emxArray_real_T *) NULL;
    }

    /* compute the length of the buffers */
    int* pt_send_buffer_length = (int*) calloc(num_proc, sizeof(int));
    int* pt_recv_buffer_length = (int*) calloc(num_proc, sizeof(int));
    for (i = 0; i< num_proc; i++)
    {
	pt_send_buffer_length[i] = 0;
	pt_recv_buffer_length[i] = 0;
    }
    for(i = 1; i<=num_pt; i++)
    {
	cur_head = mesh->ps_pinfo->head[I1dm(i)];
	master = (mesh->ps_pinfo->pdata[I1dm(cur_head)]).proc;
	if (master == rank)	/* the current proc is the master, send to all other */
	{
	    cur_node = mesh->ps_pinfo->pdata[I1dm(cur_head)].next;
	    while(cur_node!=-1)
	    {
		cur_proc = mesh->ps_pinfo->pdata[I1dm(cur_node)].proc;
		pt_send_buffer_length[cur_proc]++;
		cur_node = mesh->ps_pinfo->pdata[I1dm(cur_node)].next;
	    }
	}
	else	/* the current proc is not the master, recv this point from master */
	    pt_recv_buffer_length[master]++;
    }

    /* Here is a problem, we want the send and recv buffers have points with the same order,
     * which means that at least one buffer should be sorted.
     * We can see that the send buffer has points sorted with the same order on this proc,
     * we need to sort the recv buffer corresponding to the order on the sent proc
     */
    /* remoteid is used to store the local id of the points on the send proc, we need to sort the points by this id */
    int** remoteid = (int**) calloc(num_proc, sizeof(int*));	    

    /* allocate memory for index and buffer array, also for remoteid */
    for (i = 0; i<num_nb_proc; i++)
    {
	cur_proc = mesh->nb_proc->data[i];
	if(pt_recv_buffer_length[cur_proc]!=0)		/*  for the recv case. 
					 		 *  In this case, we need to sort the points on this proc, 
					 		 *  so allocate memory for remoteid */
	{
	    buffer_size[0] = pt_recv_buffer_length[cur_proc];
	    mesh->ps_recv_index[cur_proc] = emxCreateND_int32_T(1, buffer_size);
	    mesh->ps_recv_buffer[cur_proc] = emxCreate_real_T(buffer_size[0], 3);
	    remoteid[cur_proc] = (int*) calloc(pt_recv_buffer_length[cur_proc], sizeof(int));
	}
	if(pt_send_buffer_length[cur_proc]!=0)	/* for the send case */
	{
	    buffer_size[0] = pt_send_buffer_length[cur_proc];
	    mesh->ps_send_index[cur_proc] = emxCreateND_int32_T(1, buffer_size);
	    mesh->ps_send_buffer[cur_proc] = emxCreate_real_T(buffer_size[0], 3);
	}
    }


    int* cur_size;
    int* p_send_index = (int*) calloc(num_proc, sizeof(int));	/* index to the end of the list */
    int* p_recv_index = (int*) calloc(num_proc, sizeof(int));	/* index to the end of the list */
    int tmp_id;
    int tmp_pt;
    for(i = 0; i<num_proc; i++)
    {
	p_send_index[i] = 1;
	p_recv_index[i] = 1;
    }
    for(i = 1; i<=num_pt; i++)
    {
	cur_head = mesh->ps_pinfo->head[I1dm(i)];
	master = (mesh->ps_pinfo->pdata[I1dm(cur_head)]).proc;
	if (master == rank)	/* the current proc is the master, send to all others */
	{
	    cur_node = mesh->ps_pinfo->pdata[I1dm(cur_head)].next;
	    while(cur_node!=-1)
	    {
		cur_proc = mesh->ps_pinfo->pdata[I1dm(cur_node)].proc;

		mesh->ps_send_index[cur_proc]->data[I1dm(p_send_index[cur_proc])]= i;
		cur_size = mesh->ps_send_buffer[cur_proc]->size;
		mesh->ps_send_buffer[cur_proc]->data[I2dm(p_send_index[cur_proc],1,cur_size)]
		    = mesh->ps->data[I2dm(i, 1, mesh->ps->size)];
		mesh->ps_send_buffer[cur_proc]->data[I2dm(p_send_index[cur_proc],2,cur_size)]
		    = mesh->ps->data[I2dm(i, 2, mesh->ps->size)];
		mesh->ps_send_buffer[cur_proc]->data[I2dm(p_send_index[cur_proc],3,cur_size)]
		    = mesh->ps->data[I2dm(i, 3, mesh->ps->size)];

		p_send_index[cur_proc]++;
		cur_node = mesh->ps_pinfo->pdata[I1dm(cur_node)].next;
	    }
	}
	else	/* the current proc is not the master, recv this point from master, sorted by insertion sort */
	{

	    remoteid[master][I1dm(p_recv_index[master])] = mesh->ps_pinfo->pdata[I1dm(cur_head)].lindex;
	    mesh->ps_recv_index[master]->data[I1dm(p_recv_index[master])] = i;

	    if(p_recv_index[master]>1)	/* sort by the key of remoteid[master], the value is mesh->ps_recv_index[master]->data[] */
	    {
		for (j = p_recv_index[master]; j>1; j-- )
		{
		    if(remoteid[master][I1dm(j)]<remoteid[master][I1dm(j-1)])
		    {
			tmp_pt = mesh->ps_recv_index[master]->data[I1dm(j)];
			mesh->ps_recv_index[master]->data[I1dm(j)] = mesh->ps_recv_index[master]->data[I1dm(j-1)];
			mesh->ps_recv_index[master]->data[I1dm(j-1)] = tmp_pt;

			tmp_id = remoteid[master][I1dm(j)];
			remoteid[master][I1dm(j)] = remoteid[master][I1dm(j-1)];
			remoteid[master][I1dm(j-1)] = tmp_id;
		    }
		    else
			break;
		}
	    }
	    p_recv_index[master]++;
	}
    }

    free(pt_send_buffer_length);
    free(pt_recv_buffer_length);
    free(p_send_index);
    free(p_recv_index);
    for (i = 0; i<num_proc; i++)
	if(remoteid[i]!=NULL)
	    free(remoteid[i]);

    free(remoteid);	   
}

void hpCleanMeshByPinfo(hiPropMesh* mesh)
{
    int rank, num_proc;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_proc);
    int i,j,k;

    /* 1. Identify the tris to be saved */
    int num_tri = mesh->tris->size[0];
    int num_pt = mesh->ps->size[0];
    int* tris_to_save = (int*)calloc(num_tri, sizeof(int));
    int num_tri_to_save = 0;
    int* tri_save_flag = (int*) calloc(num_tri, sizeof(int));	/* flag 1 to the tri we need to keep,
								 * otherwise 0 */

    int master, head;
    for (i = 1; i<=num_tri; i++)
    {
	head = mesh->tris_pinfo->head[I1dm(i)];
	master = mesh->tris_pinfo->pdata[I1dm(head)].proc;
	if(master==rank)
	{
	    tris_to_save[num_tri_to_save] = i;
	    num_tri_to_save++;
	    tri_save_flag[I1dm(i)] = 1;
	}
	else
	    tri_save_flag[I1dm(i)] = 0;
    }

    /* 2. make a new mesh->point array, delete the old, use pt_save_flag and pt_new_index
     *    to do this and also to update the info on other procs later
     */
    int* pt_save_flag = (int*) calloc(num_pt, sizeof(int));	/* flag 1 to the point we need to keep, 
								 * otherwise 0 */
    int* pt_new_index = (int*) calloc(num_pt, sizeof(int));	/* if current point is to be saved,
								 * save the new index here */
    for (i = 0; i< num_pt; i++)
    {
	pt_save_flag[i] = 0;
	pt_new_index[i] = 0;
    }

    int num_pt_to_save = 0;
    int cur_tri;
    int cur_pt;
    for(i = 0; i<num_tri_to_save; i++)
    {
	cur_tri = tris_to_save[i];
	for(j = 1; j<=3; j++)
	{
	    cur_pt = mesh->tris->data[I2dm(cur_tri,j,mesh->tris->size)];
	    pt_save_flag[I1dm(cur_pt)] = 1;
	}
    }

    for (i = 0; i<num_pt; i++)
	num_pt_to_save += pt_save_flag[i];

    emxArray_real_T* newpts = emxCreate_real_T(num_pt_to_save, 3);
    cur_pt = 1;
    for(i = 1; i<=num_pt_to_save; i++)
    {
	while(pt_save_flag[I1dm(cur_pt)]!=1)
	    cur_pt++;

	pt_new_index[I1dm(cur_pt)] = i;
	for(j = 1; j<=3; j++)
	    newpts->data[I2dm(i, j, newpts->size)] = mesh->ps->data[I2dm(cur_pt,j,mesh->ps->size)];
	cur_pt++;
    }

    emxDestroyArray_real_T(mesh->ps);
    mesh->ps = newpts;

    /* 3. make a new mesh->tris array, delete the old one */
    emxArray_int32_T* newtris = emxCreate_int32_T(num_tri_to_save, 3);
    int p, old_p;
    for (i = 1; i<=num_tri_to_save; i++)
    {
	cur_tri = tris_to_save[I1dm(i)];
	for(j = 1; j<=3; j++)
	{
	    old_p = mesh->tris->data[I2dm(cur_tri, j, mesh->tris->size)];	/* old point index */
	    p = pt_new_index[I1dm(old_p)];	/* new point index */
	    newtris->data[I2dm(i,j,newtris->size)] = p;
	}
    }

    emxDestroyArray_int32_T(mesh->tris);
    mesh->tris = newtris;

    /* 4. Send the pt_save_flag and pt_new_index info to all neighbour procs */
    int num_nb_proc = mesh->nb_proc->size[0];
    int tag[3] = {1,2,3};
    MPI_Request* request_send = (MPI_Request*) malloc( 3*num_nb_proc * sizeof(MPI_Request));
    int dest;

    for (i = 0; i<num_nb_proc; i++)
    {
	dest = mesh->nb_proc->data[i];
	MPI_Isend(&num_pt,1,MPI_INT,dest,tag[0],MPI_COMM_WORLD, &request_send[3*i]);
	MPI_Isend(pt_save_flag, num_pt, MPI_INT, dest, tag[1], MPI_COMM_WORLD, &request_send[3*i+1]);
	MPI_Isend(pt_new_index, num_pt, MPI_INT, dest, tag[2], MPI_COMM_WORLD, &request_send[3*i+2]);
    }

    /* 5. Recv pt_save_flag and pt_new_index info from all neighbours and update mesh->ps_pinfo */
    int* num_pt_to_recv = (int*)calloc(num_nb_proc, sizeof(int));
    int** pt_flag = (int**) calloc(num_proc, sizeof(int*));
    int** pt_index = (int**) calloc(num_proc, sizeof(int*));
    int source;
    MPI_Status status;

    for(i = 0; i<num_proc; i++)
    {
	pt_flag[i] = (int*)NULL;
	pt_index[i] = (int*)NULL;
    }


    MPI_Request* req_recv_num = (MPI_Request*) malloc(num_nb_proc*sizeof(MPI_Request));
    int recv_index;
    for (i = 0; i<num_nb_proc; i++)
    {
	source = mesh->nb_proc->data[i];
	MPI_Irecv(&num_pt_to_recv[i], 1, MPI_INT, source, tag[0], MPI_COMM_WORLD, &req_recv_num[i]);
    }

    for(i = 0; i<num_nb_proc; i++)
    {
	MPI_Waitany(num_nb_proc, req_recv_num, &recv_index, &status);
	source = status.MPI_SOURCE;
	pt_flag[source] = (int*) calloc(num_pt_to_recv[recv_index], sizeof(int));
	pt_index[source] = (int*) calloc(num_pt_to_recv[recv_index], sizeof(int));
	MPI_Recv(pt_flag[source], num_pt_to_recv[recv_index], MPI_INT, source, tag[1], MPI_COMM_WORLD, &status);
	MPI_Recv(pt_index[source], num_pt_to_recv[recv_index], MPI_INT, source, tag[2],MPI_COMM_WORLD, &status);
    }

    pt_flag[rank] = pt_save_flag;
    pt_index[rank] = pt_new_index;

    /* To make a new mesh->ps_pinfo */
    hpPInfoList *old_ps_pinfo = mesh->ps_pinfo;
    mesh->ps_pinfo = (hpPInfoList *)NULL;
    hpInitPInfo(mesh);

    cur_pt = 1;
    int new_head, old_head, next_node;
    hpPInfoNode *old_node, *new_node;
    for(i = 1; i<=mesh->ps->size[0]; i++)
    {
	while(pt_save_flag[I1dm(cur_pt)]!=1)
	    cur_pt++;
	new_head = mesh->ps_pinfo->head[I1dm(i)];
	old_head = old_ps_pinfo->head[I1dm(cur_pt)];
	old_node = &(old_ps_pinfo->pdata[I1dm(old_head)]);
	new_node = &(mesh->ps_pinfo->pdata[I1dm(new_head)]);

	while(pt_flag[old_node->proc][I1dm(old_node->lindex)]==0)
	{
	    next_node = old_node->next;
	    old_node = &(old_ps_pinfo->pdata[I1dm(next_node)]);
	}
	new_node->proc = old_node->proc;
	new_node->lindex = pt_index[old_node->proc][I1dm(old_node->lindex)];
	while(old_node->next!=-1)
	{
	    next_node = old_node->next;
	    old_node = &(old_ps_pinfo->pdata[I1dm(next_node)]);
	    if(pt_flag[old_node->proc][I1dm(old_node->lindex)]==0)
		continue;

	    hpEnsurePInfoCapacity(mesh->ps_pinfo);
	    mesh->ps_pinfo->allocated_len++;
	    new_node->next = mesh->ps_pinfo->allocated_len;
	    new_node = &(mesh->ps_pinfo->pdata[I1dm(new_node->next)]);
	    new_node->proc = old_node->proc;
	    new_node->lindex = pt_index[new_node->proc][I1dm(old_node->lindex)];
	    new_node->next = -1;
	}
	mesh->ps_pinfo->tail[I1dm(i)] = mesh->ps_pinfo->allocated_len;
    }

    hpDeletePInfoList(&old_ps_pinfo);

    free(tris_to_save);
    free(tri_save_flag);
    free(num_pt_to_recv);
    free(req_recv_num);
    MPI_Status* status_send = (MPI_Status*) malloc(3*num_nb_proc*sizeof(MPI_Status));
    MPI_Waitall(3*num_nb_proc, request_send, status_send);
    for(i = 0; i<num_proc; i++)
    {
	free(pt_flag[i]);
	free(pt_index[i]);
    }
    free(pt_flag);
    free(pt_index);
    free(request_send);
    free(status_send);

    /* 6. update nb_proc list since it might change */
    emxDestroyArray_int32_T(mesh->nb_proc);
    int nb_proc_size[1] = {0};
    int* nb_proc_bool = (int*)calloc(num_proc,sizeof(int));
    for(i = 0; i<num_proc; i++)
	nb_proc_bool[i] = 0;

    num_pt = mesh->ps->size[0];
    int cur_head, cur_proc, cur_node;
    for(i = 1; i<=num_pt; i++)
    {
	cur_head = mesh->ps_pinfo->head[I1dm(i)];
	cur_proc = (mesh->ps_pinfo->pdata[I1dm(cur_head)]).proc;
	nb_proc_bool[cur_proc] = 1;
	cur_node = mesh->ps_pinfo->pdata[I1dm(cur_head)].next;
	while(cur_node!=-1)
	{
	    cur_proc = mesh->ps_pinfo->pdata[I1dm(cur_node)].proc;
	    nb_proc_bool[cur_proc] = 1;
	    cur_node = mesh->ps_pinfo->pdata[I1dm(cur_node)].next;
	}
    }

    for(j = 0; j<num_proc; j++)
	nb_proc_size[0]+=nb_proc_bool[j];
    nb_proc_size[0]--;		/* to exclude itself */
    mesh->nb_proc = emxCreateND_int32_T(1,nb_proc_size);

    k=0;
    for (j = 0; j<num_proc; j++)
	if((j!=rank)&&(nb_proc_bool[j]==1))
	    mesh->nb_proc->data[k++] = j;

    free(nb_proc_bool);

    /* 7. since no overlapping trianlges, free mesh->tris_pinfo 
     * and then reinitialize it*/
    hpDeletePInfoList(&(mesh->tris_pinfo));

    int num_tris_tmp = mesh->tris->size[0];
    int tris_estimate_tmp = 2*num_tris_tmp;

    mesh->tris_pinfo = (hpPInfoList *) calloc(1, sizeof(hpPInfoList));
    mesh->tris_pinfo->pdata = (hpPInfoNode *) calloc(tris_estimate_tmp, sizeof(hpPInfoNode));

    mesh->tris_pinfo->head = (int *) calloc(num_tris_tmp, sizeof(int));
    mesh->tris_pinfo->tail = (int *) calloc(num_tris_tmp, sizeof(int));

    mesh->tris_pinfo->max_len = tris_estimate_tmp;

    for (i = 1; i <= num_tris_tmp; i++)
    {
	(mesh->tris_pinfo->pdata[I1dm(i)]).proc = rank;
	(mesh->tris_pinfo->pdata[I1dm(i)]).lindex = i;
	(mesh->tris_pinfo->pdata[I1dm(i)]).next = -1;
	mesh->tris_pinfo->head[I1dm(i)] = i;
	mesh->tris_pinfo->tail[I1dm(i)] = i;
    }
    mesh->tris_pinfo->allocated_len = num_tris_tmp;

    mesh->num_int_ps = mesh->ps->size[0];
    mesh->num_int_tris = mesh->tris->size[0];
    mesh->num_int_pspinfo = mesh->ps_pinfo->allocated_len;
    mesh->is_clean = 1;
}

void hpCollectAllSharedPs(const hiPropMesh *mesh, emxArray_int32_T **out_psid)
{
    hpPInfoList *ps_pinfo = mesh->ps_pinfo;
    int num_nb_proc = mesh->nb_proc->size[0];

    int i;
    int num_all_proc, cur_proc;
    MPI_Comm_size(MPI_COMM_WORLD, &num_all_proc);
    MPI_Comm_rank(MPI_COMM_WORLD, &cur_proc);

    int *num_overlay_ps = (int *) calloc(num_nb_proc, sizeof(int));

    /* nb proc mapping[i] stores the index for proc i in
     * mesh->proc->data (starts from 0) */
    int *nb_proc_mapping = (int *) calloc(num_all_proc, sizeof(int));

    /* Initialize nb_proc_mapping, for non nb proc, map to -1 */
    for (i = 0; i < num_all_proc; i++)
	nb_proc_mapping[i] = -1;
    
    /* construct the nb proc mapping */
    for (i = 1; i <= num_nb_proc; i++)
    {
	int nb_proc_id = mesh->nb_proc->data[I1dm(i)];
	nb_proc_mapping[nb_proc_id] = i-1;
    }


    /* Traverse the ps_pinfo to fill num_overlay_ps */
    for (i = 1; i <= mesh->ps->size[0]; i++)
    {
	int next_node = ps_pinfo->head[I1dm(i)];
	while (next_node != -1)
	{
	    int proc_id = ps_pinfo->pdata[I1dm(next_node)].proc;
	    if (proc_id != cur_proc)
		num_overlay_ps[nb_proc_mapping[proc_id]]++;
	    next_node = ps_pinfo->pdata[I1dm(next_node)].next;
	}
    }
    
    /* Create out_psid[i] based on num_overlay_ps */
    for (i = 0; i < num_nb_proc; i++)
	out_psid[i] = emxCreateND_int32_T(1, &(num_overlay_ps[i]) );
    
    /* use this pointer to denote how many elements has been filled
     * in out_psid[i] */
    int *cur_ps_index = (int *) calloc(num_nb_proc, sizeof(int));

    /* Traverse the ps_pinfo to fill out_psid */
    for (i = 1; i <= mesh->ps->size[0]; i++)
    {
	int next_node = ps_pinfo->head[I1dm(i)];
	while (next_node != -1)
	{
	    int proc_id = ps_pinfo->pdata[I1dm(next_node)].proc;
	    if (proc_id != cur_proc)
	    {
		int mapped_index = nb_proc_mapping[proc_id];
		(out_psid[mapped_index])->data[cur_ps_index[mapped_index]] = i;
		cur_ps_index[mapped_index]++;
	    }
	    next_node = ps_pinfo->pdata[I1dm(next_node)].next;
	}
    }

    free(cur_ps_index);
    free(num_overlay_ps);
    free(nb_proc_mapping);

}

void hpBuildBdboxGhostPsTrisForSend(const hiPropMesh *mesh,
				    const int nb_proc_index,
				    const double *bd_box,
				    emxArray_int32_T **ps_ring_proc,
				    emxArray_int32_T **tris_ring_proc,
				    emxArray_real_T **buffer_ps,
				    emxArray_int32_T **buffer_tris)
{
    int i, j;
    emxArray_int32_T *tris = mesh->tris;
    emxArray_real_T *ps = mesh->ps;
    int num_tris = tris->size[0];
    int num_ps = ps->size[0];
    hpPInfoNode *tris_pdata = mesh->tris_pinfo->pdata;
    int *tris_phead = mesh->tris_pinfo->head;

    int nb_proc = mesh->nb_proc->data[I1dm(nb_proc_index)];

    double xL = bd_box[nb_proc*6];
    double xU = bd_box[nb_proc*6+1];
    double yL = bd_box[nb_proc*6+2];
    double yU = bd_box[nb_proc*6+3];
    double zL = bd_box[nb_proc*6+4];
    double zU = bd_box[nb_proc*6+5];



    double max_len = xU - xL;
    if ( (yU-yL) > max_len)
	max_len = yU-yL;
    if ( (zU-zL) > max_len)
	max_len = zU-zL;

    double eps = 1e-10*max_len;

    /* relax the bounding box */
    xL -= eps; xU += eps;
    yL -= eps; yU += eps;
    zL -= eps; zU += eps;

    unsigned char *tris_flag = (unsigned char *) calloc(mesh->tris->size[0], sizeof(unsigned char));
    unsigned char *ps_flag = (unsigned char *) calloc(mesh->ps->size[0], sizeof(unsigned char));

    for (i = 1; i <= num_tris; i++)
    {
	int psi = tris->data[I2dm(i,1,tris->size)]; /* first point */
	double trixL = ps->data[I2dm(psi,1,ps->size)];
	double trixU = trixL;
	double triyL = ps->data[I2dm(psi,2,ps->size)];
	double triyU = triyL;
	double trizL = ps->data[I2dm(psi,3,ps->size)];
	double trizU = trizL;
	for (j = 2; j <= 3; j++)
	{
	    psi = tris->data[I2dm(i,j,tris->size)];
	    if(ps->data[I2dm(psi,1,ps->size)] < trixL)
		trixL = ps->data[I2dm(psi,1,ps->size)];
	    else if(ps->data[I2dm(psi,1,ps->size)] > trixU)
		trixU = ps->data[I2dm(psi,1,ps->size)];

	    if(ps->data[I2dm(psi,2,ps->size)] < triyL)
		triyL = ps->data[I2dm(psi,2,ps->size)];
	    else if(ps->data[I2dm(psi,2,ps->size)] > triyU)
		triyU = ps->data[I2dm(psi,2,ps->size)];

	    if(ps->data[I2dm(psi,3,ps->size)] < trizL)
		trizL = ps->data[I2dm(psi,3,ps->size)];
	    else if(ps->data[I2dm(psi,3,ps->size)] > trizU)
		trizU = ps->data[I2dm(psi,3,ps->size)];
	}

	if ( ( (trixL >= xL) && (trixL <= xU) && (triyL >= yL) && (triyL <= yU) && (trizL >= zL) && (trizL <= zU) ) ||
	     ( (trixL >= xL) && (trixL <= xU) && (triyL >= yL) && (triyL <= yU) && (trizU >= zL) && (trizU <= zU) ) ||
	     ( (trixL >= xL) && (trixL <= xU) && (triyU >= yL) && (triyU <= yU) && (trizL >= zL) && (trizL <= zU) ) ||
	     ( (trixL >= xL) && (trixL <= xU) && (triyU >= yL) && (triyU <= yU) && (trizU >= zL) && (trizU <= zU) ) ||
	     ( (trixU >= xL) && (trixU <= xU) && (triyL >= yL) && (triyL <= yU) && (trizL >= zL) && (trizL <= zU) ) ||
	     ( (trixU >= xL) && (trixU <= xU) && (triyL >= yL) && (triyL <= yU) && (trizU >= zL) && (trizU <= zU) ) ||
	     ( (trixU >= xL) && (trixU <= xU) && (triyU >= yL) && (triyU <= yU) && (trizL >= zL) && (trizL <= zU) ) ||
	     ( (trixU >= xL) && (trixU <= xU) && (triyU >= yL) && (triyU <= yU) && (trizU >= zL) && (trizU <= zU) )
	   ) /* one of tri_bd_box point is in the big bd_box */
	{
	    tris_flag[I1dm(i)] = 1;
	    /* Only send the tris & points not existing on the nb proc */

	    int next_node = tris_phead[I1dm(i)];
	    while (next_node != -1)
	    {
		int proc_id = tris_pdata[I1dm(next_node)].proc;
		if (proc_id == nb_proc)
		{
		    tris_flag[I1dm(i)] = 0;
		    break;
		}
		else
		    next_node = tris_pdata[I1dm(next_node)].next;
	    }

	    if (tris_flag[I1dm(i)] == 1)
	    {
		for (j = 1; j <= 3; j++)
		{
		    psi = tris->data[I2dm(i,j,tris->size)];
		    ps_flag[I1dm(psi)] = 1;
		}
	    }
	}
    }

    int num_buf_ps = 0;
    int num_buf_tris = 0;

    for (i = 1; i <= num_tris; i++)
    {
	if (tris_flag[I1dm(i)] == 1)
	    num_buf_tris++;
    }
    for (i = 1; i <= num_ps; i++)
    {
	if (ps_flag[I1dm(i)] == 1)
	    num_buf_ps++;
    }

    (*ps_ring_proc) = emxCreateND_int32_T(1, &num_buf_ps);
    (*tris_ring_proc) = emxCreateND_int32_T(1, &num_buf_tris);

    (*buffer_ps) = emxCreate_real_T(num_buf_ps, 3);
    (*buffer_tris) = emxCreate_int32_T(num_buf_tris, 3);

    /* fill the ps_ring_proc & tris_ring_proc */

    j = 0;
    for (i = 1; i <= num_tris; i++)
    {
	if (tris_flag[I1dm(i)] == 1)
	{
	    (*tris_ring_proc)->data[j] = i;
	    j++;
	}
    }

    j = 0;
    for (i = 1; i <= num_ps; i++)
    {
	if (ps_flag[I1dm(i)] == 1)
	{
	    (*ps_ring_proc)->data[j] = i;
	    j++;
	}
    }

    /* fill the buffer_ps and buffer_tris */

    int *ps_mapping = (int *) calloc(ps->size[0], sizeof(int));

    for (j = 1; j <= num_buf_ps; j++)
    {
	int cur_buf_ps_index = (*ps_ring_proc)->data[I1dm(j)];
	(*buffer_ps)->data[I2dm(j,1,(*buffer_ps)->size)] =
	    ps->data[I2dm(cur_buf_ps_index,1,ps->size)];
	(*buffer_ps)->data[I2dm(j,2,(*buffer_ps)->size)] =
	    ps->data[I2dm(cur_buf_ps_index,2,ps->size)];
	(*buffer_ps)->data[I2dm(j,3,(*buffer_ps)->size)] =
	    ps->data[I2dm(cur_buf_ps_index,3,ps->size)];

	ps_mapping[I1dm(cur_buf_ps_index)] = j;
    }
    for (j = 1; j <= num_buf_tris; j++)
    {
	int cur_buf_tris_index = (*tris_ring_proc)->data[I1dm(j)];
	(*buffer_tris)->data[I2dm(j,1,(*buffer_tris)->size)] =
	    ps_mapping[tris->data[I2dm(cur_buf_tris_index,1,tris->size)]-1];
	(*buffer_tris)->data[I2dm(j,2,(*buffer_tris)->size)] =
	    ps_mapping[tris->data[I2dm(cur_buf_tris_index,2,tris->size)]-1];
	(*buffer_tris)->data[I2dm(j,3,(*buffer_tris)->size)] =
	    ps_mapping[tris->data[I2dm(cur_buf_tris_index,3,tris->size)]-1];
    }

    free(ps_flag);
    free(tris_flag);
    free(ps_mapping);
}

void hpBuildGhostPsTrisForSend(const hiPropMesh *mesh,
			       const int nb_proc_index,
			       const real_T num_ring,
			       emxArray_int32_T *psid_proc,
			       emxArray_int32_T **ps_ring_proc,
			       emxArray_int32_T **tris_ring_proc,
			       emxArray_real_T **buffer_ps,
			       emxArray_int32_T **buffer_tris)
{
    /* Get nring nb between current proc and all nb processors  
     *
     * Point positions stored in k_i*3 double matrices buffer_ps[i] where
     * k = # of points in the n-ring buffer for mesh->nb_proc->data[i].
     * Triangle indices mapped to the index for buffer_ps[i] and stored
     * in buffer_tris[i];
     */
    int *ps_mapping = (int *) calloc(mesh->ps->size[0], sizeof(int));
    int j;

    {
	hpCollectNRingTris(mesh, nb_proc_index, psid_proc, num_ring,
			   ps_ring_proc, tris_ring_proc);

	int num_ps_buffer = (*ps_ring_proc)->size[0];
	int num_tris_buffer = (*tris_ring_proc)->size[0];

	
	(*buffer_ps) = emxCreate_real_T(num_ps_buffer, 3);
	(*buffer_tris) = emxCreate_int32_T(num_tris_buffer, 3);

	for (j = 1; j <= num_ps_buffer; j++)
	{
	    int cur_buf_ps_index = (*ps_ring_proc)->data[I1dm(j)];
	    (*buffer_ps)->data[I2dm(j,1,(*buffer_ps)->size)] =
		mesh->ps->data[I2dm(cur_buf_ps_index,1,mesh->ps->size)];
	    (*buffer_ps)->data[I2dm(j,2,(*buffer_ps)->size)] =
		mesh->ps->data[I2dm(cur_buf_ps_index,2,mesh->ps->size)];
	    (*buffer_ps)->data[I2dm(j,3,(*buffer_ps)->size)] =
		mesh->ps->data[I2dm(cur_buf_ps_index,3,mesh->ps->size)];

	    ps_mapping[I1dm(cur_buf_ps_index)] = j;
	}
	for (j = 1; j <= num_tris_buffer; j++)
	{
	    int cur_buf_tris_index = (*tris_ring_proc)->data[I1dm(j)];
	    (*buffer_tris)->data[I2dm(j,1,(*buffer_tris)->size)] =
		ps_mapping[mesh->tris->data[I2dm(cur_buf_tris_index,1,mesh->tris->size)]-1];
	    (*buffer_tris)->data[I2dm(j,2,(*buffer_tris)->size)] =
		ps_mapping[mesh->tris->data[I2dm(cur_buf_tris_index,2,mesh->tris->size)]-1];
	    (*buffer_tris)->data[I2dm(j,3,(*buffer_tris)->size)] =
		ps_mapping[mesh->tris->data[I2dm(cur_buf_tris_index,3,mesh->tris->size)]-1];
	}
	/************* Debugging output **********************************
	char rank_str[5];
	char nb_rank_str[5];
	numIntoString(cur_proc,4,rank_str);
	numIntoString(mesh->nb_proc->data[I1dm(i)], 4, nb_rank_str);
	char debug_out_name[250];
	sprintf(debug_out_name, "debugout-p%s-to-p%s.vtk", rank_str, nb_rank_str);
	hpDebugOutput(mesh, ps_ring_proc[I1dm(i)], tris_ring_proc[I1dm(i)], debug_out_name);
	******************************************************************/
    }
    free(ps_mapping);

}

void hpAddProcInfoForGhostPsTris(hiPropMesh *mesh,
				 const int nb_proc_index,
				 emxArray_int32_T *ps_ring_proc,
				 emxArray_int32_T *tris_ring_proc)
{

     /* Fill and build the temp pinfo information on each master processor 
     * Step 1, For original pinfo, the new target processor is added as a new
     * node with proc = new_proc_id, lindex = -1 (unknown) to the tail
     */

    hpPInfoList *ps_pinfo = mesh->ps_pinfo;
    hpPInfoList *tris_pinfo = mesh->tris_pinfo;

    int j;

    int num_ps_buffer = ps_ring_proc->size[0];
    int num_tris_buffer = tris_ring_proc->size[0];
    int target_proc_id = mesh->nb_proc->data[I1dm(nb_proc_index)];

    /* Finish Step 1 */
    {
	for (j = 1; j <= num_ps_buffer; j++)
	{
	    unsigned char overlay_flag = 0;
	    int cur_ps_index = ps_ring_proc->data[I1dm(j)];
	    int next_node = ps_pinfo->head[I1dm(cur_ps_index)];
	    while(next_node != -1)
	    {
		if (ps_pinfo->pdata[I1dm(next_node)].proc == target_proc_id)
		    overlay_flag = 1;

		next_node = ps_pinfo->pdata[I1dm(next_node)].next;
	    }
	    if (overlay_flag == 0)
	    {
		int cur_tail = ps_pinfo->tail[I1dm(cur_ps_index)];
		hpEnsurePInfoCapacity(ps_pinfo);
		ps_pinfo->allocated_len++;
		int new_tail = ps_pinfo->allocated_len;
		ps_pinfo->pdata[I1dm(new_tail)].next = -1;
		ps_pinfo->pdata[I1dm(new_tail)].lindex = -1;
		ps_pinfo->pdata[I1dm(new_tail)].proc = target_proc_id;
		ps_pinfo->pdata[I1dm(cur_tail)].next = new_tail;
		ps_pinfo->tail[I1dm(cur_ps_index)] = new_tail;
	    }
	}

	for (j = 1; j <= num_tris_buffer; j++)
	{
	    unsigned char overlay_flag = 0;
	    int cur_tri_index = tris_ring_proc->data[I1dm(j)];
	    int next_node = tris_pinfo->head[I1dm(cur_tri_index)];
	    while(next_node != -1)
	    {
		if (tris_pinfo->pdata[I1dm(next_node)].proc == target_proc_id)
		    overlay_flag = 1;
		
		next_node = tris_pinfo->pdata[I1dm(next_node)].next;
	    }
	    if (overlay_flag == 0)
	    {
		int cur_tail = tris_pinfo->tail[I1dm(cur_tri_index)];
		hpEnsurePInfoCapacity(tris_pinfo);
		tris_pinfo->allocated_len++;
		int new_tail = tris_pinfo->allocated_len;
		tris_pinfo->pdata[I1dm(new_tail)].next = -1;
		tris_pinfo->pdata[I1dm(new_tail)].lindex = -1;
		tris_pinfo->pdata[I1dm(new_tail)].proc = target_proc_id;
		tris_pinfo->pdata[I1dm(cur_tail)].next = new_tail;
		tris_pinfo->tail[I1dm(cur_tri_index)] = new_tail;
	    }
	}
    }
}

void hpBuildGhostPsTrisPInfoForSend(const hiPropMesh *mesh,
				    const int nb_proc_index,
				    emxArray_int32_T *ps_ring_proc,
				    emxArray_int32_T *tris_ring_proc,
				    int **buffer_ps_pinfo_tag,
				    int **buffer_ps_pinfo_lindex,
				    int **buffer_ps_pinfo_proc,
				    int **buffer_tris_pinfo_tag,
				    int **buffer_tris_pinfo_lindex,
				    int **buffer_tris_pinfo_proc)
{
    /* Fill and build the temp pinfo information on each master processor 
     * Step 2, Use the updated original pinfo to build the temp pinfo for MPI
     * send
     */

    /* buffer_ps_pinfo_tag[I1dm(i)][I1dm(j)] to buffer_ps_pinfo_tag[I1dm(i)][I1dm(j+1)]-1
     * are the index of pinfo_lindex & pinfo_proc for buffer_ps[I1dm(j)]
     * buffer_ps_pinfo_lindex[I1md(i)] has all the local index (starts from 0)
     * buffer_ps_pinfo_proc[I1dm(i)] has all the proc information (starts from
     * 0)
     *
     * Same for tris*/

    hpPInfoList *ps_pinfo = mesh->ps_pinfo;
    hpPInfoList *tris_pinfo = mesh->tris_pinfo;

    int buffer_ps_pinfo_length = 0;
    int buffer_tris_pinfo_length = 0;
    
    int j;

    int num_ps_buffer = ps_ring_proc->size[0];
    int num_tris_buffer = tris_ring_proc->size[0];

    /* Fill buffer_ps_pinfo_tag & buffer_tris_pinfo_tag */
    {
	(*buffer_ps_pinfo_tag) = (int *) calloc(num_ps_buffer+1, sizeof(int));
	(*buffer_tris_pinfo_tag) = (int *) calloc(num_tris_buffer+1, sizeof(int));

	(*buffer_ps_pinfo_tag)[0] = 0;
	(*buffer_tris_pinfo_tag)[0] = 0;

	for (j = 1; j <= num_ps_buffer; j++)
	{
	    int num_pinfo_data = 0;
	    int cur_ps_index = ps_ring_proc->data[I1dm(j)];
	    int next_node = ps_pinfo->head[I1dm(cur_ps_index)];
	    while(next_node != -1)
	    {
		num_pinfo_data++;
		next_node = ps_pinfo->pdata[I1dm(next_node)].next;
	    }
	    buffer_ps_pinfo_length += num_pinfo_data;
	    (*buffer_ps_pinfo_tag)[j] = buffer_ps_pinfo_length;

	}

	for (j = 1; j <= num_tris_buffer; j++)
	{
	    int num_pinfo_data = 0;
	    int cur_tri_index = tris_ring_proc->data[I1dm(j)];
	    int next_node = tris_pinfo->head[I1dm(cur_tri_index)];
	    while(next_node != -1)
	    {
		num_pinfo_data++;
		next_node = tris_pinfo->pdata[I1dm(next_node)].next;
	    }
	    buffer_tris_pinfo_length += num_pinfo_data;
	    (*buffer_tris_pinfo_tag)[j] = buffer_tris_pinfo_length;
	}

    }

    /* Fill in the buffer_ps/tris_pinfo_lindex & buffer_ps/tris_pinfo_proc in order */
    {
	(*buffer_ps_pinfo_lindex) = (int *) calloc(buffer_ps_pinfo_length, sizeof(int));
	(*buffer_ps_pinfo_proc) = (int *) calloc(buffer_ps_pinfo_length, sizeof(int));

	(*buffer_tris_pinfo_lindex) = (int *) calloc(buffer_tris_pinfo_length, sizeof(int));
	(*buffer_tris_pinfo_proc) = (int *) calloc(buffer_tris_pinfo_length, sizeof(int));

	int cur_ps_pinfo = 0;
	for (j = 1; j <= num_ps_buffer; j++)
	{
	    int cur_ps_index = ps_ring_proc->data[I1dm(j)];
	    int next_node = ps_pinfo->head[I1dm(cur_ps_index)];
	    while(next_node != -1)
	    {
		(*buffer_ps_pinfo_lindex)[cur_ps_pinfo] = ps_pinfo->pdata[I1dm(next_node)].lindex;
		(*buffer_ps_pinfo_proc)[cur_ps_pinfo] = ps_pinfo->pdata[I1dm(next_node)].proc;
		next_node = ps_pinfo->pdata[I1dm(next_node)].next;
		cur_ps_pinfo++;
	    }
	}

	int cur_tris_pinfo = 0;
	for (j = 1; j <= num_tris_buffer; j++)
	{
	    int cur_tri_index = tris_ring_proc->data[I1dm(j)];
	    int next_node = tris_pinfo->head[I1dm(cur_tri_index)];
	    while(next_node != -1)
	    {
		(*buffer_tris_pinfo_lindex)[cur_tris_pinfo] = tris_pinfo->pdata[I1dm(next_node)].lindex;
		(*buffer_tris_pinfo_proc)[cur_tris_pinfo] = tris_pinfo->pdata[I1dm(next_node)].proc;
		next_node = tris_pinfo->pdata[I1dm(next_node)].next;
		cur_tris_pinfo++;
	    }
	}
    }
}


void hpUpdateNbWithPInfo(hiPropMesh *mesh)
{
    int i;
    int num_proc;
    int cur_proc;
    MPI_Comm_rank(MPI_COMM_WORLD, &cur_proc);
    MPI_Comm_size(MPI_COMM_WORLD, &num_proc);

    int *head = mesh->ps_pinfo->head;
    hpPInfoNode *pdata = mesh->ps_pinfo->pdata;

    unsigned char *nb_flag = (unsigned char *) calloc(num_proc, sizeof(unsigned char));

    for (i = 1; i <= mesh->ps->size[0]; i++)
    {
	int next_node = head[I1dm(i)];
	while(next_node != -1)
	{
	    nb_flag[pdata[next_node].proc] = 1;
	    next_node = pdata[next_node].next;
	}
    }

    int new_num_nbp = 0;
    
    for (i = 0; i < cur_proc; i++)
    {
	if (nb_flag[i] == 1)
	    new_num_nbp++;
    }
    for (i = cur_proc+1; i < num_proc; i++)
    {
	if (nb_flag[i] == 1)
	    new_num_nbp++;
    }

    emxArray_int32_T *new_nb_proc = emxCreateND_int32_T(1, &new_num_nbp);

    int j = 0;
    for (i = 0; i < cur_proc; i++)
    {
	if (nb_flag[i] == 1)
	{
	    new_nb_proc->data[j] = i;
	    j++;
	}
    }
    for (i = cur_proc+1; i < num_proc; i++)
    {
	if (nb_flag[i] == 1)
	{
	    new_nb_proc->data[j] = i;
	    j++;
	}
    }

    emxFree_int32_T(&(mesh->nb_proc));
    mesh->nb_proc = new_nb_proc;

    free(nb_flag);
}


void hpCollectAllGhostPs(hiPropMesh *mesh,
			 const int nbp_index,
			 int *sizep,
			 int **ppinfol)
{
    int *head = mesh->ps_pinfo->head;
    hpPInfoNode *pdata = mesh->ps_pinfo->pdata;

    int i;
    int ip = I1dm(nbp_index);
    int cur_proc;
    MPI_Comm_rank(MPI_COMM_WORLD, &cur_proc);
    int rcv_proc = mesh->nb_proc->data[ip];
    int nump = 0;

    for (i = 1; i <= mesh->ps->size[0]; i++)
    {
	int cur_head = head[I1dm(i)];
	if (pdata[I1dm(cur_head)].proc == rcv_proc)
	    nump++;
    }

    if (nump != 0)
    {
	sizep[ip] = nump;
	ppinfol[ip] = (int *) calloc(2*nump, sizeof(int));
	int j = 0;

	for (i = 1; i <= mesh->ps->size[0]; i++)
	{
	    int cur_head = head[I1dm(i)];
	    if (pdata[I1dm(cur_head)].proc == rcv_proc)
	    {
		ppinfol[ip][2*j] = pdata[I1dm(cur_head)].lindex;
		ppinfol[ip][2*j+1] = i;
		j++;
	    }
	}
    }
    else
    {
	sizep[ip] = 0;
	ppinfol[ip] = (int *) NULL;
    }
}

void hpCollectAllGhostTris(hiPropMesh *mesh,
			   const int nbp_index,
			   int *sizet,
			   int **tpinfol)
{
    int *head = mesh->tris_pinfo->head;
    hpPInfoNode *pdata = mesh->tris_pinfo->pdata;

    int i;
    int ip = I1dm(nbp_index);
    int cur_proc;
    MPI_Comm_rank(MPI_COMM_WORLD, &cur_proc);
    int rcv_proc = mesh->nb_proc->data[ip];
    int numt = 0;

    for (i = 1; i <= mesh->tris->size[0]; i++)
    {
	int cur_head = head[I1dm(i)];
	if (pdata[I1dm(cur_head)].proc == rcv_proc)
	    numt++;
    }

    if (numt != 0)
    {
	sizet[ip] = numt;
	tpinfol[ip] = (int *) calloc(2*numt, sizeof(int));


	int j = 0;

	for (i = 1; i <= mesh->tris->size[0]; i++)
	{
	    int cur_head = head[I1dm(i)];
	    if (pdata[I1dm(cur_head)].proc == rcv_proc)
	    {
		tpinfol[ip][2*j] = pdata[I1dm(cur_head)].lindex;
		tpinfol[ip][2*j+1] = i;
		j++;
	    }
	}
    }
    else
    {
	sizet[ip] = 0;
	tpinfol[ip] = (int *) NULL;
    }
}

void hpMergeOverlayPsPInfo(hiPropMesh *mesh,
			   const int rcv_id,
			   int nump,
			   int *ppinfol)
{
    hpPInfoList *ps_pinfo = mesh->ps_pinfo;
    int *head = ps_pinfo->head;
    int *tail = ps_pinfo->tail;
    hpPInfoNode *pdata = ps_pinfo->pdata;

    int i;
    for (i = 1; i <= nump; i++)
    {
	int ps_index = ppinfol[2*(i-1)];
	int next_node = head[I1dm(ps_index)];
	while(next_node != -1)
	{
	    if (pdata[I1dm(next_node)].proc == rcv_id)
	    {
		pdata[I1dm(next_node)].lindex = ppinfol[2*i-1];
		break;
	    }
	    else
		next_node = pdata[I1dm(next_node)].next;
	}
	if (next_node == -1)
	{
	    hpEnsurePInfoCapacity(ps_pinfo);
	    ps_pinfo->allocated_len++;
	    int cur_tail = tail[I1dm(ps_index)];
	    int new_tail = ps_pinfo->allocated_len;

	    pdata[I1dm(new_tail)].lindex = ppinfol[2*i-1];
	    pdata[I1dm(new_tail)].proc = rcv_id;
	    pdata[I1dm(new_tail)].next = -1;

	    pdata[I1dm(cur_tail)].next = new_tail;
	    tail[I1dm(ps_index)] = new_tail;
	}
    }
}

void hpMergeOverlayTrisPInfo(hiPropMesh *mesh,
			     const int rcv_id,
			     int numt,
			     int *tpinfol)
{
    hpPInfoList *tris_pinfo = mesh->tris_pinfo;
    int *head = tris_pinfo->head;
    int *tail = tris_pinfo->tail;
    hpPInfoNode *pdata = tris_pinfo->pdata;

    int i;
    for (i = 1; i <= numt; i++)
    {
	int tris_index = tpinfol[2*(i-1)];
	int next_node = head[I1dm(tris_index)];
	while(next_node != -1)
	{
	    if (pdata[I1dm(next_node)].proc == rcv_id)
	    {
		pdata[I1dm(next_node)].lindex = tpinfol[2*i-1];
		break;
	    }
	    else
		next_node = pdata[I1dm(next_node)].next;
	}
	if (next_node == -1)
	{
	    hpEnsurePInfoCapacity(tris_pinfo);
	    tris_pinfo->allocated_len++;
	    int cur_tail = tail[I1dm(tris_index)];
	    int new_tail = tris_pinfo->allocated_len;

	    pdata[I1dm(new_tail)].lindex = tpinfol[2*i-1];
	    pdata[I1dm(new_tail)].proc = rcv_id;
	    pdata[I1dm(new_tail)].next = -1;

	    pdata[I1dm(cur_tail)].next = new_tail;
	    tail[I1dm(tris_index)] = new_tail;
	}
    }
}

void hpUpdateMasterPInfo(hiPropMesh *mesh)
{
    int cur_proc;
    MPI_Comm_rank(MPI_COMM_WORLD, &cur_proc);
    int i;

    int num_nb_proc = mesh->nb_proc->size[0];

    int *send_size_ps = (int *) calloc(num_nb_proc, sizeof(int));
    int *send_size_tris = (int *) calloc(num_nb_proc, sizeof(int));

    int **buffer_ps_pinfo_lindex = (int **) calloc(num_nb_proc, sizeof(int *));
    int **buffer_tris_pinfo_lindex = (int **) calloc(num_nb_proc, sizeof(int *));

    int num_all_send_rqst = 4*num_nb_proc;

    MPI_Request* send_rqst_list = (MPI_Request *) calloc(num_all_send_rqst, sizeof(MPI_Request));

    for (i = 0; i < num_all_send_rqst; i++)
	send_rqst_list[i] = MPI_REQUEST_NULL;

    MPI_Status* send_status_list = (MPI_Status *) calloc(num_all_send_rqst, sizeof(MPI_Status));

    MPI_Request* recv_req_list_ps = (MPI_Request *) calloc(num_nb_proc, sizeof(MPI_Request));
    MPI_Request* recv_req_list_tris = (MPI_Request *) calloc(num_nb_proc, sizeof(MPI_Request));

    int *recv_size_ps = (int *) calloc (num_nb_proc, sizeof(int));
    int *recv_size_tris = (int *) calloc (num_nb_proc, sizeof(int));

    int cur_rqst = 0;

    int tag_ps_size = 0;
    int tag_tris_size = 100;

    int tag_ps_pinfo2 = 51;
    int tag_tris_pinfo2 = 61;

    for (i = 1; i <= num_nb_proc; i++)
    {
	hpCollectAllGhostPs(mesh, i, send_size_ps,  buffer_ps_pinfo_lindex);
	MPI_Isend(&(send_size_ps[I1dm(i)]), 1, MPI_INT,
		mesh->nb_proc->data[I1dm(i)], tag_ps_size, MPI_COMM_WORLD, &(send_rqst_list[cur_rqst++]));
	if (send_size_ps[I1dm(i)] != 0)
	    MPI_Isend(buffer_ps_pinfo_lindex[I1dm(i)], 2*send_size_ps[I1dm(i)], MPI_INT,
		    mesh->nb_proc->data[I1dm(i)], tag_ps_pinfo2, MPI_COMM_WORLD, &(send_rqst_list[cur_rqst++]));

    }

    for (i = 1; i <= num_nb_proc; i++)
    {
	hpCollectAllGhostTris(mesh, i, send_size_tris, buffer_tris_pinfo_lindex);

	MPI_Isend(&(send_size_tris[I1dm(i)]), 1, MPI_INT,
		mesh->nb_proc->data[I1dm(i)], tag_tris_size, MPI_COMM_WORLD, &(send_rqst_list[cur_rqst++]));
	if (send_size_tris[I1dm(i)] != 0)
	    MPI_Isend(buffer_tris_pinfo_lindex[I1dm(i)], 2*send_size_tris[I1dm(i)], MPI_INT,
		    mesh->nb_proc->data[I1dm(i)], tag_tris_pinfo2, MPI_COMM_WORLD, &(send_rqst_list[cur_rqst++]));
    }


	/* Recv ps pinfo */
    for (i = 1; i <= num_nb_proc; i++)
	MPI_Irecv(&(recv_size_ps[I1dm(i)]), 1, MPI_INT, mesh->nb_proc->data[I1dm(i)], tag_ps_size, MPI_COMM_WORLD, &(recv_req_list_ps[I1dm(i)]));

    for (i = 1; i <= num_nb_proc; i++)
    {
	int *buf_ppinfo_lindex_recv;

	int num_buf_ps_recv;

	MPI_Status tmp_status;
	MPI_Status recv_status;

	int recv_index;
	int proc_recv;

	MPI_Waitany(num_nb_proc, recv_req_list_ps, &recv_index, &recv_status);
	proc_recv = recv_status.MPI_SOURCE;

	num_buf_ps_recv = recv_size_ps[recv_index];

	if (num_buf_ps_recv != 0)
	{
	    buf_ppinfo_lindex_recv = (int *) calloc(2*num_buf_ps_recv, sizeof(int));

	    MPI_Recv(buf_ppinfo_lindex_recv, 2*num_buf_ps_recv, MPI_INT, proc_recv,
		    tag_ps_pinfo2, MPI_COMM_WORLD, &tmp_status);

	    hpMergeOverlayPsPInfo(mesh, proc_recv, num_buf_ps_recv, buf_ppinfo_lindex_recv);

	free(buf_ppinfo_lindex_recv);
	}

    }

	/* Recv tris pinfo */

    for (i = 1; i <= num_nb_proc; i++)
	MPI_Irecv(&(recv_size_tris[I1dm(i)]), 1, MPI_INT, mesh->nb_proc->data[I1dm(i)], tag_tris_size, MPI_COMM_WORLD, &(recv_req_list_tris[I1dm(i)]));

    for (i = 1; i <= num_nb_proc; i++)
    {
	int *buf_tpinfo_lindex_recv;

	int num_buf_tris_recv;

	MPI_Status tmp_status;
	MPI_Status recv_status;

	int recv_index;
	int proc_recv;

	MPI_Waitany(num_nb_proc, recv_req_list_tris, &recv_index, &recv_status);
	proc_recv = recv_status.MPI_SOURCE;

	num_buf_tris_recv = recv_size_tris[recv_index];

	if (num_buf_tris_recv != 0)
	{

	    buf_tpinfo_lindex_recv = (int *) calloc(2*num_buf_tris_recv, sizeof(int));

	    MPI_Recv(buf_tpinfo_lindex_recv, 2*num_buf_tris_recv, MPI_INT, proc_recv,
		    tag_tris_pinfo2, MPI_COMM_WORLD, &tmp_status);

	    hpMergeOverlayTrisPInfo(mesh, proc_recv, num_buf_tris_recv, buf_tpinfo_lindex_recv);

	    free(buf_tpinfo_lindex_recv);
	}
    }

    free(recv_req_list_ps);
    free(recv_req_list_tris);
    free(recv_size_ps);
    free(recv_size_tris);

    /* Wait until all the array are sent */

    MPI_Waitall(num_all_send_rqst, send_rqst_list, send_status_list);

    /* Free the array for send */

    free(send_rqst_list);
    free(send_status_list);

    for (i = 1; i <= num_nb_proc; i++)
    {
	free(buffer_ps_pinfo_lindex[I1dm(i)]);
	free(buffer_tris_pinfo_lindex[I1dm(i)]);
    }

    free(send_size_ps);
    free(send_size_tris);

    free(buffer_ps_pinfo_lindex);
    free(buffer_tris_pinfo_lindex);
}

void hpCollectAllOverlayPs(hiPropMesh *mesh,
			   const int nbp_index,
			   int *sizep,
			   int **ppinfot,
			   int **ppinfol,
			   int **ppinfop)
{
    int *head = mesh->ps_pinfo->head;
    hpPInfoNode *pdata = mesh->ps_pinfo->pdata;

    int i;
    int ip = I1dm(nbp_index);
    int cur_proc;
    MPI_Comm_rank(MPI_COMM_WORLD, &cur_proc);
    int rcv_proc = mesh->nb_proc->data[ip];
    int nump = 0;

    unsigned char *overlay = (unsigned char *) calloc(mesh->ps->size[0], sizeof(unsigned char));

    /* first round, fill the nump and overlay */
    for (i = 1; i <= mesh->ps->size[0]; i++)
    {
	int cur_head = head[I1dm(i)];
	if (pdata[I1dm(cur_head)].proc == cur_proc)
	{
	    int next_node = pdata[I1dm(cur_head)].next;
	    while(next_node != -1)
	    {
		if (pdata[I1dm(next_node)].proc == rcv_proc)
		{
		    overlay[I1dm(i)] = 1;
		    nump++;
		    break;
		}
		next_node = pdata[I1dm(next_node)].next;
	    }
	}
    }


    if (nump != 0)
    {
	/* second round, fill the ppinfot[ip] */
	sizep[ip] = nump;
	ppinfot[ip] = (int *) calloc(nump+1, sizeof(int));
	ppinfot[ip][0] = 0;

	int j = 1;
	int num_pinfo_all = 0;
	for (i = 1; i <= mesh->ps->size[0]; i++)
	{
	    if (overlay[I1dm(i)] == 1)
	    {
		int num_pinfo_cur = 0;
		int next_node = head[I1dm(i)];

		while(next_node != -1)
		{
		    num_pinfo_cur++;
		    next_node = pdata[I1dm(next_node)].next;
		}
		num_pinfo_all += num_pinfo_cur;
		ppinfot[ip][j++] = num_pinfo_all;
	    }
	}

	/* third rould, fill the ppinfol[ip] and ppinfop[ip] */
	ppinfol[ip] = (int *) calloc(num_pinfo_all, sizeof(int));
	ppinfop[ip] = (int *) calloc(num_pinfo_all, sizeof(int));

	j = 0;
	for (i = 1; i <= mesh->ps->size[0]; i++)
	{
	    if (overlay[I1dm(i)] == 1)
	    {
		int next_node = head[I1dm(i)];

		while(next_node != -1)
		{
		    ppinfol[ip][j] = pdata[I1dm(next_node)].lindex;
		    ppinfop[ip][j] = pdata[I1dm(next_node)].proc;
		    j++;
		    next_node = pdata[I1dm(next_node)].next;
		}
	    }
	}
    }
    else
    {
	sizep[ip] = 0;
	ppinfot[ip] = (int *) NULL;
	ppinfol[ip] = (int *) NULL;
	ppinfop[ip] = (int *) NULL;
    }

    free(overlay);
}

void hpCollectAllOverlayTris(hiPropMesh *mesh,
			     const int nbp_index,
			     int *sizet,
			     int **tpinfot,
			     int **tpinfol,
			     int **tpinfop)
{
    int *head = mesh->tris_pinfo->head;
    hpPInfoNode *pdata = mesh->tris_pinfo->pdata;

    int i;
    int ip = I1dm(nbp_index);
    int cur_proc;
    MPI_Comm_rank(MPI_COMM_WORLD, &cur_proc);
    int rcv_proc = mesh->nb_proc->data[ip];
    int numt = 0;

    unsigned char *overlay = (unsigned char *) calloc(mesh->tris->size[0], sizeof(unsigned char));

    /* first round, fill the nump and overlay */
    for (i = 1; i <= mesh->tris->size[0]; i++)
    {
	int cur_head = head[I1dm(i)];
	if (pdata[I1dm(cur_head)].proc == cur_proc)
	{
	    int next_node = pdata[I1dm(cur_head)].next;
	    while(next_node != -1)
	    {
		if (pdata[I1dm(next_node)].proc == rcv_proc)
		{
		    overlay[I1dm(i)] = 1;
		    numt++;
		    break;
		}
		next_node = pdata[I1dm(next_node)].next;
	    }
	}
    }


    if (numt != 0)
    {
	/* second round, fill the ppinfot[ip] */
	sizet[ip] = numt;
	tpinfot[ip] = (int *) calloc(numt+1, sizeof(int));
	tpinfot[ip][0] = 0;

	int j = 1;
	int num_pinfo_all = 0;
	for (i = 1; i <= mesh->tris->size[0]; i++)
	{
	    if (overlay[I1dm(i)] == 1)
	    {
		int num_pinfo_cur = 0;
		int next_node = head[I1dm(i)];

		while(next_node != -1)
		{
		    num_pinfo_cur++;
		    next_node = pdata[I1dm(next_node)].next;
		}
		num_pinfo_all += num_pinfo_cur;
		tpinfot[ip][j++] = num_pinfo_all;
	    }
	}

	/* third rould, fill the ppinfol[ip] and ppinfop[ip] */
	tpinfol[ip] = (int *) calloc(num_pinfo_all, sizeof(int));
	tpinfop[ip] = (int *) calloc(num_pinfo_all, sizeof(int));

	j = 0;
	for (i = 1; i <= mesh->tris->size[0]; i++)
	{
	    if (overlay[I1dm(i)] == 1)
	    {
		int next_node = head[I1dm(i)];

		while(next_node != -1)
		{
		    tpinfol[ip][j] = pdata[I1dm(next_node)].lindex;
		    tpinfop[ip][j] = pdata[I1dm(next_node)].proc;
		    j++;
		    next_node = pdata[I1dm(next_node)].next;
		}
	    }
	}
    }
    else
    {
	sizet[ip] = 0;
	tpinfot[ip] = (int *) NULL;
	tpinfol[ip] = (int *) NULL;
	tpinfop[ip] = (int *) NULL;
    }
    free(overlay);
}

void hpMergeGhostPsPInfo(hiPropMesh *mesh,
			 const int rcv_id,
			 int nump,
			 int *ppinfot,
			 int *ppinfol,
			 int *ppinfop)
{
    hpPInfoList *ps_pinfo = mesh->ps_pinfo;
    int *head = ps_pinfo->head;
    int *tail = ps_pinfo->tail;
    hpPInfoNode *pdata = ps_pinfo->pdata;

    int cur_proc;
    MPI_Comm_rank(MPI_COMM_WORLD, &cur_proc);


    int i,j;

    for (i = 1; i <= nump; i++)
    {
	/* first locate the point */
	int ps_index;
	for (j = ppinfot[i-1]; j <= ppinfot[i]-1; j++)
	{
	    if (ppinfop[j] == cur_proc)
	    {
		ps_index = ppinfol[j];
		break;
	    }
	}

	/* then merge the pinfo list */
	for (j = ppinfot[i-1]; j <= ppinfot[i]-1; j++)
	{
	    int cur_node = head[I1dm(ps_index)];
	    while(cur_node != -1)
	    {
		if (ppinfop[j] == pdata[I1dm(cur_node)].proc)
		{
		    pdata[I1dm(cur_node)].lindex = ppinfol[j];
		    break;
		}
		cur_node = pdata[I1dm(cur_node)].next;
	    }
	    if (cur_node == -1)
	    {
		hpEnsurePInfoCapacity(ps_pinfo);
		ps_pinfo->allocated_len++;
		int cur_tail = tail[I1dm(ps_index)];
		int new_tail = ps_pinfo->allocated_len;
		pdata[I1dm(new_tail)].lindex = ppinfol[j];
		pdata[I1dm(new_tail)].proc = ppinfop[j];
		pdata[I1dm(new_tail)].next = -1;
		pdata[I1dm(cur_tail)].next = new_tail;
		tail[I1dm(ps_index)] = new_tail;
	    }
	}
    }

}

void hpMergeGhostTrisPInfo(hiPropMesh *mesh,
			   const int rcv_id,
			   int numt,
			   int *tpinfot,
			   int *tpinfol,
			   int *tpinfop)
{
    hpPInfoList *tris_pinfo = mesh->tris_pinfo;
    int *head = tris_pinfo->head;
    int *tail = tris_pinfo->tail;
    hpPInfoNode *pdata = tris_pinfo->pdata;

    int cur_proc;
    MPI_Comm_rank(MPI_COMM_WORLD, &cur_proc);


    int i,j;

    for (i = 1; i <= numt; i++)
    {
	/* first locate the point */
	int tris_index;
	for (j = tpinfot[i-1]; j <= tpinfot[i]-1; j++)
	{
	    if (tpinfop[j] == cur_proc)
	    {
		tris_index = tpinfol[j];
		break;
	    }
	}

	/* then merge the pinfo list */
	for (j = tpinfot[i-1]; j <= tpinfot[i]-1; j++)
	{
	    int cur_node = head[I1dm(tris_index)];
	    while(cur_node != -1)
	    {
		if (tpinfop[j] == pdata[I1dm(cur_node)].proc)
		{
		    pdata[I1dm(cur_node)].lindex = tpinfol[j];
		    break;
		}
		cur_node = pdata[I1dm(cur_node)].next;
		
	    }
	    /* if a new proc info */
	    if (cur_node == -1)
	    {
		hpEnsurePInfoCapacity(tris_pinfo);
		tris_pinfo->allocated_len++;
		int cur_tail = tail[I1dm(tris_index)];
		int new_tail = tris_pinfo->allocated_len;
		pdata[I1dm(new_tail)].lindex = tpinfol[j];
		pdata[I1dm(new_tail)].proc = tpinfop[j];
		pdata[I1dm(new_tail)].next = -1;
		pdata[I1dm(cur_tail)].next = new_tail;
		tail[I1dm(tris_index)] = new_tail;
	    }
	}
    }
}


void hpUpdateAllPInfoFromMaster(hiPropMesh *mesh)
{
    int cur_proc;
    MPI_Comm_rank(MPI_COMM_WORLD, &cur_proc);
    int i;

    int num_nb_proc = mesh->nb_proc->size[0];

    int *send_size_ps = (int *) calloc(num_nb_proc, sizeof(int));
    int *send_size_tris = (int *) calloc(num_nb_proc, sizeof(int));

    int **buffer_ps_pinfo_tag = (int **) calloc(num_nb_proc, sizeof(int *));
    int **buffer_ps_pinfo_lindex = (int **) calloc(num_nb_proc, sizeof(int *));
    int **buffer_ps_pinfo_proc = (int **) calloc(num_nb_proc, sizeof(int *));

    int **buffer_tris_pinfo_tag = (int **) calloc(num_nb_proc, sizeof(int *));
    int **buffer_tris_pinfo_lindex = (int **) calloc(num_nb_proc, sizeof(int *));
    int **buffer_tris_pinfo_proc = (int **) calloc(num_nb_proc, sizeof(int *));

    int num_all_send_rqst = 8*num_nb_proc;

    MPI_Request* send_rqst_list = (MPI_Request *) calloc(num_all_send_rqst, sizeof(MPI_Request));

    for (i = 0; i < num_all_send_rqst; i++)
	send_rqst_list[i] = MPI_REQUEST_NULL;

    MPI_Status* send_status_list = (MPI_Status *) calloc(num_all_send_rqst, sizeof(MPI_Status));

    MPI_Request* recv_req_list_ps = (MPI_Request *) calloc(num_nb_proc, sizeof(MPI_Request));
    MPI_Request* recv_req_list_tris = (MPI_Request *) calloc(num_nb_proc, sizeof(MPI_Request));

    int *recv_size_ps = (int *) calloc (num_nb_proc, sizeof(int));
    int *recv_size_tris = (int *) calloc (num_nb_proc, sizeof(int));

    int cur_rqst = 0;

    int tag_ps_size = 0;
    int tag_tris_size = 10;

    int tag_ps_pinfo1 = 50;
    int tag_ps_pinfo2 = 51;
    int tag_ps_pinfo3 = 52;

    int tag_tris_pinfo1 = 60;
    int tag_tris_pinfo2 = 61;
    int tag_tris_pinfo3 = 62;


    for (i = 1; i <= num_nb_proc; i++)
    {
	hpCollectAllOverlayPs(mesh, i, send_size_ps, buffer_ps_pinfo_tag, buffer_ps_pinfo_lindex, buffer_ps_pinfo_proc);
	MPI_Isend(&(send_size_ps[I1dm(i)]), 1, MPI_INT,
		mesh->nb_proc->data[I1dm(i)], tag_ps_size, MPI_COMM_WORLD, &(send_rqst_list[cur_rqst++]));
	if (send_size_ps[I1dm(i)] != 0)
	{
	    MPI_Isend(buffer_ps_pinfo_tag[I1dm(i)], send_size_ps[I1dm(i)]+1, MPI_INT,
		    mesh->nb_proc->data[I1dm(i)], tag_ps_pinfo1, MPI_COMM_WORLD, &(send_rqst_list[cur_rqst++]));
	    MPI_Isend(buffer_ps_pinfo_lindex[I1dm(i)], buffer_ps_pinfo_tag[I1dm(i)][send_size_ps[I1dm(i)]], MPI_INT,
		    mesh->nb_proc->data[I1dm(i)], tag_ps_pinfo2, MPI_COMM_WORLD, &(send_rqst_list[cur_rqst++]));
	    MPI_Isend(buffer_ps_pinfo_proc[I1dm(i)], buffer_ps_pinfo_tag[I1dm(i)][send_size_ps[I1dm(i)]], MPI_INT,
		    mesh->nb_proc->data[I1dm(i)], tag_ps_pinfo3, MPI_COMM_WORLD, &(send_rqst_list[cur_rqst++]));
	}

    }

    for (i = 1; i <= num_nb_proc; i++)
    {
	hpCollectAllOverlayTris(mesh, i, send_size_tris, buffer_tris_pinfo_tag, buffer_tris_pinfo_lindex, buffer_tris_pinfo_proc);
	MPI_Isend(&(send_size_tris[I1dm(i)]), 1, MPI_INT,
		mesh->nb_proc->data[I1dm(i)], tag_tris_size, MPI_COMM_WORLD, &(send_rqst_list[cur_rqst++]));
	if (send_size_tris[I1dm(i)] != 0)
	{
	    MPI_Isend(buffer_tris_pinfo_tag[I1dm(i)], send_size_tris[I1dm(i)]+1, MPI_INT,
		    mesh->nb_proc->data[I1dm(i)], tag_tris_pinfo1, MPI_COMM_WORLD, &(send_rqst_list[cur_rqst++]));
	    MPI_Isend(buffer_tris_pinfo_lindex[I1dm(i)], buffer_tris_pinfo_tag[I1dm(i)][send_size_tris[I1dm(i)]], MPI_INT,
		    mesh->nb_proc->data[I1dm(i)], tag_tris_pinfo2, MPI_COMM_WORLD, &(send_rqst_list[cur_rqst++]));
	    MPI_Isend(buffer_tris_pinfo_proc[I1dm(i)], buffer_tris_pinfo_tag[I1dm(i)][send_size_tris[I1dm(i)]], MPI_INT,
		    mesh->nb_proc->data[I1dm(i)], tag_tris_pinfo3, MPI_COMM_WORLD, &(send_rqst_list[cur_rqst++]));
	}

    }


	/* Recv ps pinfo */
    for (i = 1; i <= num_nb_proc; i++)
	MPI_Irecv(&(recv_size_ps[I1dm(i)]), 1, MPI_INT, mesh->nb_proc->data[I1dm(i)], tag_ps_size, MPI_COMM_WORLD, &(recv_req_list_ps[I1dm(i)]));

    for (i = 1; i <= num_nb_proc; i++)
    {
	int *buf_ppinfo_tag_recv;
	int *buf_ppinfo_lindex_recv;
	int *buf_ppinfo_proc_recv;

	int num_buf_ps_recv;
	int num_buf_ps_pinfo_recv;

	MPI_Status tmp_status;
	MPI_Status recv_status;

	int recv_index;
	int proc_recv;

	MPI_Waitany(num_nb_proc, recv_req_list_ps, &recv_index, &recv_status);
	proc_recv = recv_status.MPI_SOURCE;

	num_buf_ps_recv = recv_size_ps[recv_index];

	if (num_buf_ps_recv != 0)
	{
	    buf_ppinfo_tag_recv = (int *) calloc(num_buf_ps_recv+1, sizeof(int));

	    MPI_Recv(buf_ppinfo_tag_recv, num_buf_ps_recv+1, MPI_INT, proc_recv,
		    tag_ps_pinfo1, MPI_COMM_WORLD, &tmp_status);

	    num_buf_ps_pinfo_recv = buf_ppinfo_tag_recv[num_buf_ps_recv];

	    buf_ppinfo_lindex_recv = (int *) calloc(num_buf_ps_pinfo_recv, sizeof(int));
	    buf_ppinfo_proc_recv = (int *) calloc(num_buf_ps_pinfo_recv, sizeof(int));

	    MPI_Recv(buf_ppinfo_lindex_recv, num_buf_ps_pinfo_recv, MPI_INT, proc_recv,
		    tag_ps_pinfo2, MPI_COMM_WORLD, &tmp_status);
	    MPI_Recv(buf_ppinfo_proc_recv, num_buf_ps_pinfo_recv, MPI_INT, proc_recv,
		    tag_ps_pinfo3, MPI_COMM_WORLD, &tmp_status);

	    hpMergeGhostPsPInfo(mesh, proc_recv, num_buf_ps_recv,
		    buf_ppinfo_tag_recv, buf_ppinfo_lindex_recv, buf_ppinfo_proc_recv);
	free(buf_ppinfo_tag_recv);
	free(buf_ppinfo_lindex_recv);
	free(buf_ppinfo_proc_recv);
	}

    }

	/* Recv tris pinfo */

    for (i = 1; i <= num_nb_proc; i++)
	MPI_Irecv(&(recv_size_tris[I1dm(i)]), 1, MPI_INT, mesh->nb_proc->data[I1dm(i)], tag_tris_size, MPI_COMM_WORLD, &(recv_req_list_tris[I1dm(i)]));

    for (i = 1; i <= num_nb_proc; i++)
    {
	int *buf_tpinfo_tag_recv;
	int *buf_tpinfo_lindex_recv;
	int *buf_tpinfo_proc_recv;

	int num_buf_tris_recv;
	int num_buf_tris_pinfo_recv;

	MPI_Status tmp_status;
	MPI_Status recv_status;

	int recv_index;
	int proc_recv;

	MPI_Waitany(num_nb_proc, recv_req_list_tris, &recv_index, &recv_status);
	proc_recv = recv_status.MPI_SOURCE;

	num_buf_tris_recv = recv_size_tris[recv_index];

	if (num_buf_tris_recv != 0)
	{
	    buf_tpinfo_tag_recv = (int *) calloc(num_buf_tris_recv+1, sizeof(int));

	    MPI_Recv(buf_tpinfo_tag_recv, num_buf_tris_recv+1, MPI_INT, proc_recv,
		    tag_tris_pinfo1, MPI_COMM_WORLD, &tmp_status);

	    num_buf_tris_pinfo_recv = buf_tpinfo_tag_recv[num_buf_tris_recv];

	    buf_tpinfo_lindex_recv = (int *) calloc(num_buf_tris_pinfo_recv, sizeof(int));
	    buf_tpinfo_proc_recv = (int *) calloc(num_buf_tris_pinfo_recv, sizeof(int));

	    MPI_Recv(buf_tpinfo_lindex_recv, num_buf_tris_pinfo_recv, MPI_INT, proc_recv,
		    tag_tris_pinfo2, MPI_COMM_WORLD, &tmp_status);
	    MPI_Recv(buf_tpinfo_proc_recv, num_buf_tris_pinfo_recv, MPI_INT, proc_recv,
		    tag_tris_pinfo3, MPI_COMM_WORLD, &tmp_status);

	    hpMergeGhostTrisPInfo(mesh, proc_recv, num_buf_tris_recv,
		    buf_tpinfo_tag_recv, buf_tpinfo_lindex_recv, buf_tpinfo_proc_recv);
	    free(buf_tpinfo_tag_recv);
	    free(buf_tpinfo_lindex_recv);
	    free(buf_tpinfo_proc_recv);
	}
    }

    free(recv_req_list_ps);
    free(recv_req_list_tris);
    free(recv_size_ps);
    free(recv_size_tris);

    /* Wait until all the array are sent */

    MPI_Waitall(num_all_send_rqst, send_rqst_list, send_status_list);

    /* Free the array for send */

    free(send_rqst_list);
    free(send_status_list);

    for (i = 1; i <= num_nb_proc; i++)
    {
	free(buffer_ps_pinfo_tag[I1dm(i)]);
	free(buffer_ps_pinfo_lindex[I1dm(i)]);
	free(buffer_ps_pinfo_proc[I1dm(i)]);

	free(buffer_tris_pinfo_tag[I1dm(i)]);
	free(buffer_tris_pinfo_lindex[I1dm(i)]);
	free(buffer_tris_pinfo_proc[I1dm(i)]);
    }

    free(send_size_ps);
    free(send_size_tris);

    free(buffer_ps_pinfo_tag);
    free(buffer_ps_pinfo_lindex);
    free(buffer_ps_pinfo_proc);

    free(buffer_tris_pinfo_tag);
    free(buffer_tris_pinfo_lindex);
    free(buffer_tris_pinfo_proc);
}

void hpUpdatePInfo(hiPropMesh *mesh)
{
    hpUpdateMasterPInfo(mesh);
    hpUpdateAllPInfoFromMaster(mesh);
}



void hpBuildBoundingBoxGhost(hiPropMesh *mesh, const double *bd_box)
{
    int cur_proc;
    int num_proc;
    MPI_Comm_rank(MPI_COMM_WORLD, &cur_proc);
    MPI_Comm_size(MPI_COMM_WORLD, &num_proc);
    int i;

    double *in_all_bd_box = (double *)calloc(6*num_proc, sizeof(double));
    double *all_bd_box = (double *)calloc(6*num_proc, sizeof(double));

    for (i = 0; i < 6; i++)
	in_all_bd_box[cur_proc*6+i] = bd_box[i];

    MPI_Allreduce(in_all_bd_box, all_bd_box, 6*num_proc, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    free(in_all_bd_box);

    int num_nb_proc = num_proc - 1;

    /* temp change of nb_proc to all other processors */

    emxArray_int32_T *new_nb_proc = emxCreateND_int32_T(1, &num_nb_proc);

    int j = 0;
    for (i = 0; i < cur_proc; i++)
    {
    	new_nb_proc->data[j] = i;
	j++;
    }
    for (i = cur_proc+1; i < num_proc; i++)
    {
	new_nb_proc->data[j] = i;
	j++;
    }

    emxFree_int32_T(&(mesh->nb_proc));
    mesh->nb_proc = new_nb_proc;


    emxArray_int32_T **ps_ring_proc = (emxArray_int32_T **) calloc(num_nb_proc, sizeof(emxArray_int32_T *));
    emxArray_int32_T **tris_ring_proc = (emxArray_int32_T **) calloc(num_nb_proc, sizeof(emxArray_int32_T *));
    emxArray_real_T **buffer_ps = (emxArray_real_T **) calloc(num_nb_proc, sizeof(emxArray_real_T *));
    emxArray_int32_T **buffer_tris = (emxArray_int32_T **) calloc(num_nb_proc, sizeof(emxArray_int32_T *));


    for (i = 1; i <= num_nb_proc; i++)
    {
	hpBuildBdboxGhostPsTrisForSend(mesh, i, all_bd_box,
				  &(ps_ring_proc[I1dm(i)]),
				  &(tris_ring_proc[I1dm(i)]), 
				  &(buffer_ps[I1dm(i)]), &(buffer_tris[I1dm(i)]));
    }

    free(all_bd_box);

    hpCommPsTrisWithPInfo(mesh, ps_ring_proc, tris_ring_proc, buffer_ps, buffer_tris);
}


void hpCommPsTrisWithPInfo(hiPropMesh *mesh, emxArray_int32_T **ps_ring_proc, emxArray_int32_T **tris_ring_proc,
			   emxArray_real_T **buffer_ps, emxArray_int32_T **buffer_tris)
{
    int i;
    int num_nb_proc = mesh->nb_proc->size[0];

    /* Set up the MPI_Request list */
    int num_all_send_rqst = 10*num_nb_proc;

    MPI_Request* send_rqst_list = (MPI_Request *) calloc(num_all_send_rqst, sizeof(MPI_Request));
    MPI_Status* send_status_list = (MPI_Status *) calloc(num_all_send_rqst, sizeof(MPI_Status));

    for (i = 0; i < num_all_send_rqst; i++)
	send_rqst_list[i] = MPI_REQUEST_NULL;

    MPI_Request* recv_req_list = (MPI_Request *) calloc(num_nb_proc, sizeof(MPI_Request));

    int *recv_size = (int *) calloc (2*num_nb_proc, sizeof(int));

    int cur_rqst = 0;

    /* Build n-ring neighborhood and send*/
    int **buffer_ps_pinfo_tag = (int **) calloc(num_nb_proc, sizeof(int *));
    int **buffer_ps_pinfo_lindex = (int **) calloc(num_nb_proc, sizeof(int *));
    int **buffer_ps_pinfo_proc = (int **) calloc(num_nb_proc, sizeof(int *));

    int **buffer_tris_pinfo_tag = (int **) calloc(num_nb_proc, sizeof(int *));
    int **buffer_tris_pinfo_lindex = (int **) calloc(num_nb_proc, sizeof(int *));
    int **buffer_tris_pinfo_proc = (int **) calloc(num_nb_proc, sizeof(int *));

    int tag_ps = 0;
    int tag_tris = 10;

    int tag_ps_pinfo1 = 50;
    int tag_ps_pinfo2 = 51;
    int tag_ps_pinfo3 = 52;

    int tag_tris_pinfo1 = 60;
    int tag_tris_pinfo2 = 61;
    int tag_tris_pinfo3 = 62;


    for (i = 1; i <= num_nb_proc; i++)
    {
	hpAddProcInfoForGhostPsTris(mesh, i, ps_ring_proc[I1dm(i)], tris_ring_proc[I1dm(i)]);
    }

    for (i = 1; i <= num_nb_proc; i++)
    {
	hpBuildGhostPsTrisPInfoForSend(mesh, i, ps_ring_proc[I1dm(i)], tris_ring_proc[I1dm(i)],
		&(buffer_ps_pinfo_tag[I1dm(i)]),
		&(buffer_ps_pinfo_lindex[I1dm(i)]),
		&(buffer_ps_pinfo_proc[I1dm(i)]),
		&(buffer_tris_pinfo_tag[I1dm(i)]),
		&(buffer_tris_pinfo_lindex[I1dm(i)]),
		&(buffer_tris_pinfo_proc[I1dm(i)]));

    }

    /* Free the buffer ps and tris index */
    for (i = 1; i <= num_nb_proc; i++)
    {
	emxFree_int32_T(&(ps_ring_proc[I1dm(i)]));
	emxFree_int32_T(&(tris_ring_proc[I1dm(i)]));
    }
    free(ps_ring_proc);
    free(tris_ring_proc);

    /* send all the information to different processors */
    for (i = 1; i <= num_nb_proc; i++)
    {

	isend2D_real_T(buffer_ps[I1dm(i)], mesh->nb_proc->data[I1dm(i)],
		       tag_ps, MPI_COMM_WORLD, &(send_rqst_list[cur_rqst]), &(send_rqst_list[cur_rqst+1]));
	cur_rqst += 2;

	isend2D_int32_T(buffer_tris[I1dm(i)], mesh->nb_proc->data[I1dm(i)],
		       tag_tris, MPI_COMM_WORLD, &(send_rqst_list[cur_rqst]), &(send_rqst_list[cur_rqst+1]));
	cur_rqst += 2;

	MPI_Isend(buffer_ps_pinfo_tag[I1dm(i)], (buffer_ps[I1dm(i)])->size[0]+1, MPI_INT,
		  mesh->nb_proc->data[I1dm(i)], tag_ps_pinfo1, MPI_COMM_WORLD, &(send_rqst_list[cur_rqst++]));
	MPI_Isend(buffer_ps_pinfo_lindex[I1dm(i)], buffer_ps_pinfo_tag[I1dm(i)][(buffer_ps[I1dm(i)])->size[0]], MPI_INT,
		  mesh->nb_proc->data[I1dm(i)], tag_ps_pinfo2, MPI_COMM_WORLD, &(send_rqst_list[cur_rqst++]));
	MPI_Isend(buffer_ps_pinfo_proc[I1dm(i)], buffer_ps_pinfo_tag[I1dm(i)][(buffer_ps[I1dm(i)])->size[0]], MPI_INT,
		  mesh->nb_proc->data[I1dm(i)], tag_ps_pinfo3, MPI_COMM_WORLD, &(send_rqst_list[cur_rqst++]));


	MPI_Isend(buffer_tris_pinfo_tag[I1dm(i)], (buffer_tris[I1dm(i)])->size[0]+1, MPI_INT,
		  mesh->nb_proc->data[I1dm(i)], tag_tris_pinfo1, MPI_COMM_WORLD, &(send_rqst_list[cur_rqst++]));
	MPI_Isend(buffer_tris_pinfo_lindex[I1dm(i)], buffer_tris_pinfo_tag[I1dm(i)][(buffer_tris[I1dm(i)])->size[0]], MPI_INT,
		  mesh->nb_proc->data[I1dm(i)], tag_tris_pinfo2, MPI_COMM_WORLD, &(send_rqst_list[cur_rqst++]));
	MPI_Isend(buffer_tris_pinfo_proc[I1dm(i)], buffer_tris_pinfo_tag[I1dm(i)][(buffer_tris[I1dm(i)])->size[0]], MPI_INT,
		  mesh->nb_proc->data[I1dm(i)], tag_tris_pinfo3, MPI_COMM_WORLD, &(send_rqst_list[cur_rqst++]));
    }



    /* Receive buffer points and tris with temp pinfo */

    for (i = 1; i <= num_nb_proc; i++)
	MPI_Irecv(&(recv_size[2*I1dm(i)]), 2, MPI_INT, mesh->nb_proc->data[I1dm(i)], tag_ps+1, MPI_COMM_WORLD, &(recv_req_list[I1dm(i)]));

    for (i = 1; i <= num_nb_proc; i++)
    {
	emxArray_real_T *buffer_ps_recv;
	emxArray_int32_T *buffer_tris_recv;

	int *buf_ppinfo_tag_recv;
	int *buf_ppinfo_lindex_recv;
	int *buf_ppinfo_proc_recv;

	int *buf_tpinfo_tag_recv;
	int *buf_tpinfo_lindex_recv;
	int *buf_tpinfo_proc_recv;

	int num_buf_ps_recv, num_buf_tris_recv;
	int num_buf_ps_pinfo_recv;
	int num_buf_tris_pinfo_recv;

	MPI_Status tmp_status;
	MPI_Status recv_status1, recv_status2;

	int recv_index;
	int proc_recv;

	MPI_Waitany(num_nb_proc, recv_req_list, &recv_index, &recv_status1);
	proc_recv = recv_status1.MPI_SOURCE;

	buffer_ps_recv = emxCreate_real_T(recv_size[2*recv_index], recv_size[2*recv_index+1]);

	MPI_Recv(buffer_ps_recv->data, recv_size[2*recv_index]*recv_size[2*recv_index+1], MPI_DOUBLE, proc_recv, tag_ps+2, MPI_COMM_WORLD, &recv_status2);

	recv2D_int32_T(&buffer_tris_recv, proc_recv, tag_tris, MPI_COMM_WORLD);

	num_buf_ps_recv = buffer_ps_recv->size[0];
	num_buf_tris_recv = buffer_tris_recv->size[0];

	/* Recv ps pinfo */
	buf_ppinfo_tag_recv = (int *) calloc(num_buf_ps_recv+1, sizeof(int));

	MPI_Recv(buf_ppinfo_tag_recv, num_buf_ps_recv+1, MPI_INT, proc_recv,
		 tag_ps_pinfo1, MPI_COMM_WORLD, &tmp_status);
	
	num_buf_ps_pinfo_recv = buf_ppinfo_tag_recv[num_buf_ps_recv];

	buf_ppinfo_lindex_recv = (int *) calloc(num_buf_ps_pinfo_recv, sizeof(int));
	buf_ppinfo_proc_recv = (int *) calloc(num_buf_ps_pinfo_recv, sizeof(int));

	MPI_Recv(buf_ppinfo_lindex_recv, num_buf_ps_pinfo_recv, MPI_INT, proc_recv,
		 tag_ps_pinfo2, MPI_COMM_WORLD, &tmp_status);
	MPI_Recv(buf_ppinfo_proc_recv, num_buf_ps_pinfo_recv, MPI_INT, proc_recv,
		 tag_ps_pinfo3, MPI_COMM_WORLD, &tmp_status);

	/* Recv tris pinfo */
	buf_tpinfo_tag_recv = (int *) calloc(num_buf_tris_recv+1, sizeof(int));

	MPI_Recv(buf_tpinfo_tag_recv, num_buf_tris_recv+1, MPI_INT, proc_recv,
		 tag_tris_pinfo1, MPI_COMM_WORLD, &tmp_status);
	
	num_buf_tris_pinfo_recv = buf_tpinfo_tag_recv[num_buf_tris_recv];

	buf_tpinfo_lindex_recv = (int *) calloc(num_buf_tris_pinfo_recv, sizeof(int));
	buf_tpinfo_proc_recv = (int *) calloc(num_buf_tris_pinfo_recv, sizeof(int));

	MPI_Recv(buf_tpinfo_lindex_recv, num_buf_tris_pinfo_recv, MPI_INT, proc_recv,
		 tag_tris_pinfo2, MPI_COMM_WORLD, &tmp_status);
	MPI_Recv(buf_tpinfo_proc_recv, num_buf_tris_pinfo_recv, MPI_INT, proc_recv,
		 tag_tris_pinfo3, MPI_COMM_WORLD, &tmp_status);

	hpAttachNRingGhostWithPInfo(mesh, proc_recv, buffer_ps_recv, buffer_tris_recv,
		buf_ppinfo_tag_recv, buf_ppinfo_lindex_recv, buf_ppinfo_proc_recv,
		buf_tpinfo_tag_recv, buf_tpinfo_lindex_recv, buf_tpinfo_proc_recv);

	emxFree_real_T(&buffer_ps_recv);
	emxFree_int32_T(&buffer_tris_recv);

	free(buf_ppinfo_tag_recv);
	free(buf_ppinfo_lindex_recv);
	free(buf_ppinfo_proc_recv);

	free(buf_tpinfo_tag_recv);
	free(buf_tpinfo_lindex_recv);
	free(buf_tpinfo_proc_recv);
    }

    free(recv_req_list);
    free(recv_size);

    /* Wait until all the array are sent */

    MPI_Waitall(num_all_send_rqst, send_rqst_list, send_status_list);

    /* Free the array for send */

    free(send_rqst_list);
    free(send_status_list);

    for (i = 1; i <= num_nb_proc; i++)
    {
	emxFree_real_T(&(buffer_ps[I1dm(i)]));
	emxFree_int32_T(&(buffer_tris[I1dm(i)]));
	free(buffer_ps_pinfo_tag[I1dm(i)]);
	free(buffer_ps_pinfo_lindex[I1dm(i)]);
	free(buffer_ps_pinfo_proc[I1dm(i)]);

	free(buffer_tris_pinfo_tag[I1dm(i)]);
	free(buffer_tris_pinfo_lindex[I1dm(i)]);
	free(buffer_tris_pinfo_proc[I1dm(i)]);
    }

    free(buffer_ps);
    free(buffer_tris);
    free(buffer_ps_pinfo_tag);
    free(buffer_ps_pinfo_lindex);
    free(buffer_ps_pinfo_proc);

    free(buffer_tris_pinfo_tag);
    free(buffer_tris_pinfo_lindex);
    free(buffer_tris_pinfo_proc);

    /* Merge pinfo for ps/tris to get a fully updated pinfo list */
    hpUpdatePInfo(mesh);

    /* Update nb_proc information based on the new pinfo */
    hpUpdateNbWithPInfo(mesh);

}


void hpBuildNRingGhost(hiPropMesh *mesh, const real_T num_ring)
{
    int i;

    int num_nb_proc = mesh->nb_proc->size[0];


    emxArray_int32_T **psid_proc = (emxArray_int32_T **)
	calloc(num_nb_proc, sizeof(emxArray_int32_T *));

    /* Get the overlapping points for building up n-ring neighborhood */
    hpCollectAllSharedPs(mesh, psid_proc);

    /* Build n-ring neighborhood and send*/
    emxArray_int32_T **ps_ring_proc = (emxArray_int32_T **) calloc(num_nb_proc, sizeof(emxArray_int32_T *));
    emxArray_int32_T **tris_ring_proc = (emxArray_int32_T **) calloc(num_nb_proc, sizeof(emxArray_int32_T *));
    emxArray_real_T **buffer_ps = (emxArray_real_T **) calloc(num_nb_proc, sizeof(emxArray_real_T *));
    emxArray_int32_T **buffer_tris = (emxArray_int32_T **) calloc(num_nb_proc, sizeof(emxArray_int32_T *));

    for (i = 1; i <= num_nb_proc; i++)
    {
	hpBuildGhostPsTrisForSend(mesh, i, num_ring, psid_proc[I1dm(i)],
				  &(ps_ring_proc[I1dm(i)]),
				  &(tris_ring_proc[I1dm(i)]), 
				  &(buffer_ps[I1dm(i)]), &(buffer_tris[I1dm(i)]));
    }

    /* Free the psid_proc, ps_ring_proc, tris_ring_proc, buffer_ps, buffer_tris
     * are freed in the hpCommPsTrisWithPInfo function */

    for (i = 1; i <= num_nb_proc; i++)
	emxFree_int32_T(&(psid_proc[I1dm(i)]));
    free(psid_proc);

    /* Build and communicate the ghost based on ps_ring_proc, tris_ring_proc,
     * buffer_ps, buffer_tris */

    hpCommPsTrisWithPInfo(mesh, ps_ring_proc, tris_ring_proc, buffer_ps, buffer_tris);
}


void hpAttachNRingGhostWithPInfo(hiPropMesh *mesh,
				 const int rcv_id,
				 emxArray_real_T *bps,
				 emxArray_int32_T *btris,
				 int *ppinfot,
				 int *ppinfol,
				 int *ppinfop,
				 int *tpinfot,
				 int *tpinfol,
				 int *tpinfop)
{

    int i,j;
    int cur_proc;
    MPI_Comm_rank(MPI_COMM_WORLD, &cur_proc);

    emxArray_real_T *ps = mesh->ps;
    emxArray_int32_T *tris = mesh->tris;

    int num_ps_old = ps->size[0];
    int num_tris_old = tris->size[0];

    hpPInfoList *ps_pinfo = mesh->ps_pinfo;
    hpPInfoList *tris_pinfo = mesh->tris_pinfo;

    int num_buf_ps = bps->size[0];
    int num_buf_tris = btris->size[0];

    /*
    int num_buf_ps_pinfo = ppinfot[num_buf_ps];
    int num_buf_tris_pinfo = tpinfot[num_buf_tris];
    */

    int *ps_map = (int *) calloc(num_buf_ps, sizeof(int));
    int *tris_map = (int *) calloc(num_buf_tris, sizeof(int));

    unsigned char *buf_ps_flag = (unsigned char *) calloc(num_buf_ps, sizeof(unsigned char));
    unsigned char *buf_tris_flag = (unsigned char *) calloc(num_buf_tris, sizeof(unsigned char));

    int num_add_ps = 0;
    int num_add_tris = 0;


    /* Calculate # of new ps/tris and allocated the memory
     * Update ps_map, have all new ps and tris flagged */

    for (i = 1; i <= num_buf_ps; i++)
    {
	buf_ps_flag[I1dm(i)] = 1;
	/* If already existing, do not add a new point */
	for(j = ppinfot[i-1]; j <= ppinfot[i]-1; j++)
	{
	    if ((ppinfop[j] == cur_proc) && (ppinfol[j] != -1))
	    {
		buf_ps_flag[I1dm(i)] = 0;
		ps_map[I1dm(i)] = ppinfol[j];
		break;
	    }
	}
	if (buf_ps_flag[I1dm(i)] == 1)
	{
	    /* Still could be some existing point attached from other
	     * processors. */
	    int master_proc_buf = ppinfop[ppinfot[i-1]];
	    int master_index_buf = ppinfol[ppinfot[i-1]];
	    for (j = 1; j <= num_ps_old; j++)
	    {
		int head_cur = ps_pinfo->head[I1dm(j)];
		int master_proc_cur = ps_pinfo->pdata[I1dm(head_cur)].proc;
		int master_index_cur = ps_pinfo->pdata[I1dm(head_cur)].lindex;
		if ((master_proc_buf == master_proc_cur) &&
		    (master_index_buf == master_index_cur) )
		{
		    buf_ps_flag[I1dm(i)] = 0;
		    ps_map[I1dm(i)] = j;
		    break;
		}
	    }
	    /* now a new point */
	    if (buf_ps_flag[I1dm(i)] == 1)
	    {
		num_add_ps++;
		ps_map[I1dm(i)] = num_ps_old + num_add_ps;
	    }
	}
    }

    for (i = 1; i <= num_buf_tris; i++)
    {
	buf_tris_flag[I1dm(i)] = 1;
	for(j = tpinfot[i-1]; j <= tpinfot[i]-1; j++)
	{
	    if ((tpinfop[j] == cur_proc) && (tpinfol[j] != -1))
	    {
		buf_tris_flag[I1dm(i)] = 0;
		tris_map[I1dm(i)] = tpinfol[j];
		break;
	    }
	}
	if (buf_tris_flag[I1dm(i)] == 1)
	{
	    /* Still could be some existing triangle attached from other
	     * processors. */
	    int master_proc_buf = tpinfop[tpinfot[i-1]];
	    int master_index_buf = tpinfol[tpinfot[i-1]];
	    for (j = 1; j <= num_tris_old; j++)
	    {
		int head_cur = tris_pinfo->head[I1dm(j)];
		int master_proc_cur = tris_pinfo->pdata[I1dm(head_cur)].proc;
		int master_index_cur = tris_pinfo->pdata[I1dm(head_cur)].lindex;
		if ((master_proc_buf == master_proc_cur) &&
		    (master_index_buf == master_index_cur) )
		{
		    buf_tris_flag[I1dm(i)] = 0;
		    tris_map[I1dm(i)] = j;
		    break;
		}
	    }
	    /* now a new triangle */
	    if (buf_tris_flag[I1dm(i)] == 1)
	    {
		num_add_tris++;
		tris_map[I1dm(i)] = num_tris_old + num_add_tris;
	    }
	}
    }
    
    /* Allocated more space for ps and tris */

    addRowToArray_real_T(ps, num_add_ps);
    addRowToArray_int32_T(tris, num_add_tris);

    /* Also need to allocated more space for head and tail in pinfolist */

    int *new_head_ps = (int *) calloc(num_ps_old + num_add_ps, sizeof(int));
    int *new_tail_ps = (int *) calloc(num_ps_old + num_add_ps, sizeof(int));

    int *new_head_tris = (int *) calloc(num_tris_old + num_add_tris, sizeof(int));
    int *new_tail_tris = (int *) calloc(num_tris_old + num_add_tris, sizeof(int));

    memcpy(new_head_ps, ps_pinfo->head, num_ps_old*sizeof(int));
    memcpy(new_tail_ps, ps_pinfo->tail, num_ps_old*sizeof(int));

    memcpy(new_head_tris, tris_pinfo->head, num_tris_old*sizeof(int));
    memcpy(new_tail_tris, tris_pinfo->tail, num_tris_old*sizeof(int));

    free(ps_pinfo->head);
    free(ps_pinfo->tail);
    free(tris_pinfo->head);
    free(tris_pinfo->tail);

    ps_pinfo->head = new_head_ps;
    ps_pinfo->tail = new_tail_ps;

    tris_pinfo->head = new_head_tris;
    tris_pinfo->tail = new_tail_tris;

    /* Add each point, merge pinfo */

    for (i = 1; i <= num_buf_ps; i++)
    {
	/* If not a new point, update the pinfo */

	if (buf_ps_flag[I1dm(i)] == 0)
	{
	    int ps_index = ps_map[I1dm(i)];
	    for(j = ppinfot[i-1]; j <= ppinfot[i]-1; j++)
	    {
		if (ppinfol[j] == -1)
		{
		    int next_node = ps_pinfo->head[I1dm(ps_index)];
		    while (next_node != -1)
		    {
			if(ps_pinfo->pdata[I1dm(next_node)].proc == ppinfop[j])
			    break;
			else
			    next_node = ps_pinfo->pdata[I1dm(next_node)].next;
		    }
		    if (next_node == -1)
		    {
			hpEnsurePInfoCapacity(ps_pinfo);
			int cur_tail = ps_pinfo->tail[I1dm(ps_index)];
			ps_pinfo->allocated_len++;
			int new_tail = ps_pinfo->allocated_len; /* new node */
			ps_pinfo->pdata[I1dm(new_tail)].proc = ppinfop[j];
			ps_pinfo->pdata[I1dm(new_tail)].lindex = -1;
			ps_pinfo->pdata[I1dm(new_tail)].next = -1;

			ps_pinfo->pdata[I1dm(cur_tail)].next = new_tail;
			ps_pinfo->tail[I1dm(ps_index)] = new_tail;
		    }
		}
	    }
	}
	/* If a new point, add the point, the pinfo and update
	 * based on the current local index */
	else
	{

	    int ps_index = ps_map[I1dm(i)];

	    ps->data[I2dm(ps_index,1,ps->size)] = bps->data[I2dm(i,1,bps->size)];
	    ps->data[I2dm(ps_index,2,ps->size)] = bps->data[I2dm(i,2,bps->size)];
	    ps->data[I2dm(ps_index,3,ps->size)] = bps->data[I2dm(i,3,bps->size)];

	    /* Deal with head ---> tail - 1 */
	    int cur_node;
	    int new_head = ps_pinfo->allocated_len + 1;
	    for (j = ppinfot[i-1]; j < ppinfot[i]-1; j++)
	    {
		hpEnsurePInfoCapacity(ps_pinfo);
		ps_pinfo->allocated_len++;
		cur_node = ps_pinfo->allocated_len;
		ps_pinfo->pdata[I1dm(cur_node)].proc = ppinfop[j];
		ps_pinfo->pdata[I1dm(cur_node)].next = cur_node+1;

		if (ppinfop[j] == cur_proc)
		    ps_pinfo->pdata[I1dm(cur_node)].lindex = ps_index;
		else
		    ps_pinfo->pdata[I1dm(cur_node)].lindex = ppinfol[j];
	    }
	    ps_pinfo->head[I1dm(ps_index)] = new_head;

	    /* Deal with tail */
	    j = ppinfot[i]-1;
	    hpEnsurePInfoCapacity(ps_pinfo);
	    ps_pinfo->allocated_len++;
	    cur_node = ps_pinfo->allocated_len;
	    ps_pinfo->pdata[I1dm(cur_node)].proc = ppinfop[j];
	    ps_pinfo->pdata[I1dm(cur_node)].next = -1;

	    if (ppinfop[j] == cur_proc)
		ps_pinfo->pdata[I1dm(cur_node)].lindex = ps_index;
	    else
		ps_pinfo->pdata[I1dm(cur_node)].lindex = ppinfol[j];
	    ps_pinfo->tail[I1dm(ps_index)] = cur_node;
	}
    }

    /* Add each triangle, merge pinfo */

    for (i = 1; i <= num_buf_tris; i++)
    {
	/* If not a new triangle, update the pinfo */

	if (buf_tris_flag[I1dm(i)] == 0)
	{
	    int tris_index = tris_map[I1dm(i)];
	    for(j = tpinfot[i-1]; j <= tpinfot[i]-1; j++)
	    {
		if (tpinfol[j] == -1)
		{
		    int next_node = tris_pinfo->head[I1dm(tris_index)];
		    while (next_node != -1)
		    {
			if(tris_pinfo->pdata[I1dm(next_node)].proc == tpinfop[j])
			    break;
			else
			    next_node = tris_pinfo->pdata[I1dm(next_node)].next;
		    }

		    if (next_node == -1)
		    {
			hpEnsurePInfoCapacity(tris_pinfo);
			int cur_tail = tris_pinfo->tail[I1dm(tris_index)];
			tris_pinfo->allocated_len++; /* new node */
			int new_tail = tris_pinfo->allocated_len; 

			tris_pinfo->pdata[I1dm(tris_pinfo->allocated_len)].proc = tpinfop[j];
			tris_pinfo->pdata[I1dm(tris_pinfo->allocated_len)].lindex = -1;
			tris_pinfo->pdata[I1dm(tris_pinfo->allocated_len)].next = -1;

			tris_pinfo->pdata[I1dm(cur_tail)].next = new_tail;
			tris_pinfo->tail[I1dm(tris_index)] = new_tail;
		    }
		}
	    }
	}
	/* If a new point, add the point, the pinfo and update
	 * based on the current local index */
	else
	{
	    int tris_index = tris_map[I1dm(i)];

	    int recv_tri_index1 = btris->data[I2dm(i,1,btris->size)];
	    int recv_tri_index2 = btris->data[I2dm(i,2,btris->size)];
	    int recv_tri_index3 = btris->data[I2dm(i,3,btris->size)];

	    tris->data[I2dm(tris_index,1,tris->size)] = ps_map[I1dm(recv_tri_index1)];
	    tris->data[I2dm(tris_index,2,tris->size)] = ps_map[I1dm(recv_tri_index2)];
	    tris->data[I2dm(tris_index,3,tris->size)] = ps_map[I1dm(recv_tri_index3)];

	    /* Deal with head ---> tail - 1 */
	    int cur_node;
	    int new_head = tris_pinfo->allocated_len + 1;
	    for (j = tpinfot[i-1]; j < tpinfot[i]-1; j++)
	    {
		hpEnsurePInfoCapacity(tris_pinfo);
		tris_pinfo->allocated_len++;
		cur_node = tris_pinfo->allocated_len;

		tris_pinfo->pdata[I1dm(cur_node)].proc = tpinfop[j];
		tris_pinfo->pdata[I1dm(cur_node)].next = cur_node+1;

		if (tpinfop[j] == cur_proc)
		    tris_pinfo->pdata[I1dm(cur_node)].lindex = tris_index;
		else
		    tris_pinfo->pdata[I1dm(cur_node)].lindex = tpinfol[j];
	    }
	    tris_pinfo->head[I1dm(tris_index)] = new_head;

	    /* Deal with tail */
	    hpEnsurePInfoCapacity(tris_pinfo);
	    tris_pinfo->allocated_len++;
	    cur_node = tris_pinfo->allocated_len;
	    j = tpinfot[i]-1;

	    tris_pinfo->pdata[I1dm(cur_node)].proc = tpinfop[j];
	    tris_pinfo->pdata[I1dm(cur_node)].next = -1;

	    if (tpinfop[j] == cur_proc)
		tris_pinfo->pdata[I1dm(cur_node)].lindex = tris_index;
	    else
		tris_pinfo->pdata[I1dm(cur_node)].lindex = tpinfol[j];

	    tris_pinfo->tail[I1dm(tris_index)] = cur_node;
	}
    }

    free(ps_map);
    free(tris_map);
    free(buf_ps_flag);
    free(buf_tris_flag);

}

void hpCollectNRingTris(const hiPropMesh *mesh,
			const int nb_proc_index,
			const emxArray_int32_T *in_psid,
			const real_T num_ring,
			emxArray_int32_T **out_ps,
			emxArray_int32_T **out_tris)
{
    int i, j;

    int num_ps = mesh->ps->size[0];
    int num_tris = mesh->tris->size[0];
    int max_b_numps = 128;
    int max_b_numtris = 256;

    hpPInfoNode *tris_pdata = mesh->tris_pinfo->pdata;
    int *tris_phead = mesh->tris_pinfo->head;
    int *nb_proc = mesh->nb_proc->data;
    emxArray_int32_T *tris = mesh->tris;
    int *tris_data = tris->data;


    /* For denote whether each ps and tris belongs to the 2-ring buffer for
     * in_psid. If ps_flag[I1dm(i)] = true, then point i is in the 2-ring
     * buffer, if tris_flag[I1dm(j)] = true, then triangle j is in the 2-ring
     * buffer */
    emxArray_boolean_T *ps_flag = emxCreateND_boolean_T(1, &num_ps);
    emxArray_boolean_T *tris_flag = emxCreateND_boolean_T(1, &num_tris);

    /* Used for obtain_nring_surf, initialized as false */
    emxArray_boolean_T *in_vtags = emxCreateND_boolean_T(1, &num_ps);
    emxArray_boolean_T *in_ftags = emxCreateND_boolean_T(1, &num_tris);

    /* Used for storing outputs of obtain_nring_surf */
    emxArray_int32_T *in_ngbvs = emxCreateND_int32_T(1, &max_b_numps);
    emxArray_int32_T *in_ngbfs = emxCreateND_int32_T(1, &max_b_numtris);

    int num_ps_ring, num_tris_ring;

    for (i = 1; i <= in_psid->size[0]; i++)
    {
	int cur_ps = in_psid->data[I1dm(i)];

	obtain_nring_surf(cur_ps, num_ring, 0, mesh->tris, mesh->opphe,
			  mesh->inhe, in_ngbvs, in_vtags, in_ftags, 
			  in_ngbfs, &num_ps_ring, &num_tris_ring);

       	/*cur_ps itself in the list */
	/*
	ps_flag->data[I1dm(cur_ps)] = true;
	*/

	/*
	for (j = 1; j <= num_ps_ring; j++)
	{
	    int ps_buf_index = in_ngbvs->data[I1dm(j)];
	    ps_flag->data[I1dm(ps_buf_index)] = true;
	}
	*/

	for (j = 1; j <= num_tris_ring; j++)
	{
	    /* Only send the triangles which does not exists on the nb proc */
	    int tris_buf_index = in_ngbfs->data[I1dm(j)];
	    tris_flag->data[I1dm(tris_buf_index)] = true;

	    int next_node = tris_phead[I1dm(tris_buf_index)];
	    while(next_node != -1)
	    {
		int proc_id = tris_pdata[I1dm(next_node)].proc;
		/* If also exists on other processor, do not send it */
		if (proc_id == nb_proc[I1dm(nb_proc_index)])
		{
		    tris_flag->data[I1dm(tris_buf_index)] = false;
		    break;
		}
		else
		    next_node = tris_pdata[I1dm(next_node)].next;
	    }

	    /*
	    tris_flag->data[I1dm(tris_buf_index)] = true;
	    */
	}
    }

    /* Set the points needed to be send based on tris */

    for (i = 1; i <= num_tris; i++)
    {
	if (tris_flag->data[I1dm(i)] == true)
	{
	    int bufpi[3];
	    bufpi[0] = tris_data[I2dm(i,1,tris->size)];
	    bufpi[1] = tris_data[I2dm(i,2,tris->size)];
	    bufpi[2] = tris_data[I2dm(i,3,tris->size)];
	    ps_flag->data[bufpi[0]-1] = true;
	    ps_flag->data[bufpi[1]-1] = true;
	    ps_flag->data[bufpi[2]-1] = true;
	}
    }


    /* Get total number of ps and tris in n-ring nb */

    int num_ps_ring_all = 0;
    int num_tris_ring_all = 0;

    for (i = 1; i <= num_ps; i++)
    {
	if (ps_flag->data[I1dm(i)] == true)
	    num_ps_ring_all++;
    }

    for (i = 1; i <= num_tris; i++)
    {
	if (tris_flag->data[I1dm(i)] == true)
	    num_tris_ring_all++;
    }

    /* Create n-ring ps and tris for output */
    (*out_ps) = emxCreateND_int32_T(1, &num_ps_ring_all);
    (*out_tris) = emxCreateND_int32_T(1, &num_tris_ring_all);

    /* Fill the out_ps and out_tris based on the flags */
    j = 1;
    for (i = 1; i <= num_ps; i++)
    {
	if (ps_flag->data[I1dm(i)] == true)
	{
	    (*out_ps)->data[I1dm(j)] = i;
	    j++;
	}
    }

    j = 1;
    for (i = 1; i <= num_tris; i++)
    {
	if (tris_flag->data[I1dm(i)] == true)
	{
	    (*out_tris)->data[I1dm(j)] = i;
	    j++;
	}
    }

    emxFree_int32_T(&in_ngbvs);
    emxFree_int32_T(&in_ngbfs);

    emxFree_boolean_T(&in_vtags);
    emxFree_boolean_T(&in_ftags);

    emxFree_boolean_T(&ps_flag);
    emxFree_boolean_T(&tris_flag);
}


void hpDebugOutput(const hiPropMesh *mesh, const emxArray_int32_T *debug_ps,
		   const emxArray_int32_T *debug_tris, char *debug_file_name)
{
    int j;
    /* Build ps_mapping for vtk output of debug_ps
     * for point i in mesh->ps, the new index in the vtk output is
     * ps_mapping[I1dm(i)]. If point i does not have a mapped value,
     * ps_mapping[I1dm(i)] = -1.
     */
    int *ps_mapping = (int *) calloc(mesh->ps->size[0], sizeof(int));

    for (j = 1; j <= mesh->ps->size[0]; j++)
	ps_mapping[j-1] = -1;

    for (j = 1; j <= debug_ps->size[0]; j++)
    {
	int cur_ps_id = debug_ps->data[I1dm(j)];
	ps_mapping[cur_ps_id-1] = j-1;
    }

    /* Write the data to vtk file */
    FILE* file = fopen(debug_file_name, "w");

    fprintf(file, "# vtk DataFile Version 3.0\n");
    fprintf(file, "Debug output by hiProp\n");
    fprintf(file, "ASCII\n");
    fprintf(file, "DATASET UNSTRUCTURED_GRID\n");

    fprintf(file, "POINTS %d double\n", debug_ps->size[0]);
    for (j = 1; j <= debug_ps->size[0]; j++)
    {
	int ps_id = debug_ps->data[j-1];
	fprintf(file, "%22.16lg %22.16lg %22.16lg\n",
		mesh->ps->data[I2dm(ps_id,1,mesh->ps->size)],
		mesh->ps->data[I2dm(ps_id,2,mesh->ps->size)],
		mesh->ps->data[I2dm(ps_id,3,mesh->ps->size)]);
    }
    fprintf(file, "CELLS %d %d\n", debug_tris->size[0],
	    4*debug_tris->size[0]);
    for (j = 1; j <= debug_tris->size[0]; j++)
    {
	int tri_index = debug_tris->data[I1dm(j)];
	fprintf(file, "3 %d %d %d\n",
		ps_mapping[mesh->tris->data[I2dm(tri_index,1,mesh->tris->size)]-1],
		ps_mapping[mesh->tris->data[I2dm(tri_index,2,mesh->tris->size)]-1],
		ps_mapping[mesh->tris->data[I2dm(tri_index,3,mesh->tris->size)]-1]);

    }
    fprintf(file, "CELL_TYPES %d\n", debug_tris->size[0]);
    for (j = 1; j <= debug_tris->size[0]; j++)
	fprintf(file, "5\n");
    fclose(file);
    free(ps_mapping);
}

void hpWriteUnstrMeshWithPInfo(const char *name, const hiPropMesh *mesh)
{
    FILE* file;
    int i;
    int cur_proc;
    MPI_Comm_rank(MPI_COMM_WORLD, &cur_proc);
    emxArray_real_T* points = mesh->ps;
    emxArray_int32_T* tris = mesh->tris;
    
    int *ps_head = mesh->ps_pinfo->head;
    int *ps_tail = mesh->ps_pinfo->tail;
    hpPInfoNode *ps_pdata = mesh->ps_pinfo->pdata;

    int *tri_head = mesh->tris_pinfo->head;
    int *tri_tail = mesh->tris_pinfo->tail;
    hpPInfoNode *tri_pdata = mesh->tris_pinfo->pdata;

    file = fopen(name, "w");

    fprintf(file, "# vtk DataFile Version 3.0\n");
    fprintf(file, "Mesh output by hiProp\n");
    fprintf(file, "ASCII\n");
    fprintf(file, "DATASET UNSTRUCTURED_GRID\n");

    int num_points = mesh->ps->size[0];
    int num_tris = mesh->tris->size[0];

    fprintf(file, "POINTS %d double\n", num_points);
    for (i = 1; i <= num_points; i++)
	fprintf(file, "%22.16lg %22.16lg %22.16lg\n",
		points->data[I2dm(i,1,points->size)],
		points->data[I2dm(i,2,points->size)],
		points->data[I2dm(i,3,points->size)]);

    fprintf(file, "CELLS %d %d\n", num_tris, 4*num_tris);
    for (i = 1; i <= num_tris; i++)
	fprintf(file, "3 %d %d %d\n",
		tris->data[I2dm(i,1,tris->size)]-1,
		tris->data[I2dm(i,2,tris->size)]-1,
		tris->data[I2dm(i,3,tris->size)]-1);

    fprintf(file, "CELL_TYPES %d\n", num_tris);
    for (i = 0; i<num_tris; i++)
	fprintf(file, "5\n");


    fprintf(file, "POINT_DATA %d\n", num_points);
    fprintf(file, "SCALARS point_type int 1\n");
    fprintf(file, "LOOKUP_TABLE default\n");
    for (i = 0; i < num_points; i++)
    {
	int ps_type;
	if (ps_pdata[ps_head[i]-1].proc != cur_proc)
	    ps_type = 2; /* buffer ps */
	else
	{
	    if(ps_head[i] != ps_tail[i])
		ps_type = 1; /* overlay tri */
	    else
		ps_type = 0; /* interior tri */
	}
	fprintf(file, "%d\n", ps_type);
    }

    fprintf(file, "CELL_DATA %d\n", num_tris);
    fprintf(file, "SCALARS tri_type int 1\n");
    fprintf(file, "LOOKUP_TABLE default\n");
    for (i = 0; i < num_tris; i++)
    {
	int tri_type;
	if (tri_pdata[tri_head[i]-1].proc != cur_proc)
	    tri_type = 2; /* buffer tri */
	else
	{
	    if(tri_head[i] != tri_tail[i])
		tri_type = 1; /* overlay tri */
	    else
		tri_type = 0; /* interior tri */
	}
	fprintf(file, "%d\n", tri_type);
    }

    fclose(file);
}

void hpComputeDiffops(hiPropMesh *mesh, int32_T in_degree)
{
    if( mesh->nor != ((emxArray_real_T *) NULL) )
	emxFree_real_T(&(mesh->nor));
    if( mesh->curv != ((emxArray_real_T *) NULL) )
	emxFree_real_T(&(mesh->curv));

    real_T tmp = (double) in_degree + 2.0;
    real_T in_ring = tmp/2.0;

    int num_ps = mesh->ps->size[0];

    mesh->nor = emxCreate_real_T(num_ps, 3);
    mesh->curv = emxCreate_real_T(num_ps, 2);

    emxArray_real_T *in_prdirs = emxCreate_real_T(num_ps, 3);

    compute_diffops_surf(mesh->ps, mesh->tris, in_degree, in_ring, false, mesh->nor, mesh->curv, in_prdirs, 0);

    emxFree_real_T(&(in_prdirs));
}

void hpBuildPartitionBoundary(hiPropMesh *mesh)
{
    int i;
    int *ps_phead = mesh->ps_pinfo->head;
    int *ps_ptail = mesh->ps_pinfo->tail;

    if( mesh->part_bdry != ((emxArray_int32_T *) NULL) )
	emxFree_int32_T(&(mesh->part_bdry));

    int num_ps = mesh->num_int_ps;

    int num_pbdry = 0;

    for (i = 1; i <= num_ps; i++)
    {
	if (ps_phead[I1dm(i)] != ps_ptail[I1dm(i)])
	    num_pbdry++;
    }

    mesh->part_bdry = emxCreateND_int32_T(1, &num_pbdry);

    int j = 0;
    for (i = 1; i <= num_ps; i++)
    {
	if (ps_phead[I1dm(i)] != ps_ptail[I1dm(i)])
	    mesh->part_bdry->data[j++] = i;
    }
}

void hpComputeEstimatedNormal(hiPropMesh *mesh)
{
    if (mesh->est_nor != ((emxArray_real_T *) NULL) )
	emxFree_real_T(&(mesh->est_nor));

    int num_ps = mesh->ps->size[0];

    mesh->est_nor = emxCreate_real_T(num_ps, 3);

    emxArray_real_T *xs = mesh->ps;
    emxArray_int32_T *tris = mesh->tris;
    emxArray_real_T *nrms = mesh->est_nor;

    /*Codes from here are duplicate from matlab generated function
     * average_vertex_normal tri */

  int32_T ntris;
  int32_T nv;
  int32_T i0;
  int32_T ix;
  int32_T ii;
  int32_T iy;
  real_T a[3];
  real_T b[3];
  real_T nrm[3];
  int32_T k;
  real_T y;

  /* AVERAGE_VERTEX_NORMAL_TRI Compute average vertex normal for surface  */
  /*  triangulation. */
  /*  AVERAGE_VERTEX_NORMAL_TRI(XS,TRIS,OPT,FLABEL,OPPHES) Computes the average  */
  /*  vertex normal for surface triangulation, provided vertices XS, triangles */
  /*  TRIS, weighting options OPT, face-labels FLABEL and opposite vertices in  */
  /*  mx3 OPPHES. */
  /*   */
  /*  AVERAGE_VERTEX_NORMAL_TRI(XS,TRIS) Same as above. See below for default  */
  /*  OPT and FLABEL values. */
  /*  */
  /*  AVERAGE_VERTEX_NORMAL_TRI(XS,TRIS,OPT) Same as above. See below for  */
  /*  default FLABEL values. */
  /*  */
  /*  AVERAGE_VERTEX_NORMAL_TRI(XS,TRIS,OPT,FLABEL) Same as above. */
  /*      */
  /*  Set OPT to 'area', 'unit', 'angle', 'sphere', and 'bisect'. Default is  */
  /*  'area'. If FLABEL is given, then only faces with zero labels are  */
  /*  considered. */
  /*  */
  /*  See also AVERAGE_VERTEX_NORMAL_CURV */
  ntris = tris->size[0];
  nv = xs->size[0];
  i0 = nrms->size[0] * nrms->size[1];
  nrms->size[0] = nv;
  nrms->size[1] = 3;
  emxEnsureCapacity((emxArray__common *)nrms, i0, (int32_T)sizeof(real_T));
  ix = nv * 3 - 1;
  for (i0 = 0; i0 <= ix; i0++) {
    nrms->data[i0] = 0.0;
  }

  for (ii = 0; ii + 1 <= ntris; ii++) {
    ix = tris->data[ii + (tris->size[0] << 1)];
    iy = tris->data[ii + tris->size[0]];
    for (i0 = 0; i0 < 3; i0++) {
      a[i0] = xs->data[(ix + xs->size[0] * i0) - 1] - xs->data[(iy + xs->size[0]
        * i0) - 1];
    }

    ix = tris->data[ii];
    iy = tris->data[ii + (tris->size[0] << 1)];
    for (i0 = 0; i0 < 3; i0++) {
      b[i0] = xs->data[(ix + xs->size[0] * i0) - 1] - xs->data[(iy + xs->size[0]
        * i0) - 1];
    }

    /* CROSS_COL Efficient routine for computing cross product of two  */
    /* 3-dimensional column vectors. */
    /*  CROSS_COL(A,B) Efficiently computes the cross product between */
    /*  3-dimensional column vector A, and 3-dimensional column vector B. */
    nrm[0] = a[1] * b[2] - a[2] * b[1];
    nrm[1] = a[2] * b[0] - a[0] * b[2];
    nrm[2] = a[0] * b[1] - a[1] * b[0];
    for (k = 0; k < 3; k++) {
      ix = tris->data[ii + tris->size[0] * k];
      iy = tris->data[ii + tris->size[0] * k];
      for (i0 = 0; i0 < 3; i0++) {
        a[i0] = nrms->data[(iy + nrms->size[0] * i0) - 1] + nrm[i0];
      }

      for (i0 = 0; i0 < 3; i0++) {
        nrms->data[(ix + nrms->size[0] * i0) - 1] = a[i0];
      }
    }
  }

  for (ii = 0; ii + 1 <= nv; ii++) {
    for (i0 = 0; i0 < 3; i0++) {
      nrm[i0] = nrms->data[ii + nrms->size[0] * i0];
    }

    y = 0.0;
    ix = 0;
    iy = 0;
    for (k = 0; k < 3; k++) {
      y += nrms->data[ii + nrms->size[0] * ix] * nrm[iy];
      ix++;
      iy++;
    }

    y = sqrt(y + 1.0E-100);
    for (i0 = 0; i0 < 3; i0++) {
      nrms->data[ii + nrms->size[0] * i0] /= y;
    }
  }
}

void hpUpdatedNorCurv(hiPropMesh *mesh)
{
    int i, tag_send, tag_recv, j;
    int proc_send, proc_recv;
    int num_proc, rank;

    emxArray_real_T *nor = mesh->nor;
    emxArray_real_T *curv = mesh->curv;

    emxArray_int32_T *nb_proc = mesh->nb_proc;

    hpPInfoList *ps_pinfo = mesh->ps_pinfo;
    hpPInfoNode *ps_pdata = ps_pinfo->pdata;
    int *ps_phead = ps_pinfo->head;


    MPI_Comm_size(MPI_COMM_WORLD, &num_proc);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int num_nbp = nb_proc->size[0];

    /* First build the buffer nor/curv and corresponding lindex for send */
    emxArray_real_T **buf_nor = (emxArray_real_T **) calloc(num_nbp, sizeof(emxArray_real_T *));
    emxArray_real_T **buf_curv = (emxArray_real_T **) calloc(num_nbp, sizeof(emxArray_real_T *));
    int **buf_lindex = (int **) calloc(num_nbp, sizeof(int *));

    for (i = 1; i <= num_nbp; i++)
    {
	int nb_procid = nb_proc->data[I1dm(i)];
	int num_overlay_ps = 0;
	unsigned char *ps_flag = (unsigned char *) calloc(mesh->ps->size[0], sizeof(unsigned char));

	for (j = 1; j <= mesh->ps->size[0]; j++)
	{
	    int next_node = ps_phead[I1dm(j)];
	    if (next_node != rank)
		continue;
	    else
	    {
		next_node = ps_pdata[I1dm(next_node)].next;
		while(next_node != -1)
		{
		    if (ps_pdata[I1dm(next_node)].proc == nb_procid)
		    {
			ps_flag[I1dm(j)] = 1;
			num_overlay_ps++;
			break;
		    }
		    else
			next_node = ps_pdata[I1dm(next_node)].next;
		}

	    }
	}

	buf_nor[I1dm(i)] = emxCreate_real_T(num_overlay_ps, 3);
	buf_curv[I1dm(i)] = emxCreate_real_T(num_overlay_ps, 2);
	buf_lindex[I1dm(i)]= (int *) calloc(num_overlay_ps, sizeof(int));

	emxArray_real_T *bnor = buf_nor[I1dm(i)];
	emxArray_real_T *bcurv = buf_curv[I1dm(i)];
	int *blindex = buf_lindex[I1dm(i)];

	int cur_pos = 1;
	for (j = 1; j <= mesh->ps->size[0]; j++)
	{
	    if (ps_flag[I1dm(j)] == 1)
	    {
		bnor->data[I2dm(cur_pos,1,bnor->size)] = nor->data[I2dm(j,1,nor->size)];
		bnor->data[I2dm(cur_pos,2,bnor->size)] = nor->data[I2dm(j,2,nor->size)];
		bnor->data[I2dm(cur_pos,3,bnor->size)] = nor->data[I2dm(j,3,nor->size)];
		bcurv->data[I2dm(cur_pos,1,bcurv->size)] = curv->data[I2dm(j,1,curv->size)];
		bcurv->data[I2dm(cur_pos,2,bcurv->size)] = curv->data[I2dm(j,2,curv->size)];
		blindex[I1dm(cur_pos)] = j;
		cur_pos++;
	    }
	}
	free(ps_flag);
    }

    /* Send the information */
    MPI_Request *send_req_list = (MPI_Request *) malloc( 5*num_nbp*sizeof(MPI_Request) );

    MPI_Status *send_status_list = (MPI_Status *) malloc( 5*num_nbp*sizeof(MPI_Status) );

    MPI_Request *recv_req_list = (MPI_Request *) malloc ( num_nbp*sizeof(MPI_Request) );

    /* Stores the received array size */

    int *recv_size = (int *) calloc ( 2*num_nbp, sizeof(int));
    
    for (i = 1; i <= num_nbp; i++)
    {
	proc_send = nb_proc->data[I1dm(i)];
	tag_send = proc_send;
	isend2D_real_T(buf_nor[I1dm(i)], proc_send, tag_send, MPI_COMM_WORLD,
		       &(send_req_list[I1dm(i)]), &(send_req_list[I1dm(i)+num_nbp]));
	MPI_Isend(buf_lindex[I1dm(i)], (buf_nor[I1dm(i)])->size[0], MPI_INT, proc_send, tag_send+3, MPI_COMM_WORLD, &(send_req_list[I1dm(i)+2*num_nbp])); 
	isend2D_real_T(buf_curv[I1dm(i)], proc_send, tag_send+4, MPI_COMM_WORLD,
		       &(send_req_list[I1dm(i)+3*num_nbp]), &(send_req_list[I1dm(i)+4*num_nbp]));
    }

    for (i = 1; i <= num_nbp; i++)
    {
	proc_recv = nb_proc->data[I1dm(i)];
	tag_recv = rank;
	MPI_Irecv(&(recv_size[2*I1dm(i)]), 2, MPI_INT, proc_recv, tag_recv+1, MPI_COMM_WORLD, &(recv_req_list[I1dm(i)]));
    }

    for (i = 1; i <= num_nbp; i++)
    {
	emxArray_real_T *nor_recv;
	emxArray_real_T *curv_recv;
	int *lindex_recv;

	tag_recv = rank;
	MPI_Status recv_status1;
	MPI_Status recv_status2;
	MPI_Status recv_status3;
	int recv_index;
	
	MPI_Waitany(num_nbp, recv_req_list, &recv_index, &recv_status1);

	proc_recv = recv_status1.MPI_SOURCE;

	nor_recv = emxCreate_real_T(recv_size[2*recv_index], recv_size[2*recv_index+1]);
	lindex_recv = (int *) calloc(recv_size[2*recv_index], sizeof(int));
	nor_recv = emxCreate_real_T(recv_size[2*recv_index], 2);
	MPI_Recv(nor_recv->data, recv_size[2*recv_index]*recv_size[2*recv_index+1], MPI_DOUBLE, proc_recv, tag_recv+2, MPI_COMM_WORLD, &recv_status2);
	MPI_Recv(lindex_recv, recv_size[2*recv_index], MPI_INT, proc_recv, tag_recv+3, MPI_COMM_WORLD, &recv_status3);
	recv2D_real_T(&(curv_recv), proc_recv, tag_recv+4, MPI_COMM_WORLD);

	for (j = 1; j <= recv_size[2*recv_index]; j++)
	{
	    int cur_index = lindex_recv[I1dm(j)];
	    nor->data[I2dm(cur_index,1,nor->size)] = nor_recv->data[I2dm(j,1,nor_recv->size)];
	    nor->data[I2dm(cur_index,2,nor->size)] = nor_recv->data[I2dm(j,2,nor_recv->size)];
	    nor->data[I2dm(cur_index,3,nor->size)] = nor_recv->data[I2dm(j,3,nor_recv->size)];
	    curv->data[I2dm(cur_index,1,curv->size)] = curv_recv->data[I2dm(j,1,curv_recv->size)];
	    curv->data[I2dm(cur_index,2,curv->size)] = curv_recv->data[I2dm(j,2,curv_recv->size)];
	}
	emxFree_real_T(&nor_recv);
	emxFree_real_T(&curv_recv);
	free(lindex_recv);
    }

    free(recv_size);
    free(recv_req_list);

    MPI_Waitall(5*num_nbp, send_req_list, send_status_list);


    for (i = 1; i <= num_nbp; i++)
    {
	emxFree_real_T(&(buf_nor[I1dm(i)]));
	emxFree_real_T(&(buf_curv[I1dm(i)]));
	free(buf_lindex[I1dm(i)]);
    }

    free(buf_nor);
    free(buf_curv);
    free(buf_lindex);
    free(send_req_list);
    free(send_status_list);
}


void hpUpdateEstimatedNormal(hiPropMesh *mesh)
{
    int i, tag_send, tag_recv, j;
    int proc_send, proc_recv;
    int num_proc, rank;

    emxArray_real_T *est_nor = mesh->est_nor;
    emxArray_int32_T *nb_proc = mesh->nb_proc;

    hpPInfoList *ps_pinfo = mesh->ps_pinfo;
    hpPInfoNode *ps_pdata = ps_pinfo->pdata;
    int *ps_phead = ps_pinfo->head;


    MPI_Comm_size(MPI_COMM_WORLD, &num_proc);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int num_nbp = nb_proc->size[0];

    /* First build the buffer est_nor and corresponding lindex for send */
    emxArray_real_T **buf_enor = (emxArray_real_T **) calloc(num_nbp, sizeof(emxArray_real_T *));
    int **buf_lindex = (int **) calloc(num_nbp, sizeof(int *));

    for (i = 1; i <= num_nbp; i++)
    {
	int nb_procid = nb_proc->data[I1dm(i)];
	int num_overlay_ps = 0;
	unsigned char *ps_flag = (unsigned char *) calloc(mesh->ps->size[0], sizeof(unsigned char));

	for (j = 1; j <= mesh->ps->size[0]; j++)
	{
	    int next_node = ps_phead[I1dm(j)];
	    if (next_node != rank)
		continue;
	    else
	    {
		next_node = ps_pdata[I1dm(next_node)].next;
		while(next_node != -1)
		{
		    if (ps_pdata[I1dm(next_node)].proc == nb_procid)
		    {
			ps_flag[I1dm(j)] = 1;
			num_overlay_ps++;
			break;
		    }
		    else
			next_node = ps_pdata[I1dm(next_node)].next;
		}

	    }
	}

	buf_enor[I1dm(i)] = emxCreate_real_T(num_overlay_ps, 3);
	buf_lindex[I1dm(i)]= (int *) calloc(num_overlay_ps, sizeof(int));

	emxArray_real_T *benor = buf_enor[I1dm(i)];
	int *blindex = buf_lindex[I1dm(i)];

	int cur_pos = 1;
	for (j = 1; j <= mesh->ps->size[0]; j++)
	{
	    if (ps_flag[I1dm(j)] == 1)
	    {
		benor->data[I2dm(cur_pos,1,benor->size)] = est_nor->data[I2dm(j,1,est_nor->size)];
		benor->data[I2dm(cur_pos,2,benor->size)] = est_nor->data[I2dm(j,2,est_nor->size)];
		benor->data[I2dm(cur_pos,3,benor->size)] = est_nor->data[I2dm(j,3,est_nor->size)];
		blindex[I1dm(cur_pos)] = j;
		cur_pos++;
	    }
	}
	free(ps_flag);
    }

    /* Send the information */
    MPI_Request *send_req_list = (MPI_Request *) malloc( 3*num_nbp*sizeof(MPI_Request) );

    MPI_Status *send_status_list = (MPI_Status *) malloc( 3*num_nbp*sizeof(MPI_Status) );

    MPI_Request *recv_req_list = (MPI_Request *) malloc ( num_nbp*sizeof(MPI_Request) );

    /* Stores the received array size */

    int *recv_size = (int *) calloc ( 2*num_nbp, sizeof(int));
    
    for (i = 1; i <= num_nbp; i++)
    {
	proc_send = nb_proc->data[I1dm(i)];
	tag_send = proc_send;
	isend2D_real_T(buf_enor[I1dm(i)], proc_send, tag_send, MPI_COMM_WORLD,
		       &(send_req_list[I1dm(i)]), &(send_req_list[I1dm(i)+num_nbp]));
	MPI_Isend(buf_lindex[I1dm(i)], (buf_enor[I1dm(i)])->size[0], MPI_INT, proc_send, tag_send+3, MPI_COMM_WORLD, &(send_req_list[I1dm(i)+2*num_nbp])); 
    }

    for (i = 1; i <= num_nbp; i++)
    {
	proc_recv = nb_proc->data[I1dm(i)];
	tag_recv = rank;
	MPI_Irecv(&(recv_size[2*I1dm(i)]), 2, MPI_INT, proc_recv, tag_recv+1, MPI_COMM_WORLD, &(recv_req_list[I1dm(i)]));
    }

    for (i = 1; i <= num_nbp; i++)
    {
	emxArray_real_T *nor_recv;
	int *lindex_recv;

	tag_recv = rank;
	MPI_Status recv_status1;
	MPI_Status recv_status2;
	MPI_Status recv_status3;
	int recv_index;
	
	MPI_Waitany(num_nbp, recv_req_list, &recv_index, &recv_status1);

	proc_recv = recv_status1.MPI_SOURCE;

	nor_recv = emxCreate_real_T(recv_size[2*recv_index], recv_size[2*recv_index+1]);
	lindex_recv = (int *) calloc(recv_size[2*recv_index], sizeof(int));

	MPI_Recv(nor_recv->data, recv_size[2*recv_index]*recv_size[2*recv_index+1], MPI_DOUBLE, proc_recv, tag_recv+2, MPI_COMM_WORLD, &recv_status2);
	MPI_Recv(lindex_recv, recv_size[2*recv_index], MPI_INT, proc_recv, tag_recv+3, MPI_COMM_WORLD, &recv_status3);

	for (j = 1; j <= recv_size[2*recv_index]; j++)
	{
	    int cur_index = lindex_recv[I1dm(j)];
	    est_nor->data[I2dm(cur_index,1,est_nor->size)] = nor_recv->data[I2dm(j,1,nor_recv->size)];
	    est_nor->data[I2dm(cur_index,2,est_nor->size)] = nor_recv->data[I2dm(j,2,nor_recv->size)];
	    est_nor->data[I2dm(cur_index,3,est_nor->size)] = nor_recv->data[I2dm(j,3,nor_recv->size)];
	}
	emxFree_real_T(&nor_recv);
	free(lindex_recv);
    }

    free(recv_size);
    free(recv_req_list);

    MPI_Waitall(3*num_nbp, send_req_list, send_status_list);


    for (i = 1; i <= num_nbp; i++)
    {
	emxFree_real_T(&(buf_enor[I1dm(i)]));
	free(buf_lindex[I1dm(i)]);
    }

    free(buf_enor);
    free(buf_lindex);
    free(send_req_list);
    free(send_status_list);

}

void hpBuildPsType(hiPropMesh *mesh)
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int num_ps = mesh->ps->size[0];
    hpPInfoNode *ps_pdata = mesh->ps_pinfo->pdata;
    int *ps_phead = mesh->ps_pinfo->head;
    int *ps_ptail = mesh->ps_pinfo->tail;

    if (mesh->ps_type != (emxArray_int32_T *) NULL )
	emxFree_int32_T(&(mesh->ps_type));

    mesh->ps_type = emxCreateND_int32_T(1, &num_ps);
    int i;

    for (i = 1; i <= num_ps; i++)
    {
	int cur_head = ps_phead[I1dm(i)];
	int cur_tail = ps_ptail[I1dm(i)];
	if (ps_pdata[I1dm(cur_head)].proc == rank) /* current proc is master */
	{
	    if (cur_head != cur_tail) /* If exists on other processor */
		mesh->ps_type->data[I1dm(i)] = 1;
	}
	else /* If current proc is not master */
	    mesh->ps_type->data[I1dm(i)] = 2;
    }
}

void hpBuildPartBdryGhost(hiPropMesh *mesh, emxArray_real_T *ring_size)
{

    hpReducePartBdryGhostRingSize(mesh, ring_size);

    int i;

    int num_nb_proc = mesh->nb_proc->size[0];

    emxArray_int32_T **ps_ring_proc = (emxArray_int32_T **) calloc(num_nb_proc, sizeof(emxArray_int32_T *));
    emxArray_int32_T **tris_ring_proc = (emxArray_int32_T **) calloc(num_nb_proc, sizeof(emxArray_int32_T *));
    emxArray_real_T **buffer_ps = (emxArray_real_T **) calloc(num_nb_proc, sizeof(emxArray_real_T *));
    emxArray_int32_T **buffer_tris = (emxArray_int32_T **) calloc(num_nb_proc, sizeof(emxArray_int32_T *));

    for (i = 1; i <= num_nb_proc; i++)
    {
	hpBuildPartBdryGhostPsTrisForSend(mesh, i, ring_size,
					  &(ps_ring_proc[I1dm(i)]), &(tris_ring_proc[I1dm(i)]), 
					  &(buffer_ps[I1dm(i)]), &(buffer_tris[I1dm(i)]));
    }


    hpCommPsTrisWithPInfo(mesh, ps_ring_proc, tris_ring_proc, buffer_ps, buffer_tris);

}


void hpReducePartBdryGhostRingSize(hiPropMesh *mesh, emxArray_real_T *ring_size)
{



}


void hpBuildPartBdryGhostPsTrisForSend(const hiPropMesh *mesh,
				       const int nb_proc_index,
				       const emxArray_real_T *num_ring,
				       emxArray_int32_T **ps_ring_proc,
				       emxArray_int32_T **tris_ring_proc,
				       emxArray_real_T **buffer_ps,
				       emxArray_int32_T **buffer_tris)
{
    // Get nring nb for partition boundary 
    // between current proc and all nb processors  
    // 
    // Point positions stored in k_i*3 double matrices buffer_ps[i] where
    // k = # of points in the n-ring buffer for mesh->nb_proc->data[i].
    // Triangle indices mapped to the index for buffer_ps[i] and stored
    // in buffer_tris[i];
    //
    
    int *ps_mapping = (int *) calloc(mesh->ps->size[0], sizeof(int));
    int j;

    {
	hpCollectPartBdryNRingTris(mesh, nb_proc_index, num_ring,
				   ps_ring_proc, tris_ring_proc);

	int num_ps_buffer = (*ps_ring_proc)->size[0];
	int num_tris_buffer = (*tris_ring_proc)->size[0];

	
	(*buffer_ps) = emxCreate_real_T(num_ps_buffer, 3);
	(*buffer_tris) = emxCreate_int32_T(num_tris_buffer, 3);

	for (j = 1; j <= num_ps_buffer; j++)
	{
	    int cur_buf_ps_index = (*ps_ring_proc)->data[I1dm(j)];
	    (*buffer_ps)->data[I2dm(j,1,(*buffer_ps)->size)] =
		mesh->ps->data[I2dm(cur_buf_ps_index,1,mesh->ps->size)];
	    (*buffer_ps)->data[I2dm(j,2,(*buffer_ps)->size)] =
		mesh->ps->data[I2dm(cur_buf_ps_index,2,mesh->ps->size)];
	    (*buffer_ps)->data[I2dm(j,3,(*buffer_ps)->size)] =
		mesh->ps->data[I2dm(cur_buf_ps_index,3,mesh->ps->size)];

	    ps_mapping[I1dm(cur_buf_ps_index)] = j;
	}
	for (j = 1; j <= num_tris_buffer; j++)
	{
	    int cur_buf_tris_index = (*tris_ring_proc)->data[I1dm(j)];
	    (*buffer_tris)->data[I2dm(j,1,(*buffer_tris)->size)] =
		ps_mapping[mesh->tris->data[I2dm(cur_buf_tris_index,1,mesh->tris->size)]-1];
	    (*buffer_tris)->data[I2dm(j,2,(*buffer_tris)->size)] =
		ps_mapping[mesh->tris->data[I2dm(cur_buf_tris_index,2,mesh->tris->size)]-1];
	    (*buffer_tris)->data[I2dm(j,3,(*buffer_tris)->size)] =
		ps_mapping[mesh->tris->data[I2dm(cur_buf_tris_index,3,mesh->tris->size)]-1];
	}

    }
    free(ps_mapping);
}

void hpCollectPartBdryNRingTris(const hiPropMesh *mesh,
				const int nb_proc_index,
				const emxArray_real_T *num_ring,
				emxArray_int32_T **out_ps,
				emxArray_int32_T **out_tris)
{
    int i, j;

    int num_ps = mesh->ps->size[0];
    int num_tris = mesh->tris->size[0];
    int max_b_numps = 128;
    int max_b_numtris = 256;

    hpPInfoNode *ps_pdata = mesh->ps_pinfo->pdata;
    hpPInfoNode *tris_pdata = mesh->tris_pinfo->pdata;

    int *ps_phead = mesh->ps_pinfo->head;
    int *tris_phead = mesh->tris_pinfo->head;

    int *nb_proc = mesh->nb_proc->data;

    emxArray_int32_T *tris = mesh->tris;

    int *tris_data = tris->data;

    emxArray_int32_T *part_bdry = mesh->part_bdry;

    int dst_proc = nb_proc[I1dm(nb_proc_index)];


    // For denote whether each ps and tris belongs to the 2-ring buffer for
    // in_psid. If ps_flag[I1dm(i)] = true, then point i is in the 2-ring
    // buffer, if tris_flag[I1dm(j)] = true, then triangle j is in the 2-ring
    // buffer 
    emxArray_boolean_T *ps_flag = emxCreateND_boolean_T(1, &num_ps);
    emxArray_boolean_T *tris_flag = emxCreateND_boolean_T(1, &num_tris);

    // Used for obtain_nring_surf, initialized as false 
    emxArray_boolean_T *in_vtags = emxCreateND_boolean_T(1, &num_ps);
    emxArray_boolean_T *in_ftags = emxCreateND_boolean_T(1, &num_tris);

    // Used for storing outputs of obtain_nring_surf 
    emxArray_int32_T *in_ngbvs = emxCreateND_int32_T(1, &max_b_numps);
    emxArray_int32_T *in_ngbfs = emxCreateND_int32_T(1, &max_b_numtris);

    int num_ps_ring, num_tris_ring;

    for (i = 1; i <= part_bdry->size[0]; i++)
    {
	int cur_ps = part_bdry->data[I1dm(i)];

	// Decide whether current boundary point is between dst_proc and
	// cur_proc

	unsigned char ps_do_send = 0;
	int ps_next_node = ps_phead[I1dm(cur_ps)];
	while (ps_next_node != -1)
	{
	    int ps_proc_id = ps_pdata[I1dm(ps_next_node)].proc;
	    if (ps_proc_id == dst_proc)
	    {
		ps_do_send = 1;
		break;
	    }
	    else
		ps_next_node = ps_pdata[I1dm(ps_next_node)].next;
	}


	if (ps_do_send == 1)
	{
	    obtain_nring_surf(cur_ps, num_ring->data[I1dm(i)], 0, mesh->tris, mesh->opphe,
		    mesh->inhe, in_ngbvs, in_vtags, in_ftags, 
		    in_ngbfs, &num_ps_ring, &num_tris_ring);
	}

	for (j = 1; j <= num_tris_ring; j++)
	{
	    //  Only send the triangles which does not exists on the nb proc
	    int tris_buf_index = in_ngbfs->data[I1dm(j)];
	    tris_flag->data[I1dm(tris_buf_index)] = true;

	    int next_node = tris_phead[I1dm(tris_buf_index)];
	    while(next_node != -1)
	    {
		int proc_id = tris_pdata[I1dm(next_node)].proc;
		// If also exists on other processor, do not send it
		if (proc_id == dst_proc)
		{
		    tris_flag->data[I1dm(tris_buf_index)] = false;
		    break;
		}
		else
		    next_node = tris_pdata[I1dm(next_node)].next;
	    }
	}
    }

    // Set the points needed to be send based on tris

    for (i = 1; i <= num_tris; i++)
    {
	if (tris_flag->data[I1dm(i)] == true)
	{
	    int bufpi[3];
	    bufpi[0] = tris_data[I2dm(i,1,tris->size)];
	    bufpi[1] = tris_data[I2dm(i,2,tris->size)];
	    bufpi[2] = tris_data[I2dm(i,3,tris->size)];
	    ps_flag->data[bufpi[0]-1] = true;
	    ps_flag->data[bufpi[1]-1] = true;
	    ps_flag->data[bufpi[2]-1] = true;
	}
    }


    // Get total number of ps and tris in n-ring nb 

    int num_ps_ring_all = 0;
    int num_tris_ring_all = 0;

    for (i = 1; i <= num_ps; i++)
    {
	if (ps_flag->data[I1dm(i)] == true)
	    num_ps_ring_all++;
    }

    for (i = 1; i <= num_tris; i++)
    {
	if (tris_flag->data[I1dm(i)] == true)
	    num_tris_ring_all++;
    }

    // Create n-ring ps and tris for output 
    (*out_ps) = emxCreateND_int32_T(1, &num_ps_ring_all);
    (*out_tris) = emxCreateND_int32_T(1, &num_tris_ring_all);

    // Fill the out_ps and out_tris based on the flags 
    j = 1;
    for (i = 1; i <= num_ps; i++)
    {
	if (ps_flag->data[I1dm(i)] == true)
	{
	    (*out_ps)->data[I1dm(j)] = i;
	    j++;
	}
    }

    j = 1;
    for (i = 1; i <= num_tris; i++)
    {
	if (tris_flag->data[I1dm(i)] == true)
	{
	    (*out_tris)->data[I1dm(j)] = i;
	    j++;
	}
    }

    emxFree_int32_T(&in_ngbvs);
    emxFree_int32_T(&in_ngbfs);

    emxFree_boolean_T(&in_vtags);
    emxFree_boolean_T(&in_ftags);

    emxFree_boolean_T(&ps_flag);
    emxFree_boolean_T(&tris_flag);
}



