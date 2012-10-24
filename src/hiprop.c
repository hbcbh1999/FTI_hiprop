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

void hpInitMesh(hiPropMesh **pmesh)
{
    hiPropMesh *mesh;
    *pmesh = (hiPropMesh*) calloc(1, sizeof(hiPropMesh));
    mesh = *pmesh;
    mesh->ps = (emxArray_real_T *) NULL;
    mesh->tris = (emxArray_int32_T *) NULL;
    mesh->nor = (emxArray_real_T *) NULL;

    mesh->nb_proc = (emxArray_int32_T *) NULL;
    mesh->ps_pinfo = (hpPInfoList *) NULL;
    mesh->tris_pinfo = (hpPInfoList *) NULL;

    mesh->ps_send_index = (emxArray_int32_T **) NULL;
    mesh->ps_recv_index = (emxArray_int32_T **) NULL;
    mesh->ps_send_buffer = (emxArray_real_T **) NULL;
    mesh->ps_recv_buffer = (emxArray_real_T **) NULL;

    mesh->opphe = (emxArray_int32_T *) NULL;
    mesh->inhe = (emxArray_int32_T *) NULL;
}

void hpFreeMeshAugmentInfo(hiPropMesh *pmesh)
{
    if (pmesh->opphe != ((emxArray_int32_T *) NULL))
	emxFree_int32_T(&(pmesh->opphe));
    if (pmesh->inhe != ((emxArray_int32_T *) NULL))
	emxFree_int32_T(&(pmesh->inhe));
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
}

void hpFreeMeshBasicInfo(hiPropMesh *pmesh)
{
    if( pmesh->ps != ((emxArray_real_T *) NULL) )
	emxFree_real_T(&(pmesh->ps));
    if( pmesh->tris != ((emxArray_int32_T *) NULL) )
	emxFree_int32_T(&(pmesh->tris));
    if( pmesh->nor != ((emxArray_real_T *) NULL) )
	emxFree_real_T(&(pmesh->nor));
}

void hpFreeMesh(hiPropMesh *pmesh)
{
    hpFreeMeshUpdateInfo(pmesh);
    hpFreeMeshParallelInfo(pmesh);
    hpFreeMeshAugmentInfo(pmesh);
    hpFreeMeshBasicInfo(pmesh);
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
	fprintf(file, "%lf %lf %lf\n", 
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
	fprintf(file, "%lf %lf %lf\n",
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
	int tag)
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
	int** tri_index = (int**) malloc(num_proc*sizeof(int*));
	for(i = 0; i<num_proc; i++)
	    tri_index[i] = (int*) malloc(num_tri[i]*sizeof(int));

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
	int** pt_local = (int**)malloc(num_proc*sizeof(int*));
	for(i = 0; i<num_proc; i++)
	{
	    pt_local[i] = (int*) malloc(total_num_pt * sizeof(int));
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

	    }
	    else
	    {
	    	send2D_int32_T(p_mesh[i]->tris, i, tag, MPI_COMM_WORLD);
	    	send2D_real_T(p_mesh[i]->ps, i, tag+5, MPI_COMM_WORLD);
		
		MPI_Send(pt_index[i], p_mesh[i]->ps->size[0], MPI_INT, i, tag+10, MPI_COMM_WORLD);
		MPI_Send(&total_num_pt, 1, MPI_INT, i, tag+11, MPI_COMM_WORLD);
		for (j = 0; j<num_proc; j++)
		    MPI_Send(pt_local[j], total_num_pt, MPI_INT, i, tag+12+j, MPI_COMM_WORLD);

	    	hpDeleteMesh(&p_mesh[i]);
	    }
	}

	/* free pointers */
	for (i = 0; i<num_proc; i++)
	{
	    free(tri_index[i]);
	    free(pt_index[i]);
	    free(pt_local[i]);
	}
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
	int* l2gindex = (int*) malloc(mesh->ps->size[0]*sizeof(int));
	int** g2lindex = (int**) malloc(num_proc*sizeof(int*));
	
	MPI_Recv(l2gindex, mesh->ps->size[0], MPI_INT, root, tag+10, MPI_COMM_WORLD, &recv_stat);
	MPI_Recv(&total_num_pt, 1, MPI_INT, root, tag+11, MPI_COMM_WORLD, &recv_stat);
	for(i = 0; i<num_proc; i++)
	{
	    g2lindex[i] = (int*) calloc(total_num_pt,sizeof(int));
	    MPI_Recv(g2lindex[i], total_num_pt, MPI_INT, root, tag+12+i, MPI_COMM_WORLD, &recv_stat);
	}

    	hpConstrPInfoFromGlobalLocalInfo(mesh, g2lindex, l2gindex, rank);
	free(l2gindex);
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

    MPI_Request *send_req_list1 = (MPI_Request *) malloc( (num_proc-1)*sizeof(MPI_Request) );
    MPI_Request *send_req_list2 = (MPI_Request *) malloc( (num_proc-1)*sizeof(MPI_Request) );

    MPI_Status *send_status_list1 = (MPI_Status *) malloc( (num_proc-1)*sizeof(MPI_Status) );
    MPI_Status *send_status_list2 = (MPI_Status *) malloc( (num_proc-1)*sizeof(MPI_Status) );

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
		    	   &(send_req_list1[j]), &(send_req_list2[j]));
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

    MPI_Waitall(num_proc-1, send_req_list1, send_status_list1);
    MPI_Waitall(num_proc-1, send_req_list2, send_status_list2);

    free(send_req_list1);
    free(send_req_list2);
    free(send_status_list1);
    free(send_status_list2);


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
	hpPInfoNode *new_pdata = calloc(new_max_len, sizeof(hpPInfoNode));
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

    MPI_Request *send_req_list1 = (MPI_Request *) malloc( num_nbp*sizeof(MPI_Request) );
    MPI_Request *send_req_list2 = (MPI_Request *) malloc( num_nbp*sizeof(MPI_Request) );

    MPI_Status *send_status_list1 = (MPI_Status *) malloc( num_nbp*sizeof(MPI_Status) );
    MPI_Status *send_status_list2 = (MPI_Status *) malloc( num_nbp*sizeof(MPI_Status) );

    MPI_Request *recv_req_list = (MPI_Request *) malloc ( num_nbp*sizeof(MPI_Request) );

    /* Stores the received array size */

    int *recv_size = (int *) calloc ( 2*num_nbp, sizeof(int));
    
    for (i = 1; i <= num_nbp; i++)
    {
	proc_send = nb_proc->data[I1dm(i)];
	tag_send = proc_send;
	isend2D_real_T(ps, proc_send, tag_send, MPI_COMM_WORLD,
		       &(send_req_list1[I1dm(i)]), &(send_req_list2[I1dm(i)]));
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

    MPI_Waitall(num_nbp, send_req_list1, send_status_list1);
    MPI_Waitall(num_nbp, send_req_list2, send_status_list2);

    free(send_req_list1);
    free(send_req_list2);
    free(send_status_list1);
    free(send_status_list2);

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

    MPI_Request *ps_send_req_list1 = (MPI_Request *) malloc( num_nbp*sizeof(MPI_Request) );
    MPI_Request *ps_send_req_list2 = (MPI_Request *) malloc( num_nbp*sizeof(MPI_Request) );

    MPI_Request *tris_send_req_list1 = (MPI_Request *) malloc( num_nbp*sizeof(MPI_Request) );
    MPI_Request *tris_send_req_list2 = (MPI_Request *) malloc( num_nbp*sizeof(MPI_Request) );

    MPI_Status *ps_send_status_list1 = (MPI_Status *) malloc( num_nbp*sizeof(MPI_Status) );
    MPI_Status *ps_send_status_list2 = (MPI_Status *) malloc( num_nbp*sizeof(MPI_Status) );

    MPI_Status *tris_send_status_list1 = (MPI_Status *) malloc( num_nbp*sizeof(MPI_Status) );
    MPI_Status *tris_send_status_list2 = (MPI_Status *) malloc( num_nbp*sizeof(MPI_Status) );

    MPI_Request *ps_recv_req_list = (MPI_Request *) malloc ( num_nbp*sizeof(MPI_Request));

    int *ps_recv_size = (int *) calloc (2*num_nbp, sizeof(int));
    
    for (i = 1; i <= num_nbp; i++)
    {
	proc_send = nb_proc->data[I1dm(i)];
	ps_tag_send = proc_send;
	tris_tag_send = proc_send + 10;
	isend2D_real_T(ps, proc_send, ps_tag_send, MPI_COMM_WORLD,
		       &(ps_send_req_list1[I1dm(i)]), &(ps_send_req_list2[I1dm(i)]));
	isend2D_int32_T(tris, proc_send, tris_tag_send, MPI_COMM_WORLD, 
			&(tris_send_req_list1[I1dm(i)]), &(tris_send_req_list2[I1dm(i)]));
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

    MPI_Waitall(num_nbp, ps_send_req_list1, ps_send_status_list1);
    MPI_Waitall(num_nbp, ps_send_req_list2, ps_send_status_list2);

    MPI_Waitall(num_nbp, tris_send_req_list1, tris_send_status_list1);
    MPI_Waitall(num_nbp, tris_send_req_list2, tris_send_status_list2);

    free(ps_send_req_list1);
    free(ps_send_req_list2);
    free(tris_send_req_list1);
    free(tris_send_req_list1);

    free(ps_send_status_list1);
    free(ps_send_status_list2);

    free(tris_send_status_list1);
    free(tris_send_status_list2);
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
    int* pt_buffer_length = (int*) calloc(num_proc, sizeof(int));
    for (i = 0; i< num_proc; i++)
	pt_buffer_length[i] = 0;
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
		pt_buffer_length[cur_proc]++;
		cur_node = mesh->ps_pinfo->pdata[I1dm(cur_node)].next;
	    }
	}
	else	/* the current proc is not the master, recv this point from master */
	    pt_buffer_length[master]++;
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
	if(cur_proc<rank)		/* this proc is not the master, recv. 
					 *  In this case, we need to sort the points on this proc, 
					 *  so allocate memory for remoteid */
	{
	    buffer_size[0] = pt_buffer_length[cur_proc];
	    mesh->ps_recv_index[cur_proc] = emxCreateND_int32_T(1, buffer_size);
	    mesh->ps_recv_buffer[cur_proc] = emxCreate_real_T(buffer_size[0], 3);
	    remoteid[cur_proc] = (int*) calloc(pt_buffer_length[cur_proc], sizeof(int));
	}
	else	/* this proc is the master, send */
	{
	    buffer_size[0] = pt_buffer_length[cur_proc];
	    mesh->ps_send_index[cur_proc] = emxCreateND_int32_T(1, buffer_size);
	    mesh->ps_send_buffer[cur_proc] = emxCreate_real_T(buffer_size[0], 3);
	}
    }


    int* cur_size;
    int* p_index = (int*) calloc(num_proc, sizeof(int));	/* index to the end of the list */
    int tmp_id;
    int tmp_pt;
    for(i = 0; i<num_proc; i++)
	p_index[i] = 1;
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

		mesh->ps_send_index[cur_proc]->data[I1dm(p_index[cur_proc])]= i;
		cur_size = mesh->ps_send_buffer[cur_proc]->size;
		mesh->ps_send_buffer[cur_proc]->data[I2dm(p_index[cur_proc],1,cur_size)]
		    = mesh->ps->data[I2dm(i, 1, mesh->ps->size)];
		mesh->ps_send_buffer[cur_proc]->data[I2dm(p_index[cur_proc],2,cur_size)]
		    = mesh->ps->data[I2dm(i, 2, mesh->ps->size)];
		mesh->ps_send_buffer[cur_proc]->data[I2dm(p_index[cur_proc],3,cur_size)]
		    = mesh->ps->data[I2dm(i, 3, mesh->ps->size)];

		p_index[cur_proc]++;
		cur_node = mesh->ps_pinfo->pdata[I1dm(cur_node)].next;
	    }
	}
	else	/* the current proc is not the master, recv this point from master, sorted by insertion sort */
	{

	    remoteid[master][I1dm(p_index[master])] = mesh->ps_pinfo->pdata[I1dm(cur_head)].lindex;
	    mesh->ps_recv_index[master]->data[I1dm(p_index[master])] = i;

	    if(p_index[master]>1)	/* sort by the key of remoteid[master], the value is mesh->ps_recv_index[master]->data[] */
	    {
		for (j = p_index[master]; j>=1; j-- )
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
	    p_index[master]++;
	}
    }

    free(pt_buffer_length);
    free(p_index);
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
    MPI_Request request[3];
    int dest;

    for (i = 0; i<num_nb_proc; i++)
    {
	dest = mesh->nb_proc->data[i];
	MPI_Isend(&num_pt,1,MPI_INT,dest,tag[0],MPI_COMM_WORLD, &request[0]);
	MPI_Isend(pt_save_flag, num_pt, MPI_INT, dest, tag[1], MPI_COMM_WORLD, &request[1]);
	MPI_Isend(pt_new_index, num_pt, MPI_INT, dest, tag[2], MPI_COMM_WORLD, &request[2]);
    }

    /* 5. Recv pt_save_flag and pt_new_index info from all neighbours and update mesh->ps_pinfo */
    int num_pt_to_recv;
    int** pt_flag = (int**) calloc(num_proc, sizeof(int*));
    int** pt_index = (int**) calloc(num_proc, sizeof(int*));
    int source;
    MPI_Status status;

    for(i = 0; i<num_proc; i++)
    {
	pt_flag[i] = (int*)NULL;
	pt_index[i] = (int*)NULL;
    }

    for (i = 0; i<num_nb_proc; i++)
    {
	source = mesh->nb_proc->data[i];
	MPI_Recv(&num_pt_to_recv, 1, MPI_INT, source, tag[0], MPI_COMM_WORLD, &status);
	pt_flag[source] = (int*) calloc(num_pt_to_recv, sizeof(int));
	pt_index[source] = (int*) calloc(num_pt_to_recv, sizeof(int));
	MPI_Recv(pt_flag[source], num_pt_to_recv, MPI_INT, source, tag[1], MPI_COMM_WORLD, &status);
	MPI_Recv(pt_index[source], num_pt_to_recv, MPI_INT, source, tag[2],MPI_COMM_WORLD, &status);
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
    for(i = 0; i<num_proc; i++)
    {
	free(pt_flag[i]);
	free(pt_index[i]);
    }
    free(pt_flag);
    free(pt_index);
    free(tris_to_save);
    free(tri_save_flag);

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

    /* 7. since no overlapping trianlges, free mesh->tris_pinfo */
    hpDeletePInfoList(&(mesh->tris_pinfo));

}

void hpCollectAllOverlayPs(const hiPropMesh *mesh, emxArray_int32_T **out_psid)
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

void hpBuildGhostPsTrisForSend(const hiPropMesh *mesh,
			       const real_T num_ring,
			       emxArray_int32_T **psid_proc,
			       emxArray_int32_T **ps_ring_proc,
			       emxArray_int32_T **tris_ring_proc,
			       emxArray_real_T **buffer_ps,
			       emxArray_int32_T **buffer_tris)
{
    /* Get nring nb between current proc and all nb processors and
     * updated temp pinfo for both ps and tris.
     *
     * Point positions stored in k_i*3 double matrices buffer_ps[i] where
     * k = # of points in the n-ring buffer for mesh->nb_proc->data[i].
     * Triangle indices mapped to the index for buffer_ps[i] and stored
     * in buffer_tris[i];
     * ps/tris temp pinfo are updated from original pinfo, adding the
     * information where the ps/tri is being sent to
     */
    int *ps_mapping = (int *) calloc(mesh->ps->size[0], sizeof(int));
    int num_nb_proc = mesh->nb_proc->size[0];
    int i,j;

    for (i = 1; i <= num_nb_proc; i++)
    {
	hpCollectNRingTris(mesh, psid_proc[I1dm(i)], num_ring,
			   &(ps_ring_proc[I1dm(i)]), &(tris_ring_proc[I1dm(i)]));

	int num_ps_buffer = (ps_ring_proc[I1dm(i)])->size[0];
	int num_tris_buffer = (tris_ring_proc[I1dm(i)])->size[0];

	
	buffer_ps[I1dm(i)] = emxCreate_real_T(num_ps_buffer, 3);
	buffer_tris[I1dm(i)] = emxCreate_int32_T(num_tris_buffer, 3);

	for (j = 1; j <= num_ps_buffer; j++)
	{
	    int cur_buf_ps_index = (ps_ring_proc[I1dm(i)])->data[I1dm(j)];
	    (buffer_ps[I1dm(i)])->data[I2dm(j,1,(buffer_ps[I1dm(i)])->size)] =
		mesh->ps->data[I2dm(cur_buf_ps_index,1,mesh->ps->size)];
	    (buffer_ps[I1dm(i)])->data[I2dm(j,2,(buffer_ps[I1dm(i)])->size)] =
		mesh->ps->data[I2dm(cur_buf_ps_index,2,mesh->ps->size)];
	    (buffer_ps[I1dm(i)])->data[I2dm(j,3,(buffer_ps[I1dm(i)])->size)] =
		mesh->ps->data[I2dm(cur_buf_ps_index,3,mesh->ps->size)];

	    ps_mapping[I1dm(cur_buf_ps_index)] = j;
	}
	for (j = 1; j <= num_tris_buffer; j++)
	{
	    int cur_buf_tris_index = (tris_ring_proc[I1dm(i)])->data[I1dm(j)];
	    (buffer_tris[I1dm(i)])->data[I2dm(j,1,(buffer_tris[I1dm(i)])->size)] =
		ps_mapping[mesh->tris->data[I2dm(cur_buf_tris_index,1,mesh->tris->size)]-1];
	    (buffer_tris[I1dm(i)])->data[I2dm(j,2,(buffer_tris[I1dm(i)])->size)] =
		ps_mapping[mesh->tris->data[I2dm(cur_buf_tris_index,2,mesh->tris->size)]-1];
	    (buffer_tris[I1dm(i)])->data[I2dm(j,3,(buffer_tris[I1dm(i)])->size)] =
		ps_mapping[mesh->tris->data[I2dm(cur_buf_tris_index,3,mesh->tris->size)]-1];
	}
	/************* Debugging output **********************************
	char rank_str[5];
	char nb_rank_str[5];
	right_flush(cur_proc,4,rank_str);
	right_flush(mesh->nb_proc->data[I1dm(i)], 4, nb_rank_str);
	char debug_out_name[250];
	sprintf(debug_out_name, "debugout-p%s-to-p%s.vtk", rank_str, nb_rank_str);
	hpDebugOutput(mesh, ps_ring_proc[I1dm(i)], tris_ring_proc[I1dm(i)], debug_out_name);
	******************************************************************/
    }
    free(ps_mapping);

}

void hpBuildGhostPsTrisPInfoForSend(const hiPropMesh *mesh,
					   emxArray_int32_T **ps_ring_proc,
					   emxArray_int32_T **tris_ring_proc,
					   int **buffer_ps_pinfo_tag,
					   int **buffer_ps_pinfo_lindex,
					   int **buffer_ps_pinfo_proc,
					   int **buffer_tris_pinfo_tag,
					   int **buffer_tris_pinfo_lindex,
					   int **buffer_tris_pinfo_proc,
					   int *ps_pinfo_len,
					   int *tris_pinfo_len)
{
    /* Fill and build the temp pinfo information on each master processor 
     * Step 1, For original pinfo, the new target processor is added as a new
     * node with proc = new_proc_id, lindex = -1 (unknown) to the tail
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
    
    int num_nb_proc = mesh->nb_proc->size[0];

    int i,j;

    /* Finish Step 1 and fill buffer_ps_pinfo_tag & buffer_tris_pinfo_tag */
    for (i = 1; i <= num_nb_proc; i++)
    {
	int num_ps_buffer = (ps_ring_proc[I1dm(i)])->size[0];
	int num_tris_buffer = (tris_ring_proc[I1dm(i)])->size[0];
	int target_proc_id = mesh->nb_proc->data[I1dm(i)];

	buffer_ps_pinfo_tag[I1dm(i)] = (int *) calloc(num_ps_buffer+1, sizeof(int));
	buffer_tris_pinfo_tag[I1dm(i)] = (int *) calloc(num_tris_buffer+1, sizeof(int));

	buffer_ps_pinfo_tag[I1dm(i)][0] = 0;
	buffer_tris_pinfo_tag[I1dm(i)][0] = 0;

	for (j = 1; j <= num_ps_buffer; j++)
	{
	    int num_pinfo_data = 0;
	    unsigned char overlay_flag = 0;
	    int cur_ps_index = (ps_ring_proc[I1dm(i)])->data[I1dm(j)];
	    int next_node = ps_pinfo->head[I1dm(cur_ps_index)];
	    while(next_node != -1)
	    {
		num_pinfo_data++;
		if (ps_pinfo->pdata[I1dm(next_node)].proc == target_proc_id)
		    overlay_flag = 1;

		next_node = ps_pinfo->pdata[I1dm(next_node)].next;
	    }
	    if (overlay_flag == 0)
	    {
		int cur_tail = ps_pinfo->tail[I1dm(cur_ps_index)];
		hpEnsurePInfoCapacity(ps_pinfo);
		int new_tail = ps_pinfo->allocated_len++;
		ps_pinfo->pdata[I1dm(new_tail)].next = -1;
		ps_pinfo->pdata[I1dm(new_tail)].lindex = -1;
		ps_pinfo->pdata[I1dm(new_tail)].proc = target_proc_id;
		ps_pinfo->pdata[I1dm(cur_tail)].next = new_tail;
		num_pinfo_data++;
	    }
	    buffer_ps_pinfo_length += num_pinfo_data;
	    buffer_ps_pinfo_tag[I1dm(i)][j] = buffer_ps_pinfo_length;

	}

	for (j = 1; j <= num_tris_buffer; j++)
	{
	    int num_pinfo_data = 0;
	    unsigned char overlay_flag = 0;
	    int cur_tri_index = (tris_ring_proc[I1dm(i)])->data[I1dm(j)];
	    int next_node = tris_pinfo->head[I1dm(cur_tri_index)];
	    while(next_node != -1)
	    {
		num_pinfo_data++;
		if (tris_pinfo->pdata[I1dm(next_node)].proc == target_proc_id)
		    overlay_flag = 1;
		
		next_node = tris_pinfo->pdata[I1dm(next_node)].next;
	    }
	    if (overlay_flag == 0)
	    {
		int cur_tail = tris_pinfo->tail[I1dm(cur_tri_index)];
		hpEnsurePInfoCapacity(tris_pinfo);
		int new_tail = tris_pinfo->allocated_len++;
		tris_pinfo->pdata[I1dm(new_tail)].next = -1;
		tris_pinfo->pdata[I1dm(new_tail)].lindex = -1;
		tris_pinfo->pdata[I1dm(new_tail)].proc = target_proc_id;
		tris_pinfo->pdata[I1dm(cur_tail)].next = new_tail;
		num_pinfo_data++;
	    }
	    buffer_tris_pinfo_length += num_pinfo_data;
	    buffer_tris_pinfo_tag[I1dm(i)][j] = buffer_tris_pinfo_length;
	}

    }

    (*ps_pinfo_len) = buffer_ps_pinfo_length;
    (*tris_pinfo_len) = buffer_tris_pinfo_length;

    /* Fill in the buffer_ps/tris_pinfo_lindex & buffer_ps/tris_pinfo_proc in order */
    for (i = 1; i <= num_nb_proc; i++)
    {
	int num_ps_buffer = (ps_ring_proc[I1dm(i)])->size[0];
	int num_tris_buffer = (tris_ring_proc[I1dm(i)])->size[0];

	buffer_ps_pinfo_lindex[I1dm(i)] = (int *) calloc(buffer_ps_pinfo_length, sizeof(int));
	buffer_ps_pinfo_proc[I1dm(i)] = (int *) calloc(buffer_ps_pinfo_length, sizeof(int));

	buffer_tris_pinfo_lindex[I1dm(i)] = (int *) calloc(buffer_tris_pinfo_length, sizeof(int));
	buffer_tris_pinfo_proc[I1dm(i)] = (int *) calloc(buffer_tris_pinfo_length, sizeof(int));

	int cur_ps_pinfo = 0;
	for (j = 1; j <= num_ps_buffer; j++)
	{
	    int cur_ps_index = (ps_ring_proc[I1dm(i)])->data[I1dm(j)];
	    int next_node = ps_pinfo->head[I1dm(cur_ps_index)];
	    while(next_node != -1)
	    {
		buffer_ps_pinfo_lindex[I1dm(i)][cur_ps_pinfo] = ps_pinfo->pdata[I1dm(next_node)].lindex;
		buffer_ps_pinfo_proc[I1dm(i)][cur_ps_pinfo] = ps_pinfo->pdata[I1dm(next_node)].proc;
		next_node = ps_pinfo->pdata[I1dm(next_node)].next;
		cur_ps_pinfo++;
	    }
	}

	int cur_tris_pinfo = 0;
	for (j = 1; j <= num_tris_buffer; j++)
	{
	    int cur_tri_index = (tris_ring_proc[I1dm(i)])->data[I1dm(j)];
	    int next_node = tris_pinfo->head[I1dm(cur_tri_index)];
	    while(next_node != -1)
	    {
		buffer_tris_pinfo_lindex[I1dm(i)][cur_tris_pinfo] = tris_pinfo->pdata[I1dm(next_node)].lindex;
		buffer_tris_pinfo_proc[I1dm(i)][cur_tris_pinfo] = tris_pinfo->pdata[I1dm(next_node)].proc;
		next_node = tris_pinfo->pdata[I1dm(next_node)].next;
		cur_tris_pinfo++;
	    }
	}
    }
}




void hpBuildNRingGhost(hiPropMesh *mesh, const real_T num_ring)
{
    int cur_proc;
    MPI_Comm_rank(MPI_COMM_WORLD, &cur_proc);
    int i;

    int num_nb_proc = mesh->nb_proc->size[0];


    emxArray_int32_T **psid_proc = (emxArray_int32_T **)
	calloc(num_nb_proc, sizeof(emxArray_int32_T *));

    /* Get the overlapping points for building up n-ring neighborhood */
    hpCollectAllOverlayPs(mesh, psid_proc);


    /* Set up the MPI_Request list */
    int num_all_rqst = 10*num_nb_proc;
    MPI_Request* rqst_list = (MPI_Request *) calloc(num_all_rqst, sizeof(MPI_Request));
    MPI_Status* status_list = (MPI_Status *) calloc(num_all_rqst, sizeof(MPI_Status));
    int cur_rqst = 0;

    /* Build n-ring neighborhood and send*/
    emxArray_int32_T **ps_ring_proc = (emxArray_int32_T **) calloc(num_nb_proc, sizeof(emxArray_int32_T *));
    emxArray_int32_T **tris_ring_proc = (emxArray_int32_T **) calloc(num_nb_proc, sizeof(emxArray_int32_T *));
    emxArray_real_T **buffer_ps = (emxArray_real_T **) calloc(num_nb_proc, sizeof(emxArray_real_T *));
    emxArray_int32_T **buffer_tris = (emxArray_int32_T **) calloc(num_nb_proc, sizeof(emxArray_int32_T *));
    
    hpBuildGhostPsTrisForSend(mesh, num_ring, psid_proc, ps_ring_proc, tris_ring_proc, buffer_ps, buffer_tris);

    int tag_ps = 0;
    int tag_tris = 10;
    for (i = 1; i <= num_nb_proc; i++)
    {
	isend2D_real_T(buffer_ps[I1dm(i)], mesh->nb_proc->data[I1dm(i)],
		       tag_ps, MPI_COMM_WORLD, &(rqst_list[cur_rqst]), &(rqst_list[cur_rqst+1]));
	cur_rqst += 2;

	isend2D_int32_T(buffer_tris[I1dm(i)], mesh->nb_proc->data[I1dm(i)],
		       tag_tris, MPI_COMM_WORLD, &(rqst_list[cur_rqst]), &(rqst_list[cur_rqst+1]));
	cur_rqst += 2;
    }

    /* Get pinfo for n-ring neighborhood and send*/
    int ps_pinfo_length, tris_pinfo_length;
    int **buffer_ps_pinfo_tag = (int **) calloc(num_nb_proc, sizeof(int *));
    int **buffer_ps_pinfo_lindex = (int **) calloc(num_nb_proc, sizeof(int *));
    int **buffer_ps_pinfo_proc = (int **) calloc(num_nb_proc, sizeof(int *));

    int **buffer_tris_pinfo_tag = (int **) calloc(num_nb_proc, sizeof(int *));
    int **buffer_tris_pinfo_lindex = (int **) calloc(num_nb_proc, sizeof(int *));
    int **buffer_tris_pinfo_proc = (int **) calloc(num_nb_proc, sizeof(int *));

    hpBuildGhostPsTrisPInfoForSend(mesh, ps_ring_proc, tris_ring_proc,
	    buffer_ps_pinfo_tag, buffer_ps_pinfo_lindex, buffer_ps_pinfo_proc,
	    buffer_tris_pinfo_tag, buffer_tris_pinfo_lindex, buffer_tris_pinfo_proc,
	    &ps_pinfo_length, &tris_pinfo_length);

    int tag_ps_pinfo1 = 50;
    int tag_ps_pinfo2 = 51;
    int tag_ps_pinfo3 = 52;

    int tag_tris_pinfo1 = 60;
    int tag_tris_pinfo2 = 61;
    int tag_tris_pinfo3 = 62;

    for (i = 1; i <= num_nb_proc; i++)
    {
	int num_buf_ps_send = (buffer_ps[I1dm(i)])->size[0];
	int num_buf_ps_pinfo_send = buffer_ps_pinfo_tag[I1dm(i)][num_buf_ps_send];

	MPI_Isend(buffer_ps_pinfo_tag[I1dm(i)], num_buf_ps_send+1, MPI_INT,
		  mesh->nb_proc->data[I1dm(i)], tag_ps_pinfo1, MPI_COMM_WORLD, &(rqst_list[cur_rqst++]));
	MPI_Isend(buffer_ps_pinfo_lindex[I1dm(i)], num_buf_ps_pinfo_send, MPI_INT,
		  mesh->nb_proc->data[I1dm(i)], tag_ps_pinfo2, MPI_COMM_WORLD, &(rqst_list[cur_rqst++]));
	MPI_Isend(buffer_ps_pinfo_proc[I1dm(i)], num_buf_ps_pinfo_send, MPI_INT,
		  mesh->nb_proc->data[I1dm(i)], tag_ps_pinfo3, MPI_COMM_WORLD, &(rqst_list[cur_rqst++]));

	int num_buf_tris_send = (buffer_tris[I1dm(i)])->size[0];
	int num_buf_tris_pinfo_send = buffer_tris_pinfo_tag[I1dm(i)][num_buf_tris_send];

	MPI_Isend(buffer_tris_pinfo_tag[I1dm(i)], num_buf_tris_send+1, MPI_INT,
		  mesh->nb_proc->data[I1dm(i)], tag_tris_pinfo1, MPI_COMM_WORLD, &(rqst_list[cur_rqst++]));
	MPI_Isend(buffer_tris_pinfo_lindex[I1dm(i)], num_buf_tris_pinfo_send, MPI_INT,
		  mesh->nb_proc->data[I1dm(i)], tag_tris_pinfo2, MPI_COMM_WORLD, &(rqst_list[cur_rqst++]));
	MPI_Isend(buffer_tris_pinfo_proc[I1dm(i)], num_buf_tris_pinfo_send, MPI_INT,
		  mesh->nb_proc->data[I1dm(i)], tag_tris_pinfo3, MPI_COMM_WORLD, &(rqst_list[cur_rqst++]));
    }

    /* Free the buffer ps and tris index */
    for (i = 1; i <= num_nb_proc; i++)
    {
	emxFree_int32_T(&(psid_proc[I1dm(i)]));
	emxFree_int32_T(&(ps_ring_proc[I1dm(i)]));
	emxFree_int32_T(&(tris_ring_proc[I1dm(i)]));
    }
    free(psid_proc);
    free(ps_ring_proc);
    free(tris_ring_proc);

    /* Receive buffer points and tris with temp pinfo */

    emxArray_real_T **buffer_ps_recv = (emxArray_real_T **) calloc(num_nb_proc, sizeof(emxArray_real_T *));
    emxArray_int32_T **buffer_tris_recv = (emxArray_int32_T **) calloc(num_nb_proc, sizeof(emxArray_int32_T *));

    int **buf_ppinfo_tag_recv = (int **) calloc(num_nb_proc, sizeof(int *));
    int **buf_ppinfo_lindex_recv = (int **) calloc(num_nb_proc, sizeof(int *));
    int **buf_ppinfo_proc_recv = (int **) calloc(num_nb_proc, sizeof(int *));

    int **buf_tpinfo_tag_recv = (int **) calloc(num_nb_proc, sizeof(int *));
    int **buf_tpinfo_lindex_recv = (int **) calloc(num_nb_proc, sizeof(int *));
    int **buf_tpinfo_proc_recv = (int **) calloc(num_nb_proc, sizeof(int *));

    for (i = 1; i <= num_nb_proc; i++)
    {
	int num_buf_ps_recv, num_buf_tris_recv;
	
	int num_buf_ps_pinfo_recv;
	int num_buf_tris_pinfo_recv;
	MPI_Status tmp_status;

	recv2D_real_T(&(buffer_ps_recv[I1dm(i)]), mesh->nb_proc->data[I1dm(i)], tag_ps, MPI_COMM_WORLD);
	recv2D_int32_T(&(buffer_tris_recv[I1dm(i)]), mesh->nb_proc->data[I1dm(i)], tag_tris, MPI_COMM_WORLD);

	num_buf_ps_recv = (buffer_ps_recv[I1dm(i)])->size[0];
	num_buf_tris_recv = (buffer_tris_recv[I1dm(i)])->size[0];

	/* Recv ps pinfo */
	buf_ppinfo_tag_recv[I1dm(i)] = (int *) calloc(num_buf_ps_recv+1, sizeof(int));

	MPI_Recv(buf_ppinfo_tag_recv[I1dm(i)], num_buf_ps_recv+1, MPI_INT, mesh->nb_proc->data[I1dm(i)],
		 tag_ps_pinfo1, MPI_COMM_WORLD, &tmp_status);
	
	num_buf_ps_pinfo_recv = buf_ppinfo_tag_recv[I1dm(i)][num_buf_ps_recv];

	buf_ppinfo_lindex_recv[I1dm(i)] = (int *) calloc(num_buf_ps_pinfo_recv, sizeof(int));
	buf_ppinfo_proc_recv[I1dm(i)] = (int *) calloc(num_buf_ps_pinfo_recv, sizeof(int));

	MPI_Recv(buf_ppinfo_lindex_recv[I1dm(i)], num_buf_ps_pinfo_recv, MPI_INT, mesh->nb_proc->data[I1dm(i)],
		 tag_ps_pinfo2, MPI_COMM_WORLD, &tmp_status);
	MPI_Recv(buf_ppinfo_proc_recv[I1dm(i)], num_buf_ps_pinfo_recv, MPI_INT, mesh->nb_proc->data[I1dm(i)],
		 tag_ps_pinfo3, MPI_COMM_WORLD, &tmp_status);

	/* Recv tris pinfo */
	buf_tpinfo_tag_recv[I1dm(i)] = (int *) calloc(num_buf_tris_recv+1, sizeof(int));

	MPI_Recv(buf_tpinfo_tag_recv[I1dm(i)], num_buf_tris_recv+1, MPI_INT, mesh->nb_proc->data[I1dm(i)],
		 tag_tris_pinfo1, MPI_COMM_WORLD, &tmp_status);
	
	num_buf_tris_pinfo_recv = buf_tpinfo_tag_recv[I1dm(i)][num_buf_tris_recv];

	buf_tpinfo_lindex_recv[I1dm(i)] = (int *) calloc(num_buf_tris_pinfo_recv, sizeof(int));
	buf_tpinfo_proc_recv[I1dm(i)] = (int *) calloc(num_buf_tris_pinfo_recv, sizeof(int));

	MPI_Recv(buf_tpinfo_lindex_recv[I1dm(i)], num_buf_tris_pinfo_recv, MPI_INT, mesh->nb_proc->data[I1dm(i)],
		 tag_tris_pinfo2, MPI_COMM_WORLD, &tmp_status);
	MPI_Recv(buf_tpinfo_proc_recv[I1dm(i)], num_buf_tris_pinfo_recv, MPI_INT, mesh->nb_proc->data[I1dm(i)],
		 tag_tris_pinfo3, MPI_COMM_WORLD, &tmp_status);

    }

    /* After receiving, could do the real job */

    hpAttachNRingGhostWithPInfo(mesh, buffer_ps_recv, buffer_tris_recv,
	    buf_ppinfo_tag_recv, buf_ppinfo_lindex_recv, buf_ppinfo_proc_recv,
	    buf_tpinfo_tag_recv, buf_tpinfo_lindex_recv, buf_tpinfo_proc_recv);


    /* Wait until all the array are sent */

    MPI_Waitall(num_all_rqst, rqst_list, status_list);

    /* Free the array for send */

    free(rqst_list);
    free(status_list);

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

    /* Free the array for recv */

    for (i = 1; i <= num_nb_proc; i++)
    {
	emxFree_real_T(&(buffer_ps_recv[I1dm(i)]));
	emxFree_int32_T(&(buffer_tris_recv[I1dm(i)]));

	free(buf_ppinfo_tag_recv[I1dm(i)]); 
	free(buf_ppinfo_lindex_recv[I1dm(i)]); 
	free(buf_ppinfo_proc_recv[I1dm(i)]); 

	free(buf_tpinfo_tag_recv[I1dm(i)]); 
	free(buf_tpinfo_lindex_recv[I1dm(i)]); 
	free(buf_tpinfo_proc_recv[I1dm(i)]); 
    }

    free(buffer_ps_recv);
    free(buffer_tris_recv);
    free(buf_ppinfo_tag_recv);
    free(buf_ppinfo_lindex_recv);
    free(buf_ppinfo_proc_recv);

    free(buf_tpinfo_tag_recv);
    free(buf_tpinfo_lindex_recv);
    free(buf_tpinfo_proc_recv);

    /* Update nb_proc information based on the new pinfo */

}



void hpAttachNRingGhostWithPInfo(const hiPropMesh *mesh,
				 emxArray_real_T **bps,
				 emxArray_int32_T **btris,
				 int **ppinfot,
				 int **ppinfol,
				 int **ppinfop,
				 int **tpinfot,
				 int **tpinfol,
				 int **tpinfop)
{
    /*
    int i;
    int num_nb_proc = mesh->nb_proc->size[0];
    int cur_proc;
    MPI_Comm_rank(MPI_COMM_WORLD, &cur_proc);

    emxArray_real_T *ps = mesh->ps;
    emxArray_int32_T *tris = mesh->tris;

    hpPInfoList *ps_pinfo = mesh->ps_pinfo;
    hpPInfoList *tris_pinfo = mesh->tris_pinfo;

    for (i = 1; i <= num_nb_proc; i++)
    {
	int from_proc = mesh->nb_proc->data[I1dm(i)];
	int num_buf_ps = bps[I1dm(i)]->size[0];
	int num_buf_tris = btris[I1dm(i)]->size[0];

	int *ps_map = (int *) calloc(num_buf_ps, sizeof(int));






	free(ps_map);
    }
    */

}

void hpCollectNRingTris(const hiPropMesh *mesh,
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

	ps_flag->data[I1dm(cur_ps)] = true; /*cur_ps itself in the list */

	for (j = 1; j <= num_ps_ring; j++)
	{
	    /* j-th point in the n-ring nb */
	    int ps_buf_index = in_ngbvs->data[I1dm(j)];
	    ps_flag->data[I1dm(ps_buf_index)] = true;
	}

	for (j = 1; j <= num_tris_ring; j++)
	{
	    /* j-th triangle in the n-ring nb */
	    int tris_buf_index = in_ngbfs->data[I1dm(j)];
	    tris_flag->data[I1dm(tris_buf_index)] = true;
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
	fprintf(file, "%lf %lf %lf\n",
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

