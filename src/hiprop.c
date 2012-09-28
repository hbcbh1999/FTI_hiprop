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
    *pmesh = (hiPropMesh*) malloc(sizeof(hiPropMesh));
    mesh = *pmesh;
    mesh->ps = (emxArray_real_T *) NULL;
    mesh->tris = (emxArray_int32_T *) NULL;
    mesh->nor = (emxArray_real_T *) NULL;
    mesh->nb_proc = (emxArray_int32_T *) NULL;
    mesh->ps_pinfo = (hpPInfoList *) NULL;
    mesh->tris_pinfo = (hpPInfoList *) NULL;
}

void hpFreeMeshUpdateInfo(hiPropMesh *pmesh)
{
    /*
    int num_nb_proc = pmesh->nb_proc->size[0];
    int i;
    if (pmesh->ps_send_index != ((emxArray_int32_T *) NULL) )
    {
	for (i = 0; i < num_nb_proc; i++)
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
    if (pmesh->ps_recv_index != ((emxArray_int32_T *) NULL) )
    {
	for (i = 0; i < num_nb_proc; i++)
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
    */
}

void hpFreeMeshParallelInfo(hiPropMesh *pmesh)
{
    if (pmesh->ps_pinfo != ((hpPInfoList *) NULL) )
    {
	if (pmesh->ps_pinfo->head != ((hpPInfoNode *) NULL) )
	    free(pmesh->ps_pinfo->head);
	free(pmesh->ps_pinfo);
	pmesh->ps_pinfo = (hpPInfoList *) NULL;
    }
    if (pmesh->tris_pinfo != ((hpPInfoList *) NULL) )
    {
	if (pmesh->tris_pinfo->head != ((hpPInfoNode *) NULL) )
	    free(pmesh->tris_pinfo->head);
	free(pmesh->tris_pinfo);
	pmesh->tris_pinfo = (hpPInfoList *) NULL;
    }

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

void hpFreeMesh(hiPropMesh **pmesh)
{
    hpFreeMeshUpdateInfo((*pmesh));
    hpFreeMeshParallelInfo((*pmesh));
    hpFreeMeshBasicInfo((*pmesh));

    free((*pmesh));

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
	fprintf(file, "%lf %lf %lf\n", points->data[I2dm(i,1,points->size)], points->data[I2dm(i,2,points->size)], points->data[I2dm(i,3,points->size)]);

    fprintf(file, "POLYGONS %d %d\n", num_tris, 4*num_tris);
    for (i = 1; i <= num_tris; i++)
	fprintf(file, "3 %d %d %d\n", tris->data[I2dm(i,1,tris->size)]-1, tris->data[I2dm(i,2,tris->size)]-1, tris->data[I2dm(i,3,tris->size)]-1);

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
	fprintf(file, "%lf %lf %lf\n", points->data[I2dm(i,1,points->size)], points->data[I2dm(i,2,points->size)], points->data[I2dm(i,3,points->size)]);

    fprintf(file, "CELLS %d %d\n", num_tris, 4*num_tris);
    for (i = 1; i <= num_tris; i++)
	fprintf(file, "3 %d %d %d\n", tris->data[I2dm(i,1,tris->size)]-1, tris->data[I2dm(i,2,tris->size)]-1, tris->data[I2dm(i,3,tris->size)]-1);

    fprintf(file, "CELL_TYPES %d\n", num_tris);
    for (i = 0; i<num_tris; i++)
	fprintf(file, "5\n");
    fclose(file);
    return 1;

}

int hpMetisPartMesh(hiPropMesh* mesh, const int nparts, 
	int** tri_part, int** pt_part)
{

    //to be consistent with Metis, idx_t denote integer numbers, real_t denote floating point numbers
    //in Metis, tri and points arrays all start from index 0, which is different from HiProp,
    // so we need to convert to Metis convention, the output tri_part and pt_part are in Metis convention

    printf("entered hpMetisPartMesh\n");
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
    int i,j,k;
    int rank, num_proc;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_proc);
    printf("Entered hpDistMesh proc %d, root = %d\n", rank, root);

    // calculate the partitioned mesh on the root, then send to other processors
    if (rank == root)
    {
	if(in_mesh==NULL)
	{
	    printf("No mesh to be distributed\n");
	    return 0;
	}

	// initialize an array of pointers to the partitioned meshes to be sent
	hiPropMesh** p_mesh = (hiPropMesh**)malloc(num_proc*sizeof(hiPropMesh*));
	for(i = 0; i<num_proc; i++)
	    hpInitMesh(&p_mesh[i]);

	// an array to store the number of triangles on each processor
	int* num_tri = (int*)malloc(num_proc*sizeof(int));	
	for(i = 0; i<num_proc; i++)
	    num_tri[i] = 0;
	// an array to store the number of points on each processor
	int* num_pt = (int*)malloc(num_proc*sizeof(int));
	for(i = 0; i<num_proc; i++)
	    num_pt[i] = 0;

	int total_num_tri = in_mesh->tris->size[0];
	int total_num_pt = in_mesh->ps->size[0];

	// calculate the number of triangles on each proc
	for(i = 0; i < total_num_tri; i++)
	    num_tri[tri_part[i]]++;

	for(i = 0; i< num_proc; i++)
	{
	    printf("num_tri[%d] = %d\n", i, num_tri[i]);
	    (p_mesh[i]->tris) = emxCreate_int32_T(num_tri[i], 3);
	}


	// calculate the list of global index of triangles existing on each proc
	// tri_index[rank][i] is the global index of the ith tri on the ranked proc
	// all indices for tris here start from 0 for easy of coding
	int** tri_index = (int**) malloc(num_proc*sizeof(int*));
	for(i = 0; i<num_proc; i++)
	    tri_index[i] = (int*) malloc(num_tri[i]*sizeof(int));

	// fill tri_index by looping over all tris
	int* p = (int*)malloc(num_proc*sizeof(int));	// pointer to the end of the list
	for(i = 0; i< num_proc; i++)
	    p[i] = 0;
	int tri_rk;	// the proc rank of the current tri
	for(i = 0; i<total_num_tri; i++)
	{
	    tri_rk = tri_part[i];
	    tri_index[tri_rk][p[tri_rk]] = i;
	    p[tri_rk]++;
	}

	// construct an index table to store the local index of every point
	// all indices start from 0 for easy of coding
	// if pt_local[i][j] = -1, point[j] is not on proc[i], 
	// if pt_local[i][j] = m >= 0, the local index of point[j] on proc[i] is m.
	// looks space and time consuming, however easy to convert between globle and local index of points
	int** pt_local = (int**)malloc(num_proc*sizeof(int*));
	for(i = 0; i<num_proc; i++)
	{
	    pt_local[i] = (int*) malloc(total_num_pt * sizeof(int));
	    for(j = 0; j<total_num_pt; j++)
		pt_local[i][j] = -1;	//initialize to -1
	}

	// fill in pt_local table, calculate num_pt[] on each proc at the same time
	for (i = 0; i<total_num_pt; i++)
	{
	    for(j = 0; j<num_proc; j++)
		for(k = 0; k<num_tri[j]; k++)
		    if((in_mesh->tris->data[I2dm(tri_index[j][k]+1,1,in_mesh->tris->size)]==(i+1))
			    ||(in_mesh->tris->data[I2dm(tri_index[j][k]+1,2,in_mesh->tris->size)]==(i+1))
			    ||(in_mesh->tris->data[I2dm(tri_index[j][k]+1,3,in_mesh->tris->size)]==(i+1)))
		    {
			pt_local[j][i] = num_pt[j];
			num_pt[j]++;
			break;
		    }
	}
	for(i = 0; i<num_proc; i++)
	    printf("num_pt[%d] = %d\n", i, num_pt[i]);

	// fill in p_mesh[]->tris->data[] according to pt_local table
	int global_index;
	for( i = 0; i<num_proc; i++)
	{
	    for(j = 0; j<num_tri[i]; j++)
	    {
		global_index = in_mesh->tris->data[I2dm(tri_index[i][j]+1,1,in_mesh->tris->size)]-1;
		p_mesh[i]->tris->data[I2dm(j+1,1,p_mesh[i]->tris->size)] = pt_local[i][global_index]+1;

		global_index = in_mesh->tris->data[I2dm(tri_index[i][j]+1,2,in_mesh->tris->size)]-1;
		p_mesh[i]->tris->data[I2dm(j+1,2,p_mesh[i]->tris->size)] = pt_local[i][global_index]+1;

		global_index = in_mesh->tris->data[I2dm(tri_index[i][j]+1,3,in_mesh->tris->size)]-1;
		p_mesh[i]->tris->data[I2dm(j+1,3,p_mesh[i]->tris->size)] = pt_local[i][global_index]+1;
	    }
	}

	// pt_index is similar to tri_index
	// indices start from 0
	// pt_index[rank][i] is the global index of the ith point on the ranked proc
	// constructed using pt_local
	int** pt_index = (int**) malloc(num_proc*sizeof(int*));
	for(i = 0; i<num_proc; i++)
	{
	    pt_index[i] = (int*) malloc(num_pt[i]*sizeof(int));
	    for(j = 0; j<num_pt[i]; j++)
	    {
		for(k = 0; k<total_num_pt; k++)
		{
		    if(pt_local[i][k] == j)
			break;
		}
		if(k==total_num_pt)
		{
		    printf("Cannot find the point global index error!\n");
		    return 0;
		}
		else
		    pt_index[i][j] = k;
	    }
	}

	// finally, get in p_mesh[]->ps, :)
	for(i = 0; i< num_proc; i++)
	    (p_mesh[i]->ps) = emxCreate_real_T(num_pt[i], 3);
	// fill in p_mesh[]->ps->data with pt_index
	for (i = 0; i<num_proc; i++)
	{
	    for(j=0; j<num_pt[i]; j++)
	    {
		p_mesh[i]->ps->data[I2dm(j+1,1,p_mesh[i]->ps->size)] 
		    = in_mesh->ps->data[I2dm(pt_index[i][j]+1,1,in_mesh->ps->size)];
		p_mesh[i]->ps->data[I2dm(j+1,2,p_mesh[i]->ps->size)] 
		    = in_mesh->ps->data[I2dm(pt_index[i][j]+1,2,in_mesh->ps->size)];
		p_mesh[i]->ps->data[I2dm(j+1,3,p_mesh[i]->ps->size)] 
		    = in_mesh->ps->data[I2dm(pt_index[i][j]+1,3,in_mesh->ps->size)];
	    }
	}

	// communication
	for(i = 0; i<num_proc; i++)
	{
	    if(i==rank)
	    {
		mesh->ps = p_mesh[i]->ps;
		mesh->tris = p_mesh[i]->tris;
	    }
	    else
	    {
	    	send2D_int32_T(p_mesh[i]->tris, i, tag, MPI_COMM_WORLD);
	    	send2D_real_T(p_mesh[i]->ps, i, tag+5, MPI_COMM_WORLD);
	    	hpFreeMesh(&p_mesh[i]);
	    }
	}

	// free pointers
	for (i = 0; i<num_proc; i++)
	{
	    free(pt_index[i]);
	    free(tri_index[i]);
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

    else	//for other proc, receive the mesh info
    {
	recv2D_int32_T(&(mesh->tris),root, tag, MPI_COMM_WORLD);
	recv2D_real_T(&(mesh->ps),root, tag+5, MPI_COMM_WORLD);
    }
    printf("Leaving hpDistMesh proc %d\n", rank);
    return 1;
}

void hpGetNbProcListFromInput(hiPropMesh *mesh, int num_nb_proc, int *in_nb_proc)
{
    int i;
    int num_nb[1];
    num_nb[0] = num_nb_proc;

    mesh->nb_proc = emxCreateND_int32_T(1, num_nb);
    for (i = 1; i <= num_nb_proc; i++)
	mesh->nb_proc->data[i-1] = in_nb_proc[i-1];

}

void hpGetNbProcListAuto(hiPropMesh *mesh)
{

}

void hpInitPInfo(hiPropMesh *mesh)
{
    int i;
    int num_ps = mesh->ps->size[0];
    int num_tris = mesh->tris->size[0];

    int ps_estimate = 2*num_ps;
    int tris_estimate = 2*num_tris;

    int my_rank;

    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);


    mesh->ps_pinfo = (hpPInfoList *) malloc(sizeof(hpPInfoList));
    mesh->tris_pinfo = (hpPInfoList *) malloc(sizeof(hpPInfoList));

    mesh->ps_pinfo->head = (hpPInfoNode *) malloc(ps_estimate*sizeof(hpPInfoNode));
    mesh->tris_pinfo->head = (hpPInfoNode *) malloc(tris_estimate*sizeof(hpPInfoNode));

    mesh->ps_pinfo->max_len = ps_estimate;
    mesh->tris_pinfo->max_len = tris_estimate;

    for (i = 1; i <= num_ps; i++)
    {
	(mesh->ps_pinfo->head[i-1]).proc = my_rank;
	(mesh->ps_pinfo->head[i-1]).lindex = i;
	(mesh->ps_pinfo->head[i-1]).next = -1;
    }

    mesh->ps_pinfo->allocated_len = num_ps;
    mesh->tris_pinfo->allocated_len = num_tris;


}
