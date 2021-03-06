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
//#include "metis.h"


static void  hpBuildEstNbFromBdbox(const hiPropMesh *mesh, const double *all_bd_box, 
	emxArray_int32_T **new_nb_proc, emxArray_int8_T ***new_nb_shift_proc);


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
    mesh->nb_proc_shift = (emxArray_int8_T **) NULL;

    int i;
    for (i = 0; i < 3; ++i)
    {
	mesh->domain_len[i] = 0;
	mesh->has_periodic_boundary[i] = false;
    }

    mesh->part_bdry = (emxArray_int32_T *) NULL;
    mesh->ps_type = (emxArray_int32_T *) NULL;
    mesh->ps_pinfo = (hpPInfoList *) NULL;
    mesh->tris_pinfo = (hpPInfoList *) NULL;

    mesh->ps_send_index = (emxArray_int32_T **) NULL;
    mesh->ps_recv_index = (emxArray_int32_T **) NULL;
    mesh->ps_send_shift = (emxArray_int8_T **) NULL;

    mesh->opphe = (emxArray_int32_T *) NULL;
    mesh->inhe = (emxArray_int32_T *) NULL;
    mesh->est_nor = (emxArray_real_T *) NULL;

    mesh->nps_clean = 0;
    mesh->ntris_clean = 0;
    mesh->npspi_clean = 0;
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
    int num_proc;
    MPI_Comm_size(MPI_COMM_WORLD, &num_proc);

    int i;

    if (pmesh->ps_send_index != ((emxArray_int32_T **) NULL) )
    {
	for (i = 0; i < num_proc; i++)
	{
	    if (pmesh->ps_send_index[i] != ((emxArray_int32_T *) NULL) )
		emxFree_int32_T(&(pmesh->ps_send_index[i]));
	}
	free(pmesh->ps_send_index);
	pmesh->ps_send_index = (emxArray_int32_T **) NULL;
    }
    if (pmesh->ps_recv_index != ((emxArray_int32_T **) NULL) )
    {
	for (i = 0; i < num_proc; i++)
	{
	    if (pmesh->ps_recv_index[i] != ((emxArray_int32_T *) NULL) )
		emxFree_int32_T(&(pmesh->ps_recv_index[i]));
	}
	free(pmesh->ps_recv_index);
	pmesh->ps_recv_index = (emxArray_int32_T **) NULL;
    }
    if (pmesh->ps_send_shift != ((emxArray_int8_T **) NULL) )
    {
	for (i = 0; i < num_proc; i++)
	{
	    if (pmesh->ps_send_shift[i] != ((emxArray_int8_T *) NULL) )
		emxFree_int8_T(&(pmesh->ps_send_shift[i]));
	}
	free(pmesh->ps_send_shift);
	pmesh->ps_send_shift = (emxArray_int8_T **) NULL;
    }
}

void hpFreeMeshParallelInfo(hiPropMesh *pmesh)
{
    int i;
    hpDeletePInfoList(&(pmesh->ps_pinfo));
    hpDeletePInfoList(&(pmesh->tris_pinfo));

    if (pmesh->part_bdry != ((emxArray_int32_T *) NULL) )
	emxFree_int32_T(&(pmesh->part_bdry));

    if (pmesh->ps_type != ((emxArray_int32_T *) NULL) )
	emxFree_int32_T(&(pmesh->ps_type));

    if (pmesh->nb_proc_shift != ((emxArray_int8_T **) NULL))
    {
	for (i = 0; i < pmesh->nb_proc->size[0]; ++i)
	{
	    if (pmesh->nb_proc_shift[i] != (emxArray_int8_T *) NULL)
	    	emxFree_int8_T(&(pmesh->nb_proc_shift[i]));
	}
	free(pmesh->nb_proc_shift);
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
    if( pmesh->curv != ((emxArray_real_T *) NULL) )
	emxFree_real_T(&(pmesh->curv));
}

void hpFreeMesh(hiPropMesh *pmesh)
{
    hpFreeMeshUpdateInfo(pmesh);
    hpFreeMeshParallelInfo(pmesh);
    hpFreeMeshAugmentInfo(pmesh);
    hpFreeMeshBasicInfo(pmesh);
    pmesh->nps_clean = 0;
    pmesh->ntris_clean = 0;
    pmesh->npspi_clean = 0;
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
    if (file == NULL)
	printf("Cannot create the file %s\n", name);

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
    for (i = 0; i < (3*num_points); i++)
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
    for (i = 0; i < num_points; i++)
	for (j = 0; j < 3; j++)
	    mesh->ps->data[j*num_points+i] = pt_coord[i*3+j];

    /* triangles */
    (mesh->tris) = emxCreate_int32_T(num_tris, 3);
    for (i = 0; i < num_tris; i++)
	for (j = 0; j < 3; j++)
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



void hpGetNbProcListAuto(hiPropMesh *mesh)
{
    if (mesh->nb_proc != ((emxArray_int32_T *) NULL))
	emxFree_int32_T(&(mesh->nb_proc));

    int i, j, k, ki;
    int src, dst;
    int num_proc, rank;
    double eps = 1e-14;
    emxArray_real_T *ps = mesh->ps;
    real_T *ps_data = ps->data;
    int32_T *ps_size = ps->size;

    MPI_Comm_size(MPI_COMM_WORLD, &num_proc);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    double domain_len_x = mesh->domain_len[0];
    double domain_len_y = mesh->domain_len[1];
    double domain_len_z = mesh->domain_len[2];

    // First get the bounding box 
    // for each processor and reduce to all processor

    double bd_box[6];

   
    if (ps_size[0] >= 1)
    {
	bd_box[0]= ps_data[I2dm(1,1,ps_size)];
	bd_box[1] = bd_box[0];
	bd_box[2] = ps_data[I2dm(1,2,ps_size)];
	bd_box[3] = bd_box[2];
	bd_box[4] = ps_data[I2dm(1,3,ps_size)];
	bd_box[5] = bd_box[4];

	for (i = 2; i <= ps_size[0]; ++i)
	{
	    double x = ps_data[I2dm(i,1,ps_size)];
	    double y = ps_data[I2dm(i,2,ps_size)];
	    double z = ps_data[I2dm(i,3,ps_size)];

	    if (x < bd_box[0])
		bd_box[0] = x;
	    if (x > bd_box[1])
		bd_box[1] = x;
	    if (y < bd_box[2])
		bd_box[2] = y;
	    if (y > bd_box[3])
		bd_box[3] = y;
	    if (z < bd_box[4])
		bd_box[4] = z;
	    if (z > bd_box[5])
		bd_box[5] = z;
	}

	bd_box[0] -= eps;
	bd_box[1] += eps;
	bd_box[2] -= eps;
	bd_box[3] += eps;
	bd_box[4] -= eps;
	bd_box[5] += eps;
    }
    else
    {
	bd_box[0] = eps;
	bd_box[1] = -eps;
	bd_box[2] = eps;
	bd_box[3] = -eps;
	bd_box[4] = eps;
	bd_box[5] = -eps;
    }


    double *in_all_bd_box = (double *)calloc(6*num_proc, sizeof(double));
    double *all_bd_box = (double *)calloc(6*num_proc, sizeof(double));

    for (i = 0; i < 6; i++)
	in_all_bd_box[rank*6+i] = bd_box[i];

    MPI_Allreduce(in_all_bd_box, all_bd_box, 6*num_proc, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    free(in_all_bd_box);

    // Use bounding box to get estimated neighbor 

    boolean_T *nb_ptemp_est = (boolean_T *) calloc (num_proc, sizeof(boolean_T));
    int num_nbp_est = 0;


    for (i = 0; i < num_proc; ++i)
    {
	// Iterate all possible periodic boundary conditions 
	int x_iter = 1;
	int y_iter = 1;
	int z_iter = 1;

	if (mesh->has_periodic_boundary[0] == true)
	    x_iter = 3;
	if (mesh->has_periodic_boundary[1] == true)
	    y_iter = 3;
	if (mesh->has_periodic_boundary[2] == true)
	    z_iter = 3;

	int j1, j2, j3;
	double x_factor = 0.0;
	double y_factor = 0.0;
	double z_factor = 0.0;


	for (j1 = 0; j1 < x_iter; j1++)
	{
	    if (j1 == 1)
		x_factor = domain_len_x;
	    else if (j1 == 2)
		x_factor = -domain_len_x;
	    else
		x_factor = 0.0;

	    for (j2 = 0; j2 < y_iter; j2++)
	    {
		if (j2 == 1)
		    y_factor = domain_len_y;
		else if (j2 == 2)
		    y_factor = -domain_len_y;
		else
		    y_factor = 0.0;

		for (j3 = 0; j3 < z_iter; j3++)
		{
		    if (j3 == 1)
			z_factor = domain_len_z;
		    else if (j3 == 2)
			z_factor = -domain_len_z;
		    else
			z_factor = 0.0;


		    if (j1 == 0 && j2 == 0 && j3 == 0 && i == rank)
			continue;
		    // Actural loop starts here

		    double comxL = hpMax(bd_box[0] + x_factor, all_bd_box[i*6]);
		    double comxU = hpMin(bd_box[1] + x_factor, all_bd_box[i*6+1]);
		    double comyL = hpMax(bd_box[2] + y_factor, all_bd_box[i*6+2]);
		    double comyU = hpMin(bd_box[3] + y_factor, all_bd_box[i*6+3]);
		    double comzL = hpMax(bd_box[4] + z_factor, all_bd_box[i*6+4]);
		    double comzU = hpMin(bd_box[5] + z_factor, all_bd_box[i*6+5]);

		    if ( (comxL <= comxU) && (comyL <= comyU) && (comzL <= comzU) )
		    {
			if (nb_ptemp_est[i] == false)
			{
			    nb_ptemp_est[i] = true;
			    num_nbp_est++;
			}
		    }

		    // Actural loop ends here 
		}
	    }

	}

    }

    // Get estimated nb proc array and number of estimated nb proc 

    int *nb_ptemp_iter = (int *) calloc(num_nbp_est, sizeof(int));
    int nb_ptemp_cur = 0;

    for (i = 0; i < num_proc; i++)
    {
	if (nb_ptemp_est[i])
	    nb_ptemp_iter[nb_ptemp_cur++] = i;
    }

    free (nb_ptemp_est);

    // Allocate temporary storage for actual nb proc and nb proc shift array,
    // largest storage should be <= num_nbp_est 
    int *nb_ptemp = (int *) calloc (num_nbp_est, sizeof(int));

    for (j = 0; j < num_nbp_est; ++j)
	nb_ptemp[j] = -1;

    emxArray_int8_T **nb_shift_temp = (emxArray_int8_T **) calloc (num_nbp_est, sizeof(emxArray_int8_T *));
    int num_nbp = 0;

    MPI_Request *send_req_list = (MPI_Request *) malloc (3*num_nbp_est*sizeof(MPI_Request) );
    for (i = 0; i < 3*num_nbp_est; i++)
	send_req_list[i] = MPI_REQUEST_NULL;

    MPI_Status *send_status_list = (MPI_Status *) malloc(3*num_nbp_est*sizeof(MPI_Status) );
    MPI_Request *recv_req_list = (MPI_Request *) malloc(num_nbp_est*sizeof(MPI_Request) );

    int *num_ps_send = (int *) calloc(num_nbp_est, sizeof(int));
    double **ps_send = (double **) calloc(num_nbp_est, sizeof(double *));
    int8_T **ps_shift_send = (int8_T **) calloc(num_nbp_est, sizeof(int8_T *));


    // Iteration for each estimated nb proc

    for (i = 0; i < num_nbp_est; ++i)
    {
	num_ps_send[i] = 0;
	int target_id = nb_ptemp_iter[i];

	// Using bounding box, iterate for each possible shift to get 
	// the number of points for send 

	int x_iter = 1;
	int y_iter = 1;
	int z_iter = 1;

	if (mesh->has_periodic_boundary[0] == true)
	    x_iter = 3;
	if (mesh->has_periodic_boundary[1] == true)
	    y_iter = 3;
	if (mesh->has_periodic_boundary[2] == true)
	    z_iter = 3;

	int j1, j2, j3;
	double x_factor = 0.0;
	double y_factor = 0.0;
	double z_factor = 0.0;


	for (j1 = 0; j1 < x_iter; j1++)
	{
	    if (j1 == 1)
		x_factor = domain_len_x;
	    else if (j1 == 2)
		x_factor = -domain_len_x;
	    else
		x_factor = 0.0;

	    for (j2 = 0; j2 < y_iter; j2++)
	    {
		if (j2 == 1)
		    y_factor = domain_len_y;
		else if (j2 == 2)
		    y_factor = -domain_len_y;
		else
		    y_factor = 0.0;

		for (j3 = 0; j3 < z_iter; j3++)
		{
		    if (j3 == 1)
			z_factor = domain_len_z;
		    else if (j3 == 2)
			z_factor = -domain_len_z;
		    else
			z_factor = 0.0;


		    if (j1 == 0 && j2 == 0 && j3 == 0 && target_id == rank)
			continue;
		    // Actural loop starts here

		    double cur_bdbox_xL = all_bd_box[target_id*6] - x_factor;
		    double cur_bdbox_xU = all_bd_box[target_id*6+1] - x_factor;
		    double cur_bdbox_yL = all_bd_box[target_id*6+2] - y_factor;
		    double cur_bdbox_yU = all_bd_box[target_id*6+3] - y_factor;
		    double cur_bdbox_zL = all_bd_box[target_id*6+4] - z_factor;
		    double cur_bdbox_zU = all_bd_box[target_id*6+5] - z_factor;

		    for (j = 1; j <= ps_size[0]; ++j)
		    {
			double cur_x = ps_data[I2dm(j,1,ps_size)];
			double cur_y = ps_data[I2dm(j,2,ps_size)];
			double cur_z = ps_data[I2dm(j,3,ps_size)];

			if ( (cur_x >= cur_bdbox_xL) && (cur_x <= cur_bdbox_xU) && 
			     (cur_y >= cur_bdbox_yL) && (cur_y <= cur_bdbox_yU) &&
			     (cur_z >= cur_bdbox_zL) && (cur_z <= cur_bdbox_zU) 
			   )
			{
			    ++(num_ps_send[i]);
			}
		    }

		    // Actural loop ends here
		}
	    }
	}


	// Allocate storage for points and points shift for MPI send 

	ps_send[i] = (double *) calloc(3*num_ps_send[i], sizeof(double));
	ps_shift_send[i] = (int8_T *) calloc(3*num_ps_send[i], sizeof(int8_T));

	double *cur_ps_send = ps_send[i];
	int8_T *cur_ps_shift_send = ps_shift_send[i];

	// Re-iterate all possible shift to fill up the ps_send and ps_shift_send

	x_factor = 0.0;
	y_factor = 0.0;
	z_factor = 0.0;
	k = 0;
	ki = 0;


	for (j1 = 0; j1 < x_iter; j1++)
	{
	    if (j1 == 1)
		x_factor = domain_len_x;
	    else if (j1 == 2)
		x_factor = -domain_len_x;
	    else
		x_factor = 0.0;

	    for (j2 = 0; j2 < y_iter; j2++)
	    {
		if (j2 == 1)
		    y_factor = domain_len_y;
		else if (j2 == 2)
		    y_factor = -domain_len_y;
		else
		    y_factor = 0.0;

		for (j3 = 0; j3 < z_iter; j3++)
		{
		    if (j3 == 1)
			z_factor = domain_len_z;
		    else if (j3 == 2)
			z_factor = -domain_len_z;
		    else
			z_factor = 0.0;


		    if (j1 == 0 && j2 == 0 && j3 == 0 && target_id == rank)
			continue;
		    // Actural loop starts here 

		    double cur_bdbox_xL = all_bd_box[target_id*6] - x_factor;
		    double cur_bdbox_xU = all_bd_box[target_id*6+1] - x_factor;
		    double cur_bdbox_yL = all_bd_box[target_id*6+2] - y_factor;
		    double cur_bdbox_yU = all_bd_box[target_id*6+3] - y_factor;
		    double cur_bdbox_zL = all_bd_box[target_id*6+4] - z_factor;
		    double cur_bdbox_zU = all_bd_box[target_id*6+5] - z_factor;

		    for (j = 1; j <= ps_size[0]; ++j)
		    {
			double cur_x = ps_data[I2dm(j,1,ps_size)];
			double cur_y = ps_data[I2dm(j,2,ps_size)];
			double cur_z = ps_data[I2dm(j,3,ps_size)];

			if ( (cur_x >= cur_bdbox_xL) && (cur_x <= cur_bdbox_xU) && 
			     (cur_y >= cur_bdbox_yL) && (cur_y <= cur_bdbox_yU) &&
			     (cur_z >= cur_bdbox_zL) && (cur_z <= cur_bdbox_zU) 
			   )
			{
			    cur_ps_send[k++] = cur_x + x_factor;
			    cur_ps_send[k++] = cur_y + y_factor;
			    cur_ps_send[k++] = cur_z + z_factor;
			    cur_ps_shift_send[ki++] = (int8_T) j1;
			    cur_ps_shift_send[ki++] = (int8_T) j2;
			    cur_ps_shift_send[ki++] = (int8_T) j3;
			}
		    }

		    // Actural loop ends here
		}
	    }
	}
    }
    // Size array for receiving data     

    int *size_info = (int *) calloc(num_nbp_est, sizeof(int));


    for (i = 0; i < num_nbp_est; ++i)
    {
	dst = nb_ptemp_iter[i];
	
	MPI_Isend(&(num_ps_send[i]), 1, MPI_INT, dst, 1, MPI_COMM_WORLD, &(send_req_list[i]));
	if (num_ps_send[i] != 0)
	{
	    MPI_Isend(ps_send[i],3*num_ps_send[i], MPI_DOUBLE, dst, 2, MPI_COMM_WORLD, &(send_req_list[i+num_nbp_est])); 
	    MPI_Isend(ps_shift_send[i],3*num_ps_send[i], MPI_SIGNED_CHAR, dst, 3, MPI_COMM_WORLD, &(send_req_list[i+2*num_nbp_est])); 
	}
    }

    for (i = 0; i < num_nbp_est; ++i)
    {
	src = nb_ptemp_iter[i];
	MPI_Irecv(&(size_info[i]), 1, MPI_INT, src, 1, MPI_COMM_WORLD, &(recv_req_list[i]));
    }

    for (i = 0; i < num_nbp_est; ++i)
    {
	MPI_Status recv_status1;
	int recv_index;
	int source_id;

	// Receive any size info coming in next

	MPI_Waitany(num_nbp_est, recv_req_list, &recv_index, &recv_status1);

	source_id = recv_status1.MPI_SOURCE;
	if (size_info[recv_index] != 0)
	{
	    double *ps_recv;
	    int8_T *ps_shift_recv;

	    MPI_Status recv_status2;
	    MPI_Status recv_status3;

	    ps_recv = (double *) calloc(3*size_info[recv_index], sizeof(double));
	    ps_shift_recv = (int8_T *) calloc(3*size_info[recv_index], sizeof(int8_T));

	    // Block receiving the following information 
	    MPI_Recv(ps_recv, 3*size_info[recv_index], MPI_DOUBLE, source_id, 2, MPI_COMM_WORLD, &recv_status2);
	    MPI_Recv(ps_shift_recv, 3*size_info[recv_index], MPI_SIGNED_CHAR, source_id, 3, MPI_COMM_WORLD, &recv_status3);

	    boolean_T *flag = (boolean_T *) calloc (ps_size[0], sizeof (boolean_T));

	    // Iterate all the points to get the possible overlapping points
	    // considering all shift

	    int x_iter = 1;
	    int y_iter = 1;
	    int z_iter = 1;

	    if (mesh->has_periodic_boundary[0] == true)
		x_iter = 3;
	    if (mesh->has_periodic_boundary[1] == true)
		y_iter = 3;
	    if (mesh->has_periodic_boundary[2] == true)
		z_iter = 3;

	    int j1, j2, j3;
	    double x_factor = 0.0;
	    double y_factor = 0.0;
	    double z_factor = 0.0;


	    for (j1 = 0; j1 < x_iter; j1++)
	    {
		if (j1 == 1)
		    x_factor = domain_len_x;
		else if (j1 == 2)
		    x_factor = -domain_len_x;
		else
		    x_factor = 0.0;

		for (j2 = 0; j2 < y_iter; j2++)
		{
		    if (j2 == 1)
			y_factor = domain_len_y;
		    else if (j2 == 2)
			y_factor = -domain_len_y;
		    else
			y_factor = 0.0;

		    for (j3 = 0; j3 < z_iter; j3++)
		    {
			if (j3 == 1)
			    z_factor = domain_len_z;
			else if (j3 == 2)
			    z_factor = -domain_len_z;
			else
			    z_factor = 0.0;


			if (j1 == 0 && j2 == 0 && j3 == 0 && source_id == rank)
			    continue;
			// Actural loop starts here
			double recv_bdbox_xL = all_bd_box[6*source_id] - x_factor;
			double recv_bdbox_xU = all_bd_box[6*source_id+1] - x_factor;
			double recv_bdbox_yL = all_bd_box[6*source_id+2] - y_factor;
			double recv_bdbox_yU = all_bd_box[6*source_id+3] - y_factor;
			double recv_bdbox_zL = all_bd_box[6*source_id+4] - z_factor;
			double recv_bdbox_zU = all_bd_box[6*source_id+5] - z_factor;


			for (j = 1; j <= ps_size[0]; ++j)
			{
			    double current_x = ps_data[I2dm(j,1,ps_size)];
			    double current_y = ps_data[I2dm(j,2,ps_size)];
			    double current_z = ps_data[I2dm(j,3,ps_size)];

			    if ( (current_x >= recv_bdbox_xL) && (current_x <= recv_bdbox_xU) &&
				    (current_y >= recv_bdbox_yL) && (current_y <= recv_bdbox_yU) &&
				    (current_z >= recv_bdbox_zL) && (current_z <= recv_bdbox_zU)
			       )
			    {
				flag[j-1] = 1;
			    }
			}
		    }
		}
	    }

	    // Hash table for shifting f(a, b, c) = 100*a + 10*b + c
	    boolean_T *shift_flag = (boolean_T *) calloc(223, sizeof(boolean_T));

	    // Each received point only map to one point in ps, otherwise
	    // overlapping point exists in the original mesh
	    boolean_T *recv_flag = (boolean_T *) calloc (size_info[recv_index], sizeof(boolean_T));

	    for (j = 1; j <= ps_size[0]; ++j) 
	    {
		if (flag[j-1] == 1) //If possible overlapping
		{
		    double cur_x = ps_data[I2dm(j,1,ps_size)];
		    double cur_y = ps_data[I2dm(j,2,ps_size)];
		    double cur_z = ps_data[I2dm(j,3,ps_size)];

		    for (k = 0; k < size_info[recv_index]; ++k)
		    {
			if (recv_flag[k]) // If this received point has been mapped
			    continue;
			if ( (fabs(cur_x - ps_recv[k*3]) < eps) &&
				(fabs(cur_y - ps_recv[k*3+1]) < eps) &&
				(fabs(cur_z - ps_recv[k*3+2]) < eps)
			   )
			{
			    recv_flag[k] = 1;

			    if (num_nbp == 0)
			    {
				nb_ptemp[num_nbp] = source_id;
				num_nbp++;
			    }
			    else
			    {
				if (nb_ptemp[num_nbp-1] != source_id)
				{
				    nb_ptemp[num_nbp] = source_id;
				    num_nbp++;
				}
			    }

			    int hash_value = 100*ps_shift_recv[k*3+2] + 10*ps_shift_recv[k*3+1] + ps_shift_recv[k*3];

			    shift_flag[hash_value] = 1;

			}
		    }
		}
	    }
	    free(recv_flag);

	    int num_shift_cur_proc = 0;

	    for (k = 0; k < 223; k++)
	    {
		if (shift_flag[k])
		    num_shift_cur_proc++;
	    }

	    if (num_shift_cur_proc != 0)
	    {
		nb_shift_temp[num_nbp-1] = emxCreate_int8_T(num_shift_cur_proc, 3);

		emxArray_int8_T *cur_nb_shift = nb_shift_temp[num_nbp-1];

		ki = 1;

		for (k = 0; k < 223; k++)
		{
		    if (shift_flag[k])
		    {
			int cur_hash_value = k;
			int first_digit = cur_hash_value % 10;
			if (first_digit == 2)
			    cur_nb_shift->data[I2dm(ki,1,cur_nb_shift->size)] = -1;
			else
			    cur_nb_shift->data[I2dm(ki,1,cur_nb_shift->size)] = (int8_T) first_digit;

			cur_hash_value /= 10;
			int second_digit = cur_hash_value % 10;
			if (second_digit == 2)
			    cur_nb_shift->data[I2dm(ki,2,cur_nb_shift->size)] = -1;
			else
			    cur_nb_shift->data[I2dm(ki,2,cur_nb_shift->size)] = (int8_T) second_digit;

			cur_hash_value /= 10;
			int third_digit = cur_hash_value;
			if (third_digit == 2)
			    cur_nb_shift->data[I2dm(ki,3,cur_nb_shift->size)] = -1;
			else
			    cur_nb_shift->data[I2dm(ki,3,cur_nb_shift->size)] = (int8_T) third_digit;

			ki++;
		    }
		}
	    }

	    free(shift_flag);
	    free(flag);
	    free(ps_recv);
	    free(ps_shift_recv);
	}
    }

    free(size_info);
    free(recv_req_list);

    MPI_Waitall(3*num_nbp_est, send_req_list, send_status_list);

    free(send_req_list);
    free(send_status_list);

    for (i = 0; i < num_nbp_est; ++i)
    {
	free(ps_send[i]);
	free(ps_shift_send[i]);
    }
    free(num_ps_send);
    free(ps_send);
    free(ps_shift_send);
    free(all_bd_box);
    free(nb_ptemp_iter);
    mesh->nb_proc = emxCreateND_int32_T(1, &num_nbp);
    mesh->nb_proc_shift = (emxArray_int8_T **) calloc(num_nbp, sizeof(emxArray_int8_T *));

    for (i = 0; i < num_nbp; i++)
    {
	mesh->nb_proc->data[i] = nb_ptemp[i];
	mesh->nb_proc_shift[i] = nb_shift_temp[i];
    }

    free(nb_ptemp);
    free(nb_shift_temp);
}


void hpGetNbProcListFromInput(hiPropMesh *mesh, const int in_num_nbp, const int *in_nb_proc)
{
    if (mesh->nb_proc != ((emxArray_int32_T *) NULL))
	emxFree_int32_T(&(mesh->nb_proc));

    int i, j, k, ki;
    int src, dst;
    int num_proc, rank;
    double eps = 1e-14;
    emxArray_real_T *ps = mesh->ps;
    real_T *ps_data = ps->data;
    int32_T *ps_size = ps->size;

    MPI_Comm_size(MPI_COMM_WORLD, &num_proc);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    double domain_len_x = mesh->domain_len[0];
    double domain_len_y = mesh->domain_len[1];
    double domain_len_z = mesh->domain_len[2];

    // First get the bounding box 
    // for each processor and reduce to all processor

    double bd_box[6];

   
    if (ps_size[0] >= 1)
    {
	bd_box[0]= ps_data[I2dm(1,1,ps_size)];
	bd_box[1] = bd_box[0];
	bd_box[2] = ps_data[I2dm(1,2,ps_size)];
	bd_box[3] = bd_box[2];
	bd_box[4] = ps_data[I2dm(1,3,ps_size)];
	bd_box[5] = bd_box[4];

	for (i = 2; i <= ps_size[0]; ++i)
	{
	    double x = ps_data[I2dm(i,1,ps_size)];
	    double y = ps_data[I2dm(i,2,ps_size)];
	    double z = ps_data[I2dm(i,3,ps_size)];

	    if (x < bd_box[0])
		bd_box[0] = x;
	    if (x > bd_box[1])
		bd_box[1] = x;
	    if (y < bd_box[2])
		bd_box[2] = y;
	    if (y > bd_box[3])
		bd_box[3] = y;
	    if (z < bd_box[4])
		bd_box[4] = z;
	    if (z > bd_box[5])
		bd_box[5] = z;
	}

	bd_box[0] -= eps;
	bd_box[1] += eps;
	bd_box[2] -= eps;
	bd_box[3] += eps;
	bd_box[4] -= eps;
	bd_box[5] += eps;
    }
    else
    {
	bd_box[0] = eps;
	bd_box[1] = -eps;
	bd_box[2] = eps;
	bd_box[3] = -eps;
	bd_box[4] = eps;
	bd_box[5] = -eps;
    }


    double *in_all_bd_box = (double *)calloc(6*num_proc, sizeof(double));
    double *all_bd_box = (double *)calloc(6*num_proc, sizeof(double));

    for (i = 0; i < 6; i++)
	in_all_bd_box[rank*6+i] = bd_box[i];

    MPI_Allreduce(in_all_bd_box, all_bd_box, 6*num_proc, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    free(in_all_bd_box);

    // Use input for estimated neighbor 

    int *nb_ptemp_iter = (int *) calloc(in_num_nbp, sizeof(int));

    for (i = 0; i < in_num_nbp; i++)
    {
	nb_ptemp_iter[i] = in_nb_proc[i];
    }

    int num_nbp_est = in_num_nbp;

    // Allocate temporary storage for actual nb proc and nb proc shift array,
    // largest storage should be <= num_nbp_est 
    int *nb_ptemp = (int *) calloc (num_nbp_est, sizeof(int));

    for (j = 0; j < num_nbp_est; ++j)
	nb_ptemp[j] = -1;

    emxArray_int8_T **nb_shift_temp = (emxArray_int8_T **) calloc (num_nbp_est, sizeof(emxArray_int8_T *));
    int num_nbp = 0;

    MPI_Request *send_req_list = (MPI_Request *) malloc (3*num_nbp_est*sizeof(MPI_Request) );
    for (i = 0; i < 3*num_nbp_est; i++)
	send_req_list[i] = MPI_REQUEST_NULL;

    MPI_Status *send_status_list = (MPI_Status *) malloc(3*num_nbp_est*sizeof(MPI_Status) );

    MPI_Request *recv_req_list = (MPI_Request *) malloc(num_nbp_est*sizeof(MPI_Request) );

    int *num_ps_send = (int *) calloc(num_nbp_est, sizeof(int));
    double **ps_send = (double **) calloc(num_nbp_est, sizeof(double *));
    int8_T **ps_shift_send = (int8_T **) calloc(num_nbp_est, sizeof(int8_T *));


    // Iteration for each estimated nb proc

    for (i = 0; i < num_nbp_est; ++i)
    {
	num_ps_send[i] = 0;
	int target_id = nb_ptemp_iter[i];

	// Using bounding box, iterate for each possible shift to get 
	// the number of points for send 

	int x_iter = 1;
	int y_iter = 1;
	int z_iter = 1;

	if (mesh->has_periodic_boundary[0] == true)
	    x_iter = 3;
	if (mesh->has_periodic_boundary[1] == true)
	    y_iter = 3;
	if (mesh->has_periodic_boundary[2] == true)
	    z_iter = 3;

	int j1, j2, j3;
	double x_factor = 0.0;
	double y_factor = 0.0;
	double z_factor = 0.0;


	for (j1 = 0; j1 < x_iter; j1++)
	{
	    if (j1 == 1)
		x_factor = domain_len_x;
	    else if (j1 == 2)
		x_factor = -domain_len_x;
	    else
		x_factor = 0.0;

	    for (j2 = 0; j2 < y_iter; j2++)
	    {
		if (j2 == 1)
		    y_factor = domain_len_y;
		else if (j2 == 2)
		    y_factor = -domain_len_y;
		else
		    y_factor = 0.0;

		for (j3 = 0; j3 < z_iter; j3++)
		{
		    if (j3 == 1)
			z_factor = domain_len_z;
		    else if (j3 == 2)
			z_factor = -domain_len_z;
		    else
			z_factor = 0.0;


		    if (j1 == 0 && j2 == 0 && j3 == 0 && target_id == rank)
			continue;
		    // Actural loop starts here

		    double cur_bdbox_xL = all_bd_box[target_id*6] - x_factor;
		    double cur_bdbox_xU = all_bd_box[target_id*6+1] - x_factor;
		    double cur_bdbox_yL = all_bd_box[target_id*6+2] - y_factor;
		    double cur_bdbox_yU = all_bd_box[target_id*6+3] - y_factor;
		    double cur_bdbox_zL = all_bd_box[target_id*6+4] - z_factor;
		    double cur_bdbox_zU = all_bd_box[target_id*6+5] - z_factor;

		    for (j = 1; j <= ps_size[0]; ++j)
		    {
			double cur_x = ps_data[I2dm(j,1,ps_size)];
			double cur_y = ps_data[I2dm(j,2,ps_size)];
			double cur_z = ps_data[I2dm(j,3,ps_size)];

			if ( (cur_x >= cur_bdbox_xL) && (cur_x <= cur_bdbox_xU) && 
			     (cur_y >= cur_bdbox_yL) && (cur_y <= cur_bdbox_yU) &&
			     (cur_z >= cur_bdbox_zL) && (cur_z <= cur_bdbox_zU) 
			   )
			{
			    ++(num_ps_send[i]);
			}
		    }

		    // Actural loop ends here
		}
	    }
	}


	// Allocate storage for points and points shift for MPI send 

	ps_send[i] = (double *) calloc(3*num_ps_send[i], sizeof(double));
	ps_shift_send[i] = (int8_T *) calloc(3*num_ps_send[i], sizeof(int8_T));

	double *cur_ps_send = ps_send[i];
	int8_T *cur_ps_shift_send = ps_shift_send[i];

	// Re-iterate all possible shift to fill up the ps_send and ps_shift_send

	x_factor = 0.0;
	y_factor = 0.0;
	z_factor = 0.0;
	k = 0;
	ki = 0;


	for (j1 = 0; j1 < x_iter; j1++)
	{
	    if (j1 == 1)
		x_factor = domain_len_x;
	    else if (j1 == 2)
		x_factor = -domain_len_x;
	    else
		x_factor = 0.0;

	    for (j2 = 0; j2 < y_iter; j2++)
	    {
		if (j2 == 1)
		    y_factor = domain_len_y;
		else if (j2 == 2)
		    y_factor = -domain_len_y;
		else
		    y_factor = 0.0;

		for (j3 = 0; j3 < z_iter; j3++)
		{
		    if (j3 == 1)
			z_factor = domain_len_z;
		    else if (j3 == 2)
			z_factor = -domain_len_z;
		    else
			z_factor = 0.0;


		    if (j1 == 0 && j2 == 0 && j3 == 0 && target_id == rank)
			continue;
		    // Actural loop starts here 

		    double cur_bdbox_xL = all_bd_box[target_id*6] - x_factor;
		    double cur_bdbox_xU = all_bd_box[target_id*6+1] - x_factor;
		    double cur_bdbox_yL = all_bd_box[target_id*6+2] - y_factor;
		    double cur_bdbox_yU = all_bd_box[target_id*6+3] - y_factor;
		    double cur_bdbox_zL = all_bd_box[target_id*6+4] - z_factor;
		    double cur_bdbox_zU = all_bd_box[target_id*6+5] - z_factor;

		    for (j = 1; j <= ps_size[0]; ++j)
		    {
			double cur_x = ps_data[I2dm(j,1,ps_size)];
			double cur_y = ps_data[I2dm(j,2,ps_size)];
			double cur_z = ps_data[I2dm(j,3,ps_size)];

			if ( (cur_x >= cur_bdbox_xL) && (cur_x <= cur_bdbox_xU) && 
			     (cur_y >= cur_bdbox_yL) && (cur_y <= cur_bdbox_yU) &&
			     (cur_z >= cur_bdbox_zL) && (cur_z <= cur_bdbox_zU) 
			   )
			{
			    cur_ps_send[k++] = cur_x + x_factor;
			    cur_ps_send[k++] = cur_y + y_factor;
			    cur_ps_send[k++] = cur_z + z_factor;
			    cur_ps_shift_send[ki++] = (int8_T) j1;
			    cur_ps_shift_send[ki++] = (int8_T) j2;
			    cur_ps_shift_send[ki++] = (int8_T) j3;
			}
		    }

		    // Actural loop ends here
		}
	    }
	}
    }
  
    // Size array for receiving data     

    int *size_info = (int *) calloc(num_nbp_est, sizeof(int));


    for (i = 0; i < num_nbp_est; ++i)
    {
	dst = nb_ptemp_iter[i];
	
	MPI_Isend(&(num_ps_send[i]), 1, MPI_INT, dst, 1, MPI_COMM_WORLD, &(send_req_list[i]));

	if (num_ps_send[i] != 0)
	{
	    MPI_Isend(ps_send[i],3*num_ps_send[i], MPI_DOUBLE, dst, 2, MPI_COMM_WORLD, &(send_req_list[i+num_nbp_est])); 
	    MPI_Isend(ps_shift_send[i],3*num_ps_send[i], MPI_SIGNED_CHAR, dst, 3, MPI_COMM_WORLD, &(send_req_list[i+2*num_nbp_est])); 
	}
    }

    for (i = 0; i < num_nbp_est; ++i)
    {
	src = nb_ptemp_iter[i];
	MPI_Irecv(&(size_info[i]), 1, MPI_INT, src, 1, MPI_COMM_WORLD, &(recv_req_list[i]));
    }

    for (i = 0; i < num_nbp_est; ++i)
    {
	int recv_index;
	int source_id;
	MPI_Status recv_status1;

	// Receive any size info coming in next

	MPI_Waitany(num_nbp_est, recv_req_list, &recv_index, &recv_status1);
	source_id = recv_status1.MPI_SOURCE;

	if (size_info[recv_index] != 0)
	{
	    double *ps_recv;
	    int8_T *ps_shift_recv;

	    MPI_Status recv_status2;
	    MPI_Status recv_status3;

	    ps_recv = (double *) calloc(3*size_info[recv_index], sizeof(double));
	    ps_shift_recv = (int8_T *) calloc(3*size_info[recv_index], sizeof(int8_T));

	    // Block receiving the following information 
	    MPI_Recv(ps_recv, 3*size_info[recv_index], MPI_DOUBLE, source_id, 2, MPI_COMM_WORLD, &recv_status2);
	    MPI_Recv(ps_shift_recv, 3*size_info[recv_index], MPI_SIGNED_CHAR, source_id, 3, MPI_COMM_WORLD, &recv_status3);

	    boolean_T *flag = (boolean_T *) calloc (ps_size[0], sizeof (boolean_T));

	    // Iterate all the points to get the possible overlapping points
	    // considering all shift

	    int x_iter = 1;
	    int y_iter = 1;
	    int z_iter = 1;

	    if (mesh->has_periodic_boundary[0] == true)
		x_iter = 3;
	    if (mesh->has_periodic_boundary[1] == true)
		y_iter = 3;
	    if (mesh->has_periodic_boundary[2] == true)
		z_iter = 3;

	    int j1, j2, j3;
	    double x_factor = 0.0;
	    double y_factor = 0.0;
	    double z_factor = 0.0;


	    for (j1 = 0; j1 < x_iter; j1++)
	    {
		if (j1 == 1)
		    x_factor = domain_len_x;
		else if (j1 == 2)
		    x_factor = -domain_len_x;
		else
		    x_factor = 0.0;

		for (j2 = 0; j2 < y_iter; j2++)
		{
		    if (j2 == 1)
			y_factor = domain_len_y;
		    else if (j2 == 2)
			y_factor = -domain_len_y;
		    else
			y_factor = 0.0;

		    for (j3 = 0; j3 < z_iter; j3++)
		    {
			if (j3 == 1)
			    z_factor = domain_len_z;
			else if (j3 == 2)
			    z_factor = -domain_len_z;
			else
			    z_factor = 0.0;


			if (j1 == 0 && j2 == 0 && j3 == 0 && source_id == rank)
			    continue;
			// Actural loop starts here
			double recv_bdbox_xL = all_bd_box[6*source_id] - x_factor;
			double recv_bdbox_xU = all_bd_box[6*source_id+1] - x_factor;
			double recv_bdbox_yL = all_bd_box[6*source_id+2] - y_factor;
			double recv_bdbox_yU = all_bd_box[6*source_id+3] - y_factor;
			double recv_bdbox_zL = all_bd_box[6*source_id+4] - z_factor;
			double recv_bdbox_zU = all_bd_box[6*source_id+5] - z_factor;


			for (j = 1; j <= ps_size[0]; ++j)
			{
			    double current_x = ps_data[I2dm(j,1,ps_size)];
			    double current_y = ps_data[I2dm(j,2,ps_size)];
			    double current_z = ps_data[I2dm(j,3,ps_size)];

			    if ( (current_x >= recv_bdbox_xL) && (current_x <= recv_bdbox_xU) &&
				    (current_y >= recv_bdbox_yL) && (current_y <= recv_bdbox_yU) &&
				    (current_z >= recv_bdbox_zL) && (current_z <= recv_bdbox_zU)
			       )
			    {
				flag[j-1] = 1;
			    }
			}
		    }
		}
	    }

	    // Hash table for shifting f(a, b, c) = 100*a + 10*b + c
	    boolean_T *shift_flag = (boolean_T *) calloc(223, sizeof(boolean_T));

	    // Each received point only map to one point in ps, otherwise
	    // overlapping point exists in the original mesh
	    boolean_T *recv_flag = (boolean_T *) calloc (size_info[recv_index], sizeof(boolean_T));

	    for (j = 1; j <= ps_size[0]; ++j) 
	    {
		if (flag[j-1] == 1) //If possible overlapping
		{
		    double cur_x = ps_data[I2dm(j,1,ps_size)];
		    double cur_y = ps_data[I2dm(j,2,ps_size)];
		    double cur_z = ps_data[I2dm(j,3,ps_size)];

		    for (k = 0; k < size_info[recv_index]; ++k)
		    {
			if (recv_flag[k]) // If this received point has been mapped
			    continue;
			if ( (fabs(cur_x - ps_recv[k*3]) < eps) &&
				(fabs(cur_y - ps_recv[k*3+1]) < eps) &&
				(fabs(cur_z - ps_recv[k*3+2]) < eps)
			   )
			{
			    recv_flag[k] = 1;

			    if (num_nbp == 0)
			    {
				nb_ptemp[num_nbp] = source_id;
				num_nbp++;
			    }
			    else
			    {
				if (nb_ptemp[num_nbp-1] != source_id)
				{
				    nb_ptemp[num_nbp] = source_id;
				    num_nbp++;
				}
			    }

			    int hash_value = 100*ps_shift_recv[k*3+2] + 10*ps_shift_recv[k*3+1] + ps_shift_recv[k*3];

			    shift_flag[hash_value] = 1;

			}
		    }
		}
	    }
	    free(recv_flag);

	    int num_shift_cur_proc = 0;

	    for (k = 0; k < 223; k++)
	    {
		if (shift_flag[k] == 1)
		    num_shift_cur_proc++;
	    }

	    if (num_shift_cur_proc != 0)
	    {
		nb_shift_temp[num_nbp-1] = emxCreate_int8_T(num_shift_cur_proc, 3);

		emxArray_int8_T *cur_nb_shift = nb_shift_temp[num_nbp-1];

		ki = 1;

		for (k = 0; k < 223; k++)
		{
		    if (shift_flag[k] == 1)
		    {
			int cur_hash_value = k;
			int first_digit = cur_hash_value % 10;
			if (first_digit == 2)
			    cur_nb_shift->data[I2dm(ki,1,cur_nb_shift->size)] = -1;
			else
			    cur_nb_shift->data[I2dm(ki,1,cur_nb_shift->size)] = (int8_T) first_digit;

			cur_hash_value /= 10;
			int second_digit = cur_hash_value % 10;
			if (second_digit == 2)
			    cur_nb_shift->data[I2dm(ki,2,cur_nb_shift->size)] = -1;
			else
			    cur_nb_shift->data[I2dm(ki,2,cur_nb_shift->size)] = (int8_T) second_digit;

			cur_hash_value /= 10;
			int third_digit = cur_hash_value;
			if (third_digit == 2)
			    cur_nb_shift->data[I2dm(ki,3,cur_nb_shift->size)] = -1;
			else
			    cur_nb_shift->data[I2dm(ki,3,cur_nb_shift->size)] = (int8_T) third_digit;

			ki++;
		    }
		}
	    }

	    free(shift_flag);
	    free(flag);
	    free(ps_recv);
	    free(ps_shift_recv);
	}
    }
    free(size_info);
    free(recv_req_list);

    MPI_Waitall(3*num_nbp_est, send_req_list, send_status_list);

    free(send_req_list);
    free(send_status_list);

    for (i = 0; i < num_nbp_est; ++i)
    {
	free(ps_send[i]);
	free(ps_shift_send[i]);
    }
    free(num_ps_send);
    free(ps_send);
    free(ps_shift_send);
    free(all_bd_box);
    free(nb_ptemp_iter);
    
    mesh->nb_proc = emxCreateND_int32_T(1, &num_nbp);
    mesh->nb_proc_shift = (emxArray_int8_T **) calloc(num_nbp, sizeof(emxArray_int8_T *));

    for (i = 0; i < num_nbp; i++)
    {
	mesh->nb_proc->data[i] = nb_ptemp[i];
	mesh->nb_proc_shift[i] = nb_shift_temp[i];
    }

    free(nb_ptemp);
    free(nb_shift_temp);
}

void hpInitDomainBoundaryInfo(hiPropMesh *pmesh, const double *domain, const boolean_T *bdry)
{
    int i;

    for (i = 0; i < 3; ++i)
    {
	pmesh->domain_len[i] = 0;
	pmesh->has_periodic_boundary[i] = false;
    }


    // User specify domain and periodic boundary information

    pmesh->domain_len[0] = domain[0];
    pmesh->domain_len[1] = domain[1];
    pmesh->domain_len[2] = domain[2];
 

    pmesh->has_periodic_boundary[0] = bdry[0];
    pmesh->has_periodic_boundary[1] = bdry[1];
    pmesh->has_periodic_boundary[2] = bdry[2];
}

void hpInitPInfo(hiPropMesh *mesh)
{
    hpDeletePInfoList(&(mesh->ps_pinfo));
    hpDeletePInfoList(&(mesh->tris_pinfo));

    int i;
    int num_ps = mesh->ps->size[0];
    int num_tris = mesh->tris->size[0];


    // Initialize the Pinfo array to be twice as long as #of ps/tris
    // The Pinfo array would increase 10% each time needed
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

    hpPInfoNode *ps_pdata = mesh->ps_pinfo->pdata;
    int *ps_phead = mesh->ps_pinfo->head;
    int *ps_ptail = mesh->ps_pinfo->tail;

    hpPInfoNode *tris_pdata = mesh->tris_pinfo->pdata;
    int *tris_phead = mesh->tris_pinfo->head;
    int *tris_ptail = mesh->tris_pinfo->tail;


    for (i = 1; i <= num_ps; i++)
    {
	(ps_pdata[I1dm(i)]).proc = my_rank;
	(ps_pdata[I1dm(i)]).lindex = i;
	(ps_pdata[I1dm(i)]).shift[0] = 0;
	(ps_pdata[I1dm(i)]).shift[1] = 0;
	(ps_pdata[I1dm(i)]).shift[2] = 0;
	(ps_pdata[I1dm(i)]).next = -1;
	ps_phead[I1dm(i)] = i;
	ps_ptail[I1dm(i)] = i;
    }
    for (i = 1; i <= num_tris; i++)
    {
	(tris_pdata[I1dm(i)]).proc = my_rank;
	(tris_pdata[I1dm(i)]).lindex = i;
	(tris_pdata[I1dm(i)]).shift[0] = 0;
	(tris_pdata[I1dm(i)]).shift[1] = 0;
	(tris_pdata[I1dm(i)]).shift[2] = 0;
	(tris_pdata[I1dm(i)]).next = -1;
	tris_phead[I1dm(i)] = i;
	tris_ptail[I1dm(i)] = i;
    }
    mesh->ps_pinfo->allocated_len = num_ps;
    mesh->tris_pinfo->allocated_len = num_tris;

}

boolean_T hpEnsurePInfoCapacity(hpPInfoList *pinfo)
{
    if (pinfo->allocated_len >= pinfo->max_len)
    {
	double len_temp = (pinfo->max_len+1) * 1.1;
	int new_max_len = (int) (len_temp); // Increase 10%
	hpPInfoNode *new_pdata = (hpPInfoNode *) calloc(new_max_len, sizeof(hpPInfoNode));
	memcpy(new_pdata, pinfo->pdata, pinfo->allocated_len*sizeof(hpPInfoNode));
	free(pinfo->pdata);

	pinfo->pdata = new_pdata;
	pinfo->max_len = new_max_len;

	return 1;
    }
    else
	return 0;
}

void hpBuildPInfoNoOverlappingTris(hiPropMesh *mesh)
{
    int i, j, k, ki, ks, is;
    int src, dst;
    int num_proc, rank;
    double eps = 1e-14;
    emxArray_real_T *ps = mesh->ps;

    real_T *ps_data = ps->data;
    int32_T *ps_size = ps->size;

    emxArray_int32_T *nb_proc = mesh->nb_proc;
    int num_nbp = nb_proc->size[0];

    hpPInfoList *ps_pinfo = mesh->ps_pinfo;
    hpPInfoNode *ps_pdata = ps_pinfo->pdata;
    int *ps_phead = ps_pinfo->head;
    int *ps_ptail = ps_pinfo->tail;

    MPI_Comm_size(MPI_COMM_WORLD, &num_proc);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    double domain_len_x = mesh->domain_len[0];
    double domain_len_y = mesh->domain_len[1];
    double domain_len_z = mesh->domain_len[2];

    // First get the bounding box 
    // for each processor and reduce to all processor

    double bd_box[6];

    if (ps_size[0] > 0)
    {
	bd_box[0]= ps_data[I2dm(1,1,ps_size)];
	bd_box[1] = bd_box[0];
	bd_box[2] = ps_data[I2dm(1,2,ps_size)];
	bd_box[3] = bd_box[2];
	bd_box[4] = ps_data[I2dm(1,3,ps_size)];
	bd_box[5] = bd_box[4];

	for (i = 2; i <= ps_size[0]; ++i)
	{
	    double x = ps_data[I2dm(i,1,ps_size)];
	    double y = ps_data[I2dm(i,2,ps_size)];
	    double z = ps_data[I2dm(i,3,ps_size)];

	    if (x < bd_box[0])
		bd_box[0] = x;
	    if (x > bd_box[1])
		bd_box[1] = x;
	    if (y < bd_box[2])
		bd_box[2] = y;
	    if (y > bd_box[3])
		bd_box[3] = y;
	    if (z < bd_box[4])
		bd_box[4] = z;
	    if (z > bd_box[5])
		bd_box[5] = z;
	}

	bd_box[0] -= eps;
	bd_box[1] += eps;
	bd_box[2] -= eps;
	bd_box[3] += eps;
	bd_box[4] -= eps;
	bd_box[5] += eps;
    }
    else
    {
	bd_box[0] = eps;
	bd_box[1] = -eps;
	bd_box[2] = eps;
	bd_box[3] = -eps;
	bd_box[4] = eps;
	bd_box[5] = -eps;
    }


    double *in_all_bd_box = (double *)calloc(6*num_proc, sizeof(double));
    double *all_bd_box = (double *)calloc(6*num_proc, sizeof(double));

    for (i = 0; i < 6; i++)
	in_all_bd_box[rank*6+i] = bd_box[i];

    MPI_Allreduce(in_all_bd_box, all_bd_box, 6*num_proc, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    free(in_all_bd_box);

    MPI_Request *send_req_list = (MPI_Request *) malloc (4*num_nbp*sizeof(MPI_Request) );
    for (i = 0; i < 4*num_nbp; i++)
	send_req_list[i] = MPI_REQUEST_NULL;

    MPI_Status *send_status_list = (MPI_Status *) malloc(4*num_nbp*sizeof(MPI_Status) );

    MPI_Request *recv_req_list = (MPI_Request *) malloc(num_nbp*sizeof(MPI_Request) );

    int *num_ps_send = (int *) calloc(num_nbp, sizeof(int));
    double **ps_send = (double **) calloc(num_nbp, sizeof(double *));

    int **ps_index_send = (int **) calloc(num_nbp, sizeof(int *));
    int8_T **ps_shift_send = (int8_T **) calloc(num_nbp, sizeof(int8_T *));

    for (i = 0; i < num_nbp; ++i)
    {
	num_ps_send[i] = 0;
	int target_id = nb_proc->data[i];
	emxArray_int8_T *nb_shift = (mesh->nb_proc_shift)[i];

	int num_shifts = nb_shift->size[0];

	// First get the number of points needed to be sent
	for (is = 1; is <= num_shifts; ++is)
	{
	    double x_factor = 0.0;
	    double y_factor = 0.0;
	    double z_factor = 0.0;

	    int8_T sx, sy, sz;

	    sx = nb_shift->data[I2dm(is,1,nb_shift->size)];
	    sy = nb_shift->data[I2dm(is,2,nb_shift->size)];
	    sz = nb_shift->data[I2dm(is,3,nb_shift->size)];

	    x_factor = ((double) -sx) * domain_len_x;
	    y_factor = ((double) -sy) * domain_len_y;
	    z_factor = ((double) -sz) * domain_len_z;

	    double cur_bdbox_xL = all_bd_box[target_id*6] - x_factor;
	    double cur_bdbox_xU = all_bd_box[target_id*6+1] - x_factor;
	    double cur_bdbox_yL = all_bd_box[target_id*6+2] - y_factor;
	    double cur_bdbox_yU = all_bd_box[target_id*6+3] - y_factor;
	    double cur_bdbox_zL = all_bd_box[target_id*6+4] - z_factor;
	    double cur_bdbox_zU = all_bd_box[target_id*6+5] - z_factor;

	    for (j = 1; j <= ps_size[0]; ++j)
	    {
		double cur_x = ps_data[I2dm(j,1,ps_size)];
		double cur_y = ps_data[I2dm(j,2,ps_size)];
		double cur_z = ps_data[I2dm(j,3,ps_size)];

		if ( (cur_x >= cur_bdbox_xL) && (cur_x <= cur_bdbox_xU) && 
		     (cur_y >= cur_bdbox_yL) && (cur_y <= cur_bdbox_yU) &&
		     (cur_z >= cur_bdbox_zL) && (cur_z <= cur_bdbox_zU) 
		   )
		{
		    ++(num_ps_send[i]);
		}
	    }

	}

	//Allocate storage for ps postions, ps indices and ps shifts to send

	ps_send[i] = (double *) calloc(3*num_ps_send[i], sizeof(double));
	ps_index_send[i] = (int *) calloc(num_ps_send[i], sizeof(int));
	ps_shift_send[i] = (int8_T *) calloc(3*num_ps_send[i], sizeof(int8_T));

	double *cur_ps_send = ps_send[i];
	int *cur_ps_index_send = ps_index_send[i];
	int8_T *cur_ps_shift_send = ps_shift_send[i];

	k = 0;
	ki = 0;
	ks = 0;

	//Fill up the arrays for send
	
	if (num_ps_send[i] != 0)
	{
	    for (is = 1; is <= num_shifts; ++is)
	    {
		double x_factor = 0.0;
		double y_factor = 0.0;
		double z_factor = 0.0;

		int8_T sx, sy, sz;
		int8_T sx_send, sy_send, sz_send;

		sx = nb_shift->data[I2dm(is,1,nb_shift->size)];
		sy = nb_shift->data[I2dm(is,2,nb_shift->size)];
		sz = nb_shift->data[I2dm(is,3,nb_shift->size)];

		sx_send = -sx;
		sy_send = -sy;
		sz_send = -sz;

		x_factor = ((double) -sx)*domain_len_x;
		y_factor = ((double) -sy)*domain_len_y;
		z_factor = ((double) -sz)*domain_len_z;

		double cur_bdbox_xL = all_bd_box[target_id*6] - x_factor;
		double cur_bdbox_xU = all_bd_box[target_id*6+1] - x_factor;
		double cur_bdbox_yL = all_bd_box[target_id*6+2] - y_factor;
		double cur_bdbox_yU = all_bd_box[target_id*6+3] - y_factor;
		double cur_bdbox_zL = all_bd_box[target_id*6+4] - z_factor;
		double cur_bdbox_zU = all_bd_box[target_id*6+5] - z_factor;

		for (j = 1; j <= ps_size[0]; ++j)
		{
		    double cur_x = ps_data[I2dm(j,1,ps_size)];
		    double cur_y = ps_data[I2dm(j,2,ps_size)];
		    double cur_z = ps_data[I2dm(j,3,ps_size)];

		    if ( (cur_x >= cur_bdbox_xL) && (cur_x <= cur_bdbox_xU) && 
			    (cur_y >= cur_bdbox_yL) && (cur_y <= cur_bdbox_yU) &&
			    (cur_z >= cur_bdbox_zL) && (cur_z <= cur_bdbox_zU) 
		       )
		    {
			cur_ps_send[k++] = cur_x + x_factor;
			cur_ps_send[k++] = cur_y + y_factor;
			cur_ps_send[k++] = cur_z + z_factor;
			cur_ps_index_send[ki++] = j;
			cur_ps_shift_send[ks++] = sx_send;
			cur_ps_shift_send[ks++] = sy_send;
			cur_ps_shift_send[ks++] = sz_send;
		    }
		}

	    }
	}
    }

    int *size_info = (int *) calloc(num_nbp, sizeof(int));

    for (i = 0; i < num_nbp; ++i)
    {
	dst = nb_proc->data[i];

	int tag1 = 1;
	int tag2 = 2;
	int tag3 = 3;
	int tag4 = 4;

	MPI_Isend(&(num_ps_send[i]), 1, MPI_INT, dst, tag1, MPI_COMM_WORLD, &(send_req_list[i]));

	if (num_ps_send[i] != 0)
	{
	    MPI_Isend(ps_send[i], 3*num_ps_send[i], MPI_DOUBLE, dst, tag2, MPI_COMM_WORLD, &(send_req_list[i+num_nbp])); 
	    MPI_Isend(ps_index_send[i], num_ps_send[i], MPI_INT, dst, tag3, MPI_COMM_WORLD, &(send_req_list[i+2*num_nbp])); 
	    MPI_Isend(ps_shift_send[i], 3*num_ps_send[i], MPI_SIGNED_CHAR, dst, tag4, MPI_COMM_WORLD, &(send_req_list[i+3*num_nbp]));
	}
    }

    for (i = 0; i < num_nbp; ++i)
    {
	src = nb_proc->data[i];

	MPI_Irecv(&(size_info[i]), 1, MPI_INT, src, 1, MPI_COMM_WORLD, &(recv_req_list[i]));
    }

    for (i = 0; i < num_nbp; ++i)
    {
	int recv_index;
	int source_id;
	MPI_Status recv_status1;

	MPI_Waitany(num_nbp, recv_req_list, &recv_index, &recv_status1);
	source_id = recv_status1.MPI_SOURCE;

	if (size_info[recv_index] != 0)
	{
	    double *ps_recv;
	    int *ps_index_recv;
	    int8_T *ps_shift_recv;
	    MPI_Status recv_status2;
	    MPI_Status recv_status3;
	    MPI_Status recv_status4;

	    ps_recv = (double *) calloc(3*size_info[recv_index], sizeof(double));
	    ps_index_recv = (int *) calloc(size_info[recv_index], sizeof(int));
	    ps_shift_recv = (int8_T *) calloc(3*size_info[recv_index], sizeof(int8_T));

	    MPI_Recv(ps_recv, 3*size_info[recv_index], MPI_DOUBLE, source_id, 2, MPI_COMM_WORLD, &recv_status2);
	    MPI_Recv(ps_index_recv, size_info[recv_index], MPI_INT, source_id, 3, MPI_COMM_WORLD, &recv_status3);
	    MPI_Recv(ps_shift_recv, 3*size_info[recv_index], MPI_SIGNED_CHAR, source_id, 4, MPI_COMM_WORLD, &recv_status4);


	    // All possible points for floating point comparison
	    boolean_T *flag = (boolean_T *) calloc (ps->size[0], sizeof (boolean_T));

	    emxArray_int8_T *nb_shift = (mesh->nb_proc_shift)[recv_index];
	    int num_shifts = nb_shift->size[0];

	    //Iterate all shifts
	    for (is = 1; is <= num_shifts; ++is)
	    {
		double x_factor = 0.0;
		double y_factor = 0.0;
		double z_factor = 0.0;

		int8_T sx, sy, sz;

		sx = nb_shift->data[I2dm(is,1,nb_shift->size)];
		sy = nb_shift->data[I2dm(is,2,nb_shift->size)];
		sz = nb_shift->data[I2dm(is,3,nb_shift->size)];

		x_factor = ((double) -sx)*domain_len_x;
		y_factor = ((double) -sy)*domain_len_y;
		z_factor = ((double) -sz)*domain_len_z;

		double cur_bdbox_xL = all_bd_box[source_id*6] - x_factor;
		double cur_bdbox_xU = all_bd_box[source_id*6+1] - x_factor;
		double cur_bdbox_yL = all_bd_box[source_id*6+2] - y_factor;
		double cur_bdbox_yU = all_bd_box[source_id*6+3] - y_factor;
		double cur_bdbox_zL = all_bd_box[source_id*6+4] - z_factor;
		double cur_bdbox_zU = all_bd_box[source_id*6+5] - z_factor;

		for (j = 1; j <= ps_size[0]; ++j)
		{
		    double cur_x = ps_data[I2dm(j,1,ps_size)];
		    double cur_y = ps_data[I2dm(j,2,ps_size)];
		    double cur_z = ps_data[I2dm(j,3,ps_size)];

		    if ( (cur_x >= cur_bdbox_xL) && (cur_x <= cur_bdbox_xU) && 
			    (cur_y >= cur_bdbox_yL) && (cur_y <= cur_bdbox_yU) &&
			    (cur_z >= cur_bdbox_zL) && (cur_z <= cur_bdbox_zU) 
		       )
		    {
			flag[j-1] = 1;
		    }
		}

	    }

	    // Each point received could only be mapped to one point in the original
	    // mesh
	    boolean_T *recv_flag = (boolean_T *) calloc (size_info[recv_index], sizeof(boolean_T));

	    for (j = 1; j <= ps_size[0]; ++j) 
	    {
		if (flag[j-1] == 1)
		{
		    double cur_x = ps_data[I2dm(j,1,ps_size)];
		    double cur_y = ps_data[I2dm(j,2,ps_size)];
		    double cur_z = ps_data[I2dm(j,3,ps_size)];

		    for (k = 0; k < size_info[recv_index]; ++k)
		    {
			if (recv_flag[k] == 1)
			    continue;

			if ( (fabs(cur_x - ps_recv[k*3]) < eps) &&
				(fabs(cur_y - ps_recv[k*3+1]) < eps) &&
				(fabs(cur_z - ps_recv[k*3+2]) < eps)
			   )
			{
			    recv_flag[k] = 1;
			    if(hpEnsurePInfoCapacity(ps_pinfo))
				ps_pdata = ps_pinfo->pdata;
			    ps_pinfo->allocated_len++;
			    int cur_head = ps_phead[I1dm(j)];
			    int cur_tail = ps_ptail[I1dm(j)];
			    int cur_master_proc = ps_pdata[I1dm(cur_head)].proc;
			    if (source_id < cur_master_proc)
			    {
				ps_phead[I1dm(j)] = ps_pinfo->allocated_len;
				ps_pdata[I1dm(ps_pinfo->allocated_len)].proc = source_id;
				ps_pdata[I1dm(ps_pinfo->allocated_len)].lindex = ps_index_recv[k];
				ps_pdata[I1dm(ps_pinfo->allocated_len)].shift[0] = ps_shift_recv[k*3];
				ps_pdata[I1dm(ps_pinfo->allocated_len)].shift[1] = ps_shift_recv[k*3+1];
				ps_pdata[I1dm(ps_pinfo->allocated_len)].shift[2] = ps_shift_recv[k*3+2];
				ps_pdata[I1dm(ps_pinfo->allocated_len)].next = cur_head;
			    }
			    else if (source_id > cur_master_proc)
			    {
				ps_ptail[I1dm(j)] = ps_pinfo->allocated_len;
				ps_pdata[I1dm(ps_pinfo->allocated_len)].proc = source_id;
				ps_pdata[I1dm(ps_pinfo->allocated_len)].lindex = ps_index_recv[k];
				ps_pdata[I1dm(ps_pinfo->allocated_len)].shift[0] = ps_shift_recv[k*3];
				ps_pdata[I1dm(ps_pinfo->allocated_len)].shift[1] = ps_shift_recv[k*3+1];
				ps_pdata[I1dm(ps_pinfo->allocated_len)].shift[2] = ps_shift_recv[k*3+2];
				ps_pdata[I1dm(ps_pinfo->allocated_len)].next = -1;
				ps_pdata[I1dm(cur_tail)].next = ps_pinfo->allocated_len;
			    }
			    else // If two pinfo has same processor id, then use local index to
				// decide which one is master
			    {

				if (ps_pdata[I1dm(cur_head)].lindex > ps_index_recv[k])
				{
				    ps_phead[I1dm(j)] = ps_pinfo->allocated_len;
				    ps_pdata[I1dm(ps_pinfo->allocated_len)].proc = source_id;
				    ps_pdata[I1dm(ps_pinfo->allocated_len)].lindex = ps_index_recv[k];
				    ps_pdata[I1dm(ps_pinfo->allocated_len)].shift[0] = ps_shift_recv[k*3];
				    ps_pdata[I1dm(ps_pinfo->allocated_len)].shift[1] = ps_shift_recv[k*3+1];
				    ps_pdata[I1dm(ps_pinfo->allocated_len)].shift[2] = ps_shift_recv[k*3+2];
				    ps_pdata[I1dm(ps_pinfo->allocated_len)].next = cur_head;
				}
				else if (ps_pinfo->pdata[I1dm(cur_head)].lindex < ps_index_recv[k])
				{
				    ps_ptail[I1dm(j)] = ps_pinfo->allocated_len;
				    ps_pdata[I1dm(ps_pinfo->allocated_len)].proc = source_id;
				    ps_pdata[I1dm(ps_pinfo->allocated_len)].lindex = ps_index_recv[k];
				    ps_pdata[I1dm(ps_pinfo->allocated_len)].shift[0] = ps_shift_recv[k*3];
				    ps_pdata[I1dm(ps_pinfo->allocated_len)].shift[1] = ps_shift_recv[k*3+1];
				    ps_pdata[I1dm(ps_pinfo->allocated_len)].shift[2] = ps_shift_recv[k*3+2];
				    ps_pdata[I1dm(ps_pinfo->allocated_len)].next = -1;
				    ps_pdata[I1dm(cur_tail)].next = ps_pinfo->allocated_len;
				}
				else
				{
				    printf("\n Receiving element already in the PInfo list!\n");
				    exit(0);
				}
			    }
			}
		    }
		}
	    }
	    free(recv_flag);
	    free(ps_recv);
	    free(ps_index_recv);
	    free(ps_shift_recv);
	    free(flag);
	}
    }

    free(size_info);
    free(recv_req_list);

    MPI_Waitall(4*num_nbp, send_req_list, send_status_list);

    free(send_req_list);
    free(send_status_list);

    for (i = 0; i < num_nbp; ++i)
    {
	free(ps_send[i]);
	free(ps_index_send[i]);
	free(ps_shift_send[i]);
    }
    free(num_ps_send);
    free(ps_send);
    free(ps_index_send);
    free(ps_shift_send);
    free(all_bd_box);

    mesh->nps_clean = mesh->ps->size[0];
    mesh->ntris_clean = mesh->tris->size[0];
    mesh->npspi_clean = mesh->ps_pinfo->allocated_len;
    mesh->is_clean = 1;

}

void hpBuildPInfoWithOverlappingTris(hiPropMesh *mesh)
{
    int i, j, k, is;
    int src, dst;
    int num_proc, rank;
    double eps = 1e-14;

    emxArray_real_T *ps = mesh->ps;
    emxArray_int32_T *tris = mesh->tris;

    real_T *ps_data = ps->data;
    int32_T *ps_size = ps->size;

    int32_T *tris_data = tris->data;
    int32_T *tris_size = tris->size;

    emxArray_int32_T *nb_proc = mesh->nb_proc;
    int num_nbp = nb_proc->size[0];

    hpPInfoList *ps_pinfo = mesh->ps_pinfo;
    hpPInfoList *tris_pinfo = mesh->tris_pinfo;

    hpPInfoNode *ps_pdata = ps_pinfo->pdata;
    int *ps_phead = ps_pinfo->head;
    int *ps_ptail = ps_pinfo->tail;

    hpPInfoNode *tris_pdata = tris_pinfo->pdata;
    int *tris_phead = tris_pinfo->head;
    int *tris_ptail = tris_pinfo->tail;
    unsigned char is_overlapping_tri = 0;

    MPI_Comm_size(MPI_COMM_WORLD, &num_proc);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    double domain_len_x = mesh->domain_len[0];
    double domain_len_y = mesh->domain_len[1];
    double domain_len_z = mesh->domain_len[2];

    // First get the bounding box 
    // for each processor and reduce to all processor

    double bd_box[6];
    if(ps_size[0] > 0)
    {
	bd_box[0]= ps_data[I2dm(1,1,ps_size)];
	bd_box[1] = bd_box[0];
	bd_box[2] = ps_data[I2dm(1,2,ps_size)];
	bd_box[3] = bd_box[2];
	bd_box[4] = ps_data[I2dm(1,3,ps_size)];
	bd_box[5] = bd_box[4];

	for (i = 2; i <= ps_size[0]; ++i)
	{
	    double x = ps_data[I2dm(i,1,ps_size)];
	    double y = ps_data[I2dm(i,2,ps_size)];
	    double z = ps_data[I2dm(i,3,ps_size)];

	    if (x < bd_box[0])
		bd_box[0] = x;
	    if (x > bd_box[1])
		bd_box[1] = x;
	    if (y < bd_box[2])
		bd_box[2] = y;
	    if (y > bd_box[3])
		bd_box[3] = y;
	    if (z < bd_box[4])
		bd_box[4] = z;
	    if (z > bd_box[5])
		bd_box[5] = z;
	}
	bd_box[0] -= eps;
	bd_box[1] += eps;
	bd_box[2] -= eps;
	bd_box[3] += eps;
	bd_box[4] -= eps;
	bd_box[5] += eps;
    }
    else
    {
	bd_box[0] = eps;
	bd_box[1] = -eps;	
	bd_box[2] = eps;
	bd_box[3] = -eps;
	bd_box[4] = eps;
	bd_box[5] = -eps;
    }

    double *in_all_bd_box = (double *)calloc(6*num_proc, sizeof(double));
    double *all_bd_box = (double *)calloc(6*num_proc, sizeof(double));

    for (i = 0; i < 6; i++)
	in_all_bd_box[rank*6+i] = bd_box[i];

    MPI_Allreduce(in_all_bd_box, all_bd_box, 6*num_proc, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    free(in_all_bd_box);


    int *num_ps_send = (int *) calloc(num_nbp, sizeof(int));
    int *num_tris_send = (int *) calloc(num_nbp, sizeof(int));

    double **ps_send = (double **) calloc(num_nbp, sizeof(double *));
    int **tris_send = (int **) calloc(num_nbp, sizeof(int *));

    int **ps_index_send = (int **) calloc(num_nbp, sizeof(int *));
    int8_T **ps_shift_send = (int8_T **) calloc(num_nbp, sizeof(int8_T *));

    int **tris_index_send = (int **) calloc(num_nbp, sizeof(int *));
    int8_T **tris_shift_send = (int8_T **) calloc(num_nbp, sizeof(int8_T *));

    for (i = 0; i < num_nbp; ++i)
    {
	num_ps_send[i] = 0;
	num_tris_send[i] = 0;
	int target_id = nb_proc->data[i];

	emxArray_int8_T *nb_shift = (mesh->nb_proc_shift)[i];
	int num_shifts = nb_shift->size[0];

	//First get num_ps_send and num_tris_send
	for (is = 1; is <= num_shifts; ++is)
	{
	    boolean_T *ps_flag = (boolean_T *) calloc(ps_size[0], sizeof(boolean_T));
	    double x_factor = 0.0;
	    double y_factor = 0.0;
	    double z_factor = 0.0;

	    int8_T sx, sy, sz;

	    sx = nb_shift->data[I2dm(is,1,nb_shift->size)];
	    sy = nb_shift->data[I2dm(is,2,nb_shift->size)];
	    sz = nb_shift->data[I2dm(is,3,nb_shift->size)];

	    x_factor = ((double) -sx)*domain_len_x;
	    y_factor = ((double) -sy)*domain_len_y;
	    z_factor = ((double) -sz)*domain_len_z;

	    double cur_bdbox_xL = all_bd_box[target_id*6] - x_factor;
	    double cur_bdbox_xU = all_bd_box[target_id*6+1] - x_factor;
	    double cur_bdbox_yL = all_bd_box[target_id*6+2] - y_factor;
	    double cur_bdbox_yU = all_bd_box[target_id*6+3] - y_factor;
	    double cur_bdbox_zL = all_bd_box[target_id*6+4] - z_factor;
	    double cur_bdbox_zU = all_bd_box[target_id*6+5] - z_factor;

	    for (j = 1; j <= ps_size[0]; ++j)
	    {
		double cur_x = ps_data[I2dm(j,1,ps_size)];
		double cur_y = ps_data[I2dm(j,2,ps_size)];
		double cur_z = ps_data[I2dm(j,3,ps_size)];

		if ( (cur_x >= cur_bdbox_xL) && (cur_x <= cur_bdbox_xU) && 
		     (cur_y >= cur_bdbox_yL) && (cur_y <= cur_bdbox_yU) &&
		     (cur_z >= cur_bdbox_zL) && (cur_z <= cur_bdbox_zU) 
		   )
		{
		    (num_ps_send[i])++;
		    ps_flag[I1dm(j)] = 1;
		}
	    }

	    for (j = 1; j <= tris_size[0]; ++j)
	    {
		int pi1 = tris_data[I2dm(j,1,tris_size)];
		int pi2 = tris_data[I2dm(j,2,tris_size)];
		int pi3 = tris_data[I2dm(j,3,tris_size)];

		if (ps_flag[pi1-1] && ps_flag[pi2-1] && ps_flag[pi3-1])
		    (num_tris_send[i])++;
	    }
	    free(ps_flag);
	}

	//Allocate the storage for send
	
	ps_send[i] = (double *) calloc(3*num_ps_send[i], sizeof(double));
	ps_index_send[i] = (int *) calloc(num_ps_send[i], sizeof(int));
	ps_shift_send[i] = (int8_T *) calloc(3*num_ps_send[i], sizeof(int8_T));

	tris_send[i] = (int *) calloc(3*num_tris_send[i], sizeof(int));
	tris_index_send[i] = (int *) calloc(num_tris_send[i], sizeof(int));
	tris_shift_send[i] = (int8_T *) calloc(3*num_tris_send[i], sizeof(int8_T));

	// Fill up the arrays for send
	double *cur_ps_send = ps_send[i];
	int *cur_ps_index_send = ps_index_send[i];
	int8_T * cur_ps_shift_send = ps_shift_send[i];

	int *cur_tris_send = tris_send[i];
	int *cur_tris_index_send = tris_index_send[i];
	int8_T *cur_tris_shift_send = tris_shift_send[i];

	int k_ps = 0;
	int ki_ps = 0;
	int ks_ps = 0;

	int k_tris = 0;
	int ki_tris = 0;
	int ks_tris = 0;

	for (is = 1; is <= num_shifts; ++is)
	{
	    // ps_map maps the current ps id to ps id in the send array
	    // same point could be sent several times with different shift

	    int *ps_map = (int *) calloc(ps_size[0], sizeof(int));
	    double x_factor = 0.0;
	    double y_factor = 0.0;
	    double z_factor = 0.0;

	    int8_T sx, sy, sz;
	    int8_T sx_send, sy_send, sz_send;

	    sx = nb_shift->data[I2dm(is,1,nb_shift->size)];
	    sy = nb_shift->data[I2dm(is,2,nb_shift->size)];
	    sz = nb_shift->data[I2dm(is,3,nb_shift->size)];

	    sx_send = -sx;
	    sy_send = -sy;
	    sz_send = -sz;

	    x_factor = ((double) -sx)*domain_len_x;
	    y_factor = ((double) -sy)*domain_len_y;
	    z_factor = ((double) -sz)*domain_len_z;

	    double cur_bdbox_xL = all_bd_box[target_id*6] - x_factor;
	    double cur_bdbox_xU = all_bd_box[target_id*6+1] - x_factor;
	    double cur_bdbox_yL = all_bd_box[target_id*6+2] - y_factor;
	    double cur_bdbox_yU = all_bd_box[target_id*6+3] - y_factor;
	    double cur_bdbox_zL = all_bd_box[target_id*6+4] - z_factor;
	    double cur_bdbox_zU = all_bd_box[target_id*6+5] - z_factor;

	    if (num_ps_send[i] != 0)
	    {
		for (j = 1; j <= ps_size[0]; ++j)
		{
		    double cur_x = ps_data[I2dm(j,1,ps_size)];
		    double cur_y = ps_data[I2dm(j,2,ps_size)];
		    double cur_z = ps_data[I2dm(j,3,ps_size)];

		    if ( (cur_x >= cur_bdbox_xL) && (cur_x <= cur_bdbox_xU) && 
			    (cur_y >= cur_bdbox_yL) && (cur_y <= cur_bdbox_yU) &&
			    (cur_z >= cur_bdbox_zL) && (cur_z <= cur_bdbox_zU) 
		       )
		    {
			cur_ps_send[k_ps++] = cur_x + x_factor;
			cur_ps_send[k_ps++] = cur_y + y_factor;
			cur_ps_send[k_ps++] = cur_z + z_factor;
			cur_ps_index_send[ki_ps++] = j;
			cur_ps_shift_send[ks_ps++] = sx_send;
			cur_ps_shift_send[ks_ps++] = sy_send;
			cur_ps_shift_send[ks_ps++] = sz_send;
			ps_map[I1dm(j)] = ki_ps;
		    }
		}
	    }


	    if (num_tris_send[i] != 0)
	    {
		for (j = 1; j <= tris_size[0]; ++j)
		{
		    int pi1 = tris_data[I2dm(j,1,tris_size)];
		    int pi2 = tris_data[I2dm(j,2,tris_size)];
		    int pi3 = tris_data[I2dm(j,3,tris_size)];

		    if (   (ps_map[pi1-1] != 0) 
			    && (ps_map[pi2-1] != 0)
			    && (ps_map[pi3-1] != 0)  )
		    {
			cur_tris_send[k_tris++] = ps_map[pi1-1];
			cur_tris_send[k_tris++] = ps_map[pi2-1];
			cur_tris_send[k_tris++] = ps_map[pi3-1];

			cur_tris_index_send[ki_tris++] = j;

			cur_tris_shift_send[ks_tris++] = sx_send;
			cur_tris_shift_send[ks_tris++] = sy_send;
			cur_tris_shift_send[ks_tris++] = sz_send;
		    }
		}
	    }
	    free(ps_map);
	}
    }

    // Combine 2 dimension values to one array for send
    int *size_send = (int *) calloc(2*num_nbp, sizeof(int));
    for (i = 0; i < num_nbp; ++i)
    {
	size_send[2*i] = num_ps_send[i];
	size_send[2*i+1] = num_tris_send[i];
    }
    free(num_ps_send);
    free(num_tris_send);

    MPI_Request *send_req_list = (MPI_Request *) malloc (7*num_nbp*sizeof(MPI_Request) );
    for (i = 0; i < 7*num_nbp; i++)
	send_req_list[i] = MPI_REQUEST_NULL;

    MPI_Status *send_status_list = (MPI_Status *) malloc(7*num_nbp*sizeof(MPI_Status) );

    MPI_Request *recv_req_list = (MPI_Request *) malloc(num_nbp*sizeof(MPI_Request) );

    int *size_recv = (int *) calloc(2*num_nbp, sizeof(int));

    for (i = 0; i < num_nbp; ++i)
    {
	dst = nb_proc->data[i];

	MPI_Isend(&(size_send[2*i]), 2, MPI_INT, dst, 1, MPI_COMM_WORLD, &(send_req_list[i]));

	if (size_send[2*i] != 0)
	{
	    MPI_Isend(ps_send[i], 3*size_send[2*i], MPI_DOUBLE, dst, 2, MPI_COMM_WORLD, &(send_req_list[i+num_nbp])); 
	    MPI_Isend(ps_index_send[i], size_send[2*i], MPI_INT, dst, 3, MPI_COMM_WORLD, &(send_req_list[i+2*num_nbp]));
	    MPI_Isend(ps_shift_send[i], 3*size_send[2*i], MPI_SIGNED_CHAR, dst, 4, MPI_COMM_WORLD, &(send_req_list[i+3*num_nbp]));

	}
	if (size_send[2*i+1] != 0)
	{
	    MPI_Isend(tris_send[i], 3*size_send[2*i+1], MPI_INT, dst, 5, MPI_COMM_WORLD, &(send_req_list[i+4*num_nbp]));
	    MPI_Isend(tris_index_send[i], size_send[2*i+1], MPI_INT, dst, 6, MPI_COMM_WORLD, &(send_req_list[i+5*num_nbp]));
	    MPI_Isend(tris_shift_send[i], 3*size_send[2*i+1], MPI_SIGNED_CHAR, dst, 7, MPI_COMM_WORLD, &(send_req_list[i+6*num_nbp]));
	}
    }

    for (i = 0; i < num_nbp; ++i)
    {
	src = nb_proc->data[i];

	MPI_Irecv(&(size_recv[2*i]), 2, MPI_INT, src, 1, MPI_COMM_WORLD, &(recv_req_list[i]));
    }

    for (i = 0; i < num_nbp; ++i)
    {

	int recv_index;
	int source_id;
	MPI_Status recv_status1;

	MPI_Waitany(num_nbp, recv_req_list, &recv_index, &recv_status1);
	source_id = recv_status1.MPI_SOURCE;

	if ( (size_recv[2*recv_index] != 0) || (size_recv[2*recv_index+1] != 0) )
	{
	    double *ps_recv;
	    int *ps_index_recv;
	    int8_T *ps_shift_recv;
	    MPI_Status recv_status2;
	    MPI_Status recv_status3;
	    MPI_Status recv_status4;
	    ps_recv = (double *) calloc(3*size_recv[2*recv_index], sizeof(double));
	    ps_index_recv = (int *) calloc(size_recv[2*recv_index], sizeof(int));
	    ps_shift_recv = (int8_T *) calloc(3*size_recv[2*recv_index], sizeof(int8_T));

	    if (size_recv[2*recv_index] != 0)
	    {
		MPI_Recv(ps_recv, 3*size_recv[2*recv_index], MPI_DOUBLE, source_id, 2, MPI_COMM_WORLD, &recv_status2);
		MPI_Recv(ps_index_recv, size_recv[2*recv_index], MPI_INT, source_id, 3, MPI_COMM_WORLD, &recv_status3);
		MPI_Recv(ps_shift_recv, 3*size_recv[2*recv_index], MPI_SIGNED_CHAR, source_id, 4, MPI_COMM_WORLD, &recv_status4);
	    }

	    int *tris_recv;
	    int *tris_index_recv;
	    int8_T *tris_shift_recv;
	    MPI_Status recv_status5;
	    MPI_Status recv_status6;
	    MPI_Status recv_status7;
	    tris_recv = (int *) calloc(3*size_recv[2*recv_index+1], sizeof(int));
	    tris_index_recv = (int *) calloc(size_recv[2*recv_index+1], sizeof(int));
	    tris_shift_recv = (int8_T *) calloc(3*size_recv[2*recv_index+1], sizeof(int8_T));
	    
	    if (size_recv[2*recv_index+1] != 0)
	    {
		MPI_Recv(tris_recv, 3*size_recv[2*recv_index+1], MPI_INT, source_id, 5, MPI_COMM_WORLD, &recv_status5);
		MPI_Recv(tris_index_recv, size_recv[2*recv_index+1], MPI_INT, source_id, 6, MPI_COMM_WORLD, &recv_status6);
		MPI_Recv(tris_shift_recv, 3*size_recv[2*recv_index+1], MPI_SIGNED_CHAR, source_id, 7, MPI_COMM_WORLD, &recv_status7);
	    }

	    // Build up the possible ps and tris for the current processor 
	    // for comparing with ps and tris received from source_id

	    boolean_T *ps_flag = (boolean_T *) calloc (ps_size[0], sizeof (boolean_T));
	    boolean_T *tris_flag = (boolean_T *) calloc (tris_size[0], sizeof (boolean_T));

	    emxArray_int8_T *nb_shift = (mesh->nb_proc_shift)[recv_index];
	    int num_shifts = nb_shift->size[0];


	    for (is = 1; is <= num_shifts; ++is)
	    {
		double x_factor = 0.0;
		double y_factor = 0.0;
		double z_factor = 0.0;

		int8_T sx, sy, sz;

		sx = nb_shift->data[I2dm(is,1,nb_shift->size)];
		sy = nb_shift->data[I2dm(is,2,nb_shift->size)];
		sz = nb_shift->data[I2dm(is,3,nb_shift->size)];

		x_factor = ((double) -sx)*domain_len_x;
		y_factor = ((double) -sy)*domain_len_y;
		z_factor = ((double) -sz)*domain_len_z;

		double cur_bdbox_xL = all_bd_box[source_id*6] - x_factor;
		double cur_bdbox_xU = all_bd_box[source_id*6+1] - x_factor;
		double cur_bdbox_yL = all_bd_box[source_id*6+2] - y_factor;
		double cur_bdbox_yU = all_bd_box[source_id*6+3] - y_factor;
		double cur_bdbox_zL = all_bd_box[source_id*6+4] - z_factor;
		double cur_bdbox_zU = all_bd_box[source_id*6+5] - z_factor;


		if (size_recv[2*recv_index] != 0)
		{
		    for (j = 1; j <= ps_size[0]; ++j)
		    {
			double cur_x = ps_data[I2dm(j,1,ps_size)];
			double cur_y = ps_data[I2dm(j,2,ps_size)];
			double cur_z = ps_data[I2dm(j,3,ps_size)];

			if ( (cur_x >= cur_bdbox_xL) && (cur_x <= cur_bdbox_xU) && 
				(cur_y >= cur_bdbox_yL) && (cur_y <= cur_bdbox_yU) &&
				(cur_z >= cur_bdbox_zL) && (cur_z <= cur_bdbox_zU) 
			   )
			{
			    ps_flag[j-1] = 1;
			}
		    }
		}
		if (size_recv[2*recv_index+1] != 0)
		{
		    for (j = 1; j <= tris_size[0]; ++j)
		    {
			int pi1 = tris_data[I2dm(j,1,tris_size)];
			int pi2 = tris_data[I2dm(j,2,tris_size)];
			int pi3 = tris_data[I2dm(j,3,tris_size)];

			if ( ps_flag[I1dm(pi1)] && ps_flag[I1dm(pi2)] && ps_flag[I1dm(pi3)] )
			{
			    tris_flag[j-1] = 1;
			}
		    }
		}
	    }

	    // recv_ps_map[I1dm(i)] is the local index of a point 
	    // which the i-th point in the receiving ps array corresponding to.
	    // Used for building up tris

	    int *recv_ps_map = (int *) calloc(size_recv[2*recv_index], sizeof(int));

	    if (size_recv[2*recv_index] != 0)
	    {
		boolean_T *recv_ps_flag = (boolean_T *)calloc(size_recv[2*recv_index], sizeof(boolean_T));

		// Build the pinfo for ps

		for (j = 1; j <= ps_size[0]; ++j) 
		{
		    if (ps_flag[j-1])
		    {

			double cur_x = ps_data[I2dm(j,1,ps_size)];
			double cur_y = ps_data[I2dm(j,2,ps_size)];
			double cur_z = ps_data[I2dm(j,3,ps_size)];

			for (k = 0; k < size_recv[2*recv_index]; ++k)
			{
			    if (recv_ps_flag[k])
				continue;

			    if ( (fabs(cur_x - ps_recv[k*3]) < eps) &&
				    (fabs(cur_y - ps_recv[k*3+1]) < eps) &&
				    (fabs(cur_z - ps_recv[k*3+2]) < eps)
			       )
			    {
				recv_ps_flag[k] = 1;
				recv_ps_map[k] = j;
				if(hpEnsurePInfoCapacity(ps_pinfo))
				    ps_pdata = ps_pinfo->pdata;
				ps_pinfo->allocated_len++;
				int cur_head = ps_phead[I1dm(j)];
				int cur_tail = ps_ptail[I1dm(j)];
				int cur_master_proc = ps_pdata[I1dm(cur_head)].proc;
				if (source_id < cur_master_proc)
				{
				    ps_phead[I1dm(j)] = ps_pinfo->allocated_len;
				    ps_pdata[I1dm(ps_pinfo->allocated_len)].proc = source_id;
				    ps_pdata[I1dm(ps_pinfo->allocated_len)].lindex = ps_index_recv[k];
				    ps_pdata[I1dm(ps_pinfo->allocated_len)].shift[0] = ps_shift_recv[k*3];
				    ps_pdata[I1dm(ps_pinfo->allocated_len)].shift[1] = ps_shift_recv[k*3+1];
				    ps_pdata[I1dm(ps_pinfo->allocated_len)].shift[2] = ps_shift_recv[k*3+2];
				    ps_pdata[I1dm(ps_pinfo->allocated_len)].next = cur_head;
				}
				else if (source_id > cur_master_proc)
				{
				    ps_ptail[I1dm(j)] = ps_pinfo->allocated_len;
				    ps_pdata[I1dm(ps_pinfo->allocated_len)].proc = source_id;
				    ps_pdata[I1dm(ps_pinfo->allocated_len)].lindex = ps_index_recv[k];
				    ps_pdata[I1dm(ps_pinfo->allocated_len)].shift[0] = ps_shift_recv[k*3];
				    ps_pdata[I1dm(ps_pinfo->allocated_len)].shift[1] = ps_shift_recv[k*3+1];
				    ps_pdata[I1dm(ps_pinfo->allocated_len)].shift[2] = ps_shift_recv[k*3+2];
				    ps_pdata[I1dm(ps_pinfo->allocated_len)].next = -1;
				    ps_pdata[I1dm(cur_tail)].next = ps_pinfo->allocated_len;
				}
				else
				{
				    // Using local index to decide

				    if (ps_pdata[I1dm(cur_head)].lindex > ps_index_recv[k])
				    {
					ps_phead[I1dm(j)] = ps_pinfo->allocated_len;
					ps_pdata[I1dm(ps_pinfo->allocated_len)].proc = source_id;
					ps_pdata[I1dm(ps_pinfo->allocated_len)].lindex = ps_index_recv[k];
					ps_pdata[I1dm(ps_pinfo->allocated_len)].shift[0] = ps_shift_recv[k*3];
					ps_pdata[I1dm(ps_pinfo->allocated_len)].shift[1] = ps_shift_recv[k*3+1];
					ps_pdata[I1dm(ps_pinfo->allocated_len)].shift[2] = ps_shift_recv[k*3+2];
					ps_pdata[I1dm(ps_pinfo->allocated_len)].next = cur_head;
				    }
				    else if (ps_pdata[I1dm(cur_head)].lindex < ps_index_recv[k])
				    {
					ps_ptail[I1dm(j)] = ps_pinfo->allocated_len;
					ps_pdata[I1dm(ps_pinfo->allocated_len)].proc = source_id;
					ps_pdata[I1dm(ps_pinfo->allocated_len)].lindex = ps_index_recv[k];
					ps_pdata[I1dm(ps_pinfo->allocated_len)].shift[0] = ps_shift_recv[k*3];
					ps_pdata[I1dm(ps_pinfo->allocated_len)].shift[1] = ps_shift_recv[k*3+1];
					ps_pdata[I1dm(ps_pinfo->allocated_len)].shift[2] = ps_shift_recv[k*3+2];
					ps_pdata[I1dm(ps_pinfo->allocated_len)].next = -1;
					ps_pdata[I1dm(cur_tail)].next = ps_pinfo->allocated_len;
				    }
				    else
				    {
					printf("\n Receiving element already in the PInfo list!\n");
					exit(0);
				    }
				}
			    }
			}
		    }
		}
		free(recv_ps_flag);
	    }

	    if (size_recv[2*recv_index+1] != 0)
	    {
		boolean_T *recv_tris_flag = (boolean_T *)calloc(size_recv[2*recv_index+1], sizeof(boolean_T));

		// Build the tris pinfo
		for (j = 1; j <= tris_size[0]; ++j) 
		{
		    if (tris_flag[j-1])
		    {
			int p1 = tris_data[I2dm(j,1,tris_size)];
			int p2 = tris_data[I2dm(j,2,tris_size)];
			int p3 = tris_data[I2dm(j,3,tris_size)];

			for (k = 0; k < size_recv[2*recv_index+1]; ++k)
			{
			    if (recv_tris_flag[k] == 1)
				continue;
			    if ( (p1 == recv_ps_map[I1dm(tris_recv[k*3])]) &&
				    (p2 == recv_ps_map[I1dm(tris_recv[k*3+1])]) &&
				    (p3 == recv_ps_map[I1dm(tris_recv[k*3+2])])
			       )
			    {
				recv_tris_flag[k] = 1;
				is_overlapping_tri = 1;
				if(hpEnsurePInfoCapacity(tris_pinfo))
				    tris_pdata = tris_pinfo->pdata;
				tris_pinfo->allocated_len++;
				int cur_head = tris_phead[I1dm(j)];
				int cur_tail = tris_ptail[I1dm(j)];
				int cur_master_proc = tris_pdata[I1dm(cur_head)].proc;
				if (source_id < cur_master_proc)
				{
				    tris_phead[I1dm(j)] = tris_pinfo->allocated_len;
				    tris_pdata[I1dm(tris_pinfo->allocated_len)].proc = source_id;
				    tris_pdata[I1dm(tris_pinfo->allocated_len)].lindex = tris_index_recv[k];
				    tris_pdata[I1dm(tris_pinfo->allocated_len)].shift[0] = tris_shift_recv[k*3];
				    tris_pdata[I1dm(tris_pinfo->allocated_len)].shift[1] = tris_shift_recv[k*3+1];
				    tris_pdata[I1dm(tris_pinfo->allocated_len)].shift[2] = tris_shift_recv[k*3+2];
				    tris_pdata[I1dm(tris_pinfo->allocated_len)].next = cur_head;
				}
				else if (source_id > cur_master_proc)
				{
				    tris_ptail[I1dm(j)] = tris_pinfo->allocated_len;
				    tris_pdata[I1dm(tris_pinfo->allocated_len)].proc = source_id;
				    tris_pdata[I1dm(tris_pinfo->allocated_len)].lindex = tris_index_recv[k];
				    tris_pdata[I1dm(tris_pinfo->allocated_len)].shift[0] = tris_shift_recv[k*3];
				    tris_pdata[I1dm(tris_pinfo->allocated_len)].shift[1] = tris_shift_recv[k*3+1];
				    tris_pdata[I1dm(tris_pinfo->allocated_len)].shift[2] = tris_shift_recv[k*3+2];
				    tris_pdata[I1dm(tris_pinfo->allocated_len)].next = -1;
				    tris_pdata[I1dm(cur_tail)].next = tris_pinfo->allocated_len;
				}
				else
				{
				    if (tris_pdata[I1dm(cur_head)].lindex > tris_index_recv[k])
				    {
					tris_phead[I1dm(j)] = tris_pinfo->allocated_len;
					tris_pdata[I1dm(tris_pinfo->allocated_len)].proc = source_id;
					tris_pdata[I1dm(tris_pinfo->allocated_len)].lindex = tris_index_recv[k];
					tris_pdata[I1dm(tris_pinfo->allocated_len)].shift[0] = tris_shift_recv[k*3];
					tris_pdata[I1dm(tris_pinfo->allocated_len)].shift[1] = tris_shift_recv[k*3+1];
					tris_pdata[I1dm(tris_pinfo->allocated_len)].shift[2] = tris_shift_recv[k*3+2];
					tris_pdata[I1dm(tris_pinfo->allocated_len)].next = cur_head;
				    }
				    else if (tris_pdata[I1dm(cur_head)].lindex < tris_index_recv[k])
				    {
					tris_ptail[I1dm(j)] = tris_pinfo->allocated_len;
					tris_pdata[I1dm(tris_pinfo->allocated_len)].proc = source_id;
					tris_pdata[I1dm(tris_pinfo->allocated_len)].lindex = tris_index_recv[k];
					tris_pdata[I1dm(tris_pinfo->allocated_len)].shift[0] = tris_shift_recv[k*3];
					tris_pdata[I1dm(tris_pinfo->allocated_len)].shift[1] = tris_shift_recv[k*3+1];
					tris_pdata[I1dm(tris_pinfo->allocated_len)].shift[2] = tris_shift_recv[k*3+2];

					tris_pdata[I1dm(tris_pinfo->allocated_len)].next = -1;
					tris_pdata[I1dm(cur_tail)].next = tris_pinfo->allocated_len;
				    }
				    else
				    {
					printf("\n Receiving element already in the PInfo list!\n");
					exit(0);
				    }
				}
			    }
			}
		    }
		}
		free(recv_tris_flag);
	    }
	    free(recv_ps_map);

	    free(ps_recv);
	    free(tris_recv);
	    free(ps_index_recv);
	    free(tris_index_recv);
	    free(ps_shift_recv);
	    free(tris_shift_recv);
	    free(ps_flag);
	    free(tris_flag);
	}

    }

    free(size_recv);
    free(recv_req_list);

    MPI_Waitall(7*num_nbp, send_req_list, send_status_list);

    free(send_req_list);
    free(send_status_list);

    for (i = 0; i < num_nbp; ++i)
    {
	free(ps_send[i]);
	free(ps_index_send[i]);
	free(ps_shift_send[i]);
	free(tris_send[i]);
	free(tris_index_send[i]);
	free(tris_shift_send[i]);
    }
    free(size_send);
    free(ps_send);
    free(tris_send);
    free(ps_index_send);
    free(tris_index_send);
    free(ps_shift_send);
    free(tris_shift_send);

    free(all_bd_box);

    if (is_overlapping_tri == 1)
	mesh->is_clean = 0;
    else
    {
	mesh->nps_clean = mesh->ps->size[0];
	mesh->ntris_clean = mesh->tris->size[0];
	mesh->npspi_clean = mesh->ps_pinfo->allocated_len;
	mesh->is_clean = 1;
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
    int num_proc;
    int cur_head, cur_tail, cur_node;
    int master;
    int rank, i, j;
    int buffer_size[1];

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_proc);

    int *ps_phead = mesh->ps_pinfo->head;
    int *ps_ptail = mesh->ps_pinfo->tail;

    hpPInfoNode *ps_pdata = mesh->ps_pinfo->pdata;

    // initialization of pointers
    mesh->ps_send_index = (emxArray_int32_T **) calloc(num_proc, sizeof(emxArray_int32_T*));
    mesh->ps_recv_index = (emxArray_int32_T **) calloc(num_proc, sizeof(emxArray_int32_T*));
    mesh->ps_send_shift = (emxArray_int8_T **) calloc(num_proc, sizeof(emxArray_int8_T *));

    for (i = 0; i < num_proc; i++)
    {
	mesh->ps_send_index[i] = (emxArray_int32_T *) NULL;
	mesh->ps_recv_index[i] = (emxArray_int32_T *) NULL;
	mesh->ps_send_shift[i] = (emxArray_int8_T *) NULL;
    }

    // compute the length of the buffers
    int* pt_send_buffer_length = (int*) calloc(num_proc, sizeof(int));
    int* pt_recv_buffer_length = (int*) calloc(num_proc, sizeof(int));

    for(i = 1; i <= num_pt; i++)
    {
	cur_head = ps_phead[I1dm(i)];
	cur_tail = ps_ptail[I1dm(i)];
	master = ps_pdata[I1dm(cur_head)].proc;

	if (cur_head == cur_tail) // Interior point, do nothing
	    continue;

	if ( (master == rank) &&
		(ps_pdata[I1dm(cur_head)].lindex == i) ) // this point is the master point, send to all other
	{
	    cur_node = ps_pdata[I1dm(cur_head)].next;
	    while(cur_node != -1)
	    {
		pt_send_buffer_length[ps_pdata[I1dm(cur_node)].proc]++;
		cur_node = ps_pdata[I1dm(cur_node)].next;
	    }
	}
	else	// this point is not the master, recv this point from master
	    pt_recv_buffer_length[master]++;
    }

    // We want the send and recv buffers have points with the same order,
    // which means that at least one buffer should be sorted.
    // We sort the send buffer in the order master_id ---> slave_id
    // we need to sort the recv buffer corresponding to the order on the sent proc
    // It is an insertion sort for a 2D array

    // remoteid is used to store the local id of the points on the send proc, we need to sort the points by this id
    int** remoteid_recv = (int**) calloc(num_proc, sizeof(int*));
    int** remoteid_send = (int **) calloc(num_proc, sizeof(int*));
    int* len_pinfo_cur_pt = (int *) calloc(num_proc, sizeof(int*));

    // allocate memory for index and buffer array, also for remoteid
    for (i = 0; i < num_nb_proc; i++)
    {
	int cur_nb_proc = mesh->nb_proc->data[i];
	if(pt_recv_buffer_length[cur_nb_proc] != 0)	//  for the recv case. 
					 		//  In this case, we need to sort the points on this proc, 
					 		//  so allocate memory for remoteid
	{
	    buffer_size[0] = pt_recv_buffer_length[cur_nb_proc];
	    mesh->ps_recv_index[cur_nb_proc] = emxCreateND_int32_T(1, buffer_size);
	    remoteid_recv[cur_nb_proc] = (int*) calloc(pt_recv_buffer_length[cur_nb_proc], sizeof(int));
	}
	if(pt_send_buffer_length[cur_nb_proc]!=0)	// for the send case
	{
	    buffer_size[0] = pt_send_buffer_length[cur_nb_proc];
	    mesh->ps_send_index[cur_nb_proc] = emxCreateND_int32_T(1, buffer_size);
	    int shift_buffer_size = 3*buffer_size[0];
	    mesh->ps_send_shift[cur_nb_proc] = emxCreateND_int8_T(1, &shift_buffer_size);
	    remoteid_send[cur_nb_proc] = (int *) calloc(pt_send_buffer_length[cur_nb_proc], sizeof(int));
	}
    }

    int* p_send_index = (int*) calloc(num_proc, sizeof(int));	// index to the end of the list
    int* p_recv_index = (int*) calloc(num_proc, sizeof(int));	// index to the end of the list

    for(i = 0; i < num_proc; i++)
    {
	p_send_index[i] = 0;
	p_recv_index[i] = 0;
    }
    for(i = 1; i <= num_pt; i++)
    {
	cur_head = ps_phead[I1dm(i)];
	cur_tail = ps_ptail[I1dm(i)];
	master = ps_pdata[I1dm(cur_head)].proc;

	if (cur_head == cur_tail)
	    continue;

	if ( (master == rank ) &&
		(ps_pdata[I1dm(cur_head)].lindex == i) ) // the current proc is the master, send to all others
	{
	    cur_node = ps_pdata[I1dm(cur_head)].next;
	    memset(len_pinfo_cur_pt, 0, num_proc*sizeof(int));
	    while(cur_node!=-1)
	    {
		int cur_nb_proc = ps_pdata[I1dm(cur_node)].proc;
		int cur_send_id = p_send_index[cur_nb_proc];

		mesh->ps_send_index[cur_nb_proc]->data[cur_send_id] = i;
		remoteid_send[cur_nb_proc][cur_send_id] = ps_pdata[I1dm(cur_node)].lindex;
		mesh->ps_send_shift[cur_nb_proc]->data[3*cur_send_id] = ps_pdata[I1dm(cur_node)].shift[0];
		mesh->ps_send_shift[cur_nb_proc]->data[3*cur_send_id+1] = ps_pdata[I1dm(cur_node)].shift[1];
		mesh->ps_send_shift[cur_nb_proc]->data[3*cur_send_id+2] = ps_pdata[I1dm(cur_node)].shift[2];

		for (j = cur_send_id; j > p_send_index[cur_nb_proc] - len_pinfo_cur_pt[cur_nb_proc]; j--)
		{
		    if (remoteid_send[cur_nb_proc][j] < remoteid_send[cur_nb_proc][j-1])
		    {
			int8_T tmp_shiftx = mesh->ps_send_shift[cur_nb_proc]->data[3*j];
			int8_T tmp_shifty = mesh->ps_send_shift[cur_nb_proc]->data[3*j+1];
			int8_T tmp_shiftz = mesh->ps_send_shift[cur_nb_proc]->data[3*j+2];
			int tmp_id_send = remoteid_send[cur_nb_proc][j];

			mesh->ps_send_shift[cur_nb_proc]->data[3*j] = mesh->ps_send_shift[cur_nb_proc]->data[3*(j-1)];
			mesh->ps_send_shift[cur_nb_proc]->data[3*j+1] = mesh->ps_send_shift[cur_nb_proc]->data[3*(j-1)+1];
			mesh->ps_send_shift[cur_nb_proc]->data[3*j+2] = mesh->ps_send_shift[cur_nb_proc]->data[3*(j-1)+2];
			remoteid_send[cur_nb_proc][j] = remoteid_send[cur_nb_proc][j-1];

			mesh->ps_send_shift[cur_nb_proc]->data[3*(j-1)] = tmp_shiftx;
			mesh->ps_send_shift[cur_nb_proc]->data[3*(j-1)+1] = tmp_shifty;
			mesh->ps_send_shift[cur_nb_proc]->data[3*(j-1)+2] = tmp_shiftz;
			remoteid_send[cur_nb_proc][j-1] = tmp_id_send;
		    }
		    else
			break;
		}

		p_send_index[cur_nb_proc]++;
		len_pinfo_cur_pt[cur_nb_proc]++;
		cur_node = ps_pdata[I1dm(cur_node)].next;
	    }
	}
	else	// the current proc is not the master, recv this point from master, sorted by insertion sort
	{
	    remoteid_recv[master][p_recv_index[master]] = ps_pdata[I1dm(cur_head)].lindex;
	    mesh->ps_recv_index[master]->data[p_recv_index[master]] = i;

	    // sort by the key of remoteid[master], the value is mesh->ps_recv_index[master]->data[]
	    for (j = p_recv_index[master]; j > 0; j--)
	    {
		if(remoteid_recv[master][j] < remoteid_recv[master][j-1])
		{
		    int tmp_pt = mesh->ps_recv_index[master]->data[j];
		    mesh->ps_recv_index[master]->data[j] = mesh->ps_recv_index[master]->data[j-1];
		    mesh->ps_recv_index[master]->data[j-1] = tmp_pt;

		    int tmp_id = remoteid_recv[master][j];
		    remoteid_recv[master][j] = remoteid_recv[master][j-1];
		    remoteid_recv[master][j-1] = tmp_id;
		}
		else
		    break;
	    }
	    p_recv_index[master]++;
	}
    }

    free(pt_send_buffer_length);
    free(pt_recv_buffer_length);

    free(p_send_index);
    free(p_recv_index);
    for (i = 0; i < num_proc; i++)
    {
	if(remoteid_send[i] != NULL)
	    free(remoteid_send[i]);
	if(remoteid_recv[i] != NULL)
	    free(remoteid_recv[i]);
    }

    free(remoteid_send);
    free(remoteid_recv);
    free(len_pinfo_cur_pt);
}

void hpCleanMeshByPinfo(hiPropMesh* mesh)
{
    int rank, num_proc;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_proc);
    int i, j;

    emxArray_int32_T *old_tris = mesh->tris;
    int32_T *old_tris_data = old_tris->data;
    int32_T *old_tris_size = old_tris->size;

    emxArray_real_T *old_ps = mesh->ps;
    real_T *old_ps_data = old_ps->data;
    int32_T *old_ps_size = old_ps->size;

    hpPInfoList* old_ps_pinfo =  mesh->ps_pinfo;
    hpPInfoNode* old_ps_pdata = old_ps_pinfo->pdata;
    int *old_ps_phead = old_ps_pinfo->head;

    hpPInfoList* old_tris_pinfo = mesh->tris_pinfo;
    hpPInfoNode* old_tris_pdata = old_tris_pinfo->pdata;
    int *old_tris_phead = old_tris_pinfo->head;

    // 1. Identify the tris to be saved, save in tris_to_save[]

    int num_old_tri = old_tris->size[0];
    int num_old_pt = old_ps->size[0];
    int num_nb_proc = mesh->nb_proc->size[0];
    
    int num_tri_to_save = 0;

    int *tris_to_save = (int *) calloc(num_old_tri, sizeof(int));

    int head;
    for (i = 1; i <= num_old_tri; i++)
    {
	head = old_tris_phead[I1dm(i)];
	if( (old_tris_pdata[I1dm(head)].proc == rank) 
		&& (old_tris_pdata[I1dm(head)].lindex == i) )
	{
	    tris_to_save[num_tri_to_save++] = i;
	}
    }



    // 2. make a new mesh->ps array, delete the old, 
    //   use pt_new_index to save the new index, if the pt is deleted, use 0.
    //
   // if current point is to be saved, save the new index here, otherwise, 0
    int *pt_new_index = (int *) calloc(num_old_pt, sizeof(int));

    int cur_tri;
    int cur_pt;
    for(i = 0; i < num_tri_to_save; i++)
    {
	cur_tri = tris_to_save[i];
	for(j = 1; j <= 3; j++)
	{
	    cur_pt = old_tris_data[I2dm(cur_tri,j,old_tris_size)];
	    pt_new_index[I1dm(cur_pt)] = 1;	// initialize to 1 first, then add up to get the real index
	}
    }

    // Make index mapping for old to new ps
    int num_pt_to_save = 0;
    for (i = 0; i < num_old_pt; i++)
    {
	if( pt_new_index[i] == 1 )
    	    pt_new_index[i] = (++num_pt_to_save);
    }

    // Allocate the storage for new ps
    emxArray_real_T *newpts = emxCreate_real_T(num_pt_to_save, 3);
    real_T *new_ps_data = newpts->data;
    int32_T *new_ps_size = newpts->size;

    // Fill the new ps array
    for(i = 0; i < num_old_pt; i++)
    {
	if( pt_new_index[i] != 0 )
	{
	    for(j = 1; j <= 3; j++)
    		new_ps_data[I2dm(pt_new_index[i], j, new_ps_size)] = old_ps_data[I2dm((i+1), j, old_ps_size)];
	}
    }

    emxFree_real_T(&mesh->ps);
    mesh->ps = newpts;

    // 3. make a new mesh->tris array, delete the old one
    emxArray_int32_T *newtris = emxCreate_int32_T(num_tri_to_save, 3);
    int32_T *new_tris_data = newtris->data;
    int32_T *new_tris_size = newtris->size;

    int p, old_p;
    for (i = 1; i <= num_tri_to_save; i++)
    {
	cur_tri = tris_to_save[I1dm(i)];
	for(j = 1; j <= 3; j++)
	{
	    old_p = old_tris_data[I2dm(cur_tri, j, old_tris_size)];	// old point index
	    p = pt_new_index[I1dm(old_p)];	// new point index
	    new_tris_data[I2dm(i,j,new_tris_size)] = p;
	}
    }

    emxFree_int32_T(&mesh->tris);
    mesh->tris = newtris;

    free(tris_to_save);

    // 4. For the points remaining in the proc, search the old ps_pinfo list.
    //    for the ps to be saved, check the old ps_pinfo list
    //   to see if we need to send that point to the nb procs,
    //   count the number of points to be sent to each nb here.


    //    If the point exists on some neighbour in the old list, send it to
    //    that neighbour.
    //    For each nb proc, send the point new local index on the current proc, 
    //    and also the old remote index to help the remote proc to identify the point.

    //  use proc_position to locate one proc in the nb_proc list easily
    int* proc_position = (int*) calloc(num_proc, sizeof(int));
    for (i = 0; i < num_proc; i++)
	proc_position[i] = -1;


    for(i = 0; i < num_nb_proc; i++)
	proc_position[mesh->nb_proc->data[i]] = i;

    int *num_of_overlapping_ps = (int *) calloc(num_nb_proc, sizeof(int));

    for(i = 1; i <= num_old_pt; i++)
    {
	if( pt_new_index[i-1] != 0 )
	{
	    // count the number of ps to be sent to the neighbour procs
	    int next = old_ps_phead[i-1];
	    while (next != -1)
	    {
		int proc_rank = old_ps_pinfo->pdata[I1dm(next)].proc;
		if( (proc_rank != rank) || (old_ps_pdata[I1dm(next)].lindex != i) )

		    (num_of_overlapping_ps[proc_position[proc_rank]])++;
		next = old_ps_pdata[I1dm(next)].next;
	    }
	}
    }

    int tag[4] = {1,2,3,4};
    MPI_Request *request_send = (MPI_Request *) calloc(4*num_nb_proc, sizeof(MPI_Request));
    for (i = 0; i < 4*num_nb_proc; i++)
	request_send[i] = MPI_REQUEST_NULL;
    int dest;

    // for each proc, find the overlapping points in the old ps_pinfo list

    int **remote_index = (int **) calloc(num_nb_proc, sizeof(int *));
    int **local_index = (int **) calloc(num_nb_proc, sizeof(int *));
    int8_T **shift = (int8_T **) calloc(num_nb_proc, sizeof(int8_T *));
    for (i = 0; i < num_nb_proc; i++)
    {
	remote_index[i] = (int *) calloc( num_of_overlapping_ps[i], sizeof(int));
	local_index[i] = (int *) calloc( num_of_overlapping_ps[i], sizeof(int));
	shift[i] = (int8_T*) calloc( 3*num_of_overlapping_ps[i], sizeof(int8_T));
    }

    int *last_index = (int *) calloc(num_nb_proc, sizeof(int));
    
    for(i = 1; i <= num_old_pt; i++)
    {
	if( pt_new_index[i-1] != 0 )
	{
	    int next = old_ps_phead[I1dm(i)];
	    while (next != -1)
	    {
		int proc_rank = old_ps_pdata[I1dm(next)].proc;
		if( (proc_rank != rank) || (old_ps_pdata[I1dm(next)].lindex != i) )
		{
		    int cur_pos = proc_position[proc_rank];
		    int cur_last = last_index[cur_pos]++;
		    remote_index[cur_pos][cur_last] = old_ps_pdata[I1dm(next)].lindex;
		    local_index[cur_pos][cur_last] = pt_new_index[i-1];
		    shift[cur_pos][3*cur_last] = -old_ps_pdata[I1dm(next)].shift[0];
		    shift[cur_pos][3*cur_last+1] = -old_ps_pdata[I1dm(next)].shift[1];
		    shift[cur_pos][3*cur_last+2] = -old_ps_pdata[I1dm(next)].shift[2];
		}
		next = old_ps_pdata[I1dm(next)].next;
	    }

	}
    }

    free(proc_position);
    free(last_index);

    for (i = 0; i < num_nb_proc; i++)
    {
	dest = mesh->nb_proc->data[i];

	MPI_Isend(&num_of_overlapping_ps[i], 1, MPI_INT, dest, tag[0], MPI_COMM_WORLD, &request_send[3*i]);

	if (num_of_overlapping_ps[i] != 0)
	{
	    MPI_Isend(remote_index[i], num_of_overlapping_ps[i], MPI_INT, dest, tag[1], MPI_COMM_WORLD, &request_send[3*i+1]);
	    MPI_Isend(local_index[i], num_of_overlapping_ps[i], MPI_INT, dest, tag[2], MPI_COMM_WORLD, &request_send[3*i+2]);
	    MPI_Isend(shift[i], 3*num_of_overlapping_ps[i], MPI_SIGNED_CHAR, dest, tag[3], MPI_COMM_WORLD, &request_send[3*i+3]);
	}
    }



    // 5. Recv cur_local_index and cur_remote_index info from all neighbours.
    //   Note here, the received cur local index is the remote index in send,
    //   the received cur remote index is the local index in send.
    //   We should initialize a new pinfo list first, then put more nodes
    //   into the ps_pinfo list using the information received.

    int *num_pt_to_recv = (int *)calloc(num_nb_proc, sizeof(int));

    int source;
    MPI_Status status;
    MPI_Status status1;
    MPI_Status status2;
    MPI_Status status3;

    MPI_Request *req_recv_num = (MPI_Request *) calloc(num_nb_proc, sizeof(MPI_Request));
    int recv_index;
    for (i = 0; i < num_nb_proc; i++)
    {
	MPI_Irecv(&num_pt_to_recv[i], 1, MPI_INT, mesh->nb_proc->data[i], tag[0], MPI_COMM_WORLD, &req_recv_num[i]);
    }

    // Init a new pinfo list for points and triangles, do it here to have more overlapping
    // between computation and communication

    hpInitPInfo(mesh);
    hpPInfoList *new_ps_pinfo = mesh->ps_pinfo;
    hpPInfoNode *new_ps_pdata = new_ps_pinfo->pdata;
    int *new_ps_phead = new_ps_pinfo->head;
    int *new_ps_ptail = new_ps_pinfo->tail;


    for(i = 0; i < num_nb_proc; i++)
    {
	MPI_Waitany(num_nb_proc, req_recv_num, &recv_index, &status);
	source = status.MPI_SOURCE;

	int recv_num_ps = num_pt_to_recv[recv_index];
	
	if (recv_num_ps != 0)
	{
	    int *cur_local_index = (int *) calloc(recv_num_ps, sizeof(int));
	    int *cur_remote_index = (int *) calloc(recv_num_ps, sizeof(int));
	    int8_T *cur_shift = (int8_T *) calloc(3*recv_num_ps, sizeof(int8_T));

	    MPI_Recv(cur_local_index, recv_num_ps, MPI_INT, source, tag[1], MPI_COMM_WORLD, &status1);
	    MPI_Recv(cur_remote_index, recv_num_ps, MPI_INT, source, tag[2], MPI_COMM_WORLD, &status2);
	    MPI_Recv(cur_shift, 3*recv_num_ps, MPI_SIGNED_CHAR, source, tag[3], MPI_COMM_WORLD, &status3);


	    for (j = 0; j < recv_num_ps; j++ )
	    {
		int cur_pt = pt_new_index[I1dm(cur_local_index[j])];

		if(cur_pt != 0)
		{
		    if(hpEnsurePInfoCapacity(new_ps_pinfo))
			new_ps_pdata = new_ps_pinfo->pdata;
		    new_ps_pinfo->allocated_len++;
		    int cur_head = new_ps_phead[I1dm(cur_pt)];
		    int cur_tail = new_ps_ptail[I1dm(cur_pt)];
		    int cur_master_proc = new_ps_pdata[I1dm(cur_head)].proc;
		    int cur_master_index = new_ps_pdata[I1dm(cur_head)].lindex;
		    if (source < cur_master_proc)
		    {
			new_ps_phead[I1dm(cur_pt)] = new_ps_pinfo->allocated_len;
			new_ps_pdata[I1dm(new_ps_pinfo->allocated_len)].proc = source;
			new_ps_pdata[I1dm(new_ps_pinfo->allocated_len)].lindex = cur_remote_index[j];
			new_ps_pdata[I1dm(new_ps_pinfo->allocated_len)].shift[0] = cur_shift[3*j];
			new_ps_pdata[I1dm(new_ps_pinfo->allocated_len)].shift[1] = cur_shift[3*j+1];
			new_ps_pdata[I1dm(new_ps_pinfo->allocated_len)].shift[2] = cur_shift[3*j+2];
			new_ps_pdata[I1dm(new_ps_pinfo->allocated_len)].next = cur_head;
		    }
		    else if (source > cur_master_proc)
		    {
			new_ps_ptail[I1dm(cur_pt)] = new_ps_pinfo->allocated_len;

			new_ps_pdata[I1dm(new_ps_pinfo->allocated_len)].proc = source;
			new_ps_pdata[I1dm(new_ps_pinfo->allocated_len)].lindex = cur_remote_index[j];
			new_ps_pdata[I1dm(new_ps_pinfo->allocated_len)].shift[0] = cur_shift[3*j];
			new_ps_pdata[I1dm(new_ps_pinfo->allocated_len)].shift[1] = cur_shift[3*j+1];
			new_ps_pdata[I1dm(new_ps_pinfo->allocated_len)].shift[2] = cur_shift[3*j+2];

			new_ps_pinfo->pdata[I1dm(new_ps_pinfo->allocated_len)].next = -1;
			new_ps_pinfo->pdata[I1dm(cur_tail)].next = new_ps_pinfo->allocated_len;
		    }
		    else
		    {
			if (cur_remote_index[j] < cur_master_index)
			{
			    new_ps_phead[I1dm(cur_pt)] = new_ps_pinfo->allocated_len;
			    new_ps_pdata[I1dm(new_ps_pinfo->allocated_len)].proc = source;
			    new_ps_pdata[I1dm(new_ps_pinfo->allocated_len)].lindex = cur_remote_index[j];
			    new_ps_pdata[I1dm(new_ps_pinfo->allocated_len)].shift[0] = cur_shift[3*j];
			    new_ps_pdata[I1dm(new_ps_pinfo->allocated_len)].shift[1] = cur_shift[3*j+1];
			    new_ps_pdata[I1dm(new_ps_pinfo->allocated_len)].shift[2] = cur_shift[3*j+2];
			    new_ps_pdata[I1dm(new_ps_pinfo->allocated_len)].next = cur_head;
			}
			else if (cur_remote_index[j] > cur_master_index)
			{
			    new_ps_ptail[I1dm(cur_pt)] = new_ps_pinfo->allocated_len;

			    new_ps_pdata[I1dm(new_ps_pinfo->allocated_len)].proc = source;
			    new_ps_pdata[I1dm(new_ps_pinfo->allocated_len)].lindex = cur_remote_index[j];
			    new_ps_pdata[I1dm(new_ps_pinfo->allocated_len)].shift[0] = cur_shift[3*j];
			    new_ps_pdata[I1dm(new_ps_pinfo->allocated_len)].shift[1] = cur_shift[3*j+1];
			    new_ps_pdata[I1dm(new_ps_pinfo->allocated_len)].shift[2] = cur_shift[3*j+2];

			    new_ps_pinfo->pdata[I1dm(new_ps_pinfo->allocated_len)].next = -1;
			    new_ps_pinfo->pdata[I1dm(cur_tail)].next = new_ps_pinfo->allocated_len;
			}
			else
			{
			    printf("\n Receiving element already in the PInfo list!\n");
			    exit(0);
			}
		    }
		}
	    }
	    free(cur_local_index);
	    free(cur_remote_index);
	    free(cur_shift);
	}
    }


    MPI_Status *status_send = (MPI_Status *) calloc(4*num_nb_proc, sizeof(MPI_Status));
    MPI_Waitall(4*num_nb_proc, request_send, status_send);
    free(status_send);
    free(request_send);

    free(num_of_overlapping_ps);
    free(pt_new_index);
    for(i = 0; i < num_nb_proc; i++)
    {
	free(remote_index[i]);
	free(local_index[i]);
	free(shift[i]);
    }
    free(remote_index);
    free(local_index);
    free(shift);

    free(num_pt_to_recv);
    free(req_recv_num);

    hpUpdateNbWithPInfo(mesh);


    mesh->nps_clean = mesh->ps->size[0];
    mesh->ntris_clean = mesh->tris->size[0];
    mesh->npspi_clean = mesh->ps_pinfo->allocated_len;
    mesh->is_clean = 1;
    
    printf("Passed hpCleanMeshByPinfo\n");
}

void hpPrint_pinfo(hiPropMesh *mesh)
{
    printf("Getting into hpPrint_pinfo()\n\n");
    fflush(stdout);

    int i;
    printf("size of nb_proc list: %d\n", mesh->nb_proc->size[0] );
    printf("nb_proc list: ");
    for (i=1; i<=mesh->nb_proc->size[0]; i++)
	printf("%d -> ", mesh->nb_proc->data[I1dm(i)] );
    printf("\n");

    // for printing the pinfo
    printf("\nps pinfo:\n");
    for (i = 1; i <= mesh->ps->size[0]; i++)
    {
	printf("point %d: ", i);
	int next = mesh->ps_pinfo->head[I1dm(i)];
	while (next != -1)
	{

	    int cur_node = next;

	    printf("%d/%d(%d %d %d)[%d]-->", mesh->ps_pinfo->pdata[I1dm(cur_node)].proc, mesh->ps_pinfo->pdata[I1dm(cur_node)].lindex,
		    mesh->ps_pinfo->pdata[I1dm(cur_node)].shift[0], mesh->ps_pinfo->pdata[I1dm(cur_node)].shift[1],
		    mesh->ps_pinfo->pdata[I1dm(cur_node)].shift[2], mesh->ps_pinfo->pdata[I1dm(cur_node)].next);
	    next = mesh->ps_pinfo->pdata[I1dm(cur_node)].next;

	}
	printf("\n");
	printf("Head = %d, Tail = %d\n", mesh->ps_pinfo->head[I1dm(i)], mesh->ps_pinfo->tail[I1dm(i)]);
    }

    printf("tris pinfo:\n");
    for (i = 1; i <= mesh->tris->size[0]; i++)
    {
	printf("triangle %d: ", i);
	int next = mesh->tris_pinfo->head[I1dm(i)];
	while (next != -1)
	{
	    int cur_node = next;

	    printf("%d/%d(%d %d %d)-->", mesh->tris_pinfo->pdata[I1dm(cur_node)].proc, mesh->tris_pinfo->pdata[I1dm(cur_node)].lindex,
		    mesh->tris_pinfo->pdata[I1dm(cur_node)].shift[0], mesh->tris_pinfo->pdata[I1dm(cur_node)].shift[1],
		    mesh->tris_pinfo->pdata[I1dm(cur_node)].shift[2]);
	    next = mesh->tris_pinfo->pdata[I1dm(cur_node)].next;
	}
	printf("\n");
	printf("Head = %d, Tail = %d\n", mesh->tris_pinfo->head[I1dm(i)], mesh->tris_pinfo->tail[I1dm(i)]);
    }
    printf("Getting out of hpPrint_pinfo()\n");
    fflush(stdout);
}


void hpCollectAllSharedPs(const hiPropMesh *mesh, emxArray_int32_T **out_psid, emxArray_int8_T **out_ps_shift)
{
    int num_nb_proc = mesh->nb_proc->size[0];

    int i;
    int num_all_proc, cur_proc;
    MPI_Comm_size(MPI_COMM_WORLD, &num_all_proc);
    MPI_Comm_rank(MPI_COMM_WORLD, &cur_proc);

    int32_T *ps_size = mesh->ps->size;

    hpPInfoList *ps_pinfo = mesh->ps_pinfo;
    hpPInfoNode *ps_pdata = ps_pinfo->pdata;
    int *ps_phead = ps_pinfo->head;

    int *num_overlay_ps = (int *) calloc(num_nb_proc, sizeof(int));

    // nb proc mapping[i] stores the index for proc i in
    // mesh->proc->data (starts from 0) 

    int *nb_proc_mapping = (int *) calloc(num_all_proc, sizeof(int));

    // Initialize nb_proc_mapping, for non nb proc, map to -1 
    for (i = 0; i < num_all_proc; i++)
	nb_proc_mapping[i] = -1;

    // construct the nb proc mapping
    for (i = 1; i <= num_nb_proc; i++)
    {
	int nb_proc_id = mesh->nb_proc->data[I1dm(i)];
	nb_proc_mapping[nb_proc_id] = i-1;
    }

    // Traverse the ps_pinfo to fill num_overlay_ps
    for (i = 1; i <= ps_size[0]; i++)
    {
	int next_node = ps_phead[I1dm(i)];
	while (next_node != -1)
	{
	    int proc_id = ps_pdata[I1dm(next_node)].proc;
	    int local_id = ps_pdata[I1dm(next_node)].lindex;

	    if ((proc_id != cur_proc) || (local_id != i) )
		num_overlay_ps[nb_proc_mapping[proc_id]]++;
	    next_node = ps_pdata[I1dm(next_node)].next;
	}
    }
    
    // Create out_psid[i] based on num_overlay_ps
    for (i = 0; i < num_nb_proc; i++)
    {
	out_psid[i] = emxCreateND_int32_T(1, &(num_overlay_ps[i]) );
	int ps_shift_len = 3*num_overlay_ps[i];

	out_ps_shift[i] = emxCreateND_int8_T(1, &ps_shift_len);
    }


    
    // use this pointer to denote how many elements has been filled
    // in out_psid[i]
    int *cur_ps_index = (int *) calloc(num_nb_proc, sizeof(int));

    // Traverse the ps_pinfo to fill out_psid
    for (i = 1; i <= ps_size[0]; i++)
    {
	int next_node = ps_phead[I1dm(i)];
	while (next_node != -1)
	{
	    int proc_id = ps_pdata[I1dm(next_node)].proc;
	    int local_id = ps_pdata[I1dm(next_node)].lindex;
	    if ( (proc_id != cur_proc) || (local_id != i) )
	    {
		int mapped_index = nb_proc_mapping[proc_id];
		int cur_iter = cur_ps_index[mapped_index];
		(out_psid[mapped_index])->data[cur_iter] = i;
		(out_ps_shift[mapped_index])->data[cur_iter*3] = ps_pdata[I1dm(next_node)].shift[0];
		(out_ps_shift[mapped_index])->data[cur_iter*3+1] = ps_pdata[I1dm(next_node)].shift[1];
		(out_ps_shift[mapped_index])->data[cur_iter*3+2] = ps_pdata[I1dm(next_node)].shift[2];
		(cur_ps_index[mapped_index])++;
	    }
	    next_node = ps_pdata[I1dm(next_node)].next;
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
				    emxArray_int8_T **ps_shift_ring_proc,
				    emxArray_int8_T **tris_shift_ring_proc,
				    emxArray_real_T **buffer_ps,
				    emxArray_int32_T **buffer_tris)
{
    int i, j, is;

    int target_proc = mesh->nb_proc->data[I1dm(nb_proc_index)];

    emxArray_int32_T *tris = mesh->tris;
    emxArray_real_T *ps = mesh->ps;

    real_T *ps_data = ps->data;
    int32_T *ps_size = ps->size;

    int32_T *tris_data = tris->data;
    int32_T *tris_size = tris->size;

    int num_tris = tris->size[0];
    int num_ps = ps->size[0];

    hpPInfoNode *tris_pdata = mesh->tris_pinfo->pdata;
    int *tris_phead = mesh->tris_pinfo->head;

    double domain_len_x = mesh->domain_len[0];
    double domain_len_y = mesh->domain_len[1];
    double domain_len_z = mesh->domain_len[2];

    emxArray_int8_T *nb_shift = (mesh->nb_proc_shift)[I1dm(nb_proc_index)];
    int num_shifts = nb_shift->size[0];

    // target bounding box (without shift)
    double target_xL = bd_box[target_proc*6];
    double target_xU = bd_box[target_proc*6+1];
    double target_yL = bd_box[target_proc*6+2];
    double target_yU = bd_box[target_proc*6+3];
    double target_zL = bd_box[target_proc*6+4];
    double target_zU = bd_box[target_proc*6+5];

    double max_len = target_xU - target_xL;
    if ( (target_yU-target_yL) > max_len)
	max_len = target_yU-target_yL;
    if ( (target_zU-target_zL) > max_len)
	max_len = target_zU-target_zL;

    double eps = 1e-10*max_len;

    // relax the bounding box
    target_xL -= eps; target_xU += eps;
    target_yL -= eps; target_yU += eps;
    target_zL -= eps; target_zU += eps;

    int **ps_id_temp = (int **) calloc(num_shifts, sizeof(int *));
    int **tris_id_temp = (int **) calloc(num_shifts, sizeof(int *));
    int **tris_buffer_temp = (int **) calloc(num_shifts, sizeof(int *));

    int *num_ps_ring_shift = (int *) calloc(num_shifts, sizeof(int));
    int *num_tris_ring_shift = (int *) calloc(num_shifts, sizeof(int));

    int num_ps_ring_all = 0;
    int num_tris_ring_all = 0;

    for (is = 0; is < num_shifts; ++is)
    {
	boolean_T *tris_flag = (boolean_T *) calloc(num_tris, sizeof(boolean_T));
	boolean_T *ps_flag = (boolean_T *) calloc(num_ps, sizeof(boolean_T));
	int *ps_map = (int *) calloc(num_ps, sizeof(int));

	int8_T sx, sy, sz;

	sx = nb_shift->data[I2dm(is+1,1,nb_shift->size)];
	sy = nb_shift->data[I2dm(is+1,2,nb_shift->size)];
	sz = nb_shift->data[I2dm(is+1,3,nb_shift->size)];

	double x_factor = ((double)sx*domain_len_x);
	double y_factor = ((double)sy*domain_len_y);
	double z_factor = ((double)sz*domain_len_z);


	for (i = 1; i <= num_tris; i++)
	{
	    int psi = tris_data[I2dm(i,1,tris_size)]; // first point
	    double trixL = ps_data[I2dm(psi,1,ps_size)];
	    double trixU = trixL;
	    double triyL = ps_data[I2dm(psi,2,ps_size)];
	    double triyU = triyL;
	    double trizL = ps_data[I2dm(psi,3,ps_size)];
	    double trizU = trizL;
	    for (j = 2; j <= 3; j++)
	    {
		psi = tris_data[I2dm(i,j,tris_size)];
		if(ps_data[I2dm(psi,1,ps_size)] < trixL)
		    trixL = ps_data[I2dm(psi,1,ps_size)];
		else if(ps_data[I2dm(psi,1,ps_size)] > trixU)
		    trixU = ps_data[I2dm(psi,1,ps_size)];

		if(ps_data[I2dm(psi,2,ps_size)] < triyL)
		    triyL = ps_data[I2dm(psi,2,ps_size)];
		else if(ps_data[I2dm(psi,2,ps_size)] > triyU)
		    triyU = ps_data[I2dm(psi,2,ps_size)];

		if(ps_data[I2dm(psi,3,ps_size)] < trizL)
		    trizL = ps_data[I2dm(psi,3,ps_size)];
		else if(ps_data[I2dm(psi,3,ps_size)] > trizU)
		    trizU = ps_data[I2dm(psi,3,ps_size)];
	    }

	    double comxL = hpMax(target_xL + x_factor, trixL);
	    double comxU = hpMin(target_xU + x_factor, trixU);
	    double comyL = hpMax(target_yL + y_factor, triyL);
	    double comyU = hpMin(target_yU + y_factor, triyU);
	    double comzL = hpMax(target_zL + z_factor, trizL);
	    double comzU = hpMin(target_zU + z_factor, trizU);

	    if ( (comxL <= comxU) && (comyL <= comyU) && (comzL <= comzU) )	
	    {
		tris_flag[I1dm(i)] = 1;
		// Only send the tris & points not existing on the nb proc

		int next_node = tris_phead[I1dm(i)];
		while (next_node != -1)
		{
		    if ( (tris_pdata[I1dm(next_node)].proc == target_proc) &&
			    (sx == tris_pdata[I1dm(next_node)].shift[0]) &&
			    (sy == tris_pdata[I1dm(next_node)].shift[1]) &&
			    (sz == tris_pdata[I1dm(next_node)].shift[2]) )
		    {
			tris_flag[I1dm(i)] = 0;
			break;
		    }
		    else
			next_node = tris_pdata[I1dm(next_node)].next;
		}

		if (tris_flag[I1dm(i)])
		{
		    int bufpi_x = tris_data[I2dm(i,1,tris_size)];
		    int bufpi_y = tris_data[I2dm(i,2,tris_size)];
		    int bufpi_z = tris_data[I2dm(i,3,tris_size)];
		    ps_flag[I1dm(bufpi_x)] = 1;
		    ps_flag[I1dm(bufpi_y)] = 1;
		    ps_flag[I1dm(bufpi_z)] = 1;
		}
	    }
	}

	// Get number of ps for send of current shift and the sum
	// Set ps_map for building tris
	for (i = 1; i <= num_ps; i++)
	{
	    if (ps_flag[I1dm(i)])
	    {
		(num_ps_ring_shift[is])++;
		num_ps_ring_all++;

		ps_map[I1dm(i)] = num_ps_ring_all;
	    }
	}

	// Get number of tris for send of current shift and the sum
	for (i = 1; i <= num_tris; i++)
	{
	    if (tris_flag[I1dm(i)])
	    {
		(num_tris_ring_shift[is])++;
		num_tris_ring_all++;
	    }
	}

	ps_id_temp[is] = (int *) calloc(num_ps_ring_shift[is], sizeof(int));
	tris_id_temp[is] = (int *) calloc(num_tris_ring_shift[is], sizeof(int));

	tris_buffer_temp[is] = (int *) calloc(3*num_tris_ring_shift[is], sizeof(int));

	int *cur_ps_id_temp = ps_id_temp[is];
	int *cur_tris_id_temp = tris_id_temp[is];
	int *cur_tris_buffer_temp = tris_buffer_temp[is];

	int ps_iter = 0;
	int tris_iter = 0;

	for (i = 1; i <= num_ps; i++)
	{
	    if (ps_flag[I1dm(i)])
		cur_ps_id_temp[ps_iter++] = i;
	}

	for (i = 1; i <= num_tris; i++)
	{
	    if (tris_flag[I1dm(i)])
	    {
		cur_tris_id_temp[tris_iter] = i;

		cur_tris_buffer_temp[tris_iter*3] = ps_map[I1dm(tris_data[I2dm(i,1,tris_size)])];
		cur_tris_buffer_temp[tris_iter*3+1] = ps_map[I1dm(tris_data[I2dm(i,2,tris_size)])];
		cur_tris_buffer_temp[tris_iter*3+2] = ps_map[I1dm(tris_data[I2dm(i,3,tris_size)])];

		tris_iter++;
	    }
	}


	free(tris_flag);
	free(ps_flag);
	free(ps_map);
    }
    // Combine ps_ring and tris_ring for each shift to get final array

    int num_ps_ring_shift_all = 3*num_ps_ring_all;
    int num_tris_ring_shift_all = 3*num_tris_ring_all;

    (*ps_ring_proc) = emxCreateND_int32_T(1, &num_ps_ring_all);
    (*tris_ring_proc) = emxCreateND_int32_T(1, &num_tris_ring_all);

    (*buffer_tris) = emxCreate_int32_T(num_tris_ring_all, 3);
    (*buffer_ps) = emxCreate_real_T(num_ps_ring_all, 3);

    (*ps_shift_ring_proc) = emxCreateND_int8_T(1, &num_ps_ring_shift_all);
    (*tris_shift_ring_proc) = emxCreateND_int8_T(1, &num_tris_ring_shift_all);

    int ps_iter = 0;
    int tris_iter = 0;

    emxArray_int32_T *result_ps = *ps_ring_proc;
    emxArray_int32_T *result_tris = *tris_ring_proc;

    emxArray_int32_T *result_tris_buffer = *buffer_tris;
    emxArray_real_T *result_ps_buffer = *buffer_ps;

    emxArray_int8_T *result_ps_shift = *ps_shift_ring_proc;
    emxArray_int8_T *result_tris_shift = *tris_shift_ring_proc;

    for (is = 0; is < num_shifts; is++)
    {
	int8_T sx, sy, sz;

	sx = nb_shift->data[I2dm(is+1,1,nb_shift->size)];
	sy = nb_shift->data[I2dm(is+1,2,nb_shift->size)];
	sz = nb_shift->data[I2dm(is+1,3,nb_shift->size)];

	int num_ps_ring_cur_shift = num_ps_ring_shift[is];
	int num_tris_ring_cur_shift = num_tris_ring_shift[is];

	int *ps_ring_cur_shift = ps_id_temp[is];
	int *tris_ring_cur_shift = tris_id_temp[is];
	int *tris_buffer_ring_cur_shift = tris_buffer_temp[is];

	for (i = 0; i < num_ps_ring_cur_shift; i++)
	{
	    result_ps->data[ps_iter] = ps_ring_cur_shift[i];
	    result_ps_shift->data[ps_iter*3] = sx;
	    result_ps_shift->data[ps_iter*3+1] = sy;
	    result_ps_shift->data[ps_iter*3+2] = sz;

	    result_ps_buffer->data[I2dm(ps_iter+1, 1, result_ps_buffer->size)] = ps_data[I2dm(ps_ring_cur_shift[i], 1, ps_size)] - ((double)sx*domain_len_x);
	    result_ps_buffer->data[I2dm(ps_iter+1, 2, result_ps_buffer->size)] = ps_data[I2dm(ps_ring_cur_shift[i], 2, ps_size)] - ((double)sy*domain_len_y);
	    result_ps_buffer->data[I2dm(ps_iter+1, 3, result_ps_buffer->size)] = ps_data[I2dm(ps_ring_cur_shift[i], 3, ps_size)] - ((double)sz*domain_len_z);

	    ps_iter++;
	}

	for (i = 0; i < num_tris_ring_cur_shift; i++)
	{
	    result_tris->data[tris_iter] = tris_ring_cur_shift[i];
	    result_tris_shift->data[tris_iter*3] = sx;
	    result_tris_shift->data[tris_iter*3+1] = sy;
	    result_tris_shift->data[tris_iter*3+2] = sz;

	    tris_iter++;

	    result_tris_buffer->data[I2dm(tris_iter, 1, result_tris_buffer->size)] = tris_buffer_ring_cur_shift[i*3];
	    result_tris_buffer->data[I2dm(tris_iter, 2, result_tris_buffer->size)] = tris_buffer_ring_cur_shift[i*3+1];
	    result_tris_buffer->data[I2dm(tris_iter, 3, result_tris_buffer->size)] = tris_buffer_ring_cur_shift[i*3+2];
	}
    }

    for (is = 0; is < num_shifts; is++)
    {
	free(ps_id_temp[is]);
	free(tris_id_temp[is]);
	free(tris_buffer_temp[is]);
    }

    free(ps_id_temp);
    free(tris_id_temp);
    free(tris_buffer_temp);

    free(num_ps_ring_shift);
    free(num_tris_ring_shift);
 
}

void hpBuildGhostPsTrisForSend(const hiPropMesh *mesh,
	const int nb_proc_index,
	const real_T num_ring,
	emxArray_int32_T *psid_proc,
	emxArray_int8_T *ps_shift_proc,
	emxArray_int32_T **ps_ring_proc,
	emxArray_int32_T **tris_ring_proc,
	emxArray_int8_T **ps_shift_ring_proc,
	emxArray_int8_T **tris_shift_ring_proc,
	emxArray_real_T **buffer_ps,
	emxArray_int32_T **buffer_tris)
{
    // Get nring nb between current proc and all nb processors  
    //
    // Point positions stored in k_i*3 double matrices buffer_ps[i] where
    // k_i = # of points in the n-ring buffer for mesh->nb_proc->data[i].
    // Triangle indices mapped to the index for buffer_ps[i] and stored
    // in buffer_tris[i];

    emxArray_real_T *ps = mesh->ps;
    real_T *ps_data = ps->data;
    int32_T *ps_size = ps->size;

    int j;

    double domain_len_x = mesh->domain_len[0];
    double domain_len_y = mesh->domain_len[1];
    double domain_len_z = mesh->domain_len[2];

    hpCollectNRingTris(mesh, nb_proc_index, psid_proc, ps_shift_proc, num_ring,
	    ps_ring_proc, tris_ring_proc, ps_shift_ring_proc, tris_shift_ring_proc, buffer_tris);

    int num_ps_buffer = (*ps_ring_proc)->size[0];

    (*buffer_ps) = emxCreate_real_T(num_ps_buffer, 3);

    real_T *buffer_ps_data = (*buffer_ps)->data;
    int32_T *buffer_ps_size = (*buffer_ps)->size;

    for (j = 1; j <= num_ps_buffer; j++)
    {
	int cur_buf_ps_index = (*ps_ring_proc)->data[I1dm(j)];

	double shift_x, shift_y, shift_z;

	int8_T sx, sy, sz;

	sx = (*ps_shift_ring_proc)->data[3*(j-1)];
	sy = (*ps_shift_ring_proc)->data[3*(j-1)+1];
	sz = (*ps_shift_ring_proc)->data[3*(j-1)+2];

	shift_x = ((double) -sx)*domain_len_x;
	shift_y = ((double) -sy)*domain_len_y;
	shift_z = ((double) -sz)*domain_len_z;

	buffer_ps_data[I2dm(j,1,buffer_ps_size)] =
	    ps_data[I2dm(cur_buf_ps_index,1,ps_size)] + shift_x;
	buffer_ps_data[I2dm(j,2,buffer_ps_size)] =
	    ps_data[I2dm(cur_buf_ps_index,2,ps_size)] + shift_y;
	buffer_ps_data[I2dm(j,3,buffer_ps_size)] =
	    ps_data[I2dm(cur_buf_ps_index,3,ps_size)] + shift_z;
    }

    /************* Debugging output **********************************
     * char rank_str[5];
     * char nb_rank_str[5];
     * numIntoString(cur_proc,4,rank_str);
     * numIntoString(mesh->nb_proc->data[I1dm(i)], 4, nb_rank_str);
     * char debug_out_name[250];
     * sprintf(debug_out_name, "debugout-p%s-to-p%s.vtk", rank_str, nb_rank_str);
     * hpDebugOutput(mesh, ps_ring_proc[I1dm(i)], tris_ring_proc[I1dm(i)], debug_out_name);
     * ******************************************************************/
}

void hpAddProcInfoForGhostPsTris(hiPropMesh *mesh,
	const int nb_proc_index,
	emxArray_int32_T *ps_ring_proc,
	emxArray_int32_T *tris_ring_proc,
	emxArray_int8_T *ps_shift_ring_proc,
	emxArray_int8_T *tris_shift_ring_proc)
{

     // Fill and build the temp pinfo information on each master processor 
     // Step 1, For original pinfo, the new target processor is added as a new
     // node with proc = new_proc_id, lindex = -1 (unknown) to the tail

    hpPInfoList *ps_pinfo = mesh->ps_pinfo;
    hpPInfoList *tris_pinfo = mesh->tris_pinfo;


    hpPInfoNode *ps_pdata = ps_pinfo->pdata;
    int *ps_phead = ps_pinfo->head;
    int *ps_ptail = ps_pinfo->tail;

    hpPInfoNode *tris_pdata = tris_pinfo->pdata;
    int *tris_phead = tris_pinfo->head;
    int *tris_ptail = tris_pinfo->tail;

    int32_T *ps_ring_data = ps_ring_proc->data;
    int8_T *ps_shift_ring_data = ps_shift_ring_proc->data;

    int32_T *tris_ring_data = tris_ring_proc->data;
    int8_T *tris_shift_ring_data = tris_shift_ring_proc->data;

    int j;

    int num_ps_buffer = ps_ring_proc->size[0];
    int num_tris_buffer = tris_ring_proc->size[0];
    int target_proc_id = mesh->nb_proc->data[I1dm(nb_proc_index)];

    for (j = 1; j <= num_ps_buffer; j++)
    {
	boolean_T overlay_flag = 0;
	int cur_ps_index = ps_ring_data[I1dm(j)];
	int cur_sx = ps_shift_ring_data[3*(j-1)];
	int cur_sy = ps_shift_ring_data[3*(j-1)+1];
	int cur_sz = ps_shift_ring_data[3*(j-1)+2];
	int next_node = ps_phead[I1dm(cur_ps_index)];
	while(next_node != -1)
	{
	    if ( (ps_pdata[I1dm(next_node)].proc == target_proc_id) &&
		    (ps_pdata[I1dm(next_node)].shift[0] == cur_sx) &&
		    (ps_pdata[I1dm(next_node)].shift[1] == cur_sy) &&
		    (ps_pdata[I1dm(next_node)].shift[2] == cur_sz) )
	    {
		overlay_flag = 1;
		break;
	    }
	    else
		next_node = ps_pdata[I1dm(next_node)].next;
	}
	if (!overlay_flag) //If not overlapping, attach it to the end as ghost
	{
	    int cur_tail = ps_ptail[I1dm(cur_ps_index)];
	    if(hpEnsurePInfoCapacity(ps_pinfo))
		ps_pdata = ps_pinfo->pdata;
	    ps_pinfo->allocated_len++;
	    int new_tail = ps_pinfo->allocated_len;
	    ps_pdata[I1dm(new_tail)].next = -1;
	    ps_pdata[I1dm(new_tail)].lindex = -1;
	    ps_pdata[I1dm(new_tail)].proc = target_proc_id;
	    ps_pdata[I1dm(new_tail)].shift[0] = cur_sx;
	    ps_pdata[I1dm(new_tail)].shift[1] = cur_sy;
	    ps_pdata[I1dm(new_tail)].shift[2] = cur_sz;
	    ps_pdata[I1dm(cur_tail)].next = new_tail;
	    ps_ptail[I1dm(cur_ps_index)] = new_tail;
	}
    }

    for (j = 1; j <= num_tris_buffer; j++)
    {
	unsigned char overlay_flag = 0;
	int cur_sx = tris_shift_ring_data[3*(j-1)];
	int cur_sy = tris_shift_ring_data[3*(j-1)+1];
	int cur_sz = tris_shift_ring_data[3*(j-1)+2];
	int cur_tri_index = tris_ring_data[I1dm(j)];
	int next_node = tris_phead[I1dm(cur_tri_index)];
	while(next_node != -1)
	{
	    if ( (tris_pdata[I1dm(next_node)].proc == target_proc_id) &&
		    (tris_pdata[I1dm(next_node)].shift[0] == cur_sx ) &&
		    (tris_pdata[I1dm(next_node)].shift[1] == cur_sy ) &&
		    (tris_pdata[I1dm(next_node)].shift[2] == cur_sz ) )
	    {
		overlay_flag = 1;
		break;
	    }
	    else
		next_node = tris_pinfo->pdata[I1dm(next_node)].next;
	}
	if (!overlay_flag)
	{
	    int cur_tail = tris_ptail[I1dm(cur_tri_index)];
	    if(hpEnsurePInfoCapacity(tris_pinfo))
		tris_pdata = tris_pinfo->pdata;
	    tris_pinfo->allocated_len++;
	    int new_tail = tris_pinfo->allocated_len;
	    tris_pdata[I1dm(new_tail)].next = -1;
	    tris_pdata[I1dm(new_tail)].lindex = -1;
	    tris_pdata[I1dm(new_tail)].proc = target_proc_id;
	    tris_pdata[I1dm(new_tail)].shift[0] = cur_sx;
	    tris_pdata[I1dm(new_tail)].shift[1] = cur_sy;
	    tris_pdata[I1dm(new_tail)].shift[2] = cur_sz;
	    tris_pdata[I1dm(cur_tail)].next = new_tail;
	    tris_ptail[I1dm(cur_tri_index)] = new_tail;
	}
    }
}

void hpBuildGhostPsTrisPInfoForSend(const hiPropMesh *mesh,
	const int nb_proc_index,
	emxArray_int32_T *ps_ring_proc,
	emxArray_int32_T *tris_ring_proc,
	emxArray_int8_T *ps_shift_ring_proc,
	emxArray_int8_T *tris_shift_ring_proc,
	int **buffer_ps_pinfo_tag,
	int **buffer_ps_pinfo_lindex,
	int **buffer_ps_pinfo_proc,
	int8_T **buffer_ps_pinfo_shift,
	int **buffer_tris_pinfo_tag,
	int **buffer_tris_pinfo_lindex,
	int **buffer_tris_pinfo_proc,
	int8_T **buffer_tris_pinfo_shift)
{
    // buffer_ps_pinfo_tag[I1dm(i)][I1dm(j)] to buffer_ps_pinfo_tag[I1dm(i)][I1dm(j+1)]-1
    // are the index of pinfo_lindex & pinfo_proc for buffer_ps[I1dm(j)]
    // buffer_ps_pinfo_lindex[I1md(i)] has all the local index (starts from 0)
    // buffer_ps_pinfo_proc[I1dm(i)] has all the proc information (starts from
    // 0)
    //
    // Same for tris

    hpPInfoList *ps_pinfo = mesh->ps_pinfo;
    hpPInfoList *tris_pinfo = mesh->tris_pinfo;

    hpPInfoNode *ps_pdata = ps_pinfo->pdata;
    hpPInfoNode *tris_pdata = tris_pinfo->pdata;

    int *ps_phead = ps_pinfo->head;
    int *tris_phead = tris_pinfo->head;

    int *ps_ring_data = ps_ring_proc->data;
    int *tris_ring_data = tris_ring_proc->data;

    int8_T *ps_shift_ring_data = ps_shift_ring_proc->data;
    int8_T *tris_shift_ring_data = tris_shift_ring_proc->data;

    // Total length of pinfo list for send
    int buffer_ps_pinfo_length = 0;
    int buffer_tris_pinfo_length = 0;
    
    int j;

    int num_ps_buffer = ps_ring_proc->size[0];
    int num_tris_buffer = tris_ring_proc->size[0];

    // Fill buffer_ps_pinfo_tag & buffer_tris_pinfo_tag

    (*buffer_ps_pinfo_tag) = (int *) calloc(num_ps_buffer+1, sizeof(int));
    (*buffer_tris_pinfo_tag) = (int *) calloc(num_tris_buffer+1, sizeof(int));

    int *cur_ps_ptag = (*buffer_ps_pinfo_tag);
    int *cur_tris_ptag = (*buffer_tris_pinfo_tag);

    cur_ps_ptag[0] = 0;
    cur_tris_ptag[0] = 0;

    for (j = 1; j <= num_ps_buffer; j++)
    {
	int num_pinfo_data = 0;
	int cur_ps_index = ps_ring_data[I1dm(j)];
	int next_node = ps_phead[I1dm(cur_ps_index)];
	while(next_node != -1)
	{
	    num_pinfo_data++;
	    next_node = ps_pdata[I1dm(next_node)].next;
	}
	buffer_ps_pinfo_length += num_pinfo_data;
	cur_ps_ptag[j] = buffer_ps_pinfo_length;
    }

    for (j = 1; j <= num_tris_buffer; j++)
    {
	int num_pinfo_data = 0;
	int cur_tri_index = tris_ring_data[I1dm(j)];
	int next_node = tris_phead[I1dm(cur_tri_index)];
	while(next_node != -1)
	{
	    num_pinfo_data++;
	    next_node = tris_pdata[I1dm(next_node)].next;
	}
	buffer_tris_pinfo_length += num_pinfo_data;
	cur_tris_ptag[j] = buffer_tris_pinfo_length;
    }

    // Fill in the buffer_ps/tris_pinfo_lindex & buffer_ps/tris_pinfo_proc in order
    (*buffer_ps_pinfo_lindex) = (int *) calloc(buffer_ps_pinfo_length, sizeof(int));
    (*buffer_ps_pinfo_proc) = (int *) calloc(buffer_ps_pinfo_length, sizeof(int));
    (*buffer_ps_pinfo_shift) = (int8_T *) calloc(3*buffer_ps_pinfo_length, sizeof(int8_T));

    (*buffer_tris_pinfo_lindex) = (int *) calloc(buffer_tris_pinfo_length, sizeof(int));
    (*buffer_tris_pinfo_proc) = (int *) calloc(buffer_tris_pinfo_length, sizeof(int));
    (*buffer_tris_pinfo_shift) = (int8_T *) calloc(3*buffer_tris_pinfo_length, sizeof(int8_T));

    int *cur_ps_pli = (*buffer_ps_pinfo_lindex);
    int *cur_ps_pp = (*buffer_ps_pinfo_proc);
    int8_T *cur_ps_psh = (*buffer_ps_pinfo_shift);

    int *cur_tris_pli = (*buffer_tris_pinfo_lindex);
    int *cur_tris_pp = (*buffer_tris_pinfo_proc);
    int8_T *cur_tris_psh = (*buffer_tris_pinfo_shift);


    int cur_ps_pinfo = 0;
    for (j = 1; j <= num_ps_buffer; j++)
    {
	int cur_ps_index = ps_ring_data[I1dm(j)];
	int next_node = ps_phead[I1dm(cur_ps_index)];
	while(next_node != -1)
	{
	    cur_ps_pli[cur_ps_pinfo] = ps_pdata[I1dm(next_node)].lindex;
	    cur_ps_pp[cur_ps_pinfo] = ps_pdata[I1dm(next_node)].proc;
	    cur_ps_psh[3*cur_ps_pinfo] = ps_pdata[I1dm(next_node)].shift[0] - ps_shift_ring_data[3*(j-1)];
	    cur_ps_psh[3*cur_ps_pinfo+1] = ps_pdata[I1dm(next_node)].shift[1] - ps_shift_ring_data[3*(j-1)+1];
	    cur_ps_psh[3*cur_ps_pinfo+2] = ps_pdata[I1dm(next_node)].shift[2] - ps_shift_ring_data[3*(j-1)+2];
	    next_node = ps_pdata[I1dm(next_node)].next;
	    cur_ps_pinfo++;
	}
    }

    int cur_tris_pinfo = 0;
    for (j = 1; j <= num_tris_buffer; j++)
    {
	int cur_tri_index = tris_ring_proc->data[I1dm(j)];
	int next_node = tris_phead[I1dm(cur_tri_index)];
	while(next_node != -1)
	{
	    cur_tris_pli[cur_tris_pinfo] = tris_pdata[I1dm(next_node)].lindex;
	    cur_tris_pp[cur_tris_pinfo] = tris_pdata[I1dm(next_node)].proc;
	    cur_tris_psh[3*cur_tris_pinfo] = tris_pdata[I1dm(next_node)].shift[0] - tris_shift_ring_data[3*(j-1)];
	    cur_tris_psh[3*cur_tris_pinfo+1] = tris_pdata[I1dm(next_node)].shift[1] - tris_shift_ring_data[3*(j-1)+1];
	    cur_tris_psh[3*cur_tris_pinfo+2] = tris_pdata[I1dm(next_node)].shift[2] - tris_shift_ring_data[3*(j-1)+2];
	    next_node = tris_pdata[I1dm(next_node)].next;
	    cur_tris_pinfo++;
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

    int *ps_phead = mesh->ps_pinfo->head;
    hpPInfoNode *ps_pdata = mesh->ps_pinfo->pdata;

    int num_ps = mesh->ps->size[0];

    int num_old_nbp = mesh->nb_proc->size[0];

    boolean_T *nb_flag = (boolean_T *) calloc(num_proc, sizeof(boolean_T));

    for (i = 1; i <= num_ps; i++)
    {
	int next_node = ps_phead[I1dm(i)];
	while(next_node != -1)
	{
	    // If it is the pinfo for the point itself, jump over
	    if ( (ps_pdata[I1dm(next_node)].proc == cur_proc ) &&
		 (ps_pdata[I1dm(next_node)].lindex == i) )
	    {
		next_node = ps_pdata[I1dm(next_node)].next;
		continue;
	    }

	    nb_flag[ps_pdata[I1dm(next_node)].proc] = 1;
	    next_node = ps_pdata[I1dm(next_node)].next;
	}
    }


    int new_num_nbp = 0;
    
    for (i = 0; i < num_proc; i++)
    {
	if (nb_flag[i])
	    new_num_nbp++;
    }

    emxArray_int32_T *new_nb_proc = emxCreateND_int32_T(1, &new_num_nbp);

    int j = 0;
    for (i = 0; i < num_proc; i++)
    {
	if (nb_flag[i])
	{
	    new_nb_proc->data[j] = i;
	    j++;
	}
    }

    emxFree_int32_T(&(mesh->nb_proc));
    mesh->nb_proc = new_nb_proc;

    emxArray_int8_T **new_nb_proc_shift = (emxArray_int8_T **)calloc(new_num_nbp, sizeof(emxArray_int8_T *));

    for (j = 0; j < new_num_nbp; j++)
    {
	int new_cur_nb_proc = new_nb_proc->data[j];
	boolean_T *shift_flag = (boolean_T *) calloc(223, sizeof(boolean_T));

	for (i = 1; i <= num_ps; i++)
	{
	    int next_node = ps_phead[I1dm(i)];
	    while(next_node != -1)
	    {
		// If it is the pinfo for the point itself or not related to cur
		// nb proc, jump over
		if ( ((ps_pdata[I1dm(next_node)].proc == cur_proc) && (ps_pdata[I1dm(next_node)].lindex == i)) 
			|| (ps_pdata[I1dm(next_node)].proc != new_cur_nb_proc) )
		   
		{
		    next_node = ps_pdata[I1dm(next_node)].next;
		    continue;
		}

		int first_d, second_d, third_d;

		if (ps_pdata[I1dm(next_node)].shift[0] < 0)
		    first_d = 2;
		else
		    first_d = ps_pdata[I1dm(next_node)].shift[0];

		if (ps_pdata[I1dm(next_node)].shift[1] < 0)
		    second_d = 2;
		else
		    second_d = ps_pdata[I1dm(next_node)].shift[1];

		if (ps_pdata[I1dm(next_node)].shift[2] < 0)
		    third_d = 2;
		else
		    third_d = ps_pdata[I1dm(next_node)].shift[2];

		int hash_value = 100*third_d + 10*second_d + first_d;

		shift_flag[hash_value] = 1;

		next_node = ps_pdata[I1dm(next_node)].next;

	    }
	}

	int num_shift_cur_proc = 0;

	int k = 0;
	for (k = 0; k < 223; k++)
	{
	    if (shift_flag[k])
		num_shift_cur_proc++;
	}

	new_nb_proc_shift[j] = emxCreate_int8_T(num_shift_cur_proc, 3);
	emxArray_int8_T *cur_nb_shift = new_nb_proc_shift[j];

	int ki = 1;

	for (k = 0; k < 223; k++)
	{
	    if (shift_flag[k] == 1)
	    {
		int cur_hash_value = k;
		int first_digit = cur_hash_value % 10;
		if (first_digit == 2)
		    cur_nb_shift->data[I2dm(ki,1,cur_nb_shift->size)] = -1;
		else
		    cur_nb_shift->data[I2dm(ki,1,cur_nb_shift->size)] = (int8_T) first_digit;

		cur_hash_value /= 10;
		int second_digit = cur_hash_value % 10;
		if (second_digit == 2)
		    cur_nb_shift->data[I2dm(ki,2,cur_nb_shift->size)] = -1;
		else
		    cur_nb_shift->data[I2dm(ki,2,cur_nb_shift->size)] = (int8_T) second_digit;

		cur_hash_value /= 10;
		int third_digit = cur_hash_value;
		if (third_digit == 2)
		    cur_nb_shift->data[I2dm(ki,3,cur_nb_shift->size)] = -1;
		else
		    cur_nb_shift->data[I2dm(ki,3,cur_nb_shift->size)] = (int8_T) third_digit;

		ki++;
	    }
	}


	free(shift_flag);
    }


    for (i = 0; i < num_old_nbp; i++)
	emxFree_int8_T(&(mesh->nb_proc_shift[i]));
    free(mesh->nb_proc_shift);
    mesh->nb_proc_shift = new_nb_proc_shift;
    free(nb_flag);
}


void hpCollectAllGhostPs(hiPropMesh *mesh,
			 const int nbp_index,
			 int *size_send,
			 int **ppinfol,
			 int8_T **ppinfos)
{
    int *ps_phead = mesh->ps_pinfo->head;
    hpPInfoNode *ps_pdata = mesh->ps_pinfo->pdata;

    int num_ps = mesh->ps->size[0];

    int i;
    int ip = I1dm(nbp_index);
    int cur_proc;
    MPI_Comm_rank(MPI_COMM_WORLD, &cur_proc);
    int rcv_proc = mesh->nb_proc->data[ip];
    int nump = 0;

    boolean_T *ps_flag = (boolean_T *) calloc(num_ps, sizeof(boolean_T));

    for (i = 1; i <= num_ps; i++)
    {
	int cur_head = ps_phead[I1dm(i)];
	if ((ps_pdata[I1dm(cur_head)].proc == rcv_proc) && (ps_pdata[I1dm(cur_head)].lindex != i))
	{
	    nump++;
	    ps_flag[i-1] = 1;
	}
    }

    if (nump != 0)
    {
	size_send[2*ip] = nump;

	//ppinfol[ip] stores the [master index, local index] pair
	ppinfol[ip] = (int *) calloc(2*nump, sizeof(int));
	ppinfos[ip] = (int8_T *) calloc(3*nump, sizeof(int8_T));

	int *cur_ppinfol = ppinfol[ip];
	int8_T *cur_ppinfos = ppinfos[ip];

	int j = 0;

	for (i = 1; i <= num_ps; i++)
	{
	    if (ps_flag[i-1])
	    {
		int cur_head = ps_phead[I1dm(i)];
		cur_ppinfol[2*j] = ps_pdata[I1dm(cur_head)].lindex;
		cur_ppinfol[2*j+1] = i;

		cur_ppinfos[3*j] = -(ps_pdata[I1dm(cur_head)].shift[0]);
		cur_ppinfos[3*j+1] = -(ps_pdata[I1dm(cur_head)].shift[1]);
		cur_ppinfos[3*j+2] = -(ps_pdata[I1dm(cur_head)].shift[2]);

		j++;
	    }
	}
    }
    else
    {
	size_send[2*ip] = 0;
	ppinfol[ip] = (int *) NULL;
	ppinfos[ip] = (int8_T *) NULL;
    }
    free(ps_flag);
}

void hpCollectAllGhostTris(hiPropMesh *mesh,
			   const int nbp_index,
			   int *size_send,
			   int **tpinfol,
			   int8_T **tpinfos)
{
    int *tris_phead = mesh->tris_pinfo->head;
    hpPInfoNode *tris_pdata = mesh->tris_pinfo->pdata;

    int num_tris = mesh->tris->size[0];

    int i;
    int ip = I1dm(nbp_index);
    int cur_proc;
    MPI_Comm_rank(MPI_COMM_WORLD, &cur_proc);
    int rcv_proc = mesh->nb_proc->data[ip];
    int numt = 0;

    boolean_T *tris_flag = (boolean_T *) calloc(num_tris, sizeof(boolean_T));

    for (i = 1; i <= num_tris; i++)
    {
	int cur_head = tris_phead[I1dm(i)];
	if ( (tris_pdata[I1dm(cur_head)].proc == rcv_proc) && (tris_pdata[I1dm(cur_head)].lindex !=i) )
	{
	    numt++;
	    tris_flag[i-1] = 1;
	}
    }

    if (numt != 0)
    {
	size_send[2*ip+1] = numt;

	tpinfol[ip] = (int *) calloc(2*numt, sizeof(int));
	tpinfos[ip] = (int8_T *) calloc(3*numt, sizeof(int8_T));

	int *cur_tpinfol = tpinfol[ip];
	int8_T *cur_tpinfos = tpinfos[ip];

	int j = 0;

	for (i = 1; i <= num_tris; i++)
	{
	    if (tris_flag[i-1])
	    {
		int cur_head = tris_phead[I1dm(i)];

		cur_tpinfol[2*j] = tris_pdata[I1dm(cur_head)].lindex;
		cur_tpinfol[2*j+1] = i;

		cur_tpinfos[3*j] = -(tris_pdata[I1dm(cur_head)].shift[0]);
		cur_tpinfos[3*j+1] = -(tris_pdata[I1dm(cur_head)].shift[1]);
		cur_tpinfos[3*j+2] = -(tris_pdata[I1dm(cur_head)].shift[2]);

		j++;
	    }
	}
    }
    else
    {
	size_send[2*ip+1] = 0;
	tpinfol[ip] = (int *) NULL;
	tpinfos[ip] = (int8_T *) NULL;
    }
    free(tris_flag);
}

void hpMergeOverlayPsPInfo(hiPropMesh *mesh,
			   const int rcv_id,
			   int nump,
			   int *ppinfol,
			   int8_T *ppinfos)
{
    hpPInfoList *ps_pinfo = mesh->ps_pinfo;
    int *ps_phead = ps_pinfo->head;
    int *ps_ptail = ps_pinfo->tail;
    hpPInfoNode *ps_pdata = ps_pinfo->pdata;

    int i;
    for (i = 0; i < nump; i++)
    {
	int ps_index = ppinfol[2*i];
	int next_node = ps_phead[I1dm(ps_index)];

	//If exists, whether = -1 or not, reset it
	
	while(next_node != -1)
	{
	    if ( (ps_pdata[I1dm(next_node)].proc == rcv_id) 
		    && (ps_pdata[I1dm(next_node)].shift[0] == ppinfos[3*i])
		    && (ps_pdata[I1dm(next_node)].shift[1] == ppinfos[3*i+1])
		    && (ps_pdata[I1dm(next_node)].shift[2] == ppinfos[3*i+2]) )
	    {
		ps_pdata[I1dm(next_node)].lindex = ppinfol[2*i+1];
		break;
	    }
	    else
		next_node = ps_pdata[I1dm(next_node)].next;
	}

	//If doesn't exist, then add it to the tail

	if (next_node == -1)
	{
	    if(hpEnsurePInfoCapacity(ps_pinfo))
		ps_pdata = ps_pinfo->pdata;
	    ps_pinfo->allocated_len++;

	    int cur_tail = ps_ptail[I1dm(ps_index)];
	    int new_tail = ps_pinfo->allocated_len;

	    ps_pdata[I1dm(new_tail)].proc = rcv_id;
	    ps_pdata[I1dm(new_tail)].lindex = ppinfol[2*i+1];
	    ps_pdata[I1dm(new_tail)].shift[0] = ppinfos[3*i];
	    ps_pdata[I1dm(new_tail)].shift[1] = ppinfos[3*i+1];
	    ps_pdata[I1dm(new_tail)].shift[2] = ppinfos[3*i+2];
	    ps_pdata[I1dm(new_tail)].next = -1;

	    ps_pdata[I1dm(cur_tail)].next = new_tail;

	    ps_ptail[I1dm(ps_index)] = new_tail;
	}
    }
}

void hpMergeOverlayTrisPInfo(hiPropMesh *mesh,
			     const int rcv_id,
			     int numt,
			     int *tpinfol,
			     int8_T *tpinfos)
{
    hpPInfoList *tris_pinfo = mesh->tris_pinfo;
    int *tris_phead = tris_pinfo->head;
    int *tris_ptail = tris_pinfo->tail;
    hpPInfoNode *tris_pdata = tris_pinfo->pdata;

    int i;
    for (i = 0; i < numt; i++)
    {
	int tris_index = tpinfol[2*i];
	int next_node = tris_phead[I1dm(tris_index)];

	//If exists, whether = -1 or not, reset it
	
	while(next_node != -1)
	{
	    if ( (tris_pdata[I1dm(next_node)].proc == rcv_id)
		    && (tris_pdata[I1dm(next_node)].shift[0] == tpinfos[3*i]) 
		    && (tris_pdata[I1dm(next_node)].shift[1] == tpinfos[3*i+1])
		    && (tris_pdata[I1dm(next_node)].shift[2] == tpinfos[3*i+2]) )
	    {
		tris_pdata[I1dm(next_node)].lindex = tpinfol[2*i+1];
		break;
	    }
	    else
		next_node = tris_pdata[I1dm(next_node)].next;
	}

	//If doesn't exist, then add it to the tail

	if (next_node == -1)
	{
	    if(hpEnsurePInfoCapacity(tris_pinfo))
		tris_pdata = tris_pinfo->pdata;
	    tris_pinfo->allocated_len++;

	    int cur_tail = tris_ptail[I1dm(tris_index)];
	    int new_tail = tris_pinfo->allocated_len;

	    tris_pdata[I1dm(new_tail)].proc = rcv_id;
	    tris_pdata[I1dm(new_tail)].lindex = tpinfol[2*i+1];
	    tris_pdata[I1dm(new_tail)].shift[0] = tpinfos[3*i];
	    tris_pdata[I1dm(new_tail)].shift[1] = tpinfos[3*i+1];
	    tris_pdata[I1dm(new_tail)].shift[2] = tpinfos[3*i+2];
	    tris_pdata[I1dm(new_tail)].next = -1;

	    tris_pdata[I1dm(cur_tail)].next = new_tail;

	    tris_ptail[I1dm(tris_index)] = new_tail;
	}
    }
}

void hpUpdateMasterPInfo(hiPropMesh *mesh)
{
    int cur_proc;
    MPI_Comm_rank(MPI_COMM_WORLD, &cur_proc);
    int i;

    int num_nbp = mesh->nb_proc->size[0];

    int *send_size = (int *) calloc(2*num_nbp, sizeof(int));

    int **buffer_ps_pinfo_lindex = (int **) calloc(num_nbp, sizeof(int *));
    int **buffer_tris_pinfo_lindex = (int **) calloc(num_nbp, sizeof(int *));

    int8_T **buffer_ps_pinfo_shift = (int8_T **) calloc(num_nbp, sizeof(int8_T *));
    int8_T **buffer_tris_pinfo_shift = (int8_T **) calloc(num_nbp, sizeof(int8_T *));

    int tag_size = 1;
    int tag_ps_pinfoi = 2;
    int tag_tris_pinfoi = 3;
    int tag_ps_pinfos = 4;
    int tag_tris_pinfos = 5;

    for (i = 1; i <= num_nbp; i++)
    {
	hpCollectAllGhostPs(mesh, i, send_size,  buffer_ps_pinfo_lindex, buffer_ps_pinfo_shift);
	hpCollectAllGhostTris(mesh, i, send_size, buffer_tris_pinfo_lindex, buffer_tris_pinfo_shift);
    }

    MPI_Request* send_rqst_list = (MPI_Request *) malloc(5*num_nbp*sizeof(MPI_Request) );

    for (i = 0; i < 5*num_nbp; ++i)
	send_rqst_list[i] = MPI_REQUEST_NULL;

    int cur_rqst = 0;

    MPI_Status* send_status_list = (MPI_Status *) malloc(5*num_nbp*sizeof(MPI_Status) );
    MPI_Request* recv_req_list = (MPI_Request *) malloc(num_nbp*sizeof(MPI_Request) );

    int *recv_size = (int *) calloc (2*num_nbp, sizeof(int));

    for (i = 0; i < num_nbp; i++)
    {
	MPI_Isend(&(send_size[2*i]), 2, MPI_INT, mesh->nb_proc->data[i], tag_size, MPI_COMM_WORLD, &(send_rqst_list[cur_rqst++]));

	if (send_size[2*i] != 0)
	{
	    MPI_Isend(buffer_ps_pinfo_lindex[i], 2*send_size[2*i], MPI_INT,
		    mesh->nb_proc->data[i], tag_ps_pinfoi, MPI_COMM_WORLD, &(send_rqst_list[cur_rqst++]));

	    MPI_Isend(buffer_ps_pinfo_shift[i], 3*send_size[2*i], MPI_SIGNED_CHAR,
		    mesh->nb_proc->data[i], tag_ps_pinfos, MPI_COMM_WORLD, &(send_rqst_list[cur_rqst++]));
	}
	if (send_size[2*i+1] != 0)
	{
	    MPI_Isend(buffer_tris_pinfo_lindex[i], 2*send_size[2*i+1], MPI_INT,
		    mesh->nb_proc->data[i], tag_tris_pinfoi, MPI_COMM_WORLD, &(send_rqst_list[cur_rqst++]));
	    MPI_Isend(buffer_tris_pinfo_shift[i], 3*send_size[2*i+1], MPI_SIGNED_CHAR,
		    mesh->nb_proc->data[i], tag_tris_pinfos, MPI_COMM_WORLD, &(send_rqst_list[cur_rqst++]));
	}
    }


	// Recv size info
    for (i = 0; i < num_nbp; i++)
    {
	MPI_Irecv(&(recv_size[2*i]), 2, MPI_INT, mesh->nb_proc->data[i], tag_size, MPI_COMM_WORLD, &(recv_req_list[i]));
    }


    for (i = 0; i < num_nbp; i++)
    {
	int *buf_ppinfo_lindex_recv;
	int *buf_tpinfo_lindex_recv;

	int8_T *buf_ppinfo_shift_recv;
	int8_T *buf_tpinfo_shift_recv;

	int num_buf_ps_recv;
	int num_buf_tris_recv;

	MPI_Status recv_status1;
	MPI_Status recv_status2;
	MPI_Status recv_status3;
	MPI_Status recv_status4;
	MPI_Status recv_status5;

	int recv_index;
	int proc_recv;

	MPI_Waitany(num_nbp, recv_req_list, &recv_index, &recv_status1);
	proc_recv = recv_status1.MPI_SOURCE;

	num_buf_ps_recv = recv_size[2*recv_index];
	num_buf_tris_recv = recv_size[2*recv_index+1];

	    
	if (num_buf_ps_recv != 0)
	{
	    buf_ppinfo_lindex_recv = (int *) calloc(2*num_buf_ps_recv, sizeof(int));
	    buf_ppinfo_shift_recv = (int8_T *) calloc(3*num_buf_ps_recv, sizeof(int8_T));

	    MPI_Recv(buf_ppinfo_lindex_recv, 2*recv_size[2*recv_index], MPI_INT, proc_recv,
		    tag_ps_pinfoi, MPI_COMM_WORLD, &recv_status2);
	    MPI_Recv(buf_ppinfo_shift_recv, 3*recv_size[2*recv_index], MPI_SIGNED_CHAR, proc_recv,
		    tag_ps_pinfos, MPI_COMM_WORLD, &recv_status3);

	    hpMergeOverlayPsPInfo(mesh, proc_recv, num_buf_ps_recv, buf_ppinfo_lindex_recv, buf_ppinfo_shift_recv);

	    free(buf_ppinfo_lindex_recv);
	    free(buf_ppinfo_shift_recv);
	}


	if (num_buf_tris_recv != 0)
	{
	    buf_tpinfo_lindex_recv = (int *) calloc(2*num_buf_tris_recv, sizeof(int));
	    buf_tpinfo_shift_recv = (int8_T *) calloc(3*num_buf_tris_recv, sizeof(int8_T));

	    MPI_Recv(buf_tpinfo_lindex_recv, 2*recv_size[2*recv_index+1], MPI_INT, proc_recv,
		    tag_tris_pinfoi, MPI_COMM_WORLD, &recv_status4);
	    MPI_Recv(buf_tpinfo_shift_recv, 3*recv_size[2*recv_index+1], MPI_SIGNED_CHAR, proc_recv,
		    tag_tris_pinfos, MPI_COMM_WORLD, &recv_status5);

	    hpMergeOverlayTrisPInfo(mesh, proc_recv, num_buf_tris_recv, buf_tpinfo_lindex_recv, buf_tpinfo_shift_recv);
	    
	    free(buf_tpinfo_lindex_recv);
	    free(buf_tpinfo_shift_recv);
	}
    }

    free(recv_req_list);
    free(recv_size);

    // Wait until all the array are sent

    MPI_Waitall(5*num_nbp, send_rqst_list, send_status_list);

    // Free the array for send

    free(send_rqst_list);
    free(send_status_list);

    for (i = 0; i < num_nbp; i++)
    {
	free(buffer_ps_pinfo_lindex[i]);
	free(buffer_ps_pinfo_shift[i]);
	free(buffer_tris_pinfo_lindex[i]);
	free(buffer_tris_pinfo_shift[i]);
    }

    free(send_size);

    free(buffer_ps_pinfo_lindex);
    free(buffer_ps_pinfo_shift);
    free(buffer_tris_pinfo_lindex);
    free(buffer_tris_pinfo_shift);
}

void hpCollectAllOverlayPs(hiPropMesh *mesh,
			   const int nbp_index,
			   int *size_send,
			   int **ppinfot,
			   int **ppinfol,
			   int **ppinfop,
			   int8_T **ppinfos)
{
    int *ps_phead = mesh->ps_pinfo->head;
    hpPInfoNode *ps_pdata = mesh->ps_pinfo->pdata;

    int i;
    int ip = nbp_index-1;
    int cur_proc;
    MPI_Comm_rank(MPI_COMM_WORLD, &cur_proc);
    int rcv_proc = mesh->nb_proc->data[ip];

    int num_ps = mesh->ps->size[0];

    int nump = 0;

    boolean_T *overlay = (boolean_T *) calloc(mesh->ps->size[0], sizeof(boolean_T));

    // first round, fill the nump and overlay
    for (i = 1; i <= num_ps; i++)
    {
	int cur_head = ps_phead[I1dm(i)];
	if ( (ps_pdata[I1dm(cur_head)].proc == cur_proc) && (ps_pdata[I1dm(cur_head)].lindex == i) )
	{
	    int next_node = ps_pdata[I1dm(cur_head)].next;
	    while(next_node != -1)
	    {
		if (ps_pdata[I1dm(next_node)].proc == rcv_proc)
		{
		    overlay[I1dm(i)] = 1;
		    nump++;
		}
		next_node = ps_pdata[I1dm(next_node)].next;
	    }
	}
    }

    if (nump != 0)
    {
	// second round, fill the ppinfot[ip]
	size_send[2*ip] = nump;
	ppinfot[ip] = (int *) calloc(nump+1, sizeof(int));

	int *cur_ppinfot = ppinfot[ip];
	cur_ppinfot[0] = 0;

	int j = 1;
	int num_pinfo_all = 0;
	for (i = 1; i <= num_ps; i++)
	{
	    if (overlay[I1dm(i)]) //At least overlapping onece, could be multiple overlapping
	    {
		int cur_head = ps_phead[I1dm(i)];
		int next_shift_node = ps_pdata[I1dm(cur_head)].next;

		while(next_shift_node != -1)
		{
		    if (ps_pdata[I1dm(next_shift_node)].proc == rcv_proc)
		    {
			int num_pinfo_cur = 0;
			int next_node = ps_phead[I1dm(i)];

			while(next_node != -1)
			{
			    num_pinfo_cur++;
			    next_node = ps_pdata[I1dm(next_node)].next;
			}
			num_pinfo_all += num_pinfo_cur;
			cur_ppinfot[j++] = num_pinfo_all;
		    }
		    next_shift_node = ps_pdata[I1dm(next_shift_node)].next;
		}
	    }
	}

	// third rould, fill the ppinfol[ip] and ppinfop[ip]
	ppinfol[ip] = (int *) calloc(num_pinfo_all, sizeof(int));
	ppinfop[ip] = (int *) calloc(num_pinfo_all, sizeof(int));
	ppinfos[ip] = (int8_T *) calloc(3*num_pinfo_all, sizeof(int8_T));

	int *cur_ppinfol = ppinfol[ip];
	int *cur_ppinfop = ppinfop[ip];
	int8_T *cur_ppinfos = ppinfos[ip];

	j = 0;
	for (i = 1; i <= num_ps; i++)
	{
	    if (overlay[I1dm(i)])
	    {
		int cur_head = ps_phead[I1dm(i)];
		int next_shift_node = ps_pdata[I1dm(cur_head)].next;

		while(next_shift_node != -1)
		{
		    if (ps_pdata[I1dm(next_shift_node)].proc == rcv_proc)
		    {
			int next_node = ps_phead[I1dm(i)];
			while(next_node != -1)
			{
			    cur_ppinfol[j] = ps_pdata[I1dm(next_node)].lindex;
			    cur_ppinfop[j] = ps_pdata[I1dm(next_node)].proc;
			    cur_ppinfos[j*3] = ps_pdata[I1dm(next_node)].shift[0] - ps_pdata[I1dm(next_shift_node)].shift[0];
			    cur_ppinfos[j*3+1] = ps_pdata[I1dm(next_node)].shift[1] - ps_pdata[I1dm(next_shift_node)].shift[1];
			    cur_ppinfos[j*3+2] = ps_pdata[I1dm(next_node)].shift[2] - ps_pdata[I1dm(next_shift_node)].shift[2];
			    j++;
			    next_node = ps_pdata[I1dm(next_node)].next;
			}
		    }
		    next_shift_node = ps_pdata[I1dm(next_shift_node)].next;
		}
	    }
	}
    }
    else
    {
	size_send[2*ip] = 0;
	ppinfot[ip] = (int *) NULL;
	ppinfol[ip] = (int *) NULL;
	ppinfop[ip] = (int *) NULL;
	ppinfos[ip] = (int8_T *) NULL;
    }

    free(overlay);
}

void hpCollectAllOverlayTris(hiPropMesh *mesh,
			     const int nbp_index,
			     int *size_send,
			     int **tpinfot,
			     int **tpinfol,
			     int **tpinfop,
			     int8_T **tpinfos)
{
    int *tris_phead = mesh->tris_pinfo->head;
    hpPInfoNode *tris_pdata = mesh->tris_pinfo->pdata;

    int i;
    int ip = nbp_index-1;
    int cur_proc;
    MPI_Comm_rank(MPI_COMM_WORLD, &cur_proc);
    int rcv_proc = mesh->nb_proc->data[ip];

    int num_tris = mesh->tris->size[0];

    int numt = 0;

    boolean_T *overlay = (boolean_T *) calloc(mesh->tris->size[0], sizeof(boolean_T));

    // first round, fill the nump and overlay
    for (i = 1; i <= num_tris; i++)
    {
	int cur_head = tris_phead[I1dm(i)];
	if ( (tris_pdata[I1dm(cur_head)].proc == cur_proc) && (tris_pdata[I1dm(cur_head)].lindex == i) )
	{
	    int next_node = tris_pdata[I1dm(cur_head)].next;
	    while(next_node != -1)
	    {
		if (tris_pdata[I1dm(next_node)].proc == rcv_proc)
		{
		    overlay[I1dm(i)] = 1;
		    numt++;
		}
		next_node = tris_pdata[I1dm(next_node)].next;
	    }
	}
    }


    if (numt != 0)
    {
	// second round, fill the ppinfot[ip]
	size_send[2*ip+1] = numt;
	tpinfot[ip] = (int *) calloc(numt+1, sizeof(int));

	int *cur_tpinfot = tpinfot[ip];

	cur_tpinfot[0] = 0;

	int j = 1;
	int num_pinfo_all = 0;
	for (i = 1; i <= num_tris; i++)
	{
	    if (overlay[I1dm(i)])
	    {
		int cur_head = tris_phead[I1dm(i)];
		int next_shift_node = tris_pdata[I1dm(cur_head)].next;

		while(next_shift_node != -1)
		{
		    if (tris_pdata[I1dm(next_shift_node)].proc == rcv_proc)
		    {
			int num_pinfo_cur = 0;
			int next_node = tris_phead[I1dm(i)];

			while(next_node != -1)
			{
			    num_pinfo_cur++;
			    next_node = tris_pdata[I1dm(next_node)].next;
			}
			num_pinfo_all += num_pinfo_cur;
			cur_tpinfot[j++] = num_pinfo_all;
		    }
		    next_shift_node = tris_pdata[I1dm(next_shift_node)].next;
		}
	    }

	}

	// third rould, fill the ppinfol[ip] and ppinfop[ip]
	tpinfol[ip] = (int *) calloc(num_pinfo_all, sizeof(int));
	tpinfop[ip] = (int *) calloc(num_pinfo_all, sizeof(int));
	tpinfos[ip] = (int8_T *) calloc(3*num_pinfo_all, sizeof(int8_T));

	int *cur_tpinfol = tpinfol[ip];
	int *cur_tpinfop = tpinfop[ip];
	int8_T *cur_tpinfos = tpinfos[ip];

	j = 0;
	for (i = 1; i <= num_tris; i++)
	{
	    if (overlay[I1dm(i)])
	    {
		int cur_head = tris_phead[I1dm(i)];
		int next_shift_node = tris_pdata[I1dm(cur_head)].next;

		while(next_shift_node != -1)
		{
		    if (tris_pdata[I1dm(next_shift_node)].proc == rcv_proc)
		    {
			int next_node = tris_phead[I1dm(i)];

			while(next_node != -1)
			{
			    cur_tpinfol[j] = tris_pdata[I1dm(next_node)].lindex;
			    cur_tpinfop[j] = tris_pdata[I1dm(next_node)].proc;
			    cur_tpinfos[j*3] = tris_pdata[I1dm(next_node)].shift[0] - tris_pdata[I1dm(next_shift_node)].shift[0];
			    cur_tpinfos[j*3+1] = tris_pdata[I1dm(next_node)].shift[1] - tris_pdata[I1dm(next_shift_node)].shift[1];
			    cur_tpinfos[j*3+2] = tris_pdata[I1dm(next_node)].shift[2] - tris_pdata[I1dm(next_shift_node)].shift[2];
			    j++;
			    next_node = tris_pdata[I1dm(next_node)].next;
			}
		    }
		    next_shift_node = tris_pdata[I1dm(next_shift_node)].next;
		}
	    }
	}
    }
    else
    {
	size_send[2*ip+1] = 0;
	tpinfot[ip] = (int *) NULL;
	tpinfol[ip] = (int *) NULL;
	tpinfop[ip] = (int *) NULL;
	tpinfos[ip] = (int8_T *) NULL;
    }
    free(overlay);
}

void hpMergeGhostPsPInfo(hiPropMesh *mesh,
			 const int rcv_id,
			 int nump,
			 int *ppinfot,
			 int *ppinfol,
			 int *ppinfop,
			 int8_T *ppinfos)
{
    hpPInfoList *ps_pinfo = mesh->ps_pinfo;
    int *ps_phead = ps_pinfo->head;
    int *ps_ptail = ps_pinfo->tail;
    hpPInfoNode *ps_pdata = ps_pinfo->pdata;

    int cur_proc;
    MPI_Comm_rank(MPI_COMM_WORLD, &cur_proc);


    int i,j;

    for (i = 0; i < nump; i++)
    {
	// first locate the point
	int ps_index;
	for (j = ppinfot[i]; j <= ppinfot[i+1]-1; j++)
	{
	    if ( (ppinfop[j] == cur_proc)
		    && (ppinfos[3*j] == 0)
		    && (ppinfos[3*j+1] == 0)
		    && (ppinfos[3*j+2] == 0) )
	    {
		ps_index = ppinfol[j];
		break;
	    }
	}

	// then merge the pinfo list
	for (j = ppinfot[i]; j <= ppinfot[i+1]-1; j++)
	{
	    int cur_node = ps_phead[I1dm(ps_index)];
	    while(cur_node != -1)
	    {
		if ( (ppinfop[j] == ps_pdata[I1dm(cur_node)].proc)
			&& (ppinfos[j*3] == ps_pdata[I1dm(cur_node)].shift[0])
			&& (ppinfos[j*3+1] == ps_pdata[I1dm(cur_node)].shift[1])
			&& (ppinfos[j*3+2] == ps_pdata[I1dm(cur_node)].shift[2]) )
		{
		    ps_pdata[I1dm(cur_node)].lindex = ppinfol[j];
		    break;
		}
		cur_node = ps_pdata[I1dm(cur_node)].next;
	    }

	    // if a new proc info
	    if (cur_node == -1)
	    {
		if(hpEnsurePInfoCapacity(ps_pinfo))
		    ps_pdata = ps_pinfo->pdata;
		ps_pinfo->allocated_len++;

		int cur_tail = ps_ptail[I1dm(ps_index)];
		int new_tail = ps_pinfo->allocated_len;
		ps_pdata[I1dm(new_tail)].lindex = ppinfol[j];
		ps_pdata[I1dm(new_tail)].proc = ppinfop[j];
		ps_pdata[I1dm(new_tail)].shift[0] = ppinfos[j*3];
		ps_pdata[I1dm(new_tail)].shift[1] = ppinfos[j*3+1];
		ps_pdata[I1dm(new_tail)].shift[2] = ppinfos[j*3+2];
		ps_pdata[I1dm(new_tail)].next = -1;

		ps_pdata[I1dm(cur_tail)].next = new_tail;
		ps_ptail[I1dm(ps_index)] = new_tail;
	    }
	}
    }

}

void hpMergeGhostTrisPInfo(hiPropMesh *mesh,
			   const int rcv_id,
			   int numt,
			   int *tpinfot,
			   int *tpinfol,
			   int *tpinfop,
			   int8_T *tpinfos)
{
    hpPInfoList *tris_pinfo = mesh->tris_pinfo;
    int *tris_phead = tris_pinfo->head;
    int *tris_ptail = tris_pinfo->tail;
    hpPInfoNode *tris_pdata = tris_pinfo->pdata;

    int cur_proc;
    MPI_Comm_rank(MPI_COMM_WORLD, &cur_proc);


    int i,j;

    for (i = 0; i < numt; i++)
    {
	// first locate the point
	int tris_index;
	for (j = tpinfot[i]; j <= tpinfot[i+1]-1; j++)
	{
	    if ( (tpinfop[j] == cur_proc)
		    && (tpinfos[3*j] == 0)
		    && (tpinfos[3*j+1] == 0)
		    && (tpinfos[3*j+2] == 0) )
	    {
		tris_index = tpinfol[j];
		break;
	    }
	}

	// then merge the pinfo list
	for (j = tpinfot[i]; j <= tpinfot[i+1]-1; j++)
	{
	    int cur_node = tris_phead[I1dm(tris_index)];
	    while(cur_node != -1)
	    {
		if ( (tpinfop[j] == tris_pdata[I1dm(cur_node)].proc)
			&& (tpinfos[j*3] == tris_pdata[I1dm(cur_node)].shift[0])
			&& (tpinfos[j*3+1] == tris_pdata[I1dm(cur_node)].shift[1])
			&& (tpinfos[j*3+2] == tris_pdata[I1dm(cur_node)].shift[2]) )
		{
		    tris_pdata[I1dm(cur_node)].lindex = tpinfol[j];
		    break;
		}
		cur_node = tris_pdata[I1dm(cur_node)].next;
		
	    }
	    // if a new proc info
	    if (cur_node == -1)
	    {
		if(hpEnsurePInfoCapacity(tris_pinfo))
		    tris_pdata = tris_pinfo->pdata;
		tris_pinfo->allocated_len++;

		int cur_tail = tris_ptail[I1dm(tris_index)];
		int new_tail = tris_pinfo->allocated_len;
		tris_pdata[I1dm(new_tail)].lindex = tpinfol[j];
		tris_pdata[I1dm(new_tail)].proc = tpinfop[j];
		tris_pdata[I1dm(new_tail)].shift[0] = tpinfos[j*3];
		tris_pdata[I1dm(new_tail)].shift[1] = tpinfos[j*3+1];
		tris_pdata[I1dm(new_tail)].shift[2] = tpinfos[j*3+2];
		tris_pdata[I1dm(new_tail)].next = -1;

		tris_pdata[I1dm(cur_tail)].next = new_tail;
		tris_ptail[I1dm(tris_index)] = new_tail;
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

    int *send_size = (int *) calloc(2*num_nb_proc, sizeof(int));

    int **buffer_ps_pinfo_tag = (int **) calloc(num_nb_proc, sizeof(int *));
    int **buffer_ps_pinfo_lindex = (int **) calloc(num_nb_proc, sizeof(int *));
    int **buffer_ps_pinfo_proc = (int **) calloc(num_nb_proc, sizeof(int *));
    int8_T **buffer_ps_pinfo_shift = (int8_T **) calloc(num_nb_proc, sizeof(int8_T *));

    int **buffer_tris_pinfo_tag = (int **) calloc(num_nb_proc, sizeof(int *));
    int **buffer_tris_pinfo_lindex = (int **) calloc(num_nb_proc, sizeof(int *));
    int **buffer_tris_pinfo_proc = (int **) calloc(num_nb_proc, sizeof(int *));
    int8_T **buffer_tris_pinfo_shift = (int8_T **) calloc(num_nb_proc, sizeof(int8_T *));

    int num_all_send_rqst = 9*num_nb_proc;
    MPI_Request* send_rqst_list = (MPI_Request *) calloc(num_all_send_rqst, sizeof(MPI_Request));

    for (i = 0; i < num_all_send_rqst; i++)
	send_rqst_list[i] = MPI_REQUEST_NULL;

    MPI_Status* send_status_list = (MPI_Status *) calloc(num_all_send_rqst, sizeof(MPI_Status));
    MPI_Request* recv_req_list = (MPI_Request *) calloc(num_nb_proc, sizeof(MPI_Request));

    int *recv_size = (int *) calloc (2*num_nb_proc, sizeof(int));

    int cur_rqst = 0;

    int tag_size = 1;

    int tag_ps_pinfot = 2;
    int tag_ps_pinfop = 3;
    int tag_ps_pinfol = 4;
    int tag_ps_pinfos = 5;

    int tag_tris_pinfot = 6;
    int tag_tris_pinfop = 7;
    int tag_tris_pinfol = 8;
    int tag_tris_pinfos = 9;


    for (i = 0; i < num_nb_proc; i++)
    {
	hpCollectAllOverlayPs(mesh, i+1, send_size, buffer_ps_pinfo_tag, buffer_ps_pinfo_lindex, buffer_ps_pinfo_proc, buffer_ps_pinfo_shift);
	hpCollectAllOverlayTris(mesh, i+1, send_size, buffer_tris_pinfo_tag, buffer_tris_pinfo_lindex, buffer_tris_pinfo_proc, buffer_tris_pinfo_shift);

	MPI_Isend(&(send_size[2*i]), 2, MPI_INT,
		mesh->nb_proc->data[i], tag_size, MPI_COMM_WORLD, &(send_rqst_list[cur_rqst++]));

	if (send_size[2*i] != 0)
	{
	    MPI_Isend(buffer_ps_pinfo_tag[i], send_size[2*i]+1, MPI_INT,
		    mesh->nb_proc->data[i], tag_ps_pinfot, MPI_COMM_WORLD, &(send_rqst_list[cur_rqst++]));
	    MPI_Isend(buffer_ps_pinfo_lindex[i], buffer_ps_pinfo_tag[i][send_size[2*i]], MPI_INT,
		    mesh->nb_proc->data[i], tag_ps_pinfol, MPI_COMM_WORLD, &(send_rqst_list[cur_rqst++]));
	    MPI_Isend(buffer_ps_pinfo_proc[i], buffer_ps_pinfo_tag[i][send_size[2*i]], MPI_INT,
		    mesh->nb_proc->data[i], tag_ps_pinfop, MPI_COMM_WORLD, &(send_rqst_list[cur_rqst++]));
	    MPI_Isend(buffer_ps_pinfo_shift[i], 3*buffer_ps_pinfo_tag[i][send_size[2*i]], MPI_SIGNED_CHAR,
		    mesh->nb_proc->data[i], tag_ps_pinfos, MPI_COMM_WORLD, &(send_rqst_list[cur_rqst++]));
	}
	if (send_size[2*i+1] != 0)
	{
	    MPI_Isend(buffer_tris_pinfo_tag[i], send_size[2*i+1]+1, MPI_INT,
		    mesh->nb_proc->data[i], tag_tris_pinfot, MPI_COMM_WORLD, &(send_rqst_list[cur_rqst++]));
	    MPI_Isend(buffer_tris_pinfo_lindex[i], buffer_tris_pinfo_tag[i][send_size[2*i+1]], MPI_INT,
		    mesh->nb_proc->data[i], tag_tris_pinfol, MPI_COMM_WORLD, &(send_rqst_list[cur_rqst++]));
	    MPI_Isend(buffer_tris_pinfo_proc[i], buffer_tris_pinfo_tag[i][send_size[2*i+1]], MPI_INT,
		    mesh->nb_proc->data[i], tag_tris_pinfop, MPI_COMM_WORLD, &(send_rqst_list[cur_rqst++]));
	    MPI_Isend(buffer_tris_pinfo_shift[i], 3*buffer_tris_pinfo_tag[i][send_size[2*i+1]], MPI_SIGNED_CHAR,
		    mesh->nb_proc->data[i], tag_tris_pinfos, MPI_COMM_WORLD, &(send_rqst_list[cur_rqst++]));
	}

    }

	// Recv ps pinfo
    for (i = 0; i < num_nb_proc; i++)
	MPI_Irecv(&(recv_size[2*i]), 2, MPI_INT, mesh->nb_proc->data[i], tag_size, MPI_COMM_WORLD, &(recv_req_list[i]));

    for (i = 0; i < num_nb_proc; i++)
    {
	int *buf_ppinfo_tag_recv;
	int *buf_ppinfo_lindex_recv;
	int *buf_ppinfo_proc_recv;
	int8_T *buf_ppinfo_shift_recv;

	int num_buf_ps_recv;
	int num_buf_ps_pinfo_recv;

	int *buf_tpinfo_tag_recv;
	int *buf_tpinfo_lindex_recv;
	int *buf_tpinfo_proc_recv;
	int8_T *buf_tpinfo_shift_recv;

	int num_buf_tris_recv;
	int num_buf_tris_pinfo_recv;

	MPI_Status recv_status;
	MPI_Status recv_status1;
	MPI_Status recv_status2;
	MPI_Status recv_status3;
	MPI_Status recv_status4;
	MPI_Status recv_status5;
	MPI_Status recv_status6;
	MPI_Status recv_status7;
	MPI_Status recv_status8;

	int recv_index;
	int proc_recv;

	MPI_Waitany(num_nb_proc, recv_req_list, &recv_index, &recv_status);
	proc_recv = recv_status.MPI_SOURCE;

	num_buf_ps_recv = recv_size[2*recv_index];
	num_buf_tris_recv = recv_size[2*recv_index+1];

	if (num_buf_ps_recv != 0)
	{
	    buf_ppinfo_tag_recv = (int *) calloc(num_buf_ps_recv+1, sizeof(int));

	    MPI_Recv(buf_ppinfo_tag_recv, num_buf_ps_recv+1, MPI_INT, proc_recv,
		    tag_ps_pinfot, MPI_COMM_WORLD, &recv_status1);

	    num_buf_ps_pinfo_recv = buf_ppinfo_tag_recv[num_buf_ps_recv];

	    buf_ppinfo_lindex_recv = (int *) calloc(num_buf_ps_pinfo_recv, sizeof(int));
	    buf_ppinfo_proc_recv = (int *) calloc(num_buf_ps_pinfo_recv, sizeof(int));
	    buf_ppinfo_shift_recv = (int8_T *) calloc(3*num_buf_ps_pinfo_recv, sizeof(int8_T));

	    MPI_Recv(buf_ppinfo_lindex_recv, num_buf_ps_pinfo_recv, MPI_INT, proc_recv,
		    tag_ps_pinfol, MPI_COMM_WORLD, &recv_status2);
	    MPI_Recv(buf_ppinfo_proc_recv, num_buf_ps_pinfo_recv, MPI_INT, proc_recv,
		    tag_ps_pinfop, MPI_COMM_WORLD, &recv_status3);
	    MPI_Recv(buf_ppinfo_shift_recv, 3*num_buf_ps_pinfo_recv, MPI_SIGNED_CHAR, proc_recv,
		    tag_ps_pinfos, MPI_COMM_WORLD, &recv_status4);

	    hpMergeGhostPsPInfo(mesh, proc_recv, num_buf_ps_recv,
		    buf_ppinfo_tag_recv, buf_ppinfo_lindex_recv, buf_ppinfo_proc_recv, buf_ppinfo_shift_recv);

	    free(buf_ppinfo_tag_recv);
	    free(buf_ppinfo_lindex_recv);
	    free(buf_ppinfo_proc_recv);
	    free(buf_ppinfo_shift_recv);
	}

	if (num_buf_tris_recv != 0)
	{
	    buf_tpinfo_tag_recv = (int *) calloc(num_buf_tris_recv+1, sizeof(int));

	    MPI_Recv(buf_tpinfo_tag_recv, num_buf_tris_recv+1, MPI_INT, proc_recv,
		    tag_tris_pinfot, MPI_COMM_WORLD, &recv_status5);

	    num_buf_tris_pinfo_recv = buf_tpinfo_tag_recv[num_buf_tris_recv];

	    buf_tpinfo_lindex_recv = (int *) calloc(num_buf_tris_pinfo_recv, sizeof(int));
	    buf_tpinfo_proc_recv = (int *) calloc(num_buf_tris_pinfo_recv, sizeof(int));
	    buf_tpinfo_shift_recv = (int8_T *) calloc(3*num_buf_tris_pinfo_recv, sizeof(int8_T));

	    MPI_Recv(buf_tpinfo_lindex_recv, num_buf_tris_pinfo_recv, MPI_INT, proc_recv,
		    tag_tris_pinfol, MPI_COMM_WORLD, &recv_status6);
	    MPI_Recv(buf_tpinfo_proc_recv, num_buf_tris_pinfo_recv, MPI_INT, proc_recv,
		    tag_tris_pinfop, MPI_COMM_WORLD, &recv_status7);
	    MPI_Recv(buf_tpinfo_shift_recv, 3*num_buf_tris_pinfo_recv, MPI_SIGNED_CHAR, proc_recv,
		    tag_tris_pinfos, MPI_COMM_WORLD, &recv_status8);

	    hpMergeGhostTrisPInfo(mesh, proc_recv, num_buf_tris_recv,
		    buf_tpinfo_tag_recv, buf_tpinfo_lindex_recv, buf_tpinfo_proc_recv, buf_tpinfo_shift_recv);
	    free(buf_tpinfo_tag_recv);
	    free(buf_tpinfo_lindex_recv);
	    free(buf_tpinfo_proc_recv);
	    free(buf_tpinfo_shift_recv);
	}
    }

    free(recv_req_list);
    free(recv_size);

    // Wait until all the array are sent

    MPI_Waitall(num_all_send_rqst, send_rqst_list, send_status_list);

    // Free the array for send

    free(send_rqst_list);
    free(send_status_list);

    for (i = 0; i < num_nb_proc; i++)
    {
	free(buffer_ps_pinfo_tag[i]);
	free(buffer_ps_pinfo_lindex[i]);
	free(buffer_ps_pinfo_proc[i]);

	free(buffer_tris_pinfo_tag[i]);
	free(buffer_tris_pinfo_lindex[i]);
	free(buffer_tris_pinfo_proc[i]);
    }

    free(send_size);

    free(buffer_ps_pinfo_tag);
    free(buffer_ps_pinfo_lindex);
    free(buffer_ps_pinfo_proc);
    free(buffer_ps_pinfo_shift);

    free(buffer_tris_pinfo_tag);
    free(buffer_tris_pinfo_lindex);
    free(buffer_tris_pinfo_proc);
    free(buffer_tris_pinfo_shift);
}

void hpUpdatePInfo(hiPropMesh *mesh)
{
    hpUpdateMasterPInfo(mesh);

    MPI_Barrier(MPI_COMM_WORLD);

    hpUpdateAllPInfoFromMaster(mesh);
}

static void  hpBuildEstNbFromBdbox(const hiPropMesh *mesh, const double *all_bd_box,
	emxArray_int32_T **new_nb_proc, emxArray_int8_T ***new_nb_shift_proc)
{
    int cur_proc;
    int num_proc;
    MPI_Comm_rank(MPI_COMM_WORLD, &cur_proc);
    MPI_Comm_size(MPI_COMM_WORLD, &num_proc);
    int i;

    double domain_len_x = mesh->domain_len[0];
    double domain_len_y = mesh->domain_len[1];
    double domain_len_z = mesh->domain_len[2];

    boolean_T *nb_ptemp_est = (boolean_T *) calloc (num_proc, sizeof(boolean_T));
    int num_nbp_est = 0;

    int *nb_num_shift_est = (int *) calloc (num_proc, sizeof(int));

    for (i = 0; i < num_proc; ++i)
    {
	// Iterate all possible periodic boundary conditions 
	int x_iter = 1;
	int y_iter = 1;
	int z_iter = 1;

	if (mesh->has_periodic_boundary[0] == true)
	    x_iter = 3;
	if (mesh->has_periodic_boundary[1] == true)
	    y_iter = 3;
	if (mesh->has_periodic_boundary[2] == true)
	    z_iter = 3;

	int j1, j2, j3;
	double x_factor = 0.0;
	double y_factor = 0.0;
	double z_factor = 0.0;

	int8_T sx,sy,sz;

	int num_cur_shift = 0;

	for (j1 = 0; j1 < x_iter; j1++)
	{
	    if (j1 > 1)
	    {
		sx = -1;
	    }
	    else
		sx = (int8_T) j1;

	    for (j2 = 0; j2 < y_iter; j2++)
	    {
		if (j2 > 1)
		    sy = -1;
		else
		    sy = (int8_T) j2;

		for (j3 = 0; j3 < z_iter; j3++)
		{
		    if (j3 > 1)
			sz = -1;
		    else
			sz = (int8_T) j3;

		    if (j1 == 0 && j2 == 0 && j3 == 0 && i == cur_proc)
			continue;

		    x_factor = ((double) sx)*domain_len_x;
		    y_factor = ((double) sy)*domain_len_y;
		    z_factor = ((double) sz)*domain_len_z;

		    // Actural loop starts here

		    double comxL = hpMax(all_bd_box[cur_proc*6] - x_factor, all_bd_box[i*6]);
		    double comxU = hpMin(all_bd_box[cur_proc*6+1] - x_factor, all_bd_box[i*6+1]);
		    double comyL = hpMax(all_bd_box[cur_proc*6+2] - y_factor, all_bd_box[i*6+2]);
		    double comyU = hpMin(all_bd_box[cur_proc*6+3] - y_factor, all_bd_box[i*6+3]);
		    double comzL = hpMax(all_bd_box[cur_proc*6+4] - z_factor, all_bd_box[i*6+4]);
		    double comzU = hpMin(all_bd_box[cur_proc*6+5] - z_factor, all_bd_box[i*6+5]);

		    if ( (comxL <= comxU) && (comyL <= comyU) && (comzL <= comzU) )
		    {
			num_cur_shift++;
			if (!(nb_ptemp_est[i]))
			{
			    nb_ptemp_est[i] = 1;
			    num_nbp_est++;
			}
		    }
		    // Actural loop ends here 
		}
	    }
	}
	nb_num_shift_est[i] = num_cur_shift;
    }

    (*new_nb_proc) = emxCreateND_int32_T(1, &num_nbp_est);
    (*new_nb_shift_proc) = (emxArray_int8_T **) calloc(num_nbp_est, sizeof(emxArray_int8_T *));

    // Set new nb_proc, allocate memory for new nb_proc_shift

    int j = 0;
    for (i = 0; i < num_proc; i++)
    {
	if (nb_ptemp_est[i])
	{
	    (*new_nb_proc)->data[j] = i;
	    int num_shift = nb_num_shift_est[i];
	    ((*new_nb_shift_proc)[j]) = emxCreate_int8_T(num_shift, 3);
	    j++;
	}
    }

    // Set new nb_proc_shift
    
    for (i = 0; i < num_nbp_est; ++i)
    {
	int cur_nb_proc = (*new_nb_proc)->data[i];
	emxArray_int8_T *cur_nb_shift_new = (*new_nb_shift_proc)[i];

	// Iterate all possible periodic boundary conditions 
	int x_iter = 1;
	int y_iter = 1;
	int z_iter = 1;

	if (mesh->has_periodic_boundary[0] == true)
	    x_iter = 3;
	if (mesh->has_periodic_boundary[1] == true)
	    y_iter = 3;
	if (mesh->has_periodic_boundary[2] == true)
	    z_iter = 3;

	int j1, j2, j3;
	double x_factor = 0.0;
	double y_factor = 0.0;
	double z_factor = 0.0;

	int8_T sx,sy,sz;

	int cur_shift_iter = 1;

	for (j1 = 0; j1 < x_iter; j1++)
	{
	    if (j1 > 1)
	    {
		sx = -1;
	    }
	    else
		sx = (int8_T) j1;

	    for (j2 = 0; j2 < y_iter; j2++)
	    {
		if (j2 > 1)
		    sy = -1;
		else
		    sy = (int8_T) j2;

		for (j3 = 0; j3 < z_iter; j3++)
		{
		    if (j3 > 1)
			sz = -1;
		    else
			sz = (int8_T) j3;

		    if (j1 == 0 && j2 == 0 && j3 == 0 && i == cur_proc)
			continue;

		    x_factor = ((double) sx)*domain_len_x;
		    y_factor = ((double) sy)*domain_len_y;
		    z_factor = ((double) sz)*domain_len_z;

		    // Actural loop starts here

		    double comxL = hpMax(all_bd_box[cur_proc*6] - x_factor, all_bd_box[cur_nb_proc*6]);
		    double comxU = hpMin(all_bd_box[cur_proc*6+1] - x_factor, all_bd_box[cur_nb_proc*6+1]);
		    double comyL = hpMax(all_bd_box[cur_proc*6+2] - y_factor, all_bd_box[cur_nb_proc*6+2]);
		    double comyU = hpMin(all_bd_box[cur_proc*6+3] - y_factor, all_bd_box[cur_nb_proc*6+3]);
		    double comzL = hpMax(all_bd_box[cur_proc*6+4] - z_factor, all_bd_box[cur_nb_proc*6+4]);
		    double comzU = hpMin(all_bd_box[cur_proc*6+5] - z_factor, all_bd_box[cur_nb_proc*6+5]);

		    if ( (comxL <= comxU) && (comyL <= comyU) && (comzL <= comzU) )
		    {
			cur_nb_shift_new->data[I2dm(cur_shift_iter, 1, cur_nb_shift_new->size)] = sx;
			cur_nb_shift_new->data[I2dm(cur_shift_iter, 2, cur_nb_shift_new->size)] = sy;
			cur_nb_shift_new->data[I2dm(cur_shift_iter, 3, cur_nb_shift_new->size)] = sz;
			cur_shift_iter++;
		    }
		    // Actural loop ends here 
		}
	    }
	}
    }

    free(nb_ptemp_est);
    free(nb_num_shift_est);
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

    // temp change of nb_proc to all other processors

    emxArray_int32_T *new_nb_proc;
    emxArray_int8_T **new_nb_proc_shift;

    hpBuildEstNbFromBdbox(mesh, all_bd_box, &new_nb_proc, &new_nb_proc_shift);

    for (i = 0; i < mesh->nb_proc->size[0]; i++)
	emxFree_int8_T(&(mesh->nb_proc_shift[i]));
    free(mesh->nb_proc_shift);
    emxFree_int32_T(&(mesh->nb_proc));

    mesh->nb_proc = new_nb_proc;
    mesh->nb_proc_shift = new_nb_proc_shift;
    int num_nb_proc = mesh->nb_proc->size[0];
    emxArray_int32_T **ps_ring_proc = (emxArray_int32_T **) calloc(num_nb_proc, sizeof(emxArray_int32_T *));
    emxArray_int32_T **tris_ring_proc = (emxArray_int32_T **) calloc(num_nb_proc, sizeof(emxArray_int32_T *));

    emxArray_int8_T **ps_shift_ring_proc = (emxArray_int8_T **) calloc(num_nb_proc, sizeof(emxArray_int8_T *));
    emxArray_int8_T **tris_shift_ring_proc = (emxArray_int8_T **) calloc(num_nb_proc, sizeof(emxArray_int8_T *));

    emxArray_real_T **buffer_ps = (emxArray_real_T **) calloc(num_nb_proc, sizeof(emxArray_real_T *));
    emxArray_int32_T **buffer_tris = (emxArray_int32_T **) calloc(num_nb_proc, sizeof(emxArray_int32_T *));


    for (i = 1; i <= num_nb_proc; i++)
    {
	hpBuildBdboxGhostPsTrisForSend(mesh, i, all_bd_box,
				  &(ps_ring_proc[I1dm(i)]),
				  &(tris_ring_proc[I1dm(i)]),
				  &(ps_shift_ring_proc[I1dm(i)]),
				  &(tris_shift_ring_proc[I1dm(i)]),
				  &(buffer_ps[I1dm(i)]), &(buffer_tris[I1dm(i)]));
    }

    free(all_bd_box);

    hpCommPsTrisWithPInfo(mesh, ps_ring_proc, tris_ring_proc, ps_shift_ring_proc, tris_shift_ring_proc, buffer_ps, buffer_tris);

    for (i = 1; i <= num_nb_proc; i++)
    {
	emxFree_int32_T(&(ps_ring_proc[I1dm(i)]));
	emxFree_int32_T(&(tris_ring_proc[I1dm(i)]));

	emxFree_int8_T(&(ps_shift_ring_proc[I1dm(i)]));
	emxFree_int8_T(&(tris_shift_ring_proc[I1dm(i)]));

	emxFree_real_T(&(buffer_ps[I1dm(i)]));
	emxFree_int32_T(&(buffer_tris[I1dm(i)]));
    }

    free(ps_ring_proc);
    free(tris_ring_proc);
    free(ps_shift_ring_proc);
    free(tris_shift_ring_proc);
    free(buffer_ps);
    free(buffer_tris);

    hpUpdatePInfo(mesh);

    hpUpdateNbWithPInfo(mesh);
}


void hpCommPsTrisWithPInfo(hiPropMesh *mesh, emxArray_int32_T **ps_ring_proc, emxArray_int32_T **tris_ring_proc,
			   emxArray_int8_T **ps_shift_ring_proc, emxArray_int8_T **tris_shift_ring_proc,
			   emxArray_real_T **buffer_ps, emxArray_int32_T **buffer_tris)
{
    int i;

    emxArray_int32_T *nb_proc = mesh->nb_proc;
    int num_nb_proc = nb_proc->size[0];

    // If there's some points need to be sent, there has to be a triangle needed
    // to be sent, vice versa
    for (i = 1; i <= num_nb_proc; i++)
    {
	if ( (ps_ring_proc[I1dm(i)]->size[0] == 0) && (tris_ring_proc[I1dm(i)]->size[0] != 0) )
	{
	    printf("\n For dst proc %d, ps send size = 0, tris send size != 0, inconsistent!\n", mesh->nb_proc->data[I1dm(i)]);
	    exit(0);
	}
	if ( (ps_ring_proc[I1dm(i)]->size[0] != 0) && (tris_ring_proc[I1dm(i)]->size[0] == 0) )
	{
	    printf("\n For dst proc %d, ps send size != 0, tris send size = 0, inconsistent!\n", mesh->nb_proc->data[I1dm(i)]);
	    exit(0);
	}

    }
    // Add the new proc info to the ps/tris for send, lindex currently unknown,
    // shift info added.
    for (i = 1; i <= num_nb_proc; i++)
    {
	hpAddProcInfoForGhostPsTris(mesh, i, ps_ring_proc[I1dm(i)], tris_ring_proc[I1dm(i)],
		ps_shift_ring_proc[I1dm(i)], tris_shift_ring_proc[I1dm(i)]);
    }


    // Build the pinfo (proc/lindex/shift) for send
    int **buffer_ps_pinfo_tag = (int **) calloc(num_nb_proc, sizeof(int *));
    int **buffer_ps_pinfo_lindex = (int **) calloc(num_nb_proc, sizeof(int *));
    int **buffer_ps_pinfo_proc = (int **) calloc(num_nb_proc, sizeof(int *));
    int8_T **buffer_ps_pinfo_shift = (int8_T **)calloc(num_nb_proc, sizeof(int8_T *));

    int **buffer_tris_pinfo_tag = (int **) calloc(num_nb_proc, sizeof(int *));
    int **buffer_tris_pinfo_lindex = (int **) calloc(num_nb_proc, sizeof(int *));
    int **buffer_tris_pinfo_proc = (int **) calloc(num_nb_proc, sizeof(int *));
    int8_T **buffer_tris_pinfo_shift = (int8_T **) calloc(num_nb_proc, sizeof(int8_T *));

    for (i = 1; i <= num_nb_proc; i++)
    {
	hpBuildGhostPsTrisPInfoForSend(mesh, i, ps_ring_proc[I1dm(i)], tris_ring_proc[I1dm(i)],
		ps_shift_ring_proc[I1dm(i)], tris_shift_ring_proc[I1dm(i)],
		&(buffer_ps_pinfo_tag[I1dm(i)]),
		&(buffer_ps_pinfo_lindex[I1dm(i)]),
		&(buffer_ps_pinfo_proc[I1dm(i)]),
		&(buffer_ps_pinfo_shift[I1dm(i)]),
		&(buffer_tris_pinfo_tag[I1dm(i)]),
		&(buffer_tris_pinfo_lindex[I1dm(i)]),
		&(buffer_tris_pinfo_proc[I1dm(i)]),
		&(buffer_tris_pinfo_shift[I1dm(i)]));

    }

    // Set up the MPI_Request list for send
    int num_all_send_rqst = 11*num_nb_proc;

    MPI_Request* send_rqst_list = (MPI_Request *) calloc(num_all_send_rqst, sizeof(MPI_Request));
    MPI_Status* send_status_list = (MPI_Status *) calloc(num_all_send_rqst, sizeof(MPI_Status));

    for (i = 0; i < num_all_send_rqst; i++)
	send_rqst_list[i] = MPI_REQUEST_NULL;


    // Set up send size in the order of ps size, tris size

    int *size_send = (int *) calloc(2*num_nb_proc, sizeof(int));
    for (i = 0; i < num_nb_proc; ++i)
    {
	size_send[2*i] = (buffer_ps[i])->size[0];
	size_send[2*i+1] = (buffer_tris[i])->size[0];
    }

    MPI_Request* recv_req_list = (MPI_Request *) calloc(num_nb_proc, sizeof(MPI_Request));

    int *recv_size = (int *) calloc (2*num_nb_proc, sizeof(int));

    int cur_rqst = 0;

    // send all the information to different processors
    for (i = 0; i < num_nb_proc; ++i)
    {
	int tag_size;
	int tag_ps, tag_ps_tag, tag_ps_li, tag_ps_proc, tag_ps_shift;
	int tag_tris, tag_tris_tag, tag_tris_li, tag_tris_proc, tag_tris_shift;

	tag_size = 1;
	tag_ps = 11; tag_ps_tag = 12; tag_ps_li = 13; tag_ps_proc = 14; tag_ps_shift = 15;
	tag_tris = 21; tag_tris_tag = 22; tag_tris_li = 23; tag_tris_proc = 24; tag_tris_shift = 25;

	int num_buf_ps = size_send[2*i];
	int num_buf_tris = size_send[2*i+1];

	int num_buf_ps_pinfo = buffer_ps_pinfo_tag[i][num_buf_ps];
	int num_buf_tris_pinfo = buffer_tris_pinfo_tag[i][num_buf_tris];

	MPI_Isend(&(size_send[2*i]), 2, MPI_INT, nb_proc->data[i], tag_size, MPI_COMM_WORLD, &(send_rqst_list[cur_rqst++]));

	if ( (size_send[2*i] != 0) && (size_send[2*i+1] != 0) )
	{
	    MPI_Isend((buffer_ps[i])->data, 3*size_send[2*i], MPI_DOUBLE, nb_proc->data[i], tag_ps, MPI_COMM_WORLD, &(send_rqst_list[cur_rqst++]));
	    MPI_Isend((buffer_tris[i])->data, 3*size_send[2*i+1], MPI_INT, nb_proc->data[i], tag_tris, MPI_COMM_WORLD, &(send_rqst_list[cur_rqst++]));

	    MPI_Isend(buffer_ps_pinfo_tag[i], num_buf_ps+1, MPI_INT,
		    nb_proc->data[i], tag_ps_tag, MPI_COMM_WORLD, &(send_rqst_list[cur_rqst++]));
	    MPI_Isend(buffer_ps_pinfo_lindex[i], num_buf_ps_pinfo, MPI_INT,
		    nb_proc->data[i], tag_ps_li, MPI_COMM_WORLD, &(send_rqst_list[cur_rqst++]));
	    MPI_Isend(buffer_ps_pinfo_proc[i], num_buf_ps_pinfo, MPI_INT,
		    nb_proc->data[i], tag_ps_proc, MPI_COMM_WORLD, &(send_rqst_list[cur_rqst++]));
	    MPI_Isend(buffer_ps_pinfo_shift[i],3*num_buf_ps_pinfo, MPI_SIGNED_CHAR,
		    nb_proc->data[i], tag_ps_shift, MPI_COMM_WORLD, &(send_rqst_list[cur_rqst++]));

	    MPI_Isend(buffer_tris_pinfo_tag[i], num_buf_tris+1, MPI_INT,
		    nb_proc->data[i], tag_tris_tag, MPI_COMM_WORLD, &(send_rqst_list[cur_rqst++]));
	    MPI_Isend(buffer_tris_pinfo_lindex[i], num_buf_tris_pinfo, MPI_INT,
		    nb_proc->data[i], tag_tris_li, MPI_COMM_WORLD, &(send_rqst_list[cur_rqst++]));
	    MPI_Isend(buffer_tris_pinfo_proc[i], num_buf_tris_pinfo, MPI_INT,
		    nb_proc->data[i], tag_tris_proc, MPI_COMM_WORLD, &(send_rqst_list[cur_rqst++]));
	    MPI_Isend(buffer_tris_pinfo_shift[i],3*num_buf_tris_pinfo, MPI_SIGNED_CHAR,
		    nb_proc->data[i], tag_tris_shift, MPI_COMM_WORLD, &(send_rqst_list[cur_rqst++]));
	}

    }

    // Receive buffer points and tris with temp pinfo

    for (i = 0; i < num_nb_proc; i++)
    {
	int tag_size_recv = 1;
	MPI_Irecv(&(recv_size[2*i]), 2, MPI_INT, nb_proc->data[i], tag_size_recv, MPI_COMM_WORLD, &(recv_req_list[i]));
    }

    for (i = 0; i < num_nb_proc; ++i)
    {
	int tag_ps, tag_ps_tag, tag_ps_li, tag_ps_proc, tag_ps_shift;
	int tag_tris, tag_tris_tag, tag_tris_li, tag_tris_proc, tag_tris_shift;

	tag_ps = 11; tag_ps_tag = 12; tag_ps_li = 13; tag_ps_proc = 14; tag_ps_shift = 15;
	tag_tris = 21; tag_tris_tag = 22; tag_tris_li = 23; tag_tris_proc = 24; tag_tris_shift = 25;

	int recv_index;
	int proc_recv;
	MPI_Status recv_status1;

	MPI_Waitany(num_nb_proc, recv_req_list, &recv_index, &recv_status1);
	proc_recv = recv_status1.MPI_SOURCE;

	if ( (recv_size[2*recv_index] != 0) && (recv_size[2*recv_index+1] != 0) )
	{
	    emxArray_real_T *buffer_ps_recv;
	    emxArray_int32_T *buffer_tris_recv;

	    int *buf_ppinfo_tag_recv;
	    int *buf_ppinfo_lindex_recv;
	    int *buf_ppinfo_proc_recv;
	    int8_T *buf_ppinfo_shift_recv;

	    int *buf_tpinfo_tag_recv;
	    int *buf_tpinfo_lindex_recv;
	    int *buf_tpinfo_proc_recv;
	    int8_T *buf_tpinfo_shift_recv;

	    int num_buf_ps_recv, num_buf_tris_recv;

	    int num_buf_ps_pinfo_recv;
	    int num_buf_tris_pinfo_recv;

	    MPI_Status recv_status_ps, recv_status_tris;
	    MPI_Status recv_status_ptag, recv_status_pli, recv_status_pp, recv_status_psh;
	    MPI_Status recv_status_ttag, recv_status_tli, recv_status_tp, recv_status_tsh;

	    buffer_ps_recv = emxCreate_real_T(recv_size[2*recv_index], 3);

	    MPI_Recv(buffer_ps_recv->data, 3*recv_size[2*recv_index], MPI_DOUBLE, proc_recv, tag_ps, MPI_COMM_WORLD, &recv_status_ps);

	    buffer_tris_recv = emxCreate_int32_T(recv_size[2*recv_index+1], 3);

	    MPI_Recv(buffer_tris_recv->data, 3*recv_size[2*recv_index+1], MPI_INT, proc_recv, tag_tris, MPI_COMM_WORLD, &recv_status_tris);

	    num_buf_ps_recv = buffer_ps_recv->size[0];
	    num_buf_tris_recv = buffer_tris_recv->size[0];

	    // Recv ps pinfo
	    buf_ppinfo_tag_recv = (int *) calloc(num_buf_ps_recv+1, sizeof(int));

	    MPI_Recv(buf_ppinfo_tag_recv, num_buf_ps_recv+1, MPI_INT, proc_recv,
		    tag_ps_tag, MPI_COMM_WORLD, &recv_status_ptag);

	    num_buf_ps_pinfo_recv = buf_ppinfo_tag_recv[num_buf_ps_recv];

	    buf_ppinfo_lindex_recv = (int *) calloc(num_buf_ps_pinfo_recv, sizeof(int));
	    buf_ppinfo_proc_recv = (int *) calloc(num_buf_ps_pinfo_recv, sizeof(int));
	    buf_ppinfo_shift_recv = (int8_T *) calloc(3*num_buf_ps_pinfo_recv, sizeof(int8_T));

	    MPI_Recv(buf_ppinfo_lindex_recv, num_buf_ps_pinfo_recv, MPI_INT, proc_recv,
		    tag_ps_li, MPI_COMM_WORLD, &recv_status_pli);
	    MPI_Recv(buf_ppinfo_proc_recv, num_buf_ps_pinfo_recv, MPI_INT, proc_recv,
		    tag_ps_proc, MPI_COMM_WORLD, &recv_status_pp);
	    MPI_Recv(buf_ppinfo_shift_recv, 3*num_buf_ps_pinfo_recv, MPI_SIGNED_CHAR, proc_recv,
		    tag_ps_shift, MPI_COMM_WORLD, &recv_status_psh);

	    // Recv tris pinfo
	    buf_tpinfo_tag_recv = (int *) calloc(num_buf_tris_recv+1, sizeof(int));

	    MPI_Recv(buf_tpinfo_tag_recv, num_buf_tris_recv+1, MPI_INT, proc_recv,
		    tag_tris_tag, MPI_COMM_WORLD, &recv_status_ttag);

	    num_buf_tris_pinfo_recv = buf_tpinfo_tag_recv[num_buf_tris_recv];

	    buf_tpinfo_lindex_recv = (int *) calloc(num_buf_tris_pinfo_recv, sizeof(int));
	    buf_tpinfo_proc_recv = (int *) calloc(num_buf_tris_pinfo_recv, sizeof(int));
	    buf_tpinfo_shift_recv = (int8_T *) calloc(3*num_buf_tris_pinfo_recv, sizeof(int8_T));

	    MPI_Recv(buf_tpinfo_lindex_recv, num_buf_tris_pinfo_recv, MPI_INT, proc_recv,
		    tag_tris_li, MPI_COMM_WORLD, &recv_status_tli);
	    MPI_Recv(buf_tpinfo_proc_recv, num_buf_tris_pinfo_recv, MPI_INT, proc_recv,
		    tag_tris_proc, MPI_COMM_WORLD, &recv_status_tp);
	    MPI_Recv(buf_tpinfo_shift_recv, 3*num_buf_tris_pinfo_recv, MPI_SIGNED_CHAR, proc_recv,
		    tag_tris_shift, MPI_COMM_WORLD, &recv_status_tsh);

	    hpAttachNRingGhostWithPInfo(mesh, proc_recv, buffer_ps_recv, buffer_tris_recv,
		    buf_ppinfo_tag_recv, buf_ppinfo_lindex_recv,  buf_ppinfo_proc_recv, buf_ppinfo_shift_recv,
		    buf_tpinfo_tag_recv, buf_tpinfo_lindex_recv,  buf_tpinfo_proc_recv, buf_tpinfo_shift_recv);

	    emxFree_real_T(&buffer_ps_recv);
	    emxFree_int32_T(&buffer_tris_recv);

	    free(buf_ppinfo_tag_recv);
	    free(buf_ppinfo_lindex_recv);
	    free(buf_ppinfo_proc_recv);
	    free(buf_ppinfo_shift_recv);

	    free(buf_tpinfo_tag_recv);
	    free(buf_tpinfo_lindex_recv);
	    free(buf_tpinfo_proc_recv);
	    free(buf_tpinfo_shift_recv);
	}
    }

    free(recv_req_list);
    free(recv_size);

    // Wait until all the array are sent

    MPI_Waitall(num_all_send_rqst, send_rqst_list, send_status_list);

    // Free the array for send

    free(send_rqst_list);
    free(send_status_list);
    free(size_send);

    for (i = 0; i < num_nb_proc; i++)
    {
	free(buffer_ps_pinfo_tag[i]);
	free(buffer_ps_pinfo_lindex[i]);
	free(buffer_ps_pinfo_proc[i]);
	free(buffer_ps_pinfo_shift[i]);

	free(buffer_tris_pinfo_tag[i]);
	free(buffer_tris_pinfo_lindex[i]);
	free(buffer_tris_pinfo_proc[i]);
	free(buffer_tris_pinfo_shift[i]);
    }

    free(buffer_ps_pinfo_tag);
    free(buffer_ps_pinfo_lindex);
    free(buffer_ps_pinfo_proc);
    free(buffer_ps_pinfo_shift);

    free(buffer_tris_pinfo_tag);
    free(buffer_tris_pinfo_lindex);
    free(buffer_tris_pinfo_proc);
    free(buffer_tris_pinfo_shift);
}


void hpBuildNRingGhost(hiPropMesh *mesh, const real_T num_ring)
{
    int i;

    int num_nb_proc = mesh->nb_proc->size[0];


    emxArray_int32_T **psid_proc = (emxArray_int32_T **)
	calloc(num_nb_proc, sizeof(emxArray_int32_T *));

    emxArray_int8_T **ps_shift_proc = (emxArray_int8_T **)
	calloc(num_nb_proc, sizeof(emxArray_int8_T *));

    // Get the overlapping points with shift for building up n-ring neighborhood
    hpCollectAllSharedPs(mesh, psid_proc, ps_shift_proc);

    // Build n-ring neighborhood and send
    emxArray_int32_T **ps_ring_proc = (emxArray_int32_T **) calloc(num_nb_proc, sizeof(emxArray_int32_T *));
    emxArray_int32_T **tris_ring_proc = (emxArray_int32_T **) calloc(num_nb_proc, sizeof(emxArray_int32_T *));

    emxArray_int8_T **ps_shift_ring_proc = (emxArray_int8_T **) calloc(num_nb_proc, sizeof(emxArray_int8_T *));
    emxArray_int8_T **tris_shift_ring_proc = (emxArray_int8_T **) calloc(num_nb_proc, sizeof(emxArray_int8_T *));

    emxArray_real_T **buffer_ps = (emxArray_real_T **) calloc(num_nb_proc, sizeof(emxArray_real_T *));
    emxArray_int32_T **buffer_tris = (emxArray_int32_T **) calloc(num_nb_proc, sizeof(emxArray_int32_T *));

    for (i = 1; i <= num_nb_proc; i++)
    {
	hpBuildGhostPsTrisForSend(mesh, i, num_ring, psid_proc[I1dm(i)], ps_shift_proc[I1dm(i)],
		&(ps_ring_proc[I1dm(i)]),
		&(tris_ring_proc[I1dm(i)]),
		&(ps_shift_ring_proc[I1dm(i)]),
		&(tris_shift_ring_proc[I1dm(i)]), 
		&(buffer_ps[I1dm(i)]), &(buffer_tris[I1dm(i)]));
    }

    for (i = 1; i <= num_nb_proc; i++)
    {
	emxFree_int32_T(&(psid_proc[I1dm(i)]));
	emxFree_int8_T(&(ps_shift_proc[I1dm(i)]));
    }
    free(psid_proc);
    free(ps_shift_proc);



    // Build and communicate the ghost based on ps_ring_proc, tris_ring_proc,
    // buffer_ps, buffer_tris
    // There could exists duplicate point in ps_ring_proc and tris_ring_proc,
    // which is same case as sending same ps/tris to the same processor from
    // different processors. This info has to be dealt with when receiving the
    // points/triangles.
    //
    // Current ps_shift_ring_proc and tris_shift_ring_proc stores the shift info for
    // each element. The format is the same as shift info in the pinfo list,
    // which is: destination + shift = source ==> destination = source - shift.

    hpCommPsTrisWithPInfo(mesh, ps_ring_proc, tris_ring_proc, ps_shift_ring_proc, tris_shift_ring_proc, buffer_ps, buffer_tris);
    for (i = 1; i <= num_nb_proc; i++)
    {
	emxFree_int32_T(&(ps_ring_proc[I1dm(i)]));
	emxFree_int32_T(&(tris_ring_proc[I1dm(i)]));

	emxFree_int8_T(&(ps_shift_ring_proc[I1dm(i)]));
	emxFree_int8_T(&(tris_shift_ring_proc[I1dm(i)]));

	emxFree_real_T(&(buffer_ps[I1dm(i)]));
	emxFree_int32_T(&(buffer_tris[I1dm(i)]));
    }

    free(ps_ring_proc);
    free(tris_ring_proc);
    free(ps_shift_ring_proc);
    free(tris_shift_ring_proc);
    free(buffer_ps);
    free(buffer_tris);

    hpUpdatePInfo(mesh);

    hpUpdateNbWithPInfo(mesh);
}


void hpAttachNRingGhostWithPInfo(hiPropMesh *mesh,
				 const int rcv_id,
				 emxArray_real_T *bps,
				 emxArray_int32_T *btris,
				 int *ppinfot,
				 int *ppinfol,
				 int *ppinfop,
				 int8_T *ppinfos,
				 int *tpinfot,
				 int *tpinfol,
				 int *tpinfop,
				 int8_T *tpinfos)
{

    int i,j;
    int cur_proc;
    MPI_Comm_rank(MPI_COMM_WORLD, &cur_proc);

    emxArray_real_T *ps = mesh->ps;
    emxArray_int32_T *tris = mesh->tris;

    real_T *ps_data = ps->data;
    int32_T *ps_size = ps->size;

    int32_T *tris_data = tris->data;
    int32_T *tris_size = tris->size;

    real_T *bps_data = bps->data;
    int32_T *bps_size = bps->size;

    int32_T *btris_data = btris->data;
    int32_T *btris_size = btris->size;


    int num_ps_old = ps->size[0];
    int num_tris_old = tris->size[0];

    hpPInfoList *ps_pinfo = mesh->ps_pinfo;
    hpPInfoList *tris_pinfo = mesh->tris_pinfo;

    hpPInfoNode *ps_pdata = ps_pinfo->pdata;
    int *ps_phead = ps_pinfo->head;
    int *ps_ptail = ps_pinfo->tail;

    hpPInfoNode *tris_pdata = tris_pinfo->pdata;
    int *tris_phead = tris_pinfo->head;
    int *tris_ptail = tris_pinfo->tail;

    int num_buf_ps = bps->size[0];
    int num_buf_tris = btris->size[0];

    int *ps_map = (int *) calloc(num_buf_ps, sizeof(int));
    int *tris_map = (int *) calloc(num_buf_tris, sizeof(int));

    boolean_T *buf_ps_flag = (boolean_T *) calloc(num_buf_ps, sizeof(boolean_T));
    boolean_T *buf_tris_flag = (boolean_T *) calloc(num_buf_tris, sizeof(boolean_T));

    int num_add_ps = 0;
    int num_add_tris = 0;


    // Calculate # of new ps/tris and allocated the memory
    // Update ps_map, have all new ps and tris flagged

    for (i = 0; i < num_buf_ps; i++)
    {
	buf_ps_flag[i] = 1;
	// If already existing, do not add a new point
	for(j = ppinfot[i]; j <= ppinfot[i+1]-1; j++)
	{
	    if ( (ppinfop[j] == cur_proc) 
		    && (ppinfol[j] != -1) 
		    && (ppinfos[j*3] == 0)// All shifts == 0 means it is the local point
		    && (ppinfos[j*3+1] == 0)
		    && (ppinfos[j*3+2] == 0) )
	    {
		buf_ps_flag[i] = 0;
		ps_map[i] = ppinfol[j];
		break;
	    }
	}

	// Still could be some existing point attached from other
	// processors or same processor different shift

	// First check the different shift from same processor

	if (buf_ps_flag[i])
	{
	    int cur_head_tag = ppinfot[i];

	    for (j = 0; j <= i-1; j++)
	    {
		int head_tag_comp = ppinfot[j];
		if ( (ppinfop[cur_head_tag] == ppinfop[head_tag_comp] )
			&& (ppinfol[cur_head_tag] == ppinfol[head_tag_comp])
			&& (ppinfos[cur_head_tag*3] == ppinfos[head_tag_comp*3])
			&& (ppinfos[cur_head_tag*3+1] == ppinfos[head_tag_comp*3+1])
			&& (ppinfos[cur_head_tag*3+2] == ppinfos[head_tag_comp*3+2]) )
		{
		    buf_ps_flag[i] = 0;
		    ps_map[i] = ps_map[j];
		    break;
		} 
	    }
	}

	// Then check possible points from other processors

	if (buf_ps_flag[i])
	{
	    for (j = 1; j <= num_ps_old; j++)
	    {
		int head_cur = ps_phead[I1dm(j)];

		if ((ps_pdata[I1dm(head_cur)].proc == ppinfop[ppinfot[i]]) &&
		    (ps_pdata[I1dm(head_cur)].lindex == ppinfol[ppinfot[i]]) &&
		    (ps_pdata[I1dm(head_cur)].shift[0] == ppinfos[ppinfot[i]*3]) &&
		    (ps_pdata[I1dm(head_cur)].shift[1] == ppinfos[ppinfot[i]*3+1]) &&
		    (ps_pdata[I1dm(head_cur)].shift[2] == ppinfos[ppinfot[i]*3+2]) )
		{
		    buf_ps_flag[i] = 0;
		    ps_map[i] = j;
		    break;
		}
	    }
	}
	// now a new point
	if (buf_ps_flag[i])
	{
	    num_add_ps++;
	    ps_map[i] = num_ps_old + num_add_ps;
	}
    }

    // For triangles

    for (i = 0; i < num_buf_tris; i++)
    {
	buf_tris_flag[i] = 1;
	for(j = tpinfot[i]; j <= tpinfot[i+1]-1; j++)
	{
	    if ((tpinfop[j] == cur_proc) 
		    && (tpinfol[j] != -1)
		    && (tpinfos[j*3] == 0)
		    && (tpinfos[j*3+1] == 0)
		    && (tpinfos[j*3+2] == 0) )
	    {
		buf_tris_flag[i] = 0;
		tris_map[i] = tpinfol[j];
		break;
	    }
	}

	// Still could be some existing triangle attached from other
	// processors or same processor different shift

	// First check the different shift from same processor

	if (buf_tris_flag[i])
	{
	    int cur_head_tag = tpinfot[i];

	    for (j = 0; j <= i-1; j++)
	    {
		int head_tag_comp = tpinfot[j];
		if ( (tpinfop[cur_head_tag] == tpinfop[head_tag_comp] )
			&& (tpinfol[cur_head_tag] == tpinfol[head_tag_comp])
			&& (tpinfos[cur_head_tag*3] == tpinfos[head_tag_comp*3])
			&& (tpinfos[cur_head_tag*3+1] == tpinfos[head_tag_comp*3+1])
			&& (tpinfos[cur_head_tag*3+2] == tpinfos[head_tag_comp*3+2]) )
		{
		    buf_tris_flag[i] = 0;
		    tris_map[i] = tris_map[j];
		    break;
		} 
	    }
	}

	// Then check possible tris from other processors

	if (buf_tris_flag[i])
	{
	    for (j = 1; j <= num_tris_old; j++)
	    {
		int head_cur = tris_pinfo->head[I1dm(j)];

		if ((tris_pdata[I1dm(head_cur)].proc == tpinfop[tpinfot[i]]) &&
		    (tris_pdata[I1dm(head_cur)].lindex == tpinfol[tpinfot[i]]) &&
		    (tris_pdata[I1dm(head_cur)].shift[0] == tpinfos[tpinfot[i]*3]) &&
		    (tris_pdata[I1dm(head_cur)].shift[1] == tpinfos[tpinfot[i]*3+1]) &&
		    (tris_pdata[I1dm(head_cur)].shift[2] == tpinfos[tpinfot[i]*3+2]) )
		{
		    buf_tris_flag[i] = 0;
		    tris_map[i] = j;
		    break;
		}
	    }

	}
	// now a new triangle
	if (buf_tris_flag[i])
	{
	    num_add_tris++;
	    tris_map[i] = num_tris_old + num_add_tris;
	}
    }

    // Allocated more space for ps and tris

    addRowToArray_real_T(ps, num_add_ps);
    addRowToArray_int32_T(tris, num_add_tris);

    // After allocation, re-assign the pointer to ps data and tris data

    ps_data = ps->data;
    ps_size = ps->size;

    tris_data = tris->data;
    tris_size = tris->size;

    // Also need to allocated more space for head and tail in pinfolist

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

    // After reallocating the head/tail array, re-assign the pointer

    ps_phead = ps_pinfo->head;
    ps_ptail = ps_pinfo->tail;

    tris_phead = tris_pinfo->head;
    tris_ptail = tris_pinfo->tail;


    // Add each point, merge pinfo
    //

    for (i = 0; i < num_buf_ps; i++)
    {
	// If not a new point, update the pinfo
	// 

	// If a new point, add the point, the pinfo and update
	// based on the current local index
	
	if (buf_ps_flag[i])
	{
	    int ps_index = ps_map[i];

	    ps_data[I2dm(ps_index,1,ps_size)] = bps_data[I2dm(i+1,1,bps_size)];
	    ps_data[I2dm(ps_index,2,ps_size)] = bps_data[I2dm(i+1,2,bps_size)];
	    ps_data[I2dm(ps_index,3,ps_size)] = bps_data[I2dm(i+1,3,bps_size)];

	    // Deal with head ---> tail - 1

	    int cur_node;
	    int new_head = ps_pinfo->allocated_len + 1;
	    for (j = ppinfot[i]; j < ppinfot[i+1]-1; j++)
	    {
		if(hpEnsurePInfoCapacity(ps_pinfo))
		    ps_pdata = ps_pinfo->pdata;
		ps_pinfo->allocated_len++;
		cur_node = ps_pinfo->allocated_len;
		ps_pdata[I1dm(cur_node)].proc = ppinfop[j];
		ps_pdata[I1dm(cur_node)].next = cur_node+1;
		ps_pdata[I1dm(cur_node)].shift[0] = ppinfos[j*3];
		ps_pdata[I1dm(cur_node)].shift[1] = ppinfos[j*3+1];
		ps_pdata[I1dm(cur_node)].shift[2] = ppinfos[j*3+2];
		if ( (ppinfop[j] == cur_proc) 
			&& (ppinfos[j*3] == 0)
			&& (ppinfos[j*3+1] == 0)
			&& (ppinfos[j*3+2] == 0) )
		    ps_pdata[I1dm(cur_node)].lindex = ps_index;
		else
		    ps_pdata[I1dm(cur_node)].lindex = ppinfol[j];
	    }
	    ps_phead[I1dm(ps_index)] = new_head;
	    // Deal with tail
	    j = ppinfot[i+1]-1;
	    if(hpEnsurePInfoCapacity(ps_pinfo))
		ps_pdata = ps_pinfo->pdata;
	    ps_pinfo->allocated_len++;
	    cur_node = ps_pinfo->allocated_len;
	    ps_pdata[I1dm(cur_node)].proc = ppinfop[j];
	    ps_pdata[I1dm(cur_node)].next = -1;
	    ps_pdata[I1dm(cur_node)].shift[0] = ppinfos[j*3];
	    ps_pdata[I1dm(cur_node)].shift[1] = ppinfos[j*3+1];
	    ps_pdata[I1dm(cur_node)].shift[2] = ppinfos[j*3+2];

	    if ((ppinfop[j] == cur_proc)
		    && (ppinfos[j*3] == 0)
		    && (ppinfos[j*3+1] == 0)
		    && (ppinfos[j*3+2] == 0) )
		ps_pdata[I1dm(cur_node)].lindex = ps_index;
	    else
		ps_pdata[I1dm(cur_node)].lindex = ppinfol[j];
	    ps_ptail[I1dm(ps_index)] = cur_node;
	}
    }
    // Add each triangle, merge pinfo

    for (i = 0; i < num_buf_tris; i++)
    {
	// If not a new triangle, update the pinfo
	// If a new triangle, add the point, the pinfo and update
	// based on the current local index
	if (buf_tris_flag[i])	
	{
	    int tris_index = tris_map[i];

	    int recv_tri_index1 = btris_data[I2dm(i+1,1,btris_size)];
	    int recv_tri_index2 = btris_data[I2dm(i+1,2,btris_size)];
	    int recv_tri_index3 = btris_data[I2dm(i+1,3,btris_size)];

	    tris_data[I2dm(tris_index,1,tris_size)] = ps_map[I1dm(recv_tri_index1)];
	    tris_data[I2dm(tris_index,2,tris_size)] = ps_map[I1dm(recv_tri_index2)];
	    tris_data[I2dm(tris_index,3,tris_size)] = ps_map[I1dm(recv_tri_index3)];

	    // Deal with head ---> tail - 1 
	    int cur_node;
	    int new_head = tris_pinfo->allocated_len + 1;
	    for (j = tpinfot[i]; j < tpinfot[i+1]-1; j++)
	    {
		if(hpEnsurePInfoCapacity(tris_pinfo))
		    tris_pdata = tris_pinfo->pdata;
		tris_pinfo->allocated_len++;
		cur_node = tris_pinfo->allocated_len;

		tris_pdata[I1dm(cur_node)].proc = tpinfop[j];
		tris_pdata[I1dm(cur_node)].next = cur_node+1;
		tris_pdata[I1dm(cur_node)].shift[0] = tpinfos[j*3];
		tris_pdata[I1dm(cur_node)].shift[1] = tpinfos[j*3+1];
		tris_pdata[I1dm(cur_node)].shift[2] = tpinfos[j*3+2];

		if ((tpinfop[j] == cur_proc)
			&& (tpinfos[j*3] == 0)
			&& (tpinfos[j*3+1] == 0)
			&& (tpinfos[j*3+2] == 0) )
		    tris_pdata[I1dm(cur_node)].lindex = tris_index;
		else
		    tris_pdata[I1dm(cur_node)].lindex = tpinfol[j];
	    }
	    tris_phead[I1dm(tris_index)] = new_head;

	    // Deal with tail
	    if(hpEnsurePInfoCapacity(tris_pinfo))
		tris_pdata = tris_pinfo->pdata;
	    tris_pinfo->allocated_len++;
	    cur_node = tris_pinfo->allocated_len;
	    j = tpinfot[i+1]-1;

	    tris_pdata[I1dm(cur_node)].proc = tpinfop[j];
	    tris_pdata[I1dm(cur_node)].next = -1;
	    tris_pdata[I1dm(cur_node)].shift[0] = tpinfos[j*3];
	    tris_pdata[I1dm(cur_node)].shift[1] = tpinfos[j*3+1];
	    tris_pdata[I1dm(cur_node)].shift[2] = tpinfos[j*3+2];

	    if ((tpinfop[j] == cur_proc)
		    && (tpinfos[j*3] == 0)
		    && (tpinfos[j*3+1] == 0)
		    && (tpinfos[j*3+2] == 0) )
		tris_pinfo->pdata[I1dm(cur_node)].lindex = tris_index;
	    else
		tris_pinfo->pdata[I1dm(cur_node)].lindex = tpinfol[j];

	    tris_ptail[I1dm(tris_index)] = cur_node;
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
			const emxArray_int8_T *in_ps_shift,
			const real_T num_ring,
			emxArray_int32_T **out_ps,
			emxArray_int32_T **out_tris,
			emxArray_int8_T **out_ps_shift,
			emxArray_int8_T **out_tris_shift,
			emxArray_int32_T **out_tris_buffer)
{
    int i, j, is;

    int target_proc = mesh->nb_proc->data[I1dm(nb_proc_index)];

    int num_ps = mesh->ps->size[0];
    int num_tris = mesh->tris->size[0];
    int max_b_numps = 128;
    int max_b_numtris = 256;

    emxArray_int8_T *nb_shift = (mesh->nb_proc_shift)[nb_proc_index-1];
    int num_shifts = nb_shift->size[0];

    hpPInfoNode *tris_pdata = mesh->tris_pinfo->pdata;
    int *tris_phead = mesh->tris_pinfo->head;

    int32_T *tris_data = mesh->tris->data;
    int32_T *tris_size = mesh->tris->size;

    // num of points/tris in the nring for each point
    int num_ps_ring = 0;
    int num_tris_ring = 0;

    // Used for obtain_nring_surf, initialized as false
    emxArray_boolean_T *in_vtags = emxCreateND_boolean_T(1, &num_ps);
    emxArray_boolean_T *in_ftags = emxCreateND_boolean_T(1, &num_tris);

    /// Used for storing outputs of obtain_nring_surf 
    emxArray_int32_T *in_ngbvs = emxCreateND_int32_T(1, &max_b_numps);
    emxArray_int32_T *in_ngbfs = emxCreateND_int32_T(1, &max_b_numtris);


    int **ps_id_temp = (int **) calloc(num_shifts, sizeof(int *));
    int **tris_id_temp = (int **) calloc(num_shifts, sizeof(int *));

    int **tris_buffer_temp = (int **) calloc(num_shifts, sizeof(int *));

    int *num_ps_ring_shift = (int *) calloc(num_shifts, sizeof(int));
    int *num_tris_ring_shift = (int *) calloc(num_shifts, sizeof(int));

    int num_ps_ring_all = 0;
    int num_tris_ring_all = 0;

    for (is = 0; is < num_shifts; ++is)
    {
	int num_ol_pt_cur_shift = 0;
	int num_ol_pt = in_psid->size[0];

	//The overlapping points related to current shift
	boolean_T *shift_flag = (boolean_T *) calloc(num_ol_pt, sizeof(boolean_T));

	//The n-ring points/tris related to current shift
	boolean_T *ps_flag = (boolean_T *) calloc(num_ps, sizeof(boolean_T));
	boolean_T *tris_flag = (boolean_T *) calloc(num_tris, sizeof(boolean_T));

	int *ps_map = (int *) calloc(num_ps, sizeof(int));

	int8_T sx, sy, sz;

	sx = nb_shift->data[I2dm(is+1,1,nb_shift->size)];
	sy = nb_shift->data[I2dm(is+1,2,nb_shift->size)];
	sz = nb_shift->data[I2dm(is+1,3,nb_shift->size)];

	// First flag the overlapping points related to current shift
	for (i = 0; i < num_ol_pt; i++)
	{
	    if ( (sx == in_ps_shift->data[i*3]) &&
		 (sy == in_ps_shift->data[i*3+1]) &&
		 (sz == in_ps_shift->data[i*3+2]) )
	    {
		shift_flag[i] = 1;
		num_ol_pt_cur_shift++;
	    }

	}

	// Get n-ring neighbor for each point, go through the 
	// pinfo for all the triangles on the n-ring of each point
	// to decide whether it need to be sent

	for (i = 0; i < num_ol_pt; i++)
	{
	    if (shift_flag[i]) //If it is related to current shift
	    {
		int cur_ps = in_psid->data[i];
		obtain_nring_surf(cur_ps, num_ring, 0, mesh->tris, mesh->opphe,
			mesh->inhe, in_ngbvs, in_vtags, in_ftags, 
			in_ngbfs, &num_ps_ring, &num_tris_ring);
		for (j = 0; j < num_tris_ring; j++)
		{
		    // Only send the triangles which does not exists on the nb proc
		    int tris_buf_index = in_ngbfs->data[j];
		    tris_flag[I1dm(tris_buf_index)] = 1;

		    int next_node = tris_phead[I1dm(tris_buf_index)];
		    while(next_node != -1)
		    {
			// If also exists on other processor, do not send it

			if ( (tris_pdata[I1dm(next_node)].proc == target_proc) &&
			     (sx == tris_pdata[I1dm(next_node)].shift[0]) &&
			     (sy == tris_pdata[I1dm(next_node)].shift[1]) &&
			     (sz == tris_pdata[I1dm(next_node)].shift[2]) )
			{
			    tris_flag[I1dm(tris_buf_index)] = 0;
			    break;

			}
			else
			    next_node = tris_pdata[I1dm(next_node)].next;
		    }

		}
	    }

	}
	// Decide which points to be sent based on tris
	
	for (i = 1; i <= num_tris; i++)
	{
	    if (tris_flag[i-1])
	    {
		int bufpi_x = tris_data[I2dm(i,1,tris_size)];
		int bufpi_y = tris_data[I2dm(i,2,tris_size)];
		int bufpi_z = tris_data[I2dm(i,3,tris_size)];
		ps_flag[I1dm(bufpi_x)] = 1;
		ps_flag[I1dm(bufpi_y)] = 1;
		ps_flag[I1dm(bufpi_z)] = 1;
	    }
	}

	// Get number of ps for send of current shift and the sum
	// Set ps_map for building tris
	for (i = 1; i <= num_ps; i++)
	{
	    if (ps_flag[I1dm(i)])
	    {
		(num_ps_ring_shift[is])++;
		num_ps_ring_all++;

		ps_map[I1dm(i)] = num_ps_ring_all;
	    }
	}

	// Get number of tris for send of current shift and the sum
	for (i = 1; i <= num_tris; i++)
	{
	    if (tris_flag[I1dm(i)])
	    {
		(num_tris_ring_shift[is])++;
		num_tris_ring_all++;
	    }
	}

	// Temporarily save result in ps_id_temp, tris_id_temp and
	// tris_buffer_temp as the storage for the final result could only be
	// allocated after looping over all shifts to get num_ps_ring_all and
	// num_tris_ring_all
	ps_id_temp[is] = (int *) calloc(num_ps_ring_shift[is], sizeof(int));
	tris_id_temp[is] = (int *) calloc(num_tris_ring_shift[is], sizeof(int));

	tris_buffer_temp[is] = (int *) calloc(3*num_tris_ring_shift[is], sizeof(int));

	int *cur_ps_id_temp = ps_id_temp[is];
	int *cur_tris_id_temp = tris_id_temp[is];
	int *cur_tris_buffer_temp = tris_buffer_temp[is];

	int ps_iter = 0;
	int tris_iter = 0;

	for (i = 1; i <= num_ps; i++)
	{
	    if (ps_flag[I1dm(i)])
		cur_ps_id_temp[ps_iter++] = i;
	}

	for (i = 1; i <= num_tris; i++)
	{
	    if (tris_flag[I1dm(i)])
	    {
		cur_tris_id_temp[tris_iter] = i;

		cur_tris_buffer_temp[tris_iter*3] = ps_map[I1dm(tris_data[I2dm(i,1,tris_size)])];
		cur_tris_buffer_temp[tris_iter*3+1] = ps_map[I1dm(tris_data[I2dm(i,2,tris_size)])];
		cur_tris_buffer_temp[tris_iter*3+2] = ps_map[I1dm(tris_data[I2dm(i,3,tris_size)])];

		tris_iter++;
	    }
	}
	free(shift_flag);
	free(ps_flag);
	free(tris_flag);
	free(ps_map);
    }

    // Combine ps_ring and tris_ring for each shift to get final array

    int num_ps_ring_shift_all = 3*num_ps_ring_all;
    int num_tris_ring_shift_all = 3*num_tris_ring_all;

    (*out_ps) = emxCreateND_int32_T(1, &num_ps_ring_all);
    (*out_tris) = emxCreateND_int32_T(1, &num_tris_ring_all);

    (*out_tris_buffer) = emxCreate_int32_T(num_tris_ring_all, 3);

    (*out_ps_shift) = emxCreateND_int8_T(1, &num_ps_ring_shift_all);
    (*out_tris_shift) = emxCreateND_int8_T(1, &num_tris_ring_shift_all);

    int ps_iter = 0;
    int tris_iter = 0;

    emxArray_int32_T *result_ps = *out_ps;
    emxArray_int32_T *result_tris = *out_tris;

    emxArray_int32_T *result_tris_buffer = *out_tris_buffer;

    emxArray_int8_T *result_ps_shift = *out_ps_shift;
    emxArray_int8_T *result_tris_shift = *out_tris_shift;

    for (is = 0; is < num_shifts; is++)
    {
	int8_T sx, sy, sz;

	sx = nb_shift->data[I2dm(is+1,1,nb_shift->size)];
	sy = nb_shift->data[I2dm(is+1,2,nb_shift->size)];
	sz = nb_shift->data[I2dm(is+1,3,nb_shift->size)];

	int num_ps_ring_cur_shift = num_ps_ring_shift[is];
	int num_tris_ring_cur_shift = num_tris_ring_shift[is];

	int *ps_ring_cur_shift = ps_id_temp[is];
	int *tris_ring_cur_shift = tris_id_temp[is];
	int *tris_buffer_ring_cur_shift = tris_buffer_temp[is];

	for (i = 0; i < num_ps_ring_cur_shift; i++)
	{
	    result_ps->data[ps_iter] = ps_ring_cur_shift[i];
	    result_ps_shift->data[ps_iter*3] = sx;
	    result_ps_shift->data[ps_iter*3+1] = sy;
	    result_ps_shift->data[ps_iter*3+2] = sz;
	    ps_iter++;
	}

	for (i = 0; i < num_tris_ring_cur_shift; i++)
	{
	    result_tris->data[tris_iter] = tris_ring_cur_shift[i];
	    result_tris_shift->data[tris_iter*3] = sx;
	    result_tris_shift->data[tris_iter*3+1] = sy;
	    result_tris_shift->data[tris_iter*3+2] = sz;

	    tris_iter++;

	    result_tris_buffer->data[I2dm(tris_iter, 1, result_tris_buffer->size)] = tris_buffer_ring_cur_shift[i*3];
	    result_tris_buffer->data[I2dm(tris_iter, 2, result_tris_buffer->size)] = tris_buffer_ring_cur_shift[i*3+1];
	    result_tris_buffer->data[I2dm(tris_iter, 3, result_tris_buffer->size)] = tris_buffer_ring_cur_shift[i*3+2];
	}
    }

    for (is = 0; is < num_shifts; is++)
    {
	free(ps_id_temp[is]);
	free(tris_id_temp[is]);
	free(tris_buffer_temp[is]);
    }

    free(ps_id_temp);
    free(tris_id_temp);
    free(tris_buffer_temp);
    free(num_ps_ring_shift);
    free(num_tris_ring_shift);


    emxFree_int32_T(&in_ngbvs);
    emxFree_int32_T(&in_ngbfs);

    emxFree_boolean_T(&in_vtags);
    emxFree_boolean_T(&in_ftags);
}


void hpDebugOutput(const hiPropMesh *mesh, const emxArray_int32_T *debug_ps,
		   const emxArray_int32_T *debug_tris, char *debug_file_name)
{
    int j;
    // Build ps_mapping for vtk output of debug_ps
    //  for point i in mesh->ps, the new index in the vtk output is
    //  ps_mapping[I1dm(i)]. If point i does not have a mapped value,
    //  ps_mapping[I1dm(i)] = -1.

    int *ps_mapping = (int *) calloc(mesh->ps->size[0], sizeof(int));

    for (j = 1; j <= mesh->ps->size[0]; j++)
	ps_mapping[j-1] = -1;

    for (j = 1; j <= debug_ps->size[0]; j++)
    {
	int cur_ps_id = debug_ps->data[I1dm(j)];
	ps_mapping[cur_ps_id-1] = j-1;
    }

    // Write the data to vtk file
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
	    ps_type = 2; // buffer ps
	else
	{
	    if(ps_pdata[ps_head[i]-1].lindex != i+1)
		ps_type = 2; // buffer ps
	    else
	    {
		if(ps_head[i] != ps_tail[i])
		    ps_type = 1; // overlay ps
		else
		    ps_type = 0; // interior ps
	    }
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
	    tri_type = 2; // buffer tri
	else
	{
	    if (tri_pdata[tri_head[i]-1].lindex != i+1)
		tri_type = 2; // buffer tri
	    else
	    {
		if(tri_head[i] != tri_tail[i])
		    tri_type = 1; // overlay tri
		else
		    tri_type = 0; // interior tri
	    }
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

    real_T tmp = (double) in_degree + 1.0;
    real_T in_ring = tmp/2.0;

    int num_ps = mesh->ps->size[0];
    int num_ps_clean = mesh->nps_clean;

    mesh->nor = emxCreate_real_T(num_ps, 3);
    mesh->curv = emxCreate_real_T(num_ps, 2);

    emxArray_real_T *in_prdirs = emxCreate_real_T(num_ps, 3);

    hpComputeEstimatedNormal(mesh);
    hpUpdateGhostPointData_real_T(mesh, mesh->est_nor, 0);

    compute_diffops_surf_cleanmesh(num_ps_clean, mesh->ps, mesh->tris, mesh->est_nor, in_degree, in_ring, false, mesh->nor, mesh->curv, in_prdirs);

    hpUpdateGhostPointData_real_T(mesh, mesh->nor, 0);
    hpUpdateGhostPointData_real_T(mesh, mesh->curv, 0);

    emxFree_real_T(&(in_prdirs));
    emxFree_real_T(&(mesh->est_nor));
}

void hpBuildPartitionBoundary(hiPropMesh *mesh)
{
    int i;
    int *ps_phead = mesh->ps_pinfo->head;
    int *ps_ptail = mesh->ps_pinfo->tail;

    if( mesh->part_bdry != ((emxArray_int32_T *) NULL) )
	emxFree_int32_T(&(mesh->part_bdry));

    int num_ps = mesh->nps_clean;

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
    int num_tris = mesh->tris->size[0];
    int num_ps_clean = mesh->nps_clean;

    mesh->est_nor = emxCreate_real_T(num_ps, 3);

    emxArray_real_T *tmp_flabel = emxCreateND_real_T(1, &num_tris);
    
    average_vertex_normal_tri_cleanmesh(num_ps_clean, mesh->ps, mesh->tris, tmp_flabel, mesh->est_nor);

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

    // First build the buffer nor/curv and corresponding lindex for send
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

    // Send the information
    MPI_Request *send_req_list = (MPI_Request *) malloc( 5*num_nbp*sizeof(MPI_Request) );

    MPI_Status *send_status_list = (MPI_Status *) malloc( 5*num_nbp*sizeof(MPI_Status) );

    MPI_Request *recv_req_list = (MPI_Request *) malloc ( num_nbp*sizeof(MPI_Request) );

    // Stores the received array size

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

    // First build the buffer est_nor and corresponding lindex for send
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

    // Send the information
    MPI_Request *send_req_list = (MPI_Request *) malloc( 3*num_nbp*sizeof(MPI_Request) );

    MPI_Status *send_status_list = (MPI_Status *) malloc( 3*num_nbp*sizeof(MPI_Status) );

    MPI_Request *recv_req_list = (MPI_Request *) malloc ( num_nbp*sizeof(MPI_Request) );

    // Stores the received array size

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
	if (ps_pdata[I1dm(cur_head)].proc == rank && ps_pdata[I1dm(cur_head)].lindex == i) // current proc is master
	{
	    if (cur_head != cur_tail) // If exists on other processor (or same processor but different position)
		mesh->ps_type->data[I1dm(i)] = 1;
	}
	else // If current proc is not master
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


    //hpCommPsTrisWithPInfo(mesh, ps_ring_proc, tris_ring_proc, buffer_ps, buffer_tris);

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


void hpUpdateGhostPointData_int32_T(hiPropMesh *mesh, emxArray_int32_T *array)
{
    int num_proc, rank, num_nbp;
    int i, j, k;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_proc);
    num_nbp = mesh->nb_proc->size[0];

    int32_T **send_buf = (int32_T **) calloc(num_nbp, sizeof(int32_T *));
    int32_T **recv_buf = (int32_T **) calloc(num_nbp, sizeof(int32_T *));

    int *send_size = (int *) calloc(num_nbp, sizeof(int));
    int *recv_size = (int *) calloc(num_nbp, sizeof(int));

    // initialize send/recv buffer
    for (i = 0; i < num_nbp; i++)
	send_buf[i] = recv_buf[i] = (int *) NULL;

    // fill in the send buffer and allocate recv buffer
    for (i = 1; i <= num_nbp; i++)
    {
	int nbp_id = mesh->nb_proc->data[I1dm(i)];
	emxArray_int32_T *cur_psi = mesh->ps_send_index[nbp_id];
	emxArray_int32_T *cur_pri = mesh->ps_recv_index[nbp_id];
	if (cur_psi != (emxArray_int32_T *) NULL)
	{
	    int num_overlay_ps = cur_psi->size[0];

	    int size_col;

	    if (array->numDimensions == 1)
		size_col = 1;
	    else if (array->numDimensions > 1)
		size_col = array->size[1];
	    else
	    {
		printf("\n Wrong dimension, less than 1 !\n");
		exit(0);
	    }

	    send_size[I1dm(i)] = num_overlay_ps*size_col;
	    send_buf[I1dm(i)] = (int *) calloc(send_size[I1dm(i)], sizeof(int));

	}
	if (cur_pri != (emxArray_int32_T *) NULL)
	{
	    int num_ghost_ps = cur_pri->size[0];

	    int size_col;

	    if (array->numDimensions == 1)
		size_col = 1;
	    else if (array->numDimensions > 1)
		size_col = array->size[1];
	    else
	    {
		printf("\n Wrong dimension, less than 1 !\n");
		exit(0);
	    }	  

	    recv_size[I1dm(i)] = num_ghost_ps*size_col;
	    recv_buf[I1dm(i)] = (int *) calloc(recv_size[I1dm(i)], sizeof(int));
	}
    }

    // communicate the send buffer to recv buffer and update

    MPI_Request *send_req_list = (MPI_Request *) malloc( num_nbp*sizeof(MPI_Request) );

    MPI_Status *send_status_list = (MPI_Status *) malloc( num_nbp*sizeof(MPI_Status) );

    MPI_Request *recv_req_list = (MPI_Request *) malloc ( num_nbp*sizeof(MPI_Request) );

    for (i = 0; i < num_nbp; i++)
    {
	send_req_list[i] = MPI_REQUEST_NULL;
	recv_req_list[i] = MPI_REQUEST_NULL;
    }

    for (i = 1; i <= num_nbp; i++)
    {
	if (send_size[I1dm(i)] != 0)
	{
	    int *cur_send_buf = send_buf[I1dm(i)];
	    int cur_pos = 0;
	    int nbp_id = mesh->nb_proc->data[I1dm(i)];
	    
	    emxArray_int32_T *cur_psi = mesh->ps_send_index[nbp_id];
    	    
	    if (array->numDimensions > 1)
	    {
		for (k = 1; k <= array->size[1]; k++)
		{
		    for (j = 1; j <= cur_psi->size[0]; j++)
		    {
			int overlay_ps_id = cur_psi->data[I1dm(j)];
			cur_send_buf[cur_pos] = array->data[I2dm(overlay_ps_id, k, array->size)];
			cur_pos++;
		    }
		}
	    }
	    else if (array->numDimensions == 1)
	    {
		for (j = 1; j <= cur_psi->size[0]; j++)
		{
		    int overlay_ps_id = cur_psi->data[I1dm(j)];
		    cur_send_buf[cur_pos] = array->data[I1dm(overlay_ps_id)];
		    cur_pos++;
		}
	    }
	    else
	    {
		printf("\n Wrong dimension, less than 1 !\n");
		exit(0);

	    }

	    
	    MPI_Isend(send_buf[I1dm(i)], send_size[I1dm(i)], MPI_INT, 
		    mesh->nb_proc->data[I1dm(i)], 1, MPI_COMM_WORLD, &(send_req_list[i-1]));
	}
    }

    for (i = 1; i <= num_nbp; i++)
    {
	if (recv_size[I1dm(i)] != 0)
	    MPI_Irecv(recv_buf[I1dm(i)], recv_size[I1dm(i)], MPI_INT, 
		    mesh->nb_proc->data[I1dm(i)], 1, MPI_COMM_WORLD, &(recv_req_list[i-1]));
    }

    for (i = 1; i <= num_nbp; i++)
    {
	int recv_index;
	int recv_proc;
	MPI_Status recv_status;

	MPI_Waitany(num_nbp, recv_req_list, &recv_index, &recv_status);

	if (recv_index == MPI_UNDEFINED) // means all the recv_requests are NULL
	    break;

	recv_proc = recv_status.MPI_SOURCE;
	emxArray_int32_T *cur_pri = mesh->ps_recv_index[recv_proc];

	int *cur_recv_buf = recv_buf[recv_index];
	int cur_recv_pos = 0;

	if (array->numDimensions > 1)
	{
	    for (k = 1; k <= array->size[1]; k++)
	    {
		for (j = 1; j <= cur_pri->size[0]; j++)
		{
		    int ghost_ps_id = cur_pri->data[I1dm(j)];
		    array->data[I2dm(ghost_ps_id, k, array->size)] = cur_recv_buf[cur_recv_pos];
		    cur_recv_pos++;
		}
	    }
	}
	else if (array->numDimensions == 1)
	{
	    for (j = 1; j <= cur_pri->size[0]; j++)
	    {
		int ghost_ps_id = cur_pri->data[I1dm(j)];
		array->data[I1dm(ghost_ps_id)] = cur_recv_buf[cur_recv_pos];
		cur_recv_pos++;
	    }
	}
	else
	{
	    printf("\n Wrong dimension, less than 1!\n");
	    exit(0);
	}

    }

    free(recv_req_list);

    MPI_Waitall(num_nbp, send_req_list, send_status_list);

    free(send_req_list);
    free(send_status_list);


    for (i = 0; i < num_nbp; i++)
    {
	free(send_buf[i]);
	free(recv_buf[i]);
    }
    free(send_buf);
    free(recv_buf);

    free(send_size);
    free(recv_size);
    
}

void hpUpdateGhostPointData_real_T(hiPropMesh *mesh, emxArray_real_T *array, boolean_T add_shift)
{
    int num_proc, rank, num_nbp;
    int i, j, k;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_proc);
    num_nbp = mesh->nb_proc->size[0];
    double *domain_len = mesh->domain_len;

    real_T **send_buf = (real_T **) calloc(num_nbp, sizeof(real_T *));
    real_T **recv_buf = (real_T **) calloc(num_nbp, sizeof(real_T *));

    int *send_size = (int *) calloc(num_nbp, sizeof(int));
    int *recv_size = (int *) calloc(num_nbp, sizeof(int));

    // initialize send/recv buffer
    for (i = 0; i < num_nbp; i++)
	send_buf[i] = recv_buf[i] = (real_T *) NULL;

    // fill in the send buffer and allocate recv buffer
    for (i = 1; i <= num_nbp; i++)
    {
	int nbp_id = mesh->nb_proc->data[I1dm(i)];
	emxArray_int32_T *cur_psi = mesh->ps_send_index[nbp_id];
	emxArray_int32_T *cur_pri = mesh->ps_recv_index[nbp_id];
	if (cur_psi != (emxArray_int32_T *) NULL)
	{
	    int num_overlay_ps = cur_psi->size[0];

	    int size_col;

	    if (array->numDimensions == 1)
		size_col = 1;
	    else if (array->numDimensions > 1)
		size_col = array->size[1];
	    else
	    {
		printf("\n Wrong dimension, less than 1 !\n");
		exit(0);
	    }

	    send_size[I1dm(i)] = num_overlay_ps*size_col;
	    send_buf[I1dm(i)] = (real_T *) calloc(send_size[I1dm(i)], sizeof(real_T));

	}
	if (cur_pri != (emxArray_int32_T *) NULL)
	{
	    int num_ghost_ps = cur_pri->size[0];

	    int size_col;

	    if (array->numDimensions == 1)
		size_col = 1;
	    else if (array->numDimensions > 1)
		size_col = array->size[1];
	    else
	    {
		printf("\n Wrong dimension, less than 1 !\n");
		exit(0);
	    }

	    recv_size[I1dm(i)] = num_ghost_ps*size_col;
	    recv_buf[I1dm(i)] = (real_T *) calloc(recv_size[I1dm(i)], sizeof(real_T));
	}
    }

    // communicate the send buffer to recv buffer and update

    MPI_Request *send_req_list = (MPI_Request *) malloc( num_nbp*sizeof(MPI_Request) );

    MPI_Status *send_status_list = (MPI_Status *) malloc( num_nbp*sizeof(MPI_Status) );

    MPI_Request *recv_req_list = (MPI_Request *) malloc ( num_nbp*sizeof(MPI_Request) );

    for (i = 0; i < num_nbp; i++)
    {
	send_req_list[i] = MPI_REQUEST_NULL;
	recv_req_list[i] = MPI_REQUEST_NULL;
    }

    for (i = 1; i <= num_nbp; i++)
    {
	if (send_size[I1dm(i)] != 0)
	{
	    real_T *cur_send_buf = send_buf[I1dm(i)];
	    int cur_pos = 0;
	    int nbp_id = mesh->nb_proc->data[I1dm(i)];
	    
	    emxArray_int32_T *cur_psi = mesh->ps_send_index[nbp_id];
	    emxArray_int8_T *cur_pss = mesh->ps_send_shift[nbp_id];

	    if (array->numDimensions > 1)
	    {
		for (k = 1; k <= array->size[1]; k++)
		{
		    for (j = 1; j <= cur_psi->size[0]; j++)
		    {
			int overlay_ps_id = cur_psi->data[I1dm(j)];
			if (add_shift)
			{
			    double factor = ((double)(-cur_pss->data[3*(j-1) + (k-1)])*domain_len[k-1]);
			    cur_send_buf[cur_pos] = array->data[I2dm(overlay_ps_id, k, array->size)] + factor;
			}
			else
			    cur_send_buf[cur_pos] = array->data[I2dm(overlay_ps_id, k, array->size)];
			cur_pos++;
		    }
		}
	    }
	    else if (array->numDimensions == 1)
	    {
		for (j = 1; j <= cur_psi->size[0]; j++)
		{
		    int overlay_ps_id = cur_psi->data[I1dm(j)];
		    cur_send_buf[cur_pos] = array->data[I1dm(overlay_ps_id)];
		    cur_pos++;
		}
	    }
	    else
	    {
		printf("\n Wrong dimension, less than 1 !\n");
		exit(0);

	    }
	    
	    MPI_Isend(send_buf[I1dm(i)], send_size[I1dm(i)], MPI_DOUBLE, 
		    mesh->nb_proc->data[I1dm(i)], 1, MPI_COMM_WORLD, &(send_req_list[i-1]));
	}
    }

    for (i = 1; i <= num_nbp; i++)
    {
	if (recv_size[I1dm(i)] != 0)
	    MPI_Irecv(recv_buf[I1dm(i)], recv_size[I1dm(i)], MPI_DOUBLE, 
		    mesh->nb_proc->data[I1dm(i)], 1, MPI_COMM_WORLD, &(recv_req_list[i-1]));
    }

    for (i = 1; i <= num_nbp; i++)
    {
	int recv_index;
	int recv_proc;
	MPI_Status recv_status;

	MPI_Waitany(num_nbp, recv_req_list, &recv_index, &recv_status);

	if (recv_index == MPI_UNDEFINED) // means all the recv_requests are NULL
	    break;

	recv_proc = recv_status.MPI_SOURCE;
	emxArray_int32_T *cur_pri = mesh->ps_recv_index[recv_proc];

	real_T *cur_recv_buf = recv_buf[recv_index];
	int cur_recv_pos = 0;

	if (array->numDimensions > 1)
	{
	    for (k = 1; k <= array->size[1]; k++)
	    {
		for (j = 1; j <= cur_pri->size[0]; j++)
		{
		    int ghost_ps_id = cur_pri->data[I1dm(j)];
		    array->data[I2dm(ghost_ps_id, k, array->size)] = cur_recv_buf[cur_recv_pos];
		    cur_recv_pos++;
		}
	    }
	}
	else if (array->numDimensions == 1)
	{
	    for (j = 1; j <= cur_pri->size[0]; j++)
	    {
		int ghost_ps_id = cur_pri->data[I1dm(j)];
		array->data[I1dm(ghost_ps_id)] = cur_recv_buf[cur_recv_pos];
		cur_recv_pos++;
	    }
	}
	else
	{
	    printf("\n Wrong dimension, less than 1!\n");
	    exit(0);
	}

    }

    free(recv_req_list);

    MPI_Waitall(num_nbp, send_req_list, send_status_list);

    free(send_req_list);
    free(send_status_list);


    for (i = 0; i < num_nbp; i++)
    {
	free(send_buf[i]);
	free(recv_buf[i]);
    }
    free(send_buf);
    free(recv_buf);

    free(send_size);
    free(recv_size);
}

void hpUpdateGhostPointData_boolean_T(hiPropMesh *mesh, emxArray_boolean_T *array)
{
    int num_proc, rank, num_nbp;
    int i, j, k;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_proc);
    num_nbp = mesh->nb_proc->size[0];

    boolean_T **send_buf = (boolean_T **) calloc(num_nbp, sizeof(boolean_T *));
    boolean_T **recv_buf = (boolean_T **) calloc(num_nbp, sizeof(boolean_T *));

    int *send_size = (int *) calloc(num_nbp, sizeof(int));
    int *recv_size = (int *) calloc(num_nbp, sizeof(int));

    // initialize send/recv buffer
    for (i = 0; i < num_nbp; i++)
	send_buf[i] = recv_buf[i] = (boolean_T *) NULL;

    // fill in the send buffer and allocate recv buffer
    for (i = 1; i <= num_nbp; i++)
    {
	int nbp_id = mesh->nb_proc->data[I1dm(i)];
	emxArray_int32_T *cur_psi = mesh->ps_send_index[nbp_id];
	emxArray_int32_T *cur_pri = mesh->ps_recv_index[nbp_id];
	if (cur_psi != (emxArray_int32_T *) NULL)
	{
	    int num_overlay_ps = cur_psi->size[0];
	    int size_col;

	    if (array->numDimensions == 1)
		size_col = 1;
	    else if (array->numDimensions > 1)
		size_col = array->size[1];
	    else
	    {
		printf("\n Wrong dimension, less than 1 !\n");
		exit(0);
	    }

	    send_size[I1dm(i)] = num_overlay_ps*size_col;
	    send_buf[I1dm(i)] = (boolean_T *) calloc(send_size[I1dm(i)], sizeof(boolean_T));

	}
	if (cur_pri != (emxArray_int32_T *) NULL)
	{
	    int num_ghost_ps = cur_pri->size[0];
	    int size_col;

	    if (array->numDimensions == 1)
		size_col = 1;
	    else if (array->numDimensions > 1)
		size_col = array->size[1];
	    else
	    {
		printf("\n Wrong dimension, less than 1 !\n");
		exit(0);
	    }

	    recv_size[I1dm(i)] = num_ghost_ps*size_col;
	    recv_buf[I1dm(i)] = (boolean_T *) calloc(recv_size[I1dm(i)], sizeof(boolean_T));
	}
    }

    // communicate the send buffer to recv buffer and update

    MPI_Request *send_req_list = (MPI_Request *) malloc( num_nbp*sizeof(MPI_Request) );

    MPI_Status *send_status_list = (MPI_Status *) malloc( num_nbp*sizeof(MPI_Status) );

    MPI_Request *recv_req_list = (MPI_Request *) malloc ( num_nbp*sizeof(MPI_Request) );

    for (i = 0; i < num_nbp; i++)
    {
	send_req_list[i] = MPI_REQUEST_NULL;
	recv_req_list[i] = MPI_REQUEST_NULL;
    }

    for (i = 1; i <= num_nbp; i++)
    {
	if (send_size[I1dm(i)] != 0)
	{
	    boolean_T *cur_send_buf = send_buf[I1dm(i)];
	    int cur_pos = 0;
	    int nbp_id = mesh->nb_proc->data[I1dm(i)];
	    
	    emxArray_int32_T *cur_psi = mesh->ps_send_index[nbp_id];
    	    
	    if (array->numDimensions > 1)
	    {
		for (k = 1; k <= array->size[1]; k++)
		{
		    for (j = 1; j <= cur_psi->size[0]; j++)
		    {
			int overlay_ps_id = cur_psi->data[I1dm(j)];
			cur_send_buf[cur_pos] = array->data[I2dm(overlay_ps_id, k, array->size)];
			cur_pos++;
		    }
		}
	    }
	    else if (array->numDimensions == 1)
	    {
		for (j = 1; j <= cur_psi->size[0]; j++)
		{
		    int overlay_ps_id = cur_psi->data[I1dm(j)];
		    cur_send_buf[cur_pos] = array->data[I1dm(overlay_ps_id)];
		    cur_pos++;
		}
	    }
	    else
	    {
		printf("\n Wrong dimension, less than 1 !\n");
		exit(0);

	    }
	    
	    MPI_Isend(send_buf[I1dm(i)], send_size[I1dm(i)], MPI_UNSIGNED_CHAR, 
		    mesh->nb_proc->data[I1dm(i)], 1, MPI_COMM_WORLD, &(send_req_list[i-1]));
	}
    }

    for (i = 1; i <= num_nbp; i++)
    {
	if (recv_size[I1dm(i)] != 0)
	    MPI_Irecv(recv_buf[I1dm(i)], recv_size[I1dm(i)], MPI_UNSIGNED_CHAR, 
		    mesh->nb_proc->data[I1dm(i)], 1, MPI_COMM_WORLD, &(recv_req_list[i-1]));
    }

    for (i = 1; i <= num_nbp; i++)
    {
	int recv_index;
	int recv_proc;
	MPI_Status recv_status;

	MPI_Waitany(num_nbp, recv_req_list, &recv_index, &recv_status);

	if (recv_index == MPI_UNDEFINED) // means all the recv_requests are NULL
	    break;

	recv_proc = recv_status.MPI_SOURCE;
	emxArray_int32_T *cur_pri = mesh->ps_recv_index[recv_proc];

	boolean_T *cur_recv_buf = recv_buf[recv_index];
	int cur_recv_pos = 0;

	if (array->numDimensions > 1)
	{
	    for (k = 1; k <= array->size[1]; k++)
	    {
		for (j = 1; j <= cur_pri->size[0]; j++)
		{
		    int ghost_ps_id = cur_pri->data[I1dm(j)];
		    array->data[I2dm(ghost_ps_id, k, array->size)] = cur_recv_buf[cur_recv_pos];
		    cur_recv_pos++;
		}
	    }
	}
	else if (array->numDimensions == 1)
	{
	    for (j = 1; j <= cur_pri->size[0]; j++)
	    {
		int ghost_ps_id = cur_pri->data[I1dm(j)];
		array->data[I1dm(ghost_ps_id)] = cur_recv_buf[cur_recv_pos];
		cur_recv_pos++;
	    }
	}
	else
	{
	    printf("\n Wrong dimension, less than 1!\n");
	    exit(0);
	}

    }

    free(recv_req_list);

    MPI_Waitall(num_nbp, send_req_list, send_status_list);

    free(send_req_list);
    free(send_status_list);


    for (i = 0; i < num_nbp; i++)
    {
	free(send_buf[i]);
	free(recv_buf[i]);
    }
    free(send_buf);
    free(recv_buf);

    free(send_size);
    free(recv_size);

}

void hpAdaptiveBuildGhost(hiPropMesh *mesh, const int32_T in_degree)
{
    int num_ps_clean = mesh->nps_clean;
    int num_ps_pbdry = mesh->part_bdry->size[0];
    emxArray_real_T *ring_clean = emxCreateND_real_T(1, &num_ps_clean);
    emxArray_real_T *ring_pbdry = emxCreateND_real_T(1, &num_ps_pbdry);

    obtain_ringsz_cleanmesh(num_ps_clean, mesh->part_bdry, mesh->ps, mesh->tris,
	    in_degree, ring_clean);

    int i;

    for (i = 1; i <= num_ps_pbdry; i++)
    {
	int cur_id = mesh->part_bdry->data[I1dm(i)];
	ring_pbdry->data[I1dm(i)] = ring_clean->data[I1dm(cur_id)];
    }

    emxFree_real_T(&ring_clean);

    hpBuildPartBdryGhost(mesh, ring_pbdry);
}


void hpMeshSmoothing(hiPropMesh *mesh, int32_T in_degree, const char* method)
{
    int num_iter = 15;

    real_T in_angletol_min = 25;
    real_T in_perfolded = 80;
    real_T in_disp_alpha = 1;
    boolean_T in_check_trank = false;
    boolean_T in_vc_flag = true;

    emxArray_char_T *in_method;


    if (method[0] == 'C')
    {
	in_method = emxCreate_char_T(1, 3);

	in_method->data[0] = 'C';
	in_method->data[1] = 'M';
	in_method->data[2] = 'F';
    }
    else if(method[0] == 'W')
    {
	in_method = emxCreate_char_T(1, 4);

	in_method->data[0] = 'W';
	in_method->data[1] = 'A';
	in_method->data[2] = 'L';
	in_method->data[3] = 'F';
    }
    else
    {
	printf("Error Method! Terminating...\n");
	exit(0);
    }


    int32_T in_verbose = (int32_T) 2;

    printf("\n clean size = %d, %d, actual size = %d %d\n", mesh->nps_clean, mesh->ntris_clean, mesh->ps->size[0], mesh->tris->size[0]);

    smooth_mesh_hisurf_cleanmesh(mesh->nps_clean, mesh->ntris_clean,
	    mesh->ps, mesh->tris, in_degree, num_iter, in_angletol_min,
	    in_perfolded, in_disp_alpha, in_check_trank, in_vc_flag, in_method,
	    in_verbose, mesh);
}


void hpDebugParallelToSerialOutput(hiPropMesh *mesh, emxArray_real_T *array, const char *outname)
{
    int i;
    int rank;
    char rank_str[5];
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    numIntoString(rank,4,rank_str);

    int num_old_ps = mesh->nps_clean;
    printf("num of old ps = %d\n", num_old_ps);

    char ptid_filename[200];
    sprintf(ptid_filename, "data/parallel/sphere_nonuni_psid-p%s.data", rank_str);
    FILE *ptinfile = fopen(ptid_filename, "r");

    int *ptid = (int *) calloc(num_old_ps, sizeof(int));

    for (i = 0; i < num_old_ps; i++)
	fscanf(ptinfile, "%d", &(ptid[i]));
    fclose(ptinfile);

    int num_all_pt = 0;

    for (i = 1; i <= mesh->ps->size[0]; i++)
    {
	int head = mesh->ps_pinfo->head[I1dm(i)];
	if (mesh->ps_pinfo->pdata[I1dm(head)].proc == rank)
	    num_all_pt++;
    }

    int out_num_all_pt = 0;

    MPI_Allreduce(&num_all_pt, &out_num_all_pt, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    printf("\n Num of all pts = %d \n", out_num_all_pt);

    double *inps1 = (double *) calloc(out_num_all_pt, sizeof(double));
    double *inps2 = (double *) calloc(out_num_all_pt, sizeof(double));
    double *inps3 = (double *) calloc(out_num_all_pt, sizeof(double));

    double *outps1 = (double *) calloc(out_num_all_pt, sizeof(double));
    double *outps2 = (double *) calloc(out_num_all_pt, sizeof(double));
    double *outps3 = (double *) calloc(out_num_all_pt, sizeof(double));


    for (i = 1; i <= mesh->ps->size[0]; i++)
    {
	int head = mesh->ps_pinfo->head[I1dm(i)];
	if (mesh->ps_pinfo->pdata[I1dm(head)].proc == rank)
	{
	    if (array->numDimensions == 2)
	    {
		inps1[ptid[i-1]-1] = array->data[I2dm(i,1,mesh->ps->size)];
		inps2[ptid[i-1]-1] = array->data[I2dm(i,2,mesh->ps->size)];
		inps3[ptid[i-1]-1] = array->data[I2dm(i,3,mesh->ps->size)];
	    }
	    else if (array->numDimensions == 1)
	    {
		inps1[ptid[i-1]-1] = array->data[I1dm(i)];
	    }
	}
    }

    if (array->numDimensions == 2)
    {
	MPI_Allreduce(inps1, outps1, out_num_all_pt, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	MPI_Allreduce(inps2, outps2, out_num_all_pt, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	MPI_Allreduce(inps3, outps3, out_num_all_pt, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    }
    else if (array->numDimensions == 1)
    {
	MPI_Allreduce(inps1, outps1, out_num_all_pt, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    }


    if (rank == 0)
    {
	char ncfilename[200];
	sprintf(ncfilename, outname);

	FILE *diffout = fopen(ncfilename, "w");

	if (array->numDimensions == 2)
	{
	    for (i = 0; i < out_num_all_pt; i++)
		fprintf(diffout, "%22.16g %22.16g %22.16g \n",
			outps1[i], outps2[i], outps3[i]);
	}
	else if(array->numDimensions == 1)
	{
	    for (i = 0; i < out_num_all_pt; i++)
		fprintf(diffout, "%22.16g\n",
		    outps1[i]);
	}
    }

    free(ptid);
    free(inps1); free(inps2); free(inps3);
    free(outps1); free(outps2); free(outps3);
}



/*
int hpMetisPartMesh(hiPropMesh* mesh, const int nparts, 
	int** tri_part, int** pt_part)
{

    //to be consistent with Metis, idx_t denote integer numbers, 
    //real_t denote floating point numbers in Metis, tri and points 
    //arrays all start from index 0, which is different from HiProp,
    //so we need to convert to Metis convention, the output tri_part 
    //and pt_part are in Metis convention
    

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
	// [i-1] is the global index of the i-th tri on the ranked proc
	
	int** tri_index = (int**) calloc(num_proc,sizeof(int*));
	for(i = 0; i<num_proc; i++)
	{
	    tri_index[i] = (int*) calloc(num_tri[i],sizeof(int));

	}

	// fill tri_index by looping over all tris 
	int* p = (int*)malloc(num_proc*sizeof(int));// pointer to the end of the list
	for(i = 0; i< num_proc; i++)
	    p[i] = 0;
	int tri_rk;	// the proc rank of the current tri 
	for(i = 1; i<=total_num_tri; i++)
	{
	    tri_rk = tri_part[i-1];	// convert because Metis 
					// convention use index starts from 0
	    tri_index[tri_rk][p[tri_rk]] = i;
	    p[tri_rk]++;
	}


	// construct an index table to store the local index of every point 
	// (global to local)
	// if pt_local[i][j-1] = -1, point[j] is not on proc[i], 
	// if pt_local[i][j-1] = m >= 0, the local index of point[j] on proc[i] is m.
	// looks space and time consuming, however easy to convert 
	// between globle and local index of points
	 
	int** pt_local = (int**)calloc(num_proc,sizeof(int*));
	for(i = 0; i<num_proc; i++)
	{
	    pt_local[i] = (int*) calloc(total_num_pt , sizeof(int));
	    for(j = 0; j<total_num_pt; j++)
		pt_local[i][j] = -1;	//initialize to -1
	}

	// fill in pt_local table, calculate num_pt[] on each proc at the same time
	// in this situation, the point local index is sorted as global index 

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

	// fill in p_mesh[]->tris->data[] according to pt_local table 
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

	// pt_index is similar to tri_index
	// pt_index[rank][i-1] is the global index of the i-th point on the ranked proc
	// constructed using pt_local
	
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


	// finally, get in p_mesh[]->ps, :)
	for(i = 0; i< num_proc; i++)
	    (p_mesh[i]->ps) = emxCreate_real_T(num_pt[i], 3);
	// fill in p_mesh[]->ps->data with pt_index 
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


	// communication of basic mesh info
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

		// Create a wrapper for points and triangles global index arrays.
		// This wrapper is an output of the function 
		 
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

	// free pointers
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

    else	//for other proc, receive the mesh info
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
	mesh->ps_pinfo->tail[I1dm(j)] = -1;	// the list is empty 
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

		if(mesh->ps_pinfo->tail[I1dm(j)]!=-1)	// the list is nonempty for this point
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
		else	// the list is empty for this point
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
    nb_proc_size[0]--;		// to exclude itself
    mesh->nb_proc = emxCreateND_int32_T(1,nb_proc_size);

    k=0;
    for (j = 0; j<num_proc; j++)
	if((j!=rank)&&(nb_proc_bool[j]==1))
	    mesh->nb_proc->data[k++] = j;

    printf("After hpConstrPInfoFromGlobalLocalInfo\n");
}
*
*/

/*
	if (!(buf_ps_flag[i]))
	{
	    int ps_index = ps_map[i];
	    for(j = ppinfot[i]; j <= ppinfot[i+1]-1; j++)
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
			int new_tail = ps_pinfo->allocated_len; //new node
			ps_pinfo->pdata[I1dm(new_tail)].proc = ppinfop[j];
			ps_pinfo->pdata[I1dm(new_tail)].lindex = -1;
			ps_pinfo->pdata[I1dm(new_tail)].next = -1;

			ps_pinfo->pdata[I1dm(cur_tail)].next = new_tail;
			ps_pinfo->tail[I1dm(ps_index)] = new_tail;
		    }
		}
	    }
	}
	*/
/*
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
			tris_pinfo->allocated_len++; // new node
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
	*/
    //////////////////Send debug test
   /* 
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    char rank_str[5];
    numIntoString(rank,4,rank_str);

    int j;
    for (j = 1 ; j <= num_nb_proc; j++)
    {
	char test_filename[256];
	sprintf(test_filename, "send_debug_%d-p%s.vtk", j, rank_str);
	FILE* file = fopen(test_filename, "w");
	int i;
	emxArray_real_T* points = buffer_ps[I1dm(j)];
	emxArray_int32_T* triangles = buffer_tris[I1dm(j)];

	fprintf(file, "# vtk DataFile Version 3.0\n");
	fprintf(file, "Mesh output by hiProp\n");
	fprintf(file, "ASCII\n");
	fprintf(file, "DATASET UNSTRUCTURED_GRID\n");

	int num_points = points->size[0];
	int num_tris = triangles->size[0];

	fprintf(file, "POINTS %d double\n", num_points);
	for (i = 1; i <= num_points; i++)
	    fprintf(file, "%22.16lg %22.16lg %22.16lg\n",
		    points->data[I2dm(i,1,points->size)],
		    points->data[I2dm(i,2,points->size)],
		    points->data[I2dm(i,3,points->size)]);

	fprintf(file, "CELLS %d %d\n", num_tris, 4*num_tris);
	for (i = 1; i <= num_tris; i++)
	    fprintf(file, "3 %d %d %d\n",
		    triangles->data[I2dm(i,1,triangles->size)]-1,
		    triangles->data[I2dm(i,2,triangles->size)]-1,
		    triangles->data[I2dm(i,3,triangles->size)]-1);

	fprintf(file, "CELL_TYPES %d\n", num_tris);
	for (i = 0; i<num_tris; i++)
	    fprintf(file, "5\n");
	fclose(file);
    }
    */
    //////////////////
/*
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
*/
