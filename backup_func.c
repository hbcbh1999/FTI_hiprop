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

