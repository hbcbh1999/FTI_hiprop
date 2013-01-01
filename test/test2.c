/*
 *	This file is to test input and output utilities in VTK format
 *
 */

#include "stdafx.h"
#include "util.h"
#include "hiprop.h"

int main(int argc, char* argv[])
{
    int i, j;
    int num_proc, rank;
    int tag = 1;
    int root = 0;


    MPI_Init(&argc, &argv);

    MPI_Comm_size(MPI_COMM_WORLD, &num_proc);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    char runlog_filename[200];
    char rank_str[5];
    numIntoString(rank,4,rank_str);
    sprintf(runlog_filename, "run-log.%s",rank_str);
    FILE *runlog_stream;
    if((runlog_stream = freopen(runlog_filename, "w", stdout)) == NULL)
	exit(-1);

    printf("\n Welcome to the test of Hi-Prop Library from proc %d\n", rank);

    hiPropMesh *mesh;
    hpInitMesh(&mesh);

    char in_filename[200];
    sprintf(in_filename, "data/parallel/%s-p%s.vtk", argv[1], rank_str);
    if (!hpReadUnstrMeshVtk3d(in_filename, mesh))
    {
	printf("Reading fail!\n");
	return 0;
    }

    hpGetNbProcListAuto(mesh);
    printf("\n GetNbProcInfo passed, proc %d \n", rank);

    hpInitPInfo(mesh);
    printf("\n InitPInfo passed, proc %d \n", rank);

    hpBuildPInfoNoOverlappingTris(mesh);
    printf("\n BuildPInfo passed, proc %d \n", rank);

    hpBuildOppositeHalfEdge(mesh);
    printf("\n BuildOppHalfEdge passed, proc %d \n", rank);

    hpBuildIncidentHalfEdge(mesh);
    printf("\n BuildIncidentHalfEdge passed, proc %d \n", rank);

    hpBuildNRingGhost(mesh, 2);
    printf("\n BuildNRingGhost passed, proc %d \n", rank);

    hpBuildOppositeHalfEdge(mesh);
    printf("\n BuildOppHalfEdge passed, proc %d \n", rank);

    hpBuildIncidentHalfEdge(mesh);
    printf("\n BuildIncidentHalfEdge passed, proc %d \n", rank);

    hpBuildPUpdateInfo(mesh);
    printf("\n BuildPUpdateInfo passed, proc %d \n", rank);

    hpComputeDiffops(mesh, 3);

    char norcurv_filename[200];
    sprintf(norcurv_filename, "norcurv-p%s.out", rank_str);
    FILE *norcurv_outfile = fopen(norcurv_filename, "w");

    for (i = 1; i <= mesh->ps->size[0]; i++)
    {
	fprintf(norcurv_outfile, "%22.16g %22.16g %22.16g %22.16g %22.16g %22.16g %22.16g %22.16g\n", 
		mesh->ps->data[I2dm(i,1,mesh->ps->size)], mesh->ps->data[I2dm(i,2,mesh->ps->size)], mesh->ps->data[I2dm(i,3,mesh->ps->size)],
		mesh->nor->data[I2dm(i,1,mesh->nor->size)],
		mesh->nor->data[I2dm(i,2,mesh->nor->size)], 
		mesh->nor->data[I2dm(i,3,mesh->nor->size)],
		mesh->curv->data[I2dm(i,1,mesh->curv->size)],
		mesh->curv->data[I2dm(i,2,mesh->curv->size)]);
    }

    /*
    hpBuildOppositeHalfEdge(mesh);
    printf("\n BuildOppHalfEdge passed, proc %d \n", rank);

    hpBuildIncidentHalfEdge(mesh);
    printf("\n BuildIncidentHalfEdge passed, proc %d \n", rank);

    hpBuildNRingGhost(mesh, 2);
    printf("\n BuildNRingGhost passed, proc %d \n", rank);

    char debug_filename[200];
    sprintf(debug_filename, "debugout-p%s.vtk", rank_str);
    hpWriteUnstrMeshWithPInfo(debug_filename, mesh);

    hpBuildOppositeHalfEdge(mesh);
    printf("\n BuildOppHalfEdge passed, proc %d \n", rank);

    hpBuildIncidentHalfEdge(mesh);
    printf("\n BuildIncidentHalfEdge passed, proc %d \n", rank);

    hpBuildNRingGhost(mesh, 2);
    printf("\n BuildNRingGhost passed, proc %d \n", rank);

    char debug_filename2[200];
    sprintf(debug_filename2, "debugout2-p%s.vtk", rank_str);
    hpWriteUnstrMeshWithPInfo(debug_filename2, mesh);
    */


    hpDeleteMesh(&mesh);

    printf("Success processor %d\n", rank);

    MPI_Finalize();

    return 1;

/*                      This part about testing obtain_nring 
    if (rank == 0)
    {
	emxArray_int32_T *ngbvs, *ngbfs;
	emxArray_boolean_T *vtags, *ftags;
	int32_T num_ring_ps, num_ring_tris;

	hpObtainNRingTris(mesh, 12, 2.0, 0, 128, 256, &ngbvs, &ngbfs, &vtags, &ftags, &num_ring_ps, &num_ring_tris);

	printf("num of ps = %d, num of tris = %d\n", num_ring_ps, num_ring_tris);

	for (i = 1; i <= ngbvs->size[0]; i++)
	    printf("point %d\n", ngbvs->data[I1dm(i)]);

	for (i = 1; i <= ngbfs->size[0]; i++)
	{
	    int tri_index = ngbfs->data[I1dm(i)];
	    printf("tris %d: (%d, %d, %d)\n", tri_index, mesh->tris->data[I2dm(tri_index,1,mesh->tris->size)], 
		    mesh->tris->data[I2dm(tri_index,2,mesh->tris->size)], mesh->tris->data[I2dm(tri_index,3,mesh->tris->size)]);  
	}

	num_ring_ps++;

	emxArray_int32_T *ps_vis = emxCreateND_int32_T(1, &num_ring_ps);
	emxArray_int32_T *tris_vis = emxCreateND_int32_T(1, &num_ring_tris);

	for (i = 1; i < num_ring_ps; i++)
	    ps_vis->data[I1dm(i)] = ngbvs->data[I1dm(i)];
	ps_vis->data[I1dm(num_ring_ps)] = 12;

	for (i = 1; i <= num_ring_tris; i++)
	    tris_vis->data[I1dm(i)] = ngbfs->data[I1dm(i)];



	emxFree_int32_T(&(ngbvs));
	emxFree_int32_T(&(ngbfs));
	emxFree_boolean_T(&(vtags));
	emxFree_boolean_T(&(ftags));

	char dfilename[200];
	sprintf(dfilename, "debout.vtk");
	hpDebugOutput(mesh, ps_vis, tris_vis, dfilename);

	emxFree_int32_T(&(ps_vis));
	emxFree_int32_T(&(tris_vis));

    }
*/
    
    /*                      This part about testing computing normals and
     *                      curvatures 
    hpComputeDiffops(mesh, 5);


    char ptid_filename[200];
    sprintf(ptid_filename, "ptid-p%s.data", rank_str);
    FILE *infile = fopen(ptid_filename, "r");

    int *ptid = (int *) calloc(num_old_ps, sizeof(int));

    for (i = 0; i < num_old_ps; i++)
	fscanf(infile, "%d", &(ptid[i]));


    fclose(infile);

    int num_all_pt = 0;

    char nor_curv_filename[200];
    sprintf(nor_curv_filename, "diffquant-p%s.out", rank_str);
    FILE *diff_outfile = fopen(nor_curv_filename, "w");

    for (i = 1; i <= mesh->ps->size[0]; i++)
    {
	int head = mesh->ps_pinfo->head[I1dm(i)];
	if (mesh->ps_pinfo->pdata[I1dm(head)].proc == rank)
	{
	    num_all_pt++;
	    fprintf(diff_outfile, "%22.16g %22.16g %22.16g %22.16g %22.16g %22.16g %22.16g %22.16g\n",
		    mesh->ps->data[I2dm(i,1,mesh->ps->size)], mesh->ps->data[I2dm(i,2,mesh->ps->size)], mesh->ps->data[I2dm(i,3,mesh->ps->size)],
		    mesh->nor->data[I2dm(i,1,mesh->nor->size)], mesh->nor->data[I2dm(i,2,mesh->nor->size)], mesh->nor->data[I2dm(i,3,mesh->nor->size)],
		    mesh->curv->data[I2dm(i,1,mesh->curv->size)], mesh->curv->data[I2dm(i,2,mesh->curv->size)]);

	}
    }

    fclose(diff_outfile);

    int out_num_all_pt = 0;

    MPI_Allreduce(&num_all_pt, &out_num_all_pt, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    printf("\n Num of all pts = %d \n", out_num_all_pt);

    double *inps1 = (double *) calloc(out_num_all_pt, sizeof(double));
    double *inps2 = (double *) calloc(out_num_all_pt, sizeof(double));
    double *inps3 = (double *) calloc(out_num_all_pt, sizeof(double));

    double *innor1 = (double *) calloc(out_num_all_pt, sizeof(double));
    double *innor2 = (double *) calloc(out_num_all_pt, sizeof(double));
    double *innor3 = (double *) calloc(out_num_all_pt, sizeof(double));

    double *incurv1 = (double *) calloc(out_num_all_pt, sizeof(double));
    double *incurv2 = (double *) calloc(out_num_all_pt, sizeof(double));

    double *outps1 = (double *) calloc(out_num_all_pt, sizeof(double));
    double *outps2 = (double *) calloc(out_num_all_pt, sizeof(double));
    double *outps3 = (double *) calloc(out_num_all_pt, sizeof(double));

    double *outnor1 = (double *) calloc(out_num_all_pt, sizeof(double));
    double *outnor2 = (double *) calloc(out_num_all_pt, sizeof(double));
    double *outnor3 = (double *) calloc(out_num_all_pt, sizeof(double));

    double *outcurv1 = (double *) calloc(out_num_all_pt, sizeof(double));
    double *outcurv2 = (double *) calloc(out_num_all_pt, sizeof(double));

    for (i = 1; i <= mesh->ps->size[0]; i++)
    {
	int head = mesh->ps_pinfo->head[I1dm(i)];
	if (mesh->ps_pinfo->pdata[I1dm(head)].proc == rank)
	{
	    inps1[ptid[i-1]-1] = mesh->ps->data[I2dm(i,1,mesh->ps->size)];
	    inps2[ptid[i-1]-1] = mesh->ps->data[I2dm(i,2,mesh->ps->size)];
	    inps3[ptid[i-1]-1] = mesh->ps->data[I2dm(i,3,mesh->ps->size)];

	    innor1[ptid[i-1]-1] = mesh->nor->data[I2dm(i,1,mesh->nor->size)];
	    innor2[ptid[i-1]-1] = mesh->nor->data[I2dm(i,2,mesh->nor->size)];
	    innor3[ptid[i-1]-1] = mesh->nor->data[I2dm(i,3,mesh->nor->size)];

	    incurv1[ptid[i-1]-1] = mesh->curv->data[I2dm(i,1,mesh->curv->size)];
	    incurv2[ptid[i-1]-1] = mesh->curv->data[I2dm(i,2,mesh->curv->size)];
	}
    }

    MPI_Allreduce(inps1, outps1, out_num_all_pt, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(inps2, outps2, out_num_all_pt, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(inps3, outps3, out_num_all_pt, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    MPI_Allreduce(innor1, outnor1, out_num_all_pt, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(innor2, outnor2, out_num_all_pt, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(innor3, outnor3, out_num_all_pt, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    MPI_Allreduce(incurv1, outcurv1, out_num_all_pt, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(incurv2, outcurv2, out_num_all_pt, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    if (rank == 0)
    {
	char ncfilename[200];
	sprintf(ncfilename, "diffquant.out");
	FILE *diffout = fopen(ncfilename, "w");

	for (i = 0; i < out_num_all_pt; i++)
	    fprintf(diffout, "%22.16g %22.16g %22.16g %22.16g %22.16g %22.16g %22.16g %22.16g\n",
		    outps1[i], outps2[i], outps3[i], outnor1[i], outnor2[i], outnor3[i], outcurv1[i], outcurv2[i]);
    }

    free(ptid);
    free(inps1); free(inps2); free(inps3);
    free(outps1); free(outps2); free(outps3);
    free(innor1); free(innor2); free(innor3);
    free(outnor1); free(outnor2); free(outnor3);
    free(incurv1); free(incurv2);
    free(outcurv1); free(outcurv2);

    */

/*                 This part about testing build ghost for bounding box 
 *
    hpBuildNRingGhost(mesh, 4);
    double *bounding_box = (double *) calloc(6, sizeof(double));

    switch(rank)
    {
	case 0:
	    bounding_box[0] = 0.4; bounding_box[1] = 1.0;
	    bounding_box[2] = 0; bounding_box[3] = 0.6;
	    bounding_box[4] = -0.1; bounding_box[5] = 0.1;
	    break;
	case 1:
	    bounding_box[0] = 0.4; bounding_box[1] = 1.0;
	    bounding_box[2] = 0.4; bounding_box[3] = 1.0;
	    bounding_box[4] = -0.1; bounding_box[5] = 0.1;
	    break;
	case 2:
	    bounding_box[0] = 0; bounding_box[1] = 0.6;
	    bounding_box[2] = 0.4; bounding_box[3] = 1.0;
	    bounding_box[4] = -0.1; bounding_box[5] = 0.1;
	    break;
	case 3:
	    bounding_box[0] = 0; bounding_box[1] = 0.6;
	    bounding_box[2] = 0; bounding_box[3] = 0.6;
	    bounding_box[4] = -0.1; bounding_box[5] = 0.1;
	    break;
	default:
	    break;
    }
    hpBuildBoundingBoxGhost(mesh, bounding_box);
    printf("\n BuildNRingGhost passed, proc %d \n", rank);
    char debug_filename[200];
    sprintf(debug_filename, "debugout-p%s.vtk", rank_str);
    hpWriteUnstrMeshWithPInfo(debug_filename, mesh);

    hpBuildOppositeHalfEdge(mesh);
    printf("\n BuildOppHalfEdge passed, proc %d \n", rank);

    hpBuildIncidentHalfEdge(mesh);
    printf("\n BuildIncidentHalfEdge passed, proc %d \n", rank);
*/


/* this part about testing walf projection
    emxArray_real_T *para = emxCreate_real_T(2, 2);
    para->data[I2dm(1,1,para->size)] = 2.0/3.0;
    para->data[I2dm(1,2,para->size)] = 1.0/3.0;
    para->data[I2dm(2,1,para->size)] = 1.0/3.0;
    para->data[I2dm(2,2,para->size)] = 2.0/3.0;

    int new_num_tris = mesh->tris->size[0];

    emxArray_real_T *result = emxCreate_real_T(new_num_tris, 6);
    
    test_walf_tri(mesh->ps, mesh->tris, 2, para, result);

    char trisid_filename[200];
    sprintf(trisid_filename, "trisid-p%s.data", rank_str);
    FILE *infile = fopen(trisid_filename, "r");

    int *trisid = (int *) calloc(num_old_tris, sizeof(int));

    for (i = 0; i < num_old_tris; i++)
	fscanf(infile, "%d", &(trisid[i]));

    fclose(infile);


    int num_interior_tris = 0;

    char pnt_filename[200];
    sprintf(pnt_filename, "pnt-p%s.out", rank_str);
    FILE *pnt_outfile = fopen(pnt_filename, "w");

    fprintf(pnt_outfile, "xi = %g eta = %g\n", para->data[I2dm(1,1,para->size)], para->data[I2dm(1,2,para->size)]);
    for (i = 1; i <= mesh->tris->size[0]; i++)
    {
	int head = mesh->tris_pinfo->head[I1dm(i)];
	if (mesh->tris_pinfo->pdata[I1dm(head)].proc == rank)
	{
	    fprintf(pnt_outfile, "%22.16g %22.16g %22.16g\n", result->data[I2dm(i,1,result->size)], result->data[I2dm(i,2,result->size)], result->data[I2dm(i,3,result->size)]);
	    num_interior_tris++;
	}

    }

    fprintf(pnt_outfile, "xi = %g eta = %g\n", para->data[I2dm(2,1,para->size)], para->data[I2dm(2,2,para->size)]);
    for (i = 1; i <= mesh->tris->size[0]; i++)
    {
	int head = mesh->tris_pinfo->head[I1dm(i)];
	if (mesh->tris_pinfo->pdata[I1dm(head)].proc == rank)
	{
	    fprintf(pnt_outfile, "%22.16g %22.16g %22.16g\n", result->data[I2dm(i,4,result->size)], result->data[I2dm(i,5,result->size)], result->data[I2dm(i,6,result->size)]);
	}

    }
    fclose(pnt_outfile);

    int out_num_all_tris = 0;

    MPI_Allreduce(&num_interior_tris, &out_num_all_tris, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    printf("\n Num of all tris = %d \n", out_num_all_tris);

    double *in1 = (double *) calloc(out_num_all_tris, sizeof(double));
    double *in2 = (double *) calloc(out_num_all_tris, sizeof(double));
    double *in3 = (double *) calloc(out_num_all_tris, sizeof(double));

    double *in4 = (double *) calloc(out_num_all_tris, sizeof(double));
    double *in5 = (double *) calloc(out_num_all_tris, sizeof(double));
    double *in6 = (double *) calloc(out_num_all_tris, sizeof(double));

    double *out1 = (double *) calloc(out_num_all_tris, sizeof(double));
    double *out2 = (double *) calloc(out_num_all_tris, sizeof(double));
    double *out3 = (double *) calloc(out_num_all_tris, sizeof(double));

    double *out4 = (double *) calloc(out_num_all_tris, sizeof(double));
    double *out5 = (double *) calloc(out_num_all_tris, sizeof(double));
    double *out6 = (double *) calloc(out_num_all_tris, sizeof(double));


    for (i = 1; i <= mesh->tris->size[0]; i++)
    {
	int head = mesh->tris_pinfo->head[I1dm(i)];
	if (mesh->tris_pinfo->pdata[I1dm(head)].proc == rank)
	{
	    in1[trisid[i-1]-1] = result->data[I2dm(i,1,result->size)];
	    in2[trisid[i-1]-1] = result->data[I2dm(i,2,result->size)];
	    in3[trisid[i-1]-1] = result->data[I2dm(i,3,result->size)];
	    in4[trisid[i-1]-1] = result->data[I2dm(i,4,result->size)];
	    in5[trisid[i-1]-1] = result->data[I2dm(i,5,result->size)];
	    in6[trisid[i-1]-1] = result->data[I2dm(i,6,result->size)];
	}
    }

    MPI_Allreduce(in1, out1, out_num_all_tris, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(in2, out2, out_num_all_tris, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(in3, out3, out_num_all_tris, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    MPI_Allreduce(in4, out4, out_num_all_tris, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(in5, out5, out_num_all_tris, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(in6, out6, out_num_all_tris, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    if (rank == 0)
    {
	char ptfilename[200];
	sprintf(ptfilename, "pnt.out");
	FILE *ptout = fopen(ptfilename, "w");

	fprintf(ptout, "xi = %g eta = %g\n", para->data[I2dm(1,1,para->size)], para->data[I2dm(1,2,para->size)]);
	for (i = 0; i < out_num_all_tris; i++)
	    fprintf(ptout, "%22.16g %22.16g %22.16g\n",
		    out1[i], out2[i], out3[i]);

	fprintf(ptout, "xi = %g eta = %g\n", para->data[I2dm(2,1,para->size)], para->data[I2dm(2,2,para->size)]);
	for (i = 0; i < out_num_all_tris; i++)
	    fprintf(ptout, "%22.16g %22.16g %22.16g\n",
		    out4[i], out5[i], out6[i]);

    }

    free(trisid);
    free(in1); free(in2); free(in3);
    free(in4); free(in5); free(in6);
    free(out1); free(out2); free(out3);
    free(out4); free(out5); free(out6);
    */

/*  This part is for testing the updating function
 *
    emxArray_real_T *test_array = emxCreate_real_T(mesh->ps->size[0], 3);

    for (i = 1; i <= mesh->ps->size[0]; i++)
    {
	test_array->data[I2dm(i, 1, test_array->size)] = (real_T) rank;
	test_array->data[I2dm(i, 2, test_array->size)] = (real_T)i;
	test_array->data[I2dm(i, 3, test_array->size)] = (real_T)i - 1.0;
    }

    hpUpdateGhostPointData_real_T(mesh, test_array);


    for (i = 1; i <= mesh->ps->size[0]; i++)
    {
	int cur_node = mesh->ps_pinfo->head[I1dm(i)];
	while(cur_node != -1)
	{
	    printf("%d/%d-->", mesh->ps_pinfo->pdata[I1dm(cur_node)].proc, mesh->ps_pinfo->pdata[I1dm(cur_node)].lindex);
	    cur_node = mesh->ps_pinfo->pdata[I1dm(cur_node)].next;
	}
	printf("   (%10.8g, %10.8g, %10.8g)\n", test_array->data[I2dm(i, 1, test_array->size)], test_array->data[I2dm(i, 2, test_array->size)], test_array->data[I2dm(i, 3, test_array->size)]);
    }


    int num_nbp = mesh->nb_proc->size[0];

    for (i = 0; i < num_proc; i++)
    {
	emxArray_int32_T *cur_psi = mesh->ps_send_index[i];
	if (cur_psi == (emxArray_int32_T *) NULL)
	    printf("\nNot sending to proc %d\n", i);
	else
	{
	    printf("\nSending to proc %d:\n", i);
	    for (j = 1; j <= cur_psi->size[0]; j++)
	    {
		int cur_id = cur_psi->data[I1dm(j)];
		int cur_node = mesh->ps_pinfo->head[I1dm(cur_id)];
		while(cur_node != -1)
		{
		    printf("%d/%d-->", mesh->ps_pinfo->pdata[I1dm(cur_node)].proc, mesh->ps_pinfo->pdata[I1dm(cur_node)].lindex);
		    cur_node = mesh->ps_pinfo->pdata[I1dm(cur_node)].next;
		}
		printf("\n");
	    }

	}
    }

    for (i = 0; i < num_proc; i++)
    {
	emxArray_int32_T *cur_pri = mesh->ps_recv_index[i];
	if (cur_pri == (emxArray_int32_T *) NULL)
	    printf("\nNot receiving from proc %d\n", i);
	else
	{
	    printf("\nReceiving from proc %d:\n", i);
	    for (j = 1; j <= cur_pri->size[0]; j++)
	    {
		int cur_id = cur_pri->data[I1dm(j)];
		int cur_node = mesh->ps_pinfo->head[I1dm(cur_id)];
		while(cur_node != -1)
		{
		    printf("%d/%d-->", mesh->ps_pinfo->pdata[I1dm(cur_node)].proc, mesh->ps_pinfo->pdata[I1dm(cur_node)].lindex);
		    cur_node = mesh->ps_pinfo->pdata[I1dm(cur_node)].next;
		}
		printf("\n");
	    }

	}
    }
*/


}


