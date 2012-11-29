/*!
 * \file util.c
 * \brief Implementation of util.h 
 *
 * \author Yijie Zhou
 * \date 2012.10.01
 */

#include "util.h"

static void b_fix(real_T *x);
static real_T length(const emxArray_int32_T *x);
/*
static void b_emxInit_int32_T(emxArray_int32_T **pEmxArray, int32_T numDimensions);
static void b_emxInit_real_T(emxArray_real_T **pEmxArray, int32_T numDimensions);


static void b_emxInit_int32_T(emxArray_int32_T **pEmxArray, int32_T
  numDimensions)
{
  emxArray_int32_T *emxArray;
  int32_T loop_ub;
  int32_T i;
  *pEmxArray = (emxArray_int32_T *)malloc(sizeof(emxArray_int32_T));
  emxArray = *pEmxArray;
  emxArray->data = (int32_T *)NULL;
  emxArray->numDimensions = numDimensions;
  emxArray->size = (int32_T *)malloc((uint32_T)(sizeof(int32_T) * numDimensions));
  emxArray->allocatedSize = 0;
  emxArray->canFreeData = TRUE;
  loop_ub = numDimensions - 1;
  for (i = 0; i <= loop_ub; i++) {
    emxArray->size[i] = 0;
  }
}

static void b_emxInit_real_T(emxArray_real_T **pEmxArray, int32_T numDimensions)
{
  emxArray_real_T *emxArray;
  int32_T loop_ub;
  int32_T i;
  *pEmxArray = (emxArray_real_T *)malloc(sizeof(emxArray_real_T));
  emxArray = *pEmxArray;
  emxArray->data = (real_T *)NULL;
  emxArray->numDimensions = numDimensions;
  emxArray->size = (int32_T *)malloc((uint32_T)(sizeof(int32_T) * numDimensions));
  emxArray->allocatedSize = 0;
  emxArray->canFreeData = TRUE;
  loop_ub = numDimensions - 1;
  for (i = 0; i <= loop_ub; i++) {
    emxArray->size[i] = 0;
  }
}
*/

void numIntoString(const int n, int ndigits, char *s)
{
	int i;

	if (n == 0)
		ndigits--;
	for (i = n; i > 0; i /= 10) ndigits--;
	
	s[0] = '\0';
	for (;ndigits > 0; ndigits--)
		(void) sprintf(s,"%s0",s);
	(void) sprintf(s,"%s%d",s,n);
	return;
}

int findString(FILE *file, const char *in_string)
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


void addColumnToArray_common(emxArray__common *emxArray, const int32_T numCol, const uint32_T elementSize)
{
    void *newData;

    int numElemOld = emxArray->size[0] * emxArray->size[1];
    int numElemNew = emxArray->size[0] * (emxArray->size[1] + numCol);

    emxArray->size[1] += numCol;

    newData = calloc((uint32_T)numElemNew, elementSize);

    if (emxArray->data != NULL) {
	memcpy(newData, emxArray->data, (uint32_T)(elementSize * numElemOld));
	free(emxArray->data);
    }
    emxArray->data = newData;
    emxArray->allocatedSize = numElemNew;
}

void addColumnToArray_int32_T(emxArray_int32_T *emxArray, const int32_T numCol)
{
    addColumnToArray_common( (emxArray__common *)emxArray, numCol, sizeof(int32_T) );
    emxArray->data = (int32_T *) emxArray->data;
}
void addColumnToArray_real_T(emxArray_real_T *emxArray, const int32_T numCol)
{
    addColumnToArray_common( (emxArray__common *)emxArray, numCol, sizeof(real_T) );
    emxArray->data = (real_T *) emxArray->data;
}
void addColumnToArray_boolean_T(emxArray_boolean_T *emxArray, const int32_T numCol)
{
    addColumnToArray_common( (emxArray__common *)emxArray, numCol, sizeof(boolean_T) );
    emxArray->data = (boolean_T *) emxArray->data;
}

void addRowToArray_int32_T(emxArray_int32_T *emxArray, const int32_T numRow)
{
    int i;
    void *newData;

    int numRowOld = emxArray->size[0];
    int numRowNew = emxArray->size[0] + numRow;

    int numColOld, numColNew;
    numColNew = numColOld = emxArray->size[1];

    emxArray->size[0] = numRowNew;

    int numElemNew = numRowNew*numColNew;

    newData = calloc((uint32_T)numElemNew, sizeof(int32_T));

    if (emxArray->data != NULL)
    {
	for (i = 0; i < numColNew; i++)
	    memcpy((int32_T *)newData + i*numRowNew, emxArray->data + i*numRowOld, (uint32_T)(sizeof(int32_T) * numRowOld));
	free(emxArray->data);
    }

    emxArray->data = (int32_T *) newData;
    emxArray->allocatedSize = numElemNew;
}
void addRowToArray_real_T(emxArray_real_T *emxArray, const int32_T numRow)
{
    int i;
    void *newData;

    int numRowOld = emxArray->size[0];
    int numRowNew = emxArray->size[0] + numRow;

    int numColOld, numColNew;
    numColNew = numColOld = emxArray->size[1];

    emxArray->size[0] = numRowNew;

    int numElemNew = numRowNew*numColNew;

    newData = calloc((uint32_T)numElemNew, sizeof(real_T));

    if (emxArray->data != NULL)
    {
	for (i = 0; i < numColNew; i++)
	    memcpy((real_T *)newData + i*numRowNew, emxArray->data + i*numRowOld, (uint32_T)(sizeof(real_T) * numRowOld));
	free(emxArray->data);
    }

    emxArray->data = (real_T *) newData;
    emxArray->allocatedSize = numElemNew;
}
void addRowToArray_boolean_T(emxArray_boolean_T *emxArray, const int32_T numRow)
{
    int i;
    void *newData;

    int numRowOld = emxArray->size[0];
    int numRowNew = emxArray->size[0] + numRow;

    int numColOld, numColNew;
    numColNew = numColOld = emxArray->size[1];

    emxArray->size[0] = numRowNew;

    int numElemNew = numRowNew*numColNew;

    newData = calloc((uint32_T)numElemNew, sizeof(boolean_T));

    if (emxArray->data != NULL)
    {
	for (i = 0; i < numColNew; i++)
	    memcpy((boolean_T *)newData + i*numRowNew, emxArray->data + i*numRowOld, (uint32_T)(sizeof(boolean_T) * numRowOld));
	free(emxArray->data);
    }

    emxArray->data = (boolean_T *) newData;
    emxArray->allocatedSize = numElemNew;
}


void printArray_int32_T(const emxArray_int32_T *emxArray)
{
    int i,j;
    for (i = 1; i <= emxArray->size[0]; i++)
    {
	for (j = 1; j <= emxArray->size[1]; j++)
	    printf("%3d ",emxArray->data[I2dm(i,j,emxArray->size)]);
	printf("\n");
    }
}

void printArray_real_T(const emxArray_real_T *emxArray)
{
    int i,j;
    for (i = 1; i <= emxArray->size[0]; i++)
    {
	for (j = 1; j <= emxArray->size[1]; j++)
	    printf("%10.8g ",emxArray->data[I2dm(i,j,emxArray->size)]);
	printf("\n");
    }
}
void printArray_boolean_T(const emxArray_boolean_T *emxArray)
{
    int i,j;
    for (i = 1; i <= emxArray->size[0]; i++)
    {
	for (j = 1; j <= emxArray->size[1]; j++)
	    printf("%u ",emxArray->data[I2dm(i,j,emxArray->size)]);
	printf("\n");
    }
}


void sendND_boolean_T(const emxArray_boolean_T *array_send, const int dst, const int tag, MPI_Comm comm)
{
    int i;
    
    int32_T num_elem_regular = array_send->numDimensions + 2;
    int32_T *common_info = (int32_T *)calloc((uint32_T) num_elem_regular, sizeof(int32_T));

    common_info[0] = array_send->numDimensions;
    for (i = 1; i <= array_send->numDimensions; i++)
	common_info[i] = array_send->size[i-1];
    common_info[i] = array_send->allocatedSize;

    MPI_Send(&num_elem_regular, 1, MPI_INT, dst, tag+1, comm);
    MPI_Send(common_info, num_elem_regular, MPI_INT, dst, tag+2, comm);
    MPI_Send(array_send->data, array_send->allocatedSize, MPI_UNSIGNED_CHAR, dst, tag+3, comm);

    free((void *)common_info);

}

void sendND_int32_T(const emxArray_int32_T *array_send, const int dst, const int tag, MPI_Comm comm)
{
    int i;
    
    int32_T num_elem_regular = array_send->numDimensions + 2;
    int32_T *common_info = (int32_T *)calloc((uint32_T) num_elem_regular, sizeof(int32_T));

    common_info[0] = array_send->numDimensions;
    for (i = 1; i <= array_send->numDimensions; i++)
	common_info[i] = array_send->size[i-1];
    common_info[i] = array_send->allocatedSize;

    MPI_Send(&num_elem_regular, 1, MPI_INT, dst, tag+1, comm);
    MPI_Send(common_info, num_elem_regular, MPI_INT, dst, tag+2, comm);
    MPI_Send(array_send->data, array_send->allocatedSize, MPI_INT, dst, tag+3, comm);

    free((void *)common_info);
}

void sendND_real_T(const emxArray_real_T *array_send, const int dst, const int tag, MPI_Comm comm)
{
    int i;
    
    int32_T num_elem_regular = array_send->numDimensions + 2;
    int32_T *common_info = (int32_T *)calloc((uint32_T) num_elem_regular, sizeof(int32_T));

    common_info[0] = array_send->numDimensions;
    for (i = 1; i <= array_send->numDimensions; i++)
	common_info[i] = array_send->size[i-1];
    common_info[i] = array_send->allocatedSize;

    MPI_Send(&num_elem_regular, 1, MPI_INT, dst, tag+1, comm);
    MPI_Send(common_info, num_elem_regular, MPI_INT, dst, tag+2, comm);
    MPI_Send(array_send->data, array_send->allocatedSize, MPI_DOUBLE, dst, tag+3, comm);

    free((void *)common_info);
}

void recvND_boolean_T(emxArray_boolean_T **array_recv, const int src, const int tag, MPI_Comm comm)
{
    int i;
    int num_elem_regular;
    MPI_Status recv_stat_1;
    MPI_Status recv_stat_2;
    MPI_Status recv_stat_3;

    MPI_Recv(&num_elem_regular, 1, MPI_INT, src, tag+1, comm, &recv_stat_1);

    int32_T *common_info = (int32_T *)calloc((uint32_T) num_elem_regular, sizeof(int32_T));

    MPI_Recv(common_info, num_elem_regular, MPI_INT, src, tag+2, comm, &recv_stat_2);

    int num_dim = common_info[0];
    int num_elem = common_info[num_elem_regular-1];

    int32_T *size = (int32_T *)calloc((uint32_T) num_dim, sizeof(int32_T));

    for (i = 1; i <= num_dim; i++)
	size[i] = common_info[i];

    emxArray_boolean_T *result = emxCreateND_boolean_T(num_dim, size);

    MPI_Recv(result->data, num_elem, MPI_UNSIGNED_CHAR, src, tag+3, comm, &recv_stat_3);

    free ((void *) common_info);
    free ((void *) size);

    (*array_recv) = result;
}

void recvND_int32_T(emxArray_int32_T **array_recv, const int src, const int tag, MPI_Comm comm)
{
    int i;
    int num_elem_regular;
    MPI_Status recv_stat_1;
    MPI_Status recv_stat_2;
    MPI_Status recv_stat_3;

    MPI_Recv(&num_elem_regular, 1, MPI_INT, src, tag+1, comm, &recv_stat_1);

    int32_T *common_info = (int32_T *)calloc((uint32_T) num_elem_regular, sizeof(int32_T));

    MPI_Recv(common_info, num_elem_regular, MPI_INT, src, tag+2, comm, &recv_stat_2);

    int num_dim = common_info[0];
    int num_elem = common_info[num_elem_regular-1];

    int32_T *size = (int32_T *)calloc((uint32_T) num_dim, sizeof(int32_T));

    for (i = 1; i <= num_dim; i++)
	size[i] = common_info[i];

    emxArray_int32_T *result = emxCreateND_int32_T(num_dim, size);

    MPI_Recv(result->data, num_elem, MPI_INT, src, tag+3, comm, &recv_stat_3);

    free ((void *) common_info);
    free ((void *) size);

    (*array_recv) = result;
}

void recvND_real_T(emxArray_real_T **array_recv, const int src, const int tag, MPI_Comm comm)
{
    int i;
    int num_elem_regular;
    MPI_Status recv_stat_1;
    MPI_Status recv_stat_2;
    MPI_Status recv_stat_3;
    MPI_Recv(&num_elem_regular, 1, MPI_INT, src, tag+1, comm, &recv_stat_1);

    int32_T *common_info = (int32_T *)calloc((uint32_T) num_elem_regular, sizeof(int32_T));

    MPI_Recv(common_info, num_elem_regular, MPI_INT, src, tag+2, comm, &recv_stat_2);

    int num_dim = common_info[0];
    int num_elem = common_info[num_elem_regular-1];

    int32_T *size = (int32_T *)calloc((uint32_T) num_dim, sizeof(int32_T));

    for (i = 1; i <= num_dim; i++)
	size[i] = common_info[i];

    emxArray_real_T *result = emxCreateND_real_T(num_dim, size);

    MPI_Recv(result->data, num_elem, MPI_DOUBLE, src, tag+3, comm, &recv_stat_3);

    free ((void *) common_info);
    free ((void *) size);

    (*array_recv) = result;
}

void send2D_boolean_T(const emxArray_boolean_T *array_send, const int dst, const int tag, MPI_Comm comm)
{
    MPI_Send(array_send->size, 2, MPI_INT, dst, tag+1, comm);
    MPI_Send(array_send->data, array_send->allocatedSize, MPI_UNSIGNED_CHAR, dst, tag+2, comm);
}

void send2D_int32_T(const emxArray_int32_T *array_send, const int dst, const int tag, MPI_Comm comm)
{
    MPI_Send(array_send->size, 2, MPI_INT, dst, tag+1, comm);
    MPI_Send(array_send->data, array_send->allocatedSize, MPI_INT, dst, tag+2, comm);

}

void send2D_real_T(const emxArray_real_T *array_send, const int dst, const int tag, MPI_Comm comm)
{
    MPI_Send(array_send->size, 2, MPI_INT, dst, tag+1, comm);
    MPI_Send(array_send->data, array_send->allocatedSize, MPI_DOUBLE, dst, tag+2, comm);

}
void recv2D_boolean_T(emxArray_boolean_T **array_recv, const int src, const int tag, MPI_Comm comm)
{
    MPI_Status recv_stat_1;
    MPI_Status recv_stat_2;

    int32_T common_info[2];

    MPI_Recv(common_info, 2, MPI_INT, src, tag+1, comm, &recv_stat_1);

    emxArray_boolean_T *result = emxCreate_boolean_T(common_info[0], common_info[1]);

    MPI_Recv(result->data, common_info[0]*common_info[1], MPI_UNSIGNED_CHAR, src, tag+2, comm, &recv_stat_2);

    (*array_recv) = result;
}


void recv2D_int32_T(emxArray_int32_T **array_recv, const int src, const int tag, MPI_Comm comm)
{
    MPI_Status recv_stat_1;
    MPI_Status recv_stat_2;

    int32_T common_info[2];

    MPI_Recv(common_info, 2, MPI_INT, src, tag+1, comm, &recv_stat_1);

    emxArray_int32_T *result = emxCreate_int32_T(common_info[0], common_info[1]);

    MPI_Recv(result->data, common_info[0]*common_info[1], MPI_INT, src, tag+2, comm, &recv_stat_2);

    (*array_recv) = result;
}

void recv2D_real_T(emxArray_real_T **array_recv, const int src, const int tag, MPI_Comm comm)
{
    MPI_Status recv_stat_1;
    MPI_Status recv_stat_2;

    int32_T common_info[2];

    MPI_Recv(common_info, 2, MPI_INT, src, tag+1, comm, &recv_stat_1);

    emxArray_real_T *result = emxCreate_real_T(common_info[0], common_info[1]);

    MPI_Recv(result->data, common_info[0]*common_info[1], MPI_DOUBLE, src, tag+2, comm, &recv_stat_2);

    (*array_recv) = result;
}

void isend2D_boolean_T(const emxArray_boolean_T *array_send, const int dst, const int tag,
		       MPI_Comm comm, MPI_Request *req_com, MPI_Request *req_data)
{
    MPI_Isend(array_send->size, 2, MPI_INT, dst, tag+1, comm, req_com);
    MPI_Isend(array_send->data, array_send->allocatedSize, MPI_UNSIGNED_CHAR, dst, tag+2, comm, req_data); 
}


void isend2D_int32_T(const emxArray_int32_T *array_send, const int dst, const int tag,
		     MPI_Comm comm, MPI_Request *req_com, MPI_Request *req_data)
{
    MPI_Isend(array_send->size, 2, MPI_INT, dst, tag+1, comm, req_com);
    MPI_Isend(array_send->data, array_send->allocatedSize, MPI_INT, dst, tag+2, comm, req_data); 
}

void isend2D_real_T(const emxArray_real_T *array_send, const int dst, const int tag,
		    MPI_Comm comm, MPI_Request *req_com, MPI_Request *req_data)
{
    MPI_Isend(array_send->size, 2, MPI_INT, dst, tag+1, comm, req_com);
    MPI_Isend(array_send->data, array_send->allocatedSize, MPI_DOUBLE, dst, tag+2, comm, req_data); 
}

void determine_opposite_halfedge_tri(int32_T nv, const emxArray_int32_T *tris,
  emxArray_int32_T *opphes)
{
  emxArray_int32_T *is_index;
  int32_T ntris;
  int32_T i0;
  int32_T ii;
  boolean_T exitg1;
  int32_T b_is_index[3];
  emxArray_int32_T *v2nv;
  emxArray_int32_T *v2he;
  int32_T ne;
  static const int8_T iv0[3] = { 1, 2, 0 };

  int32_T tris_idx_0;
  int32_T loop_ub;
  static const int8_T iv1[3] = { 2, 3, 1 };

  emxInit_int32_T(&is_index, 1);

  /* DETERMINE_OPPOSITE_HALFEDGE_TRI Determine opposite half-edges for triangle  */
  /* mesh. */
  /*  DETERMINE_OPPOSITE_HALFEDGE_TRI(NV,TRIS,OPPHES) Determines */
  /*  opposite half-edges for triangle mesh. The following explains the input */
  /*  and output arguments. */
  /*  */
  /*  OPPHES = DETERMINE_OPPOSITE_HALFEDGE_TRI(NV,TRIS) */
  /*  OPPHES = DETERMINE_OPPOSITE_HALFEDGE_TRI(NV,TRIS,OPPHES) */
  /*  Computes mapping from each half-edge to its opposite half-edge for  */
  /*  triangle mesh. */
  /*  */
  /*  Convention: Each half-edge is indicated by <face_id,local_edge_id>. */
  /*  We assign 2 bits to local_edge_id (starts from 0). */
  /*  */
  /*  See also DETERMINE_OPPOSITE_HALFEDGE */
  /* 'determine_opposite_halfedge_tri:18' coder.inline('never'); */
  /* 'determine_opposite_halfedge_tri:19' assert(isscalar(nv)&&isa(nv,'int32')); */
  /* 'determine_opposite_halfedge_tri:20' assert((size(tris,2)==3) && (size(tris,1)>=1) && isa(tris,'int32')); */
  /* 'determine_opposite_halfedge_tri:21' assert((size(opphes,2)==3) && (size(opphes,1)>=1) && isa(opphes,'int32')); */
  /* 'determine_opposite_halfedge_tri:23' nepE = int32(3); */
  /*  Number of edges per element */
  /* 'determine_opposite_halfedge_tri:24' next = int32([2,3,1]); */
  /* 'determine_opposite_halfedge_tri:25' inds = int32(1:3); */
  /* 'determine_opposite_halfedge_tri:27' ntris = int32(size(tris,1)); */
  ntris = tris->size[0];

  /* % First, build is_index to store starting position for each vertex. */
  /* 'determine_opposite_halfedge_tri:29' is_index = zeros(nv+1,1,'int32'); */
  i0 = is_index->size[0];
  is_index->size[0] = nv + 1;
  emxEnsureCapacity((emxArray__common *)is_index, i0, (int32_T)sizeof(int32_T));
  for (i0 = 0; i0 <= nv; i0++) {
    is_index->data[i0] = 0;
  }

  /* 'determine_opposite_halfedge_tri:30' for ii=1:ntris */
  ii = 0;
  exitg1 = 0U;
  while ((exitg1 == 0U) && (ii + 1 <= ntris)) {
    /* 'determine_opposite_halfedge_tri:31' if tris(ii,1)==0 */
    if (tris->data[ii] == 0) {
      /* 'determine_opposite_halfedge_tri:31' ntris=ii-1; */
      ntris = ii;
      exitg1 = 1U;
    } else {
      /* 'determine_opposite_halfedge_tri:32' is_index(tris(ii,inds)+1) = is_index(tris(ii,inds)+1) + 1; */
      for (i0 = 0; i0 < 3; i0++) {
        b_is_index[i0] = is_index->data[tris->data[ii + tris->size[0] * i0]] + 1;
      }

      for (i0 = 0; i0 < 3; i0++) {
        is_index->data[tris->data[ii + tris->size[0] * i0]] = b_is_index[i0];
      }

      ii++;
    }
  }

  /* 'determine_opposite_halfedge_tri:34' is_index(1) = 1; */
  is_index->data[0] = 1;

  /* 'determine_opposite_halfedge_tri:35' for ii=1:nv */
  for (ii = 1; ii <= nv; ii++) {
    /* 'determine_opposite_halfedge_tri:36' is_index(ii+1) = is_index(ii) + is_index(ii+1); */
    is_index->data[ii] += is_index->data[ii - 1];
  }

  emxInit_int32_T(&v2nv, 1);
  emxInit_int32_T(&v2he, 1);

  /* 'determine_opposite_halfedge_tri:39' ne = ntris*nepE; */
  ne = ntris * 3;

  /* 'determine_opposite_halfedge_tri:40' v2nv = coder.nullcopy(zeros( ne,1, 'int32')); */
  i0 = v2nv->size[0];
  v2nv->size[0] = ne;
  emxEnsureCapacity((emxArray__common *)v2nv, i0, (int32_T)sizeof(int32_T));

  /*  Vertex to next vertex in each halfedge. */
  /* 'determine_opposite_halfedge_tri:41' v2he = coder.nullcopy(zeros( ne,1, 'int32')); */
  i0 = v2he->size[0];
  v2he->size[0] = ne;
  emxEnsureCapacity((emxArray__common *)v2he, i0, (int32_T)sizeof(int32_T));

  /*  Vertex to half-edge. */
  /* 'determine_opposite_halfedge_tri:42' for ii=1:ntris */
  for (ii = 0; ii + 1 <= ntris; ii++) {
    /* 'determine_opposite_halfedge_tri:43' v2nv(is_index( tris(ii,inds))) = tris(ii,next); */
    for (i0 = 0; i0 < 3; i0++) {
      v2nv->data[is_index->data[tris->data[ii + tris->size[0] * i0] - 1] - 1] =
        tris->data[ii + tris->size[0] * iv0[i0]];
    }

    /* 'determine_opposite_halfedge_tri:44' v2he(is_index( tris(ii,inds))) = 4*ii-1+inds; */
    ne = (ii + 1) << 2;
    for (i0 = 0; i0 < 3; i0++) {
      v2he->data[is_index->data[tris->data[ii + tris->size[0] * i0] - 1] - 1] =
        i0 + ne;
    }

    /* 'determine_opposite_halfedge_tri:45' is_index(tris(ii,inds)) = is_index(tris(ii,inds)) + 1; */
    for (i0 = 0; i0 < 3; i0++) {
      b_is_index[i0] = is_index->data[tris->data[ii + tris->size[0] * i0] - 1] +
        1;
    }

    for (i0 = 0; i0 < 3; i0++) {
      is_index->data[tris->data[ii + tris->size[0] * i0] - 1] = b_is_index[i0];
    }
  }

  /* 'determine_opposite_halfedge_tri:47' for ii=nv-1:-1:1 */
  for (ii = nv - 1; ii > 0; ii--) {
    /* 'determine_opposite_halfedge_tri:47' is_index(ii+1) = is_index(ii); */
    is_index->data[ii] = is_index->data[ii - 1];
  }

  /* 'determine_opposite_halfedge_tri:48' is_index(1)=1; */
  is_index->data[0] = 1;

  /* % Set opphes */
  /* 'determine_opposite_halfedge_tri:50' if nargin<3 || isempty(opphes) */
  if (opphes->size[0] == 0) {
    /* 'determine_opposite_halfedge_tri:51' opphes = zeros(size(tris,1), nepE, 'int32'); */
    tris_idx_0 = tris->size[0];
    i0 = opphes->size[0] * opphes->size[1];
    opphes->size[0] = tris_idx_0;
    opphes->size[1] = 3;
    emxEnsureCapacity((emxArray__common *)opphes, i0, (int32_T)sizeof(int32_T));
    for (i0 = 0; i0 < 3; i0++) {
      loop_ub = tris_idx_0 - 1;
      for (ne = 0; ne <= loop_ub; ne++) {
        opphes->data[ne + opphes->size[0] * i0] = 0;
      }
    }
  } else {
    /* 'determine_opposite_halfedge_tri:52' else */
    /* 'determine_opposite_halfedge_tri:53' assert( size(opphes,1)>=ntris && size(opphes,2)>=nepE); */
    /* 'determine_opposite_halfedge_tri:54' opphes(:) = 0; */
    i0 = opphes->size[0] * opphes->size[1];
    opphes->size[1] = 3;
    emxEnsureCapacity((emxArray__common *)opphes, i0, (int32_T)sizeof(int32_T));
    for (i0 = 0; i0 < 3; i0++) {
      loop_ub = opphes->size[0] - 1;
      for (ne = 0; ne <= loop_ub; ne++) {
        opphes->data[ne + opphes->size[0] * i0] = 0;
      }
    }
  }

  /* 'determine_opposite_halfedge_tri:57' for ii=1:ntris */
  for (ii = 0; ii + 1 <= ntris; ii++) {
    /* 'determine_opposite_halfedge_tri:58' for jj=int32(1):3 */
    for (ne = 0; ne < 3; ne++) {
      /* 'determine_opposite_halfedge_tri:59' if opphes(ii,jj) */
      if (opphes->data[ii + opphes->size[0] * ne] != 0) {
      } else {
        /* 'determine_opposite_halfedge_tri:60' v = tris(ii,jj); */
        /* 'determine_opposite_halfedge_tri:60' vn = tris(ii,next(jj)); */
        /*  LOCATE: Locate index col in v2nv(first:last) */
        /* 'determine_opposite_halfedge_tri:63' found = int32(0); */
        /* 'determine_opposite_halfedge_tri:64' for index = is_index(vn):is_index(vn+1)-1 */
        loop_ub = is_index->data[tris->data[ii + tris->size[0] * (iv1[ne] - 1)]]
          - 1;
        for (tris_idx_0 = is_index->data[tris->data[ii + tris->size[0] * (iv1[ne]
              - 1)] - 1] - 1; tris_idx_0 + 1 <= loop_ub; tris_idx_0++) {
          /* 'determine_opposite_halfedge_tri:65' if v2nv(index)==v */
          if (v2nv->data[tris_idx_0] == tris->data[ii + tris->size[0] * ne]) {
            /* 'determine_opposite_halfedge_tri:66' opp = v2he(index); */
            /* 'determine_opposite_halfedge_tri:67' opphes(ii,jj) = opp; */
            opphes->data[ii + opphes->size[0] * ne] = v2he->data[tris_idx_0];

            /* opphes(heid2fid(opp),heid2leid(opp)) = ii*4+jj-1; */
            /* 'determine_opposite_halfedge_tri:69' opphes(bitshift(uint32(opp),-2),mod(opp,4)+1) = ii*4+jj-1; */
            opphes->data[((int32_T)((uint32_T)v2he->data[tris_idx_0] >> 2U) +
                          opphes->size[0] * (v2he->data[tris_idx_0] -
              ((v2he->data[tris_idx_0] >> 2) << 2))) - 1] = ((ii + 1) << 2) + ne;

            /* 'determine_opposite_halfedge_tri:71' found = found + 1; */
          }
        }

        /*  Check for consistency */
        /* 'determine_opposite_halfedge_tri:76' if found>1 */
      }
    }
  }

  emxFree_int32_T(&v2he);
  emxFree_int32_T(&v2nv);
  emxFree_int32_T(&is_index);
}

void determine_incident_halfedges(const emxArray_int32_T *elems, const
  emxArray_int32_T *opphes, emxArray_int32_T *v2he)
{
  int32_T kk;
  int32_T loop_ub;
  boolean_T guard1 = FALSE;
  uint32_T a;

  /* DETERMINE_INCIDENT_HALFEDGES Determine an incident halfedges. */
  /*  DETERMINE_INCIDENT_HALFEDGES(ELEMS,OPPHES,V2HE) Determines incident */
  /*  halfedges of each vertex for a triangular, quadrilateral, or mixed mesh.  */
  /*  It gives higher priorities to border edges. The following explains inputs */
  /*  and outputs. */
  /*  */
  /*  V2HE = DETERMINE_INCIDENT_HALFEDGES(ELEMS,OPPHES) */
  /*  V2HE = DETERMINE_INCIDENT_HALFEDGES(ELEMS,OPPHES,V2HE) */
  /*  V2HE = DETERMINE_INCIDENT_HALFEDGES(ELEMS,OPPHES,V2HE) */
  /*      ELEMS is mx3 (for triangle mesh) or mx4 (for quadrilateral mesh). */
  /*      OPPHES is mx3 (for triangle mesh) or mx4 (for quadrilateral mesh). */
  /*      V2HE is an array of size equal to number of vertices. */
  /*           It is passed by reference. */
  /*  */
  /*  See also DETERMINE_INCIDENT_HALFFACES, DETERMINE_INCIDENT_HALFVERTS */
  /* 'determine_incident_halfedges:18' coder.inline('never'); */
  /* 'determine_incident_halfedges:19' assert((size(elems,2)==3) && (size(elems,1)>=1) && isa(elems,'int32')); */
  /* 'determine_incident_halfedges:20' assert((size(opphes,2)==3) && (size(opphes,1)>=1) && isa(opphes,'int32')); */
  /* 'determine_incident_halfedges:21' assert((size(v2he,2)==1) && (size(v2he,1)>=1) && isa(v2he,'int32')); */
  /* 'determine_incident_halfedges:23' if nargin<3 */
  /* 'determine_incident_halfedges:35' else */
  /* 'determine_incident_halfedges:36' v2he(:) = 0; */
  kk = v2he->size[0];
  emxEnsureCapacity((emxArray__common *)v2he, kk, (int32_T)sizeof(int32_T));
  loop_ub = v2he->size[0] - 1;
  for (kk = 0; kk <= loop_ub; kk++) {
    v2he->data[kk] = 0;
  }

  /* 'determine_incident_halfedges:39' for kk=1:int32(size(elems,1)) */
  kk = 0;
  while ((kk + 1 <= elems->size[0]) && (!(elems->data[kk] == 0))) {
    /* 'determine_incident_halfedges:40' if elems(kk,1)==0 */
    /* 'determine_incident_halfedges:42' for lid=1:int32(size(elems,2)) */
    for (loop_ub = 0; loop_ub < 3; loop_ub++) {
      /* 'determine_incident_halfedges:43' v = elems(kk,lid); */
      /* 'determine_incident_halfedges:44' if v>0 && (v2he(v)==0 || opphes( kk,lid) == 0 || ... */
      /* 'determine_incident_halfedges:45' 	     (opphes( int32( bitshift( uint32(v2he(v)),-2)), mod(v2he(v),4)+1) && opphes( kk, lid)<0)) */
      if (elems->data[kk + elems->size[0] * loop_ub] > 0) {
        guard1 = FALSE;
        if ((v2he->data[elems->data[kk + elems->size[0] * loop_ub] - 1] == 0) ||
            (opphes->data[kk + opphes->size[0] * loop_ub] == 0)) {
          guard1 = TRUE;
        } else {
          a = (uint32_T)v2he->data[elems->data[kk + elems->size[0] * loop_ub] -
            1];
          if ((opphes->data[((int32_T)(a >> 2U) + opphes->size[0] * (v2he->
                 data[elems->data[kk + elems->size[0] * loop_ub] - 1] -
                 ((v2he->data[elems->data[kk + elems->size[0] * loop_ub] - 1] >>
                   2) << 2))) - 1] != 0) && (opphes->data[kk + opphes->size[0] *
               loop_ub] < 0)) {
            guard1 = TRUE;
          }
        }

        if (guard1 == TRUE) {
          /* 'determine_incident_halfedges:46' v2he(v) = 4*kk + lid - 1; */
          v2he->data[elems->data[kk + elems->size[0] * loop_ub] - 1] = ((kk + 1)
            << 2) + loop_ub;
        }
      }
    }

    kk++;
  }
}

static void b_fix(real_T *x)
{
  if (*x > 0.0) {
    *x = floor(*x);
  } else {
    *x = ceil(*x);
  }
}

static real_T length(const emxArray_int32_T *x)
{
  real_T n;
  if (x->size[0] == 0) {
    n = 0.0;
  } else if (x->size[0] > 1) {
    n = (real_T)x->size[0];
  } else {
    n = 1.0;
  }

  return n;
}

void obtain_nring_surf(int32_T vid, real_T ring, int32_T minpnts, const
  emxArray_int32_T *tris, const emxArray_int32_T *opphes, const emxArray_int32_T
  *v2he, emxArray_int32_T *ngbvs, emxArray_boolean_T *vtags, emxArray_boolean_T *
  ftags, emxArray_int32_T *ngbfs, int32_T *nverts, int32_T *nfaces)
{
  int32_T fid;
  int32_T lid;
  boolean_T overflow;
  int32_T maxnv;
  int32_T opp;
  int32_T maxnf;
  emxArray_int32_T *hebuf;
  int32_T fid_in;
  static const int8_T iv0[3] = { 2, 3, 1 };

  int32_T exitg4;
  static const int8_T iv1[3] = { 3, 1, 2 };

  int32_T nverts_pre;
  int32_T nfaces_pre;
  real_T ring_full;
  real_T cur_ring;
  int32_T exitg1;
  boolean_T guard1 = FALSE;
  int32_T nverts_last;
  boolean_T exitg2;
  boolean_T b0;
  boolean_T isfirst;
  int32_T exitg3;
  boolean_T guard2 = FALSE;
  boolean_T b_guard1 = FALSE;
  boolean_T b_guard2 = FALSE;

  /* OBTAIN_NRING_SURF Collect n-ring vertices and faces of a triangle mesh. */
  /*  [NGBVS,NVERTS,VTAGS,FTAGS,NGBFS,NFACES] = OBTAIN_NRING_SURF(VID,RING, ... */
  /*  MINPNTS,TRIS,OPPHES,V2HE,NGBVS,VTAGS,FTAGS,NGBFS)  Collects n-ring */
  /*  vertices and faces of a vertex and saves them into NGBVS and NGBFS, */
  /*  where n is a floating point number with 0.5 increments (1, 1.5, 2, etc.) */
  /*  We define the n-ring verticse as follows: */
  /*   - 0-ring: vertex itself */
  /*   - k-ring vertices: vertices that share an edge with (k-1)-ring vertices */
  /*   - (k+0.5)-ring vertices: k-ring plus vertices that share an element */
  /*            with two vertices of k-ring vertices. */
  /*  For triangle meshes, the k-ring vertices always form some triangles. */
  /*  */
  /*  Input arguments */
  /*    vid: vertex ID */
  /*    ring: the desired number of rings (it is a float as it can have halves) */
  /*    minpnts: the minimum number of points desired */
  /*    tris: element connectivity */
  /*    opphes: opposite half-edges */
  /*    v2he: vertex-to-halfedge mapping */
  /*    ngbvs: buffer space for neighboring vertices (not including vid itself) */
  /*    vtags: vertex tags (boolean, of length equal to number of vertices) */
  /*    ftags: face tags (boolean, of length equal to number of elements) */
  /*    ngbfs: buffer space for neighboring faces */
  /*  */
  /*  Output arguments */
  /*    ngbvs: buffer space for neighboring vertices */
  /*    nverts: number of vertices in the neighborhood */
  /*    vtags: vertex tags (boolean, of length equal to number of vertices) */
  /*    ftags: face tags (boolean, of length equal to number of elements) */
  /*    ngbfs: buffer space for neighboring faces */
  /*    nfaces: number of elements in the neighborhood */
  /*  */
  /*  Notes */
  /*   1. vtags and ftags must be set to false at input. They are reset to */
  /*      false at output. */
  /*   2. Since the vertex itself is always in ring, we do not include it in */
  /*      the output array ngbvs. */
  /*   3. If NGBVS or NGBFS is not enough to store the whole neighborhood, */
  /*      then only a subset of the neighborhood will be returned. */
  /*      The maximum number of points returned is numel(NGBVS) if NGBVS is */
  /*      given as part of the input, or 128 if not an input arguement. */
  /*      The maximum number of faces returned is numel(NGBFS) if NGBFS is */
  /*      given as part of the input, or 256 if not an input arguement. */
  /*  */
  /*  See also OBTAIN_NRING_SURF, OBTAIN_NRING_QUAD, OBTAIN_NRING_CURV, OBTAIN_NRING_VOL */
  /* 'obtain_nring_surf:49' coder.extrinsic('warning'); */
  /* 'obtain_nring_surf:50' assert(isscalar(vid)&&isa(vid,'int32')); */
  /* 'obtain_nring_surf:51' assert(isscalar(ring)&&isa(ring,'double')); */
  /* 'obtain_nring_surf:52' assert(isscalar(minpnts)&&isa(minpnts,'int32')); */
  /* 'obtain_nring_surf:53' assert((size(tris,2)==3) && (size(tris,1)>=1) && isa(tris,'int32')); */
  /* 'obtain_nring_surf:54' assert((size(opphes,2)==3) && (size(opphes,1)>=1) && isa(opphes,'int32')); */
  /* 'obtain_nring_surf:55' assert((size(v2he,2)==1) && (size(v2he,1)>=1) && isa(v2he,'int32')); */
  /* 'obtain_nring_surf:56' assert((size(ngbvs,2)==1) && (size(ngbvs,1)>=1) && isa(ngbvs,'int32')); */
  /* 'obtain_nring_surf:57' assert((size(vtags,2)==1) && (size(vtags,1)>=1) && isa(vtags,'logical')); */
  /* 'obtain_nring_surf:58' assert((size(ftags,2)==1) && (size(ftags,1)>=1) && isa(ftags,'logical')); */
  /* 'obtain_nring_surf:59' assert((size(ngbfs,2)==1) && (size(ngbfs,1)>=1) && isa(ngbfs,'int32')); */
  /* 'obtain_nring_surf:61' MAXNPNTS = int32(128); */
  /* 'obtain_nring_surf:63' assert(ring>=1 && floor(ring*2)==ring*2); */
  /* 'obtain_nring_surf:64' if nargin>=8 */
  /* 'obtain_nring_surf:64' assert( islogical( vtags)); */
  /* 'obtain_nring_surf:65' if nargin>=9 */
  /* 'obtain_nring_surf:65' assert( islogical(ftags)); */
  /* 'obtain_nring_surf:67' fid = heid2fid(v2he(vid)); */
  /*  HEID2FID   Obtains face ID from half-edge ID. */
  /* 'heid2fid:3' coder.inline('always'); */
  /* 'heid2fid:4' fid = int32(bitshift(uint32(heid), -2)); */
  fid = (int32_T)((uint32_T)v2he->data[vid - 1] >> 2U) - 1;

  /* 'obtain_nring_surf:67' lid = heid2leid(v2he(vid)); */
  /*  HEID2LEID   Obtains local edge ID within a face from half-edge ID. */
  /* 'heid2leid:3' coder.inline('always'); */
  /* 'heid2leid:4' leid = int32(bitand(uint32(heid),3))+1; */
  lid = (int32_T)((uint32_T)v2he->data[vid - 1] & 3U);

  /* 'obtain_nring_surf:68' nverts=int32(0); */
  *nverts = 0;

  /* 'obtain_nring_surf:68' nfaces=int32(0); */
  *nfaces = 0;

  /* 'obtain_nring_surf:68' overflow = false; */
  overflow = FALSE;

  /* 'obtain_nring_surf:70' if ~fid */
  if (!(fid + 1 != 0)) {
  } else {
    /* 'obtain_nring_surf:72' prv = int32([3 1 2]); */
    /* 'obtain_nring_surf:73' nxt = int32([2 3 1]); */
    /* 'obtain_nring_surf:75' if nargin>=7 && ~isempty(ngbvs) */
    if (!(ngbvs->size[0] == 0)) {
      /* 'obtain_nring_surf:76' maxnv = int32(numel(ngbvs)); */
      maxnv = ngbvs->size[0];
    } else {
      /* 'obtain_nring_surf:77' else */
      /* 'obtain_nring_surf:78' maxnv = MAXNPNTS; */
      maxnv = 128;

      /* 'obtain_nring_surf:78' ngbvs=coder.nullcopy(zeros(MAXNPNTS,1,'int32')); */
      opp = ngbvs->size[0];
      ngbvs->size[0] = 128;
      emxEnsureCapacity((emxArray__common *)ngbvs, opp, (int32_T)sizeof(int32_T));
    }

    /* 'obtain_nring_surf:81' if nargin>=10 && ~isempty(ngbfs) */
    if (!(ngbfs->size[0] == 0)) {
      /* 'obtain_nring_surf:82' maxnf = int32(numel(ngbfs)); */
      maxnf = ngbfs->size[0];
    } else {
      /* 'obtain_nring_surf:83' else */
      /* 'obtain_nring_surf:84' maxnf = 2*MAXNPNTS; */
      maxnf = 256;

      /* 'obtain_nring_surf:84' ngbfs = coder.nullcopy(zeros(maxnf,1, 'int32')); */
      opp = ngbfs->size[0];
      ngbfs->size[0] = 256;
      emxEnsureCapacity((emxArray__common *)ngbfs, opp, (int32_T)sizeof(int32_T));
    }

    emxInit_int32_T(&hebuf, 1);

    /* 'obtain_nring_surf:87' oneringonly = ring==1 && minpnts==0 && nargout<5; */
    /* 'obtain_nring_surf:88' hebuf = coder.nullcopy(zeros(maxnv,1, 'int32')); */
    opp = hebuf->size[0];
    hebuf->size[0] = maxnv;
    emxEnsureCapacity((emxArray__common *)hebuf, opp, (int32_T)sizeof(int32_T));

    /*  Optimized version for collecting one-ring vertices */
    /* 'obtain_nring_surf:91' if opphes( fid, lid) */
    if (opphes->data[fid + opphes->size[0] * lid] != 0) {
      /* 'obtain_nring_surf:92' fid_in = fid; */
      fid_in = fid + 1;
    } else {
      /* 'obtain_nring_surf:93' else */
      /* 'obtain_nring_surf:94' fid_in = int32(0); */
      fid_in = 0;

      /* 'obtain_nring_surf:96' v = tris(fid, nxt(lid)); */
      /* 'obtain_nring_surf:97' nverts = int32(1); */
      *nverts = 1;

      /* 'obtain_nring_surf:97' ngbvs( 1) = v; */
      ngbvs->data[0] = tris->data[fid + tris->size[0] * (iv0[lid] - 1)];

      /* 'obtain_nring_surf:99' if ~oneringonly */
      /* 'obtain_nring_surf:99' hebuf(1) = 0; */
      hebuf->data[0] = 0;
    }

    /*  Rotate counterclockwise order around vertex and insert vertices */
    /* 'obtain_nring_surf:103' while 1 */
    do {
      exitg4 = 0U;

      /*  Insert vertx into list */
      /* 'obtain_nring_surf:105' lid_prv = prv(lid); */
      lid = iv1[lid] - 1;

      /* 'obtain_nring_surf:106' v = tris(fid, lid_prv); */
      /* 'obtain_nring_surf:108' if nverts<maxnv && nfaces<maxnf */
      if ((*nverts < maxnv) && (*nfaces < maxnf)) {
        /* 'obtain_nring_surf:109' nverts = nverts + 1; */
        (*nverts)++;

        /* 'obtain_nring_surf:109' ngbvs( nverts) = v; */
        ngbvs->data[*nverts - 1] = tris->data[fid + tris->size[0] * lid];

        /* 'obtain_nring_surf:111' if ~oneringonly */
        /*  Save starting position for next vertex */
        /* 'obtain_nring_surf:113' hebuf(nverts) = opphes( fid, prv(lid_prv)); */
        hebuf->data[*nverts - 1] = opphes->data[fid + opphes->size[0] * (iv1[lid]
          - 1)];

        /* 'obtain_nring_surf:114' nfaces = nfaces + 1; */
        (*nfaces)++;

        /* 'obtain_nring_surf:114' ngbfs( nfaces) = fid; */
        ngbfs->data[*nfaces - 1] = fid + 1;
      } else {
        /* 'obtain_nring_surf:116' else */
        /* 'obtain_nring_surf:117' overflow = true; */
        overflow = TRUE;
      }

      /* 'obtain_nring_surf:120' opp = opphes(fid, lid_prv); */
      opp = opphes->data[fid + opphes->size[0] * lid];

      /* 'obtain_nring_surf:121' fid = heid2fid(opp); */
      /*  HEID2FID   Obtains face ID from half-edge ID. */
      /* 'heid2fid:3' coder.inline('always'); */
      /* 'heid2fid:4' fid = int32(bitshift(uint32(heid), -2)); */
      fid = (int32_T)((uint32_T)opphes->data[fid + opphes->size[0] * lid] >> 2U)
        - 1;

      /* 'obtain_nring_surf:123' if fid == fid_in */
      if (fid + 1 == fid_in) {
        exitg4 = 1U;
      } else {
        /* 'obtain_nring_surf:125' else */
        /* 'obtain_nring_surf:126' lid = heid2leid(opp); */
        /*  HEID2LEID   Obtains local edge ID within a face from half-edge ID. */
        /* 'heid2leid:3' coder.inline('always'); */
        /* 'heid2leid:4' leid = int32(bitand(uint32(heid),3))+1; */
        lid = (int32_T)((uint32_T)opp & 3U);
      }
    } while (exitg4 == 0U);

    /*  Finished cycle */
    /* 'obtain_nring_surf:130' if ring==1 && (nverts>=minpnts || nverts>=maxnv || nfaces>=maxnf || nargout<=2) */
    if ((ring == 1.0) && ((*nverts >= minpnts) || (*nverts >= maxnv) || (*nfaces
          >= maxnf))) {
      /* 'obtain_nring_surf:131' if overflow */
    } else {
      /* 'obtain_nring_surf:137' vtags(vid) = true; */
      vtags->data[vid - 1] = TRUE;

      /* 'obtain_nring_surf:138' for i=1:nverts */
      for (opp = 1; opp <= *nverts; opp++) {
        /* 'obtain_nring_surf:138' vtags(ngbvs(i))=true; */
        vtags->data[ngbvs->data[opp - 1] - 1] = TRUE;
      }

      /* 'obtain_nring_surf:139' for i=1:nfaces */
      for (opp = 1; opp <= *nfaces; opp++) {
        /* 'obtain_nring_surf:139' ftags(ngbfs(i))=true; */
        ftags->data[ngbfs->data[opp - 1] - 1] = TRUE;
      }

      /*  Define buffers and prepare tags for further processing */
      /* 'obtain_nring_surf:142' nverts_pre = int32(0); */
      nverts_pre = 0;

      /* 'obtain_nring_surf:143' nfaces_pre = int32(0); */
      nfaces_pre = 0;

      /*  Second, build full-size ring */
      /* 'obtain_nring_surf:146' ring_full = fix( ring); */
      ring_full = ring;
      b_fix(&ring_full);

      /* 'obtain_nring_surf:147' minpnts = min(minpnts, maxnv); */
      minpnts = minpnts <= maxnv ? minpnts : maxnv;

      /* 'obtain_nring_surf:149' cur_ring=1; */
      cur_ring = 1.0;

      /* 'obtain_nring_surf:150' while true */
      do {
        exitg1 = 0U;

        /* 'obtain_nring_surf:151' if cur_ring>ring_full || (cur_ring==ring_full && ring_full~=ring) */
        guard1 = FALSE;
        if ((cur_ring > ring_full) || ((cur_ring == ring_full) && (ring_full !=
              ring))) {
          /*  Collect halfring */
          /* 'obtain_nring_surf:153' nfaces_last = nfaces; */
          fid_in = *nfaces;

          /* 'obtain_nring_surf:153' nverts_last = nverts; */
          nverts_last = *nverts;

          /* 'obtain_nring_surf:154' for ii = nfaces_pre+1 : nfaces_last */
          while (nfaces_pre + 1 <= fid_in) {
            /*  take opposite vertex in opposite face */
            /* 'obtain_nring_surf:156' for jj=int32(1):3 */
            opp = 0;
            exitg2 = 0U;
            while ((exitg2 == 0U) && (opp + 1 < 4)) {
              /* 'obtain_nring_surf:157' oppe = opphes( ngbfs(ii), jj); */
              /* 'obtain_nring_surf:158' fid = heid2fid(oppe); */
              /*  HEID2FID   Obtains face ID from half-edge ID. */
              /* 'heid2fid:3' coder.inline('always'); */
              /* 'heid2fid:4' fid = int32(bitshift(uint32(heid), -2)); */
              fid = (int32_T)((uint32_T)opphes->data[(ngbfs->data[nfaces_pre] +
                opphes->size[0] * opp) - 1] >> 2U) - 1;

              /* 'obtain_nring_surf:160' if oppe && ~ftags(fid) */
              if ((opphes->data[(ngbfs->data[nfaces_pre] + opphes->size[0] * opp)
                   - 1] != 0) && (!ftags->data[fid])) {
                /* 'obtain_nring_surf:161' lid = heid2leid(oppe); */
                /*  HEID2LEID   Obtains local edge ID within a face from half-edge ID. */
                /* 'heid2leid:3' coder.inline('always'); */
                /* 'heid2leid:4' leid = int32(bitand(uint32(heid),3))+1; */
                lid = (int32_T)((uint32_T)opphes->data[(ngbfs->data[nfaces_pre]
                  + opphes->size[0] * opp) - 1] & 3U);

                /* 'obtain_nring_surf:162' v = tris( fid, prv(lid)); */
                /* 'obtain_nring_surf:164' overflow = overflow || ~vtags(v) && nverts>=length(ngbvs) || ... */
                /* 'obtain_nring_surf:165'                         ~ftags(fid) && nfaces>=length(ngbfs); */
                if (overflow || ((!vtags->data[tris->data[fid + tris->size[0] *
                                  (iv1[lid] - 1)] - 1]) && ((real_T)*nverts >=
                      length(ngbvs))) || ((!ftags->data[fid]) && ((real_T)
                      *nfaces >= length(ngbfs)))) {
                  overflow = TRUE;
                } else {
                  overflow = FALSE;
                }

                /* 'obtain_nring_surf:166' if ~ftags(fid) && ~overflow */
                if ((!ftags->data[fid]) && (!overflow)) {
                  /* 'obtain_nring_surf:167' nfaces = nfaces + 1; */
                  (*nfaces)++;

                  /* 'obtain_nring_surf:167' ngbfs( nfaces) = fid; */
                  ngbfs->data[*nfaces - 1] = fid + 1;

                  /* 'obtain_nring_surf:168' ftags(fid) = true; */
                  ftags->data[fid] = TRUE;
                }

                /* 'obtain_nring_surf:171' if ~vtags(v) && ~overflow */
                if ((!vtags->data[tris->data[fid + tris->size[0] * (iv1[lid] - 1)]
                     - 1]) && (!overflow)) {
                  /* 'obtain_nring_surf:172' nverts = nverts + 1; */
                  (*nverts)++;

                  /* 'obtain_nring_surf:172' ngbvs( nverts) = v; */
                  ngbvs->data[*nverts - 1] = tris->data[fid + tris->size[0] *
                    (iv1[lid] - 1)];

                  /* 'obtain_nring_surf:173' vtags(v) = true; */
                  vtags->data[tris->data[fid + tris->size[0] * (iv1[lid] - 1)] -
                    1] = TRUE;
                }

                exitg2 = 1U;
              } else {
                opp++;
              }
            }

            nfaces_pre++;
          }

          /* 'obtain_nring_surf:180' if nverts>=minpnts || nverts>=maxnv || nfaces>=maxnf || nfaces==nfaces_last */
          if ((*nverts >= minpnts) || (*nverts >= maxnv) || (*nfaces >= maxnf) ||
              (*nfaces == fid_in)) {
            exitg1 = 1U;
          } else {
            /* 'obtain_nring_surf:182' else */
            /*  If needs to expand, then undo the last half ring */
            /* 'obtain_nring_surf:184' for i=nverts_last+1:nverts */
            for (opp = nverts_last; opp + 1 <= *nverts; opp++) {
              /* 'obtain_nring_surf:184' vtags(ngbvs(i)) = false; */
              vtags->data[ngbvs->data[opp] - 1] = FALSE;
            }

            /* 'obtain_nring_surf:185' nverts = nverts_last; */
            *nverts = nverts_last;

            /* 'obtain_nring_surf:187' for i=nfaces_last+1:nfaces */
            for (opp = fid_in; opp + 1 <= *nfaces; opp++) {
              /* 'obtain_nring_surf:187' ftags(ngbfs(i)) = false; */
              ftags->data[ngbfs->data[opp] - 1] = FALSE;
            }

            /* 'obtain_nring_surf:188' nfaces = nfaces_last; */
            *nfaces = fid_in;
            guard1 = TRUE;
          }
        } else {
          guard1 = TRUE;
        }

        if (guard1 == TRUE) {
          /*  Collect next full level of ring */
          /* 'obtain_nring_surf:193' nverts_last = nverts; */
          nverts_last = *nverts;

          /* 'obtain_nring_surf:193' nfaces_pre = nfaces; */
          nfaces_pre = *nfaces;

          /* 'obtain_nring_surf:194' for ii=nverts_pre+1 : nverts_last */
          while (nverts_pre + 1 <= nverts_last) {
            /* 'obtain_nring_surf:195' v = ngbvs(ii); */
            /* 'obtain_nring_surf:195' fid = heid2fid(v2he(v)); */
            /*  HEID2FID   Obtains face ID from half-edge ID. */
            /* 'heid2fid:3' coder.inline('always'); */
            /* 'heid2fid:4' fid = int32(bitshift(uint32(heid), -2)); */
            fid = (int32_T)((uint32_T)v2he->data[ngbvs->data[nverts_pre] - 1] >>
                            2U) - 1;

            /* 'obtain_nring_surf:195' lid = heid2leid(v2he(v)); */
            /*  HEID2LEID   Obtains local edge ID within a face from half-edge ID. */
            /* 'heid2leid:3' coder.inline('always'); */
            /* 'heid2leid:4' leid = int32(bitand(uint32(heid),3))+1; */
            lid = (int32_T)((uint32_T)v2he->data[ngbvs->data[nverts_pre] - 1] &
                            3U);

            /*  Allow early termination of the loop if an incident halfedge */
            /*  was recorded and the vertex is not incident on a border halfedge */
            /* 'obtain_nring_surf:199' allow_early_term = hebuf(ii) && opphes(fid,lid); */
            if ((hebuf->data[nverts_pre] != 0) && (opphes->data[fid +
                 opphes->size[0] * lid] != 0)) {
              b0 = TRUE;
            } else {
              b0 = FALSE;
            }

            /* 'obtain_nring_surf:200' if allow_early_term */
            if (b0) {
              /* 'obtain_nring_surf:201' fid = heid2fid(hebuf(ii)); */
              /*  HEID2FID   Obtains face ID from half-edge ID. */
              /* 'heid2fid:3' coder.inline('always'); */
              /* 'heid2fid:4' fid = int32(bitshift(uint32(heid), -2)); */
              fid = (int32_T)((uint32_T)hebuf->data[nverts_pre] >> 2U) - 1;

              /* 'obtain_nring_surf:201' lid = heid2leid(hebuf(ii)); */
              /*  HEID2LEID   Obtains local edge ID within a face from half-edge ID. */
              /* 'heid2leid:3' coder.inline('always'); */
              /* 'heid2leid:4' leid = int32(bitand(uint32(heid),3))+1; */
              lid = (int32_T)((uint32_T)hebuf->data[nverts_pre] & 3U);
            }

            /*  */
            /* 'obtain_nring_surf:205' if opphes( fid, lid) */
            if (opphes->data[fid + opphes->size[0] * lid] != 0) {
              /* 'obtain_nring_surf:206' fid_in = fid; */
              fid_in = fid + 1;
            } else {
              /* 'obtain_nring_surf:207' else */
              /* 'obtain_nring_surf:208' fid_in = cast(0,class(fid)); */
              fid_in = 0;

              /* 'obtain_nring_surf:210' v = tris(fid, nxt(lid)); */
              /* 'obtain_nring_surf:211' overflow = overflow || ~vtags(v) && nverts>=length(ngbvs); */
              if (overflow || ((!vtags->data[tris->data[fid + tris->size[0] *
                                (iv0[lid] - 1)] - 1]) && ((real_T)*nverts >=
                    length(ngbvs)))) {
                overflow = TRUE;
              } else {
                overflow = FALSE;
              }

              /* 'obtain_nring_surf:212' if ~overflow */
              if (!overflow) {
                /* 'obtain_nring_surf:213' nverts = nverts + 1; */
                (*nverts)++;

                /* 'obtain_nring_surf:213' ngbvs( nverts) = v; */
                ngbvs->data[*nverts - 1] = tris->data[fid + tris->size[0] *
                  (iv0[lid] - 1)];

                /* 'obtain_nring_surf:213' vtags(v)=true; */
                vtags->data[tris->data[fid + tris->size[0] * (iv0[lid] - 1)] - 1]
                  = TRUE;

                /*  Save starting position for next vertex */
                /* 'obtain_nring_surf:215' hebuf(nverts) = 0; */
                hebuf->data[*nverts - 1] = 0;
              }
            }

            /*  Rotate counterclockwise around the vertex. */
            /* 'obtain_nring_surf:220' isfirst=true; */
            isfirst = TRUE;

            /* 'obtain_nring_surf:221' while true */
            do {
              exitg3 = 0U;

              /*  Insert vertx into list */
              /* 'obtain_nring_surf:223' lid_prv = prv(lid); */
              lid = iv1[lid] - 1;

              /*  Insert face into list */
              /* 'obtain_nring_surf:226' if ftags(fid) */
              guard2 = FALSE;
              if (ftags->data[fid]) {
                /* 'obtain_nring_surf:227' if allow_early_term && ~isfirst */
                if (b0 && (!isfirst)) {
                  exitg3 = 1U;
                } else {
                  guard2 = TRUE;
                }
              } else {
                /* 'obtain_nring_surf:228' else */
                /*  If the face has already been inserted, then the vertex */
                /*  must be inserted already. */
                /* 'obtain_nring_surf:231' v = tris(fid, lid_prv); */
                /* 'obtain_nring_surf:232' overflow = overflow || ~vtags(v) && nverts>=length(ngbvs) || ... */
                /* 'obtain_nring_surf:233'                     ~ftags(fid) && nfaces>=length(ngbfs); */
                b_guard1 = FALSE;
                b_guard2 = FALSE;
                if (overflow || ((!vtags->data[tris->data[fid + tris->size[0] *
                                  lid] - 1]) && ((real_T)*nverts >= length(ngbvs))))
                {
                  b_guard2 = TRUE;
                } else if (!ftags->data[fid]) {
                  if (ngbfs->size[0] == 0) {
                    opp = 0;
                  } else if (ngbfs->size[0] > 1) {
                    opp = ngbfs->size[0];
                  } else {
                    opp = 1;
                  }

                  if (*nfaces >= opp) {
                    b_guard2 = TRUE;
                  } else {
                    b_guard1 = TRUE;
                  }
                } else {
                  b_guard1 = TRUE;
                }

                if (b_guard2 == TRUE) {
                  overflow = TRUE;
                }

                if (b_guard1 == TRUE) {
                  overflow = FALSE;
                }

                /* 'obtain_nring_surf:235' if ~vtags(v) && ~overflow */
                if ((!vtags->data[tris->data[fid + tris->size[0] * lid] - 1]) &&
                    (!overflow)) {
                  /* 'obtain_nring_surf:236' nverts = nverts + 1; */
                  (*nverts)++;

                  /* 'obtain_nring_surf:236' ngbvs( nverts) = v; */
                  ngbvs->data[*nverts - 1] = tris->data[fid + tris->size[0] *
                    lid];

                  /* 'obtain_nring_surf:236' vtags(v)=true; */
                  vtags->data[tris->data[fid + tris->size[0] * lid] - 1] = TRUE;

                  /*  Save starting position for next ring */
                  /* 'obtain_nring_surf:239' hebuf(nverts) = opphes( fid, prv(lid_prv)); */
                  hebuf->data[*nverts - 1] = opphes->data[fid + opphes->size[0] *
                    (iv1[lid] - 1)];
                }

                /* 'obtain_nring_surf:242' if ~ftags(fid) && ~overflow */
                if ((!ftags->data[fid]) && (!overflow)) {
                  /* 'obtain_nring_surf:243' nfaces = nfaces + 1; */
                  (*nfaces)++;

                  /* 'obtain_nring_surf:243' ngbfs( nfaces) = fid; */
                  ngbfs->data[*nfaces - 1] = fid + 1;

                  /* 'obtain_nring_surf:243' ftags(fid)=true; */
                  ftags->data[fid] = TRUE;
                }

                /* 'obtain_nring_surf:245' isfirst = false; */
                isfirst = FALSE;
                guard2 = TRUE;
              }

              if (guard2 == TRUE) {
                /* 'obtain_nring_surf:248' opp = opphes(fid, lid_prv); */
                opp = opphes->data[fid + opphes->size[0] * lid];

                /* 'obtain_nring_surf:249' fid = heid2fid(opp); */
                /*  HEID2FID   Obtains face ID from half-edge ID. */
                /* 'heid2fid:3' coder.inline('always'); */
                /* 'heid2fid:4' fid = int32(bitshift(uint32(heid), -2)); */
                fid = (int32_T)((uint32_T)opphes->data[fid + opphes->size[0] *
                                lid] >> 2U) - 1;

                /* 'obtain_nring_surf:251' if fid == fid_in */
                if (fid + 1 == fid_in) {
                  /*  Finished cycle */
                  exitg3 = 1U;
                } else {
                  /* 'obtain_nring_surf:253' else */
                  /* 'obtain_nring_surf:254' lid = heid2leid(opp); */
                  /*  HEID2LEID   Obtains local edge ID within a face from half-edge ID. */
                  /* 'heid2leid:3' coder.inline('always'); */
                  /* 'heid2leid:4' leid = int32(bitand(uint32(heid),3))+1; */
                  lid = (int32_T)((uint32_T)opp & 3U);
                }
              }
            } while (exitg3 == 0U);

            nverts_pre++;
          }

          /* 'obtain_nring_surf:259' cur_ring = cur_ring+1; */
          cur_ring++;

          /* 'obtain_nring_surf:260' if (nverts>=minpnts && cur_ring>=ring) || nfaces==nfaces_pre || overflow */
          if (((*nverts >= minpnts) && (cur_ring >= ring)) || (*nfaces ==
               nfaces_pre) || overflow) {
            exitg1 = 1U;
          } else {
            /* 'obtain_nring_surf:264' nverts_pre = nverts_last; */
            nverts_pre = nverts_last;
          }
        }
      } while (exitg1 == 0U);

      /*  Reset flags */
      /* 'obtain_nring_surf:268' vtags(vid) = false; */
      vtags->data[vid - 1] = FALSE;

      /* 'obtain_nring_surf:269' for i=1:nverts */
      for (opp = 1; opp <= *nverts; opp++) {
        /* 'obtain_nring_surf:269' vtags(ngbvs(i))=false; */
        vtags->data[ngbvs->data[opp - 1] - 1] = FALSE;
      }

      /* 'obtain_nring_surf:270' if ~oneringonly */
      /* 'obtain_nring_surf:270' for i=1:nfaces */
      for (opp = 1; opp <= *nfaces; opp++) {
        /* 'obtain_nring_surf:270' ftags(ngbfs(i))=false; */
        ftags->data[ngbfs->data[opp - 1] - 1] = FALSE;
      }

      /* 'obtain_nring_surf:271' if overflow */
    }

    emxFree_int32_T(&hebuf);
  }
}

