/*!
 * \file util.c
 * \brief Implementation of util.h 
 *
 * \author Yijie Zhou
 * \date 2012.10.01
 */

#include "util.h"


void right_flush(int n, int ndigits, char *s)
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

int findString(
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

void addColumnToArray_common(emxArray__common *emxArray, int32_T numCol, uint32_T elementSize)
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

void addColumnToArray_int32_T(emxArray_int32_T *emxArray, int32_T numCol)
{
    addColumnToArray_common( (emxArray__common *)emxArray, numCol, sizeof(int32_T) );
    emxArray->data = (int32_T *) emxArray->data;
}
void addColumnToArray_real_T(emxArray_real_T *emxArray, int32_T numCol)
{
    addColumnToArray_common( (emxArray__common *)emxArray, numCol, sizeof(real_T) );
    emxArray->data = (real_T *) emxArray->data;
}
void addColumnToArray_boolean_T(emxArray_boolean_T *emxArray, int32_T numCol)
{
    addColumnToArray_common( (emxArray__common *)emxArray, numCol, sizeof(boolean_T) );
    emxArray->data = (boolean_T *) emxArray->data;
}

void addRowToArray_int32_T(emxArray_int32_T *emxArray, int32_T numRow)
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
void addRowToArray_real_T(emxArray_real_T *emxArray, int32_T numRow)
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
void addRowToArray_boolean_T(emxArray_boolean_T *emxArray, int32_T numRow)
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


void sendND_boolean_T(emxArray_boolean_T *array_send, int dst, int tag, MPI_Comm comm)
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

void sendND_int32_T(emxArray_int32_T *array_send, int dst, int tag, MPI_Comm comm)
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

void sendND_real_T(emxArray_real_T *array_send, int dst, int tag, MPI_Comm comm)
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

void recvND_boolean_T(emxArray_boolean_T **array_recv, int src, int tag, MPI_Comm comm)
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

void recvND_int32_T(emxArray_int32_T **array_recv, int src, int tag, MPI_Comm comm)
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

void recvND_real_T(emxArray_real_T **array_recv, int src, int tag, MPI_Comm comm)
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

void send2D_boolean_T(emxArray_boolean_T *array_send, int dst, int tag, MPI_Comm comm)
{
    MPI_Send(array_send->size, 2, MPI_INT, dst, tag+1, comm);
    MPI_Send(array_send->data, array_send->allocatedSize, MPI_UNSIGNED_CHAR, dst, tag+2, comm);
}

void send2D_int32_T(emxArray_int32_T *array_send, int dst, int tag, MPI_Comm comm)
{
    MPI_Send(array_send->size, 2, MPI_INT, dst, tag+1, comm);
    MPI_Send(array_send->data, array_send->allocatedSize, MPI_INT, dst, tag+2, comm);

}

void send2D_real_T(emxArray_real_T *array_send, int dst, int tag, MPI_Comm comm)
{
    MPI_Send(array_send->size, 2, MPI_INT, dst, tag+1, comm);
    MPI_Send(array_send->data, array_send->allocatedSize, MPI_DOUBLE, dst, tag+2, comm);

}
void recv2D_boolean_T(emxArray_boolean_T **array_recv, int src, int tag, MPI_Comm comm)
{
    MPI_Status recv_stat_1;
    MPI_Status recv_stat_2;

    int32_T common_info[2];

    MPI_Recv(common_info, 2, MPI_INT, src, tag+1, comm, &recv_stat_1);

    emxArray_boolean_T *result = emxCreate_boolean_T(common_info[0], common_info[1]);

    MPI_Recv(result->data, common_info[0]*common_info[1], MPI_UNSIGNED_CHAR, src, tag+2, comm, &recv_stat_2);

    (*array_recv) = result;
}


void recv2D_int32_T(emxArray_int32_T **array_recv, int src, int tag, MPI_Comm comm)
{
    MPI_Status recv_stat_1;
    MPI_Status recv_stat_2;

    int32_T common_info[2];

    MPI_Recv(common_info, 2, MPI_INT, src, tag+1, comm, &recv_stat_1);

    emxArray_int32_T *result = emxCreate_int32_T(common_info[0], common_info[1]);

    MPI_Recv(result->data, common_info[0]*common_info[1], MPI_INT, src, tag+2, comm, &recv_stat_2);

    (*array_recv) = result;
}

void recv2D_real_T(emxArray_real_T **array_recv, int src, int tag, MPI_Comm comm)
{
    MPI_Status recv_stat_1;
    MPI_Status recv_stat_2;

    int32_T common_info[2];

    MPI_Recv(common_info, 2, MPI_INT, src, tag+1, comm, &recv_stat_1);

    emxArray_real_T *result = emxCreate_real_T(common_info[0], common_info[1]);

    MPI_Recv(result->data, common_info[0]*common_info[1], MPI_DOUBLE, src, tag+2, comm, &recv_stat_2);

    (*array_recv) = result;
}

void isend2D_boolean_T(emxArray_boolean_T *array_send, int dst, int tag, MPI_Comm comm, MPI_Request *req_com, MPI_Request *req_data)
{
    MPI_Isend(array_send->size, 2, MPI_INT, dst, tag+1, comm, req_com);
    MPI_Isend(array_send->data, array_send->allocatedSize, MPI_UNSIGNED_CHAR, dst, tag+2, comm, req_data); 
}


void isend2D_int32_T(emxArray_int32_T *array_send, int dst, int tag, MPI_Comm comm, MPI_Request *req_com, MPI_Request *req_data)
{
    MPI_Isend(array_send->size, 2, MPI_INT, dst, tag+1, comm, req_com);
    MPI_Isend(array_send->data, array_send->allocatedSize, MPI_INT, dst, tag+2, comm, req_data); 
}

void isend2D_real_T(emxArray_real_T *array_send, int dst, int tag, MPI_Comm comm, MPI_Request *req_com, MPI_Request *req_data)
{
    MPI_Isend(array_send->size, 2, MPI_INT, dst, tag+1, comm, req_com);
    MPI_Isend(array_send->data, array_send->allocatedSize, MPI_DOUBLE, dst, tag+2, comm, req_data); 
}
