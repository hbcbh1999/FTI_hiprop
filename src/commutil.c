#include "commutil.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void sendND_boolean_T(emxArray_boolean_T *array_send, int dst, int tag, MPI_Comm comm)
{
    int i;
    
    int32_T num_elem_regular = array_send->numDimensions + 2;
    int32_T *common_info = (int32_T *)calloc((uint32_T) num_elem_regular, sizeof(int32_T));

    common_info[0] = array_send->numDimensions;
    for (i = 1; i <= array_send->numDimensions; i++)
	common_info[i] = array_send->size[i-1];
    common_info[i] = array_send->allocatedSize;

    MPI_Send(&num_elem_regular, 1, MPI_INT, dst, tag, comm);
    MPI_Send(common_info, num_elem_regular, MPI_INT, dst, tag, comm);
    MPI_Send(array_send->data, array_send->allocatedSize, MPI_UNSIGNED_CHAR, dst, tag, comm);

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

    MPI_Send(&num_elem_regular, 1, MPI_INT, dst, tag, comm);
    MPI_Send(common_info, num_elem_regular, MPI_INT, dst, tag, comm);
    MPI_Send(array_send->data, array_send->allocatedSize, MPI_INT, dst, tag, comm);

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

    MPI_Send(&num_elem_regular, 1, MPI_INT, dst, tag, comm);
    MPI_Send(common_info, num_elem_regular, MPI_INT, dst, tag, comm);
    MPI_Send(array_send->data, array_send->allocatedSize, MPI_DOUBLE, dst, tag, comm);

    free((void *)common_info);
}

void recvND_boolean_T(emxArray_boolean_T **array_recv, int src, int tag, MPI_Comm comm)
{
    int i;
    int num_elem_regular;
    MPI_Status recv_stat_1;
    MPI_Status recv_stat_2;
    MPI_Status recv_stat_3;

    MPI_Recv(&num_elem_regular, 1, MPI_INT, src, tag, comm, &recv_stat_1);

    int32_T *common_info = (int32_T *)calloc((uint32_T) num_elem_regular, sizeof(int32_T));

    MPI_Recv(common_info, num_elem_regular, MPI_INT, src, tag, comm, &recv_stat_2);

    int num_dim = common_info[0];
    int num_elem = common_info[num_elem_regular-1];

    int32_T *size = (int32_T *)calloc((uint32_T) num_dim, sizeof(int32_T));

    for (i = 1; i <= num_dim; i++)
	size[i] = common_info[i];

    emxArray_boolean_T *result = emxCreateND_boolean_T(num_dim, size);

    MPI_Recv(result->data, num_elem, MPI_UNSIGNED_CHAR, src, tag, comm, &recv_stat_3);

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

    MPI_Recv(&num_elem_regular, 1, MPI_INT, src, tag, comm, &recv_stat_1);

    int32_T *common_info = (int32_T *)calloc((uint32_T) num_elem_regular, sizeof(int32_T));

    MPI_Recv(common_info, num_elem_regular, MPI_INT, src, tag, comm, &recv_stat_2);

    int num_dim = common_info[0];
    int num_elem = common_info[num_elem_regular-1];

    int32_T *size = (int32_T *)calloc((uint32_T) num_dim, sizeof(int32_T));

    for (i = 1; i <= num_dim; i++)
	size[i] = common_info[i];

    emxArray_int32_T *result = emxCreateND_int32_T(num_dim, size);

    MPI_Recv(result->data, num_elem, MPI_INT, src, tag, comm, &recv_stat_3);

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

    MPI_Recv(&num_elem_regular, 1, MPI_INT, src, tag, comm, &recv_stat_1);

    int32_T *common_info = (int32_T *)calloc((uint32_T) num_elem_regular, sizeof(int32_T));

    MPI_Recv(common_info, num_elem_regular, MPI_INT, src, tag, comm, &recv_stat_2);

    int num_dim = common_info[0];
    int num_elem = common_info[num_elem_regular-1];

    int32_T *size = (int32_T *)calloc((uint32_T) num_dim, sizeof(int32_T));

    for (i = 1; i <= num_dim; i++)
	size[i] = common_info[i];

    emxArray_real_T *result = emxCreateND_real_T(num_dim, size);

    MPI_Recv(result->data, num_elem, MPI_DOUBLE, src, tag, comm, &recv_stat_3);

    free ((void *) common_info);
    free ((void *) size);

    (*array_recv) = result;
}

void send2D_boolean_T(emxArray_boolean_T *array_send, int dst, int tag, MPI_Comm comm)
{
    /*int32_T *common_info = (int32_T *)calloc(2, sizeof(int32_T));*/
    int32_T common_info[2];

    common_info[0] = array_send->size[0];
    common_info[1] = array_send->size[1];

    MPI_Send(common_info, 2, MPI_INT, dst, tag, comm);
    MPI_Send(array_send->data, array_send->allocatedSize, MPI_UNSIGNED_CHAR, dst, tag, comm);
}

void send2D_int32_T(emxArray_int32_T *array_send, int dst, int tag, MPI_Comm comm)
{
    /*int32_T *common_info = (int32_T *)calloc(2, sizeof(int32_T));*/
    int32_T common_info[2];

    common_info[0] = array_send->size[0];
    common_info[1] = array_send->size[1];

    MPI_Send(common_info, 2, MPI_INT, dst, tag, comm);
    MPI_Send(array_send->data, array_send->allocatedSize, MPI_INT, dst, tag, comm);

}

void send2D_real_T(emxArray_real_T *array_send, int dst, int tag, MPI_Comm comm)
{
    /*int32_T *common_info = (int32_T *)calloc(2, sizeof(int32_T));*/
    int32_T common_info[2];

    common_info[0] = array_send->size[0];
    common_info[1] = array_send->size[1];

    MPI_Send(common_info, 2, MPI_INT, dst, tag, comm);
    MPI_Send(array_send->data, array_send->allocatedSize, MPI_DOUBLE, dst, tag, comm);

}
void recv2D_boolean_T(emxArray_boolean_T **array_recv, int src, int tag, MPI_Comm comm)
{
    MPI_Status recv_stat_1;
    MPI_Status recv_stat_2;

    int32_T common_info[2];

    MPI_Recv(common_info, 2, MPI_INT, src, tag, comm, &recv_stat_1);

    emxArray_boolean_T *result = emxCreate_boolean_T(common_info[0], common_info[1]);

    MPI_Recv(result->data, common_info[0]*common_info[1], MPI_UNSIGNED_CHAR, src, tag, comm, &recv_stat_2);

    (*array_recv) = result;
}


void recv2D_int32_T(emxArray_int32_T **array_recv, int src, int tag, MPI_Comm comm)
{
    MPI_Status recv_stat_1;
    MPI_Status recv_stat_2;

    int32_T common_info[2];

    MPI_Recv(common_info, 2, MPI_INT, src, tag, comm, &recv_stat_1);

    emxArray_int32_T *result = emxCreate_int32_T(common_info[0], common_info[1]);

    MPI_Recv(result->data, common_info[0]*common_info[1], MPI_INT, src, tag, comm, &recv_stat_2);

    (*array_recv) = result;
}

void recv2D_real_T(emxArray_real_T **array_recv, int src, int tag, MPI_Comm comm)
{
    MPI_Status recv_stat_1;
    MPI_Status recv_stat_2;

    int32_T common_info[2];

    MPI_Recv(common_info, 2, MPI_INT, src, tag, comm, &recv_stat_1);

    emxArray_real_T *result = emxCreate_real_T(common_info[0], common_info[1]);

    MPI_Recv(result->data, common_info[0]*common_info[1], MPI_DOUBLE, src, tag, comm, &recv_stat_2);

    (*array_recv) = result;
}





