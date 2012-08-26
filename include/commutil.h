/*!
 * \file commutil.h
 * \brief Array communicating functions for arrays defined in datatype.h
 *
 * When sending arrays, first pack numDimensions -> size -> allocatedSize into
 * an int32_T array with size 1 + numDimensions + 1 and send to destination
 * processor. Then send data with length allocatedSize.
 *
 * When receiving arrays, first receive the general information and unpack. Then
 * receive the data with allocated space.
 *
 * \author Yijie Zhou
 * \date 2012.08.23
 */



#ifndef __COMMUTIL_H__
#define __COMMUTIL_H__

#include "stdafx.h"
#include "memutil.h"


extern void sendND_boolean_T(emxArray_boolean_T *array_send, int dst, int tag, MPI_Comm comm);
extern void sendND_int32_T(emxArray_int32_T *array_send, int dst, int tag, MPI_Comm comm);
extern void sendND_real_T(emxArray_real_T *array_send, int dst, int tag, MPI_Comm comm);


extern void recvND_boolean_T(emxArray_boolean_T **array_recv, int src, int tag, MPI_Comm comm);
extern void recvND_int32_T(emxArray_int32_T **array_recv, int src, int tag, MPI_Comm comm);
extern void recvND_real_T(emxArray_real_T **array_recv, int src, int tag, MPI_Comm comm);


extern void send2D_boolean_T(emxArray_boolean_T *array_send, int dst, int tag, MPI_Comm comm);
extern void send2D_int32_T(emxArray_int32_T *array_send, int dst, int tag, MPI_Comm comm);
extern void send2D_real_T(emxArray_real_T *array_send, int dst, int tag, MPI_Comm comm);


extern void recv2D_boolean_T(emxArray_boolean_T **array_recv, int src, int tag, MPI_Comm comm);
extern void recv2D_int32_T(emxArray_int32_T **array_recv, int src, int tag, MPI_Comm comm);
extern void recv2D_real_T(emxArray_real_T **array_recv, int src, int tag, MPI_Comm comm);

#endif
