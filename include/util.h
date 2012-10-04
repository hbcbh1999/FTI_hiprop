/*!
 * \file util.h
 * \brief basic utility functions for communication and I/O, etc.
 *
 * \detail 
 * I/O: Currently support reading and writing ascii triangular mesh
 * for POLYGON and UNSTRUCTURED_GRID.
 *
 * Communication: Array communicating functions for arrays defined in datatype.h
 * When sending N-dimensional arrays, first pack numDimensions -> size -> allocatedSize 
 * into an int32_T array with size 1 + numDimensions + 1 and send to destination
 * processor. Then send data with length allocatedSize.
 * When sending 2D arrays, first pack # of rows and # of columns into an int32_T
 * array with size 2 and send to destination processor. Then send data with
 * length allocatedSize. Both blocking and nonblocking send are supported.
 * When receiving arrays, first receive the general information and unpack. Then
 * receive the data with allocated space. The receiving function would do the
 * memory allocation.
 * 
 * \author Yijie Zhou, Chenzhe Diao 
 *
 * \date 2012.10.01
 */




#ifndef __UTIL_H__
#define __UTIL_H__

#include "emx_util.h"

/*!
 * Compute the index of A[i] in a 1D array
 * \param i index starts from 1
 */
#define I1dm(i) (i-1)

/*!
 * Compute the index of A[i][j] in a 2D array
 * \param i index in dimension 1 (starts from 1)
 * \param j index in dimension 2 (starts from 1)
 * \param size dimension information
 */
#define I2dm(i,j,size) ((j-1)*(size[0])+(i-1))

/*!
 * Compute the index of A[i][j][k] in a 3D array
 */
#define I3dm(i,j,k,size) (((k-1)*(size[1])+(j-1))*(size[0])+(i-1))

/*!
 * \brief Transform an int number into string using certain number of digits
 * \param n int number that is being transformed
 * \param ndigits number of digits for the output
 * \param in_string output string
 */
extern void right_flush(const int n, const int ndigits, char *in_string);

/*!
 * \brief Locate the current cursor after the searching string, if string not found,
 * return 0.
 * \param file file pointer
 * \param in_string string for search
 */
extern int findString(FILE* file, const char* in_string);

extern boolean_T sameTriangle(const emxArray_real_T* ps1,
			      const emxArray_int32_T* tri1,
			      const int tri_index1,
			      const emxArray_real_T* ps2,
			      const emxArray_int32_T* tri2,
			      const int tri_index2,
			      const double eps);

/*
inline int32_T I2d(int32_T i, int32_T j, int32_T *size)
{
    return (j-1)*size[0] + (i-1);
}
inline int32_T I3d(int32_T i, int32_T j, int32_T k, int32_T *size)
{
    return ((k-1)*size[1] + (j-1))*size[0] + (i-1);
}
*/


/*!
 * \brief Add column to common type array
 * 
 * \param emxArray pointer to common type array
 * \param numColAdd number of column added
 * \param elementSize size of each element
 */
extern void addColumnToArray_common(emxArray__common *emxArray, const int32_T numColAdd, const uint32_T elementSize);

/*!
 * \brief Add column to int array
 * 
 * \param emxArray pointer to int array
 * \param numColAdd number of column added
 */
extern void addColumnToArray_int32_T(emxArray_int32_T *emxArray, const int32_T numColAdd);

/*!
 * \brief Add column to real array
 * 
 * \param emxArray pointer to real array
 * \param numColAdd number of column added
 */
extern void addColumnToArray_real_T(emxArray_real_T *emxArray, const int32_T numColAdd);

/*!
 * \brief Add column to boolean array
 * 
 * \param emxArray pointer to boolean array
 * \param numColAdd number of column added
 */
extern void addColumnToArray_boolean_T(emxArray_boolean_T *emxArray, const int32_T numColAdd);

/*!
 * \brief Add row to int array
 * 
 * \param emxArray pointer to int array
 * \param numColAdd number of row added
 */
extern void addRowToArray_int32_T(emxArray_int32_T *emxArray, const int32_T numRowAdd);

/*!
 * \brief Add row to real array
 * 
 * \param emxArray pointer to real array
 * \param numColAdd number of row added
 */
extern void addRowToArray_real_T(emxArray_real_T *emxArray, const int32_T numRowAdd);

/*!
 * \brief Add row to boolean array
 * 
 * \param emxArray pointer to boolean array
 * \param numColAdd number of row added
 */
extern void addRowToArray_boolean_T(emxArray_boolean_T *emxArray, const int32_T numRowAdd);

extern void printArray_int32_T(const emxArray_int32_T *emxArray);
extern void printArray_real_T(const emxArray_real_T *emxArray);
extern void printArray_boolean_T(const emxArray_boolean_T *emxArray);

/*!
 * \brief Block sending a N-dimensional boolean array
 * \detail In the function we first send the common information including number
 * of dimensions and number of elements in each dimension, then we send the data
 * member of the array. The common info has label = tag + 1 and data info has
 * label tag + 2.
 *
 * \param array_send pointer to the array being sent
 * \param dst destination processor ID
 * \param tag tag for this non blocking send
 * \param comm MPI communicator
 */
extern void sendND_boolean_T(const emxArray_boolean_T *array_send, const int dst, const int tag, MPI_Comm comm);

/*!
 * \brief Block sending a N-dimensional int array
 * \detail In the function we first send the common information including number
 * of dimensions and number of elements in each dimension, then we send the data
 * member of the array. The common info has label = tag + 1 and data info has
 * label tag + 2.
 *
 * \param array_send pointer to the array being sent
 * \param dst destination processor ID
 * \param tag tag for this non blocking send
 * \param comm MPI communicator
 */
extern void sendND_int32_T(const emxArray_int32_T *array_send, const int dst, const int tag, MPI_Comm comm);

/*!
 * \brief Block sending a N-dimensional double array
 * \detail In the function we first send the common information including number
 * of dimensions and number of elements in each dimension, then we send the data
 * member of the array. The common info has label = tag + 1 and data info has
 * label tag + 2.
 *
 * \param array_send pointer to the array being sent
 * \param dst destination processor ID
 * \param tag tag for this non blocking send
 * \param comm MPI communicator
 */
extern void sendND_real_T(const emxArray_real_T *array_send, const int dst, const int tag, MPI_Comm comm);

/*!
 * \brief Block receiving a N-dimensional boolean array
 * \detail In the function we first receive the common information including number
 * of dimensions and number of elements in each dimension, then we create the array based
 * on the common info and then receive the data member of the array. 
 * The common info has label = tag + 1 and data info has label tag + 2.
 *
 * \param array_recv pointer to the array being received
 * \param dst source processor ID
 * \param tag tag for this non blocking send
 * \param comm MPI communicator
 */
extern void recvND_boolean_T(emxArray_boolean_T **array_recv, const int src, const int tag, MPI_Comm comm);

/*!
 * \brief Block receiving a N-dimensional int array
 * \detail In the function we first receive the common information including number
 * of dimensions and number of elements in each dimension, then we create the array based
 * on the common info and then receive the data member of the array. 
 * The common info has label = tag + 1 and data info has label tag + 2.
 *
 * \param array_recv pointer to the array being received
 * \param dst source processor ID
 * \param tag tag for this non blocking send
 * \param comm MPI communicator
 */
extern void recvND_int32_T(emxArray_int32_T **array_recv, const int src, const int tag, MPI_Comm comm);

/*!
 * \brief Block receiving a N-dimensional double array
 * \detail In the function we first receive the common information including number
 * of dimensions and number of elements in each dimension, then we create the array based
 * on the common info and then receive the data member of the array. 
 * The common info has label = tag + 1 and data info has label tag + 2.
 *
 * \param array_recv pointer to the array being received
 * \param dst source processor ID
 * \param tag tag for this non blocking send
 * \param comm MPI communicator
 */
extern void recvND_real_T(emxArray_real_T **array_recv, const int src, const int tag, MPI_Comm comm);

/*!
 * \brief Block sending a 2D boolean array
 * \detail In the function we first send the row and column number, then we send the data
 * member of the array. The common info has label = tag + 1 and data info has label tag + 2.
 *
 * \param array_send pointer to the array being sent
 * \param dst destination processor ID
 * \param tag tag for this non blocking send
 * \param comm MPI communicator
 */
extern void send2D_boolean_T(const emxArray_boolean_T *array_send, const int dst, const int tag, MPI_Comm comm);

/*!
 * \brief Block sending a 2D int array
 * \detail In the function we first send the row and column number, then we send the data
 * member of the array. The common info has label = tag + 1 and data info has label tag + 2.
 *
 * \param array_send pointer to the array being sent
 * \param dst destination processor ID
 * \param tag tag for this non blocking send
 * \param comm MPI communicator
 */
extern void send2D_int32_T(const emxArray_int32_T *array_send, const int dst, const int tag, MPI_Comm comm);

/*!
 * \brief Block sending a 2D double array
 * \detail In the function we first send the row and column number, then we send the data
 * member of the array. The common info has label = tag + 1 and data info has label tag + 2.
 *
 * \param array_send pointer to the array being sent
 * \param dst destination processor ID
 * \param tag tag for this non blocking send
 * \param comm MPI communicator
 */
extern void send2D_real_T(const emxArray_real_T *array_send, const int dst, const int tag, MPI_Comm comm);

/*!
 * \brief Block receiving a 2D boolean array
 * \detail In the function we first receive the row and column number, 
 * then we create the array and receive the data member of the array.
 * The common info has label = tag + 1 and data info has label tag + 2.
 *
 * \param array_recv pointer to the array being received
 * \param dst source processor ID
 * \param tag tag for this non blocking send
 * \param comm MPI communicator
 */
extern void recv2D_boolean_T(emxArray_boolean_T **array_recv, const int src, const int tag, MPI_Comm comm);

/*!
 * \brief Block receiving a 2D int array
 * \detail In the function we first receive the row and column number, 
 * then we create the array and receive the data member of the array.
 * The common info has label = tag + 1 and data info has label tag + 2.
 *
 * \param array_recv pointer to the array being received
 * \param dst source processor ID
 * \param tag tag for this non blocking send
 * \param comm MPI communicator
 */
extern void recv2D_int32_T(emxArray_int32_T **array_recv, const int src, const int tag, MPI_Comm comm);

/*!
 * \brief Block receiving a 2D double array
 * \detail In the function we first receive the row and column number, 
 * then we create the array and receive the data member of the array.
 * The common info has label = tag + 1 and data info has label tag + 2.
 *
 * \param array_recv pointer to the array being received
 * \param dst source processor ID
 * \param tag tag for this non blocking send
 * \param comm MPI communicator
 */
extern void recv2D_real_T(emxArray_real_T **array_recv, const int src, const int tag, MPI_Comm comm);

/*!
 * \brief NonBlock receiving a 2D boolean array
 * \detail In the function we first receive the row and column number, 
 * then we create the array and receive the data member of the array.
 * The common info has label = tag + 1 and data info has label tag + 2.
 *
 * \param array_recv pointer to the array being received
 * \param dst source processor ID
 * \param tag tag for this non blocking send
 * \param comm MPI communicator
 */
extern void isend2D_boolean_T(const emxArray_boolean_T *array_send, const int dst, const int tag,
			      MPI_Comm comm, MPI_Request *req_com, MPI_Request *req_data);

/*!
 * \brief NonBlock receiving a 2D int array
 * \detail In the function we first receive the row and column number, 
 * then we create the array and receive the data member of the array.
 * The common info has label = tag + 1 and data info has label tag + 2.
 *
 * \param array_recv pointer to the array being received
 * \param dst source processor ID
 * \param tag tag for this non blocking send
 * \param comm MPI communicator
 */
extern void isend2D_int32_T(const emxArray_int32_T *array_send, const int dst, const int tag,
			    MPI_Comm comm, MPI_Request *req_com, MPI_Request *req_data);

/*!
 * \brief NonBlock receiving a 2D double array
 * \detail In the function we first receive the row and column number, 
 * then we create the array and receive the data member of the array.
 * The common info has label = tag + 1 and data info has label tag + 2.
 *
 * \param array_recv pointer to the array being received
 * \param dst source processor ID
 * \param tag tag for this non blocking send
 * \param comm MPI communicator
 */
extern void isend2D_real_T(const emxArray_real_T *array_send, const int dst, const int tag,
			   MPI_Comm comm, MPI_Request *req_com, MPI_Request *req_data);

extern void b_fix(real_T *x);
extern real_T length(const emxArray_int32_T *x);

extern void determine_opposite_halfedge_tri(int32_T nv, const emxArray_int32_T *tris ,emxArray_int32_T *opphes);

extern void determine_incident_halfedges(const emxArray_int32_T *elems, const emxArray_int32_T *opphes, emxArray_real_T *v2he);

extern void obtain_nring_surf(int32_T vid, real_T ring, int32_T minpnts, const emxArray_int32_T *tris, const emxArray_int32_T *opphes, const emxArray_real_T *v2he, emxArray_int32_T *ngbvs, emxArray_boolean_T *vtags, emxArray_boolean_T *ftags, emxArray_int32_T *ngbfs, int32_T *nverts, int32_T *nfaces);

#endif
