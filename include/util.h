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
#include "hiprop.h"

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

#ifndef hpmax
	#define hpMax( a, b ) ( ((a) > (b)) ? (a) : (b) )
#endif

#ifndef hpmin
	#define hpMin( a, b ) ( ((a) < (b)) ? (a) : (b) )
#endif

/*!
 * \brief Transform an int number into string using certain number of digits
 * \param n int number that is being transformed
 * \param ndigits number of digits for the output
 * \param in_string output string
 */
EXTERN_C void numIntoString(const int n, const int ndigits, char *in_string);

/*!
 * \brief Locate the current cursor after the searching string, if string not found,
 * return 0.
 * \param file file pointer
 * \param in_string string for search
 */
EXTERN_C int findString(FILE* file, const char* in_string);

/*
extern inline int32_T I2d(int32_T i, int32_T j, int32_T *size)
{
    return (j-1)*size[0] + (i-1);
}

extern inline int32_T I3d(int32_T i, int32_T j, int32_T k, int32_T *size)
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
EXTERN_C void addColumnToArray_common(emxArray__common *emxArray, const int32_T numColAdd, const uint32_T elementSize);

/*!
 * \brief Add column to int array
 * 
 * \param emxArray pointer to int array
 * \param numColAdd number of column added
 */
EXTERN_C void addColumnToArray_int32_T(emxArray_int32_T *emxArray, const int32_T numColAdd);

/*!
 * \brief Add column to real array
 * 
 * \param emxArray pointer to real array
 * \param numColAdd number of column added
 */
EXTERN_C void addColumnToArray_real_T(emxArray_real_T *emxArray, const int32_T numColAdd);

/*!
 * \brief Add column to boolean array
 * 
 * \param emxArray pointer to boolean array
 * \param numColAdd number of column added
 */
EXTERN_C void addColumnToArray_boolean_T(emxArray_boolean_T *emxArray, const int32_T numColAdd);

/*!
 * \brief Add row to int array
 * 
 * \param emxArray pointer to int array
 * \param numColAdd number of row added
 */
EXTERN_C void addRowToArray_int32_T(emxArray_int32_T *emxArray, const int32_T numRowAdd);

/*!
 * \brief Add row to real array
 * 
 * \param emxArray pointer to real array
 * \param numColAdd number of row added
 */
EXTERN_C void addRowToArray_real_T(emxArray_real_T *emxArray, const int32_T numRowAdd);

/*!
 * \brief Add row to boolean array
 * 
 * \param emxArray pointer to boolean array
 * \param numColAdd number of row added
 */
EXTERN_C void addRowToArray_boolean_T(emxArray_boolean_T *emxArray, const int32_T numRowAdd);

EXTERN_C void printArray_int32_T(const emxArray_int32_T *emxArray);
EXTERN_C void printArray_real_T(const emxArray_real_T *emxArray);
EXTERN_C void printArray_boolean_T(const emxArray_boolean_T *emxArray);

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
EXTERN_C void sendND_boolean_T(const emxArray_boolean_T *array_send, const int dst, const int tag, MPI_Comm comm);

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
EXTERN_C void sendND_int32_T(const emxArray_int32_T *array_send, const int dst, const int tag, MPI_Comm comm);

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
EXTERN_C void sendND_real_T(const emxArray_real_T *array_send, const int dst, const int tag, MPI_Comm comm);

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
EXTERN_C void recvND_boolean_T(emxArray_boolean_T **array_recv, const int src, const int tag, MPI_Comm comm);

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
EXTERN_C void recvND_int32_T(emxArray_int32_T **array_recv, const int src, const int tag, MPI_Comm comm);

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
EXTERN_C void recvND_real_T(emxArray_real_T **array_recv, const int src, const int tag, MPI_Comm comm);

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
EXTERN_C void send2D_boolean_T(const emxArray_boolean_T *array_send, const int dst, const int tag, MPI_Comm comm);

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
EXTERN_C void send2D_int32_T(const emxArray_int32_T *array_send, const int dst, const int tag, MPI_Comm comm);

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
EXTERN_C void send2D_real_T(const emxArray_real_T *array_send, const int dst, const int tag, MPI_Comm comm);

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
EXTERN_C void recv2D_boolean_T(emxArray_boolean_T **array_recv, const int src, const int tag, MPI_Comm comm);

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
EXTERN_C void recv2D_int32_T(emxArray_int32_T **array_recv, const int src, const int tag, MPI_Comm comm);

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
EXTERN_C void recv2D_real_T(emxArray_real_T **array_recv, const int src, const int tag, MPI_Comm comm);

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
EXTERN_C void isend2D_boolean_T(const emxArray_boolean_T *array_send, const int dst, const int tag,
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
EXTERN_C void isend2D_int32_T(const emxArray_int32_T *array_send, const int dst, const int tag,
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
EXTERN_C void isend2D_real_T(const emxArray_real_T *array_send, const int dst, const int tag,
			   MPI_Comm comm, MPI_Request *req_com, MPI_Request *req_data);



/*!
 *
 *
 *
 */
EXTERN_C void determine_opposite_halfedge_tri(int32_T nv, const emxArray_int32_T *tris, emxArray_int32_T *opphes);

EXTERN_C void determine_incident_halfedges(const emxArray_int32_T *elems, const emxArray_int32_T *opphes, emxArray_int32_T *v2he);

EXTERN_C void obtain_nring_surf(int32_T vid, real_T ring, int32_T minpnts, const emxArray_int32_T *tris, const emxArray_int32_T *opphes, const emxArray_int32_T *v2he, emxArray_int32_T *ngbvs, emxArray_boolean_T *vtags, emxArray_boolean_T *ftags, emxArray_int32_T *ngbfs, int32_T *nverts, int32_T *nfaces);

EXTERN_C void compute_diffops_surf(const emxArray_real_T *xs, const emxArray_int32_T *tris, int32_T degree, real_T ring, boolean_T iterfit, emxArray_real_T *nrms, emxArray_real_T *curs, emxArray_real_T *prdirs, int32_T param);

EXTERN_C void test_walf_tri(const emxArray_real_T *ps, const emxArray_int32_T *tris, int32_T degree, const emxArray_real_T *param, emxArray_real_T *pnts);

EXTERN_C void average_vertex_normal_tri_cleanmesh(int32_T nv_clean, const emxArray_real_T *xs, const emxArray_int32_T *tris, const emxArray_real_T *flabel, emxArray_real_T *nrms);

EXTERN_C void obtain_ringsz_cleanmesh(int32_T nv_clean, const emxArray_int32_T *part_bdry, const emxArray_real_T *xs, const emxArray_int32_T *elems, int32_T degree, emxArray_real_T *ring_sz);

EXTERN_C void compute_diffops_surf_cleanmesh(int32_T nv_clean, const emxArray_real_T *xs, const emxArray_int32_T *tris, const emxArray_real_T *nrms_proj, int32_T degree, real_T ring, boolean_T iterfit, emxArray_real_T *nrms, emxArray_real_T *curs, emxArray_real_T *prdirs);

/*
EXTERN_C void compute_hisurf_normals(int32_T nv_clean,
				     const emxArray_real_T *xs,
				     const emxArray_int32_T *tris,
				     int32_T degree,
				     emxArray_real_T *nrms);

EXTERN_C void compute_statistics_tris_global(int32_T nt_clean,
					     const emxArray_real_T *xs,
					     const emxArray_int32_T *tris,
					     real_T *min_angle,
					     real_T *max_angle,
					     real_T *min_area,
					     real_T *max_area);

*/

EXTERN_C void smooth_mesh_hisurf_cleanmesh(int32_T nv_clean, int32_T nt_clean, emxArray_real_T *xs, const emxArray_int32_T *tris, int32_T degree, const emxArray_boolean_T *isridge, const emxArray_boolean_T *ridgeedge, const emxArray_int32_T *flabel, int32_T niter, int32_T verbose, boolean_T check_trank, hiPropMesh *pmesh);



#endif
