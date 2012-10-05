/*!
 * \file hiprop.h
 * \brief hiProp functions
 *
 * \author Yijie Zhou
 * \date 2012.10.02
 */



#ifndef __HIPROP_H__
#define __HIPROP_H__


#include "stdafx.h"
#include "metis.h"

/*!
 * \brief Parallel information element for each point/triangle in hiPropMesh
 */

typedef struct hpPInfoNode
{
    int proc;		/*!< processor ID */
    int lindex;		/*!< local index on the corresponding proc */
    int next;		/*!< index for the next node in the linked list, if next = -1, 
			  itis the last node for the list of this element */
} hpPInfoNode;

/*!
 * \brief Parallel information for points/triangles in hiPropMesh
 * \detail For each point/triangle, the parallel information is consisted of double linked
 * lists of hpPInfoNodes. The lists are stored in a array which is continuous in
 * memory. Take the parallel info for points for example, suppose # of points 
 * is N, head[I1dm(i)] is the index of the first element for the i-th point,
 * which contains the master processor ID and the local index on the master proc
 * for the point. tail[I1dm(i)] is the index of the last element for the 
 * i-th point, tail[I1dm(i)].next always equals to -1.
 * The list increases by itself for 10% of the entire number of points
 * when allocated_len > max_len. The max_len is initialized to be 2*N 
 * and allocated_len is initialized to be N. All index start from 1.
 */

typedef struct hpPInfoList
{
    hpPInfoNode *pdata;  /*!< pointer points to the first element of the list */
    int *head;		 /*!< head node index for each point/triangle */
    int *tail;		 /*!< tail node index for each point/triangle */
    int allocated_len;   /*!< allocated length */	
    int max_len;	 /*!< maximun length */

} hpPInfoList;

/*!
 * \brief hiProp Mesh data structure
 */
typedef struct hiPropMesh
{
    emxArray_real_T *ps;		/*!< point positions, # of points = n */
    emxArray_int32_T *tris;		/*!< triangles, # of triangles = m */
    emxArray_real_T *nor;		/*!< point normals */
    emxArray_int32_T *nb_proc;		/*!< neighbour processor list */
    hpPInfoList *ps_pinfo;		/*!< parallel information for points */
    hpPInfoList *tris_pinfo;		/*!< parallel information for tris */
    
    emxArray_int32_T *opphe;		/*!< opposite half edge */
    emxArray_int32_T *inhe;		/*!< incident half edge */

    emxArray_int32_T **ps_send_index;
    emxArray_real_T **ps_send_buffer;
    emxArray_int32_T **ps_recv_index;
    emxArray_real_T **ps_recv_buffer;
    
} hiPropMesh;


/*!
 * \brief Initialize a hiProp mesh and set the initial pointer to be NULL
 * \param pmesh Address of the hiProp mesh pointer
 */
extern void hpInitMesh(hiPropMesh **pmesh);
/*!
 * \brief Free a hiProp mesh update info and set the pointer to be NULL
 * \param pmesh pointer to hiProp mesh
 */
extern void hpFreeMeshUpdateInfo(hiPropMesh *pmesh);
/*!
 * \brief Free a hiProp mesh parallel info and set the pointer to be NULL
 * \param pmesh pointer to hiProp mesh
 */
extern void hpFreeMeshParallelInfo(hiPropMesh *pmesh);

/*!
 * \brief Free a hiProp mesh basic info and set the pointer to be NULL
 * \param pmesh pointer to hiProp mesh
 */
extern void hpFreeMeshBasicInfo(hiPropMesh *pmesh);
extern void hpFreeMeshAugmentInfo(hiPropMesh *pmesh);
/*!
 * \brief Free the data of a hiProp mesh 
 * \param pmesh hiProp mesh pointer
 */
extern void hpFreeMesh(hiPropMesh *pmesh);

/*!
 * \brief Delete a hiProp mesh and set the pointer to NULL
 * \param pmesh address of the hiProp mesh pointer
 */
extern void hpDeleteMesh(hiPropMesh **pmesh);

extern void hpDeletePInfoList(hpPInfoList **plist);
/*!
 * Read an ascii triangular vtk file with data type POLYGON.
 * \param name input file name
 * \param mesh mesh for storing the data read from file
 */
extern int hpReadPolyMeshVtk3d(const char *name, hiPropMesh *mesh);
/*!
 * Write an ascii triangular vtk file with data type POLYGON.
 * \param name output file name
 * \param mesh mesh for output
 */
extern int hpWritePolyMeshVtk3d(const char *name, hiPropMesh *mesh);
/*!
 * Read an ascii triangular vtk file with data type UNSTURCTURED_GRID.
 * \param name input file name
 * \param mesh mesh for storing the data read from file
 */
extern int hpReadUnstrMeshVtk3d(const char *name, hiPropMesh *mesh);
/*!
 * \brief Write an ascii triangular vtk file with data type UNSTRUCTURED_GRID.
 * \param name output file name
 * \param mesh mesh for output
 */
extern int hpWriteUnstrMeshVtk3d(const char *name, hiPropMesh *mesh);

/*!
 * \brief Partition the mesh into nparts, using the routine of METIS_PartMeshDual,
 * the partition of the mesh is based on the partition of the dual graph,
 * should be called in serial
 * \param mesh mesh to partition
 * \param nparts number of parts the mesh would be partitioned into
 * \param tri_part the address of an array of length equal to the number of triangles, 
 * 	the function will give the part index the triangle is partitioned into,
 *	memory allocated inside the function
 * \param pt_part the address of an array of length equal to the number of points, 
 * 	the function will give the part index the point is partitioned into,
 *	memory allocated inside the function
 */
extern int hpMetisPartMesh(hiPropMesh *mesh, const int nparts, int **tri_part, int **pt_part);

/*!
 * \brief Distribute the mesh according to tri_part array got in hpMetisPartMesh,
 * call hpConstrPInfoFromGlobalLocalInfo() inside, to set parallel info
 * \param root root of the communication, it should contain in_mesh and tri_part info
 * \param in_mesh the input mesh to be partitioned
 * \param mesh the output mesh after partition
 * \param tri_part the triangle partition info generated by hpMetisPartMesh
 * \param tag tag of the communication
 */
extern int hpDistMesh(int root, hiPropMesh *in_mesh, hiPropMesh *mesh, int *tri_part, int tag);

/*!
 * \brief Construct parallel information for points assuming no overlapping triangles
 * \param mesh the submesh to construct the parallel info
 * \param g2lindex a (num_proc*num_total_points) matrix,
 * g2lindex[i][I1dm(j)] is the local index of point j on proc i
 * \param l2gindex an array of length equal to the number of points in the submesh
 * l2gindex[I1dm(i)] is the global index of the i-th point on the submesh
 * \param rank the rank of the current proc
 */
extern void hpConstrPInfoFromGlobalLocalInfo(hiPropMesh *mesh,
	int** g2lindex, int* l2gindex, int rank);

/*!
 * \brief Get the neighboring processor ID and fill the nb_proc list from the
 * mesh points
 * \param mesh parallel hiPropMesh with overlapping points and triangles
 */
extern void hpGetNbProcListAuto(hiPropMesh *mesh);

/*!
 * \brief Get the neighboring processor ID and fill the nb_proc list from the
 * user given neighboring processor list
 * \param mesh parallel hiPropMesh with overlapping points and triangles
 * \param num_nb_proc number of neighboring processor
 * \param in_nb_proc array of neighboring processors with length num_nb_proc
 */
extern void hpGetNbProcListInput(hiPropMesh *mesh,
				 const int num_nb_proc, 
				 const int *in_nb_proc);

/*!
 * \brief Initialize the parallel information given a mesh
 * \detail Before communicationg, for each point i, ps_pinfo->head[I1dm(i)] =
 * ps_pinfo->tail[I1dm(i)] = i, which means both the head and tail points to the i-th
 * element of the pinfo list. For ps_pinfo->pdata[I1dm(i)], proc = current rank,
 * lindex = current local index and next = -1.
 * For tris_pinfo, a similar initialization is carried out.
 * \param mesh hiPropMesh mesh with no parallel information.
 */
extern void hpInitPInfo(hiPropMesh *mesh);

/*!
 * \brief Build the parallel information for submeshes with no overlapping
 * triangles (only with overlapping points)
 * \detail Need a hpInitPInfo before this function
 * \param mesh hiPropMesh mesh with no overlapping triangles
 */
extern void hpBuildPInfoNoOverlappingTris(hiPropMesh *mesh);

/*!
 * \brief Build the parallel information for submeshes with overlapping triangles 
 * \detail Need a hpInitPInfo before this function
 * \param mesh hiPropMesh mesh with overlapping triangles
 */
extern void hpBuildPInfoWithOverlappingTris(hiPropMesh *mesh);

/*!
 * \brief Utility function for automatically increase the parallel info list
 * when full
 * \detail When all the pre-allocated memory for this list is used up, the list
 * automatically increase itself by 10% to include more elements
 * \param pinfo A parallel information list
 */
extern void hpEnsurePInfoCapacity(hpPInfoList *pinfo);

/*!
 * \brief Build the parallel update information for submesh 
 * \detail This uses the ps_pinfo,
 * so hpBuildPInfoNoOverlappingTris or hpBuildPInfoWithOverlappingTris or hpConstrPInfoFromGlobalLocalInfo
 * should be called before this function
 * \param mesh The submesh to build the parallel update info
 */
extern void hpBuildPUpdateInfo(hiPropMesh *mesh);
extern void hpBuildOppositeHalfEdge(hiPropMesh *mesh);

extern void hpBuildIncidentHalfEdge(hiPropMesh *mesh);

extern void hpObtainNRingTris(const hiPropMesh *mesh,
			      const int32_T in_vid,
			      const real_T in_ring,
			      const int32_T in_minpnts,
			      const int32_T max_numps,
			      const int32_T max_numtris,
			      emxArray_int32_T **in_ngbvs,
			      emxArray_boolean_T **in_vtags,
			      emxArray_boolean_T **in_ftags,
			      emxArray_int32_T **in_ngbfs,
			      int32_T *in_nverts,
			      int32_T *in_nfaces);

#endif
