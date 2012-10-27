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

/*!
 * \brief Free a hiProp mesh augment info and set the pointer to be NULL
 * \param pmesh pointer to hiProp mesh
 */
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

/*!
 * \brief Wrapper for building the opposite half edge structure for triangular
 * mesh
 * \detail Fill the mesh->opphe data
 * \param mesh hiPropMesh on this processor
 */
extern void hpBuildOppositeHalfEdge(hiPropMesh *mesh);

/*!
 * \brief Wrapper for building the incident half edge structure for triangular
 * mesh
 * \detail Fill the mesh->inhe data
 * \param mesh hiPropMesh on this processor
 */
extern void hpBuildIncidentHalfEdge(hiPropMesh *mesh);

/*!
 * \brief Wrapper for getting the n-ring neighborhood of a point
 * \detail Give the address of pointers to array as input. Need to free the
 * arrays afterwards. Should already have the opposite half edge and incident
 * half edge constructed before trying to get n-ring neighborhood.
 * \param mesh hiPropMesh on this processor
 * \param in_vid point id to build n-ring neighborhood from (start from 1)
 * \param in_ring number of rings
 * \param in_minpnts minimum number of points
 * \param max_numps maximum number of points, usually set to 128
 * \param max_numtris maximum number of tris, usually set to 256
 * \param in_ngbvs address of the pointer to the array for storing the output
 * point indices
 * \param in_ngbfs address of the pointer to the array for storing the output
 * triangle indices
 * \param in_vtags address of the pointer to the array for tags of points,
 * output would be all false
 * \param in_ftags address of the pointer to the array for tags of tris,
 * output would be all false
 * \param in_nverts address of the number of output points
 * \param in_nfaces address of the number of output tris
 */
extern void hpObtainNRingTris(const hiPropMesh *mesh,
			      const int32_T in_vid,
			      const real_T in_ring,
			      const int32_T in_minpnts,
			      const int32_T max_numps,
			      const int32_T max_numtris,
			      emxArray_int32_T **in_ngbvs,
			      emxArray_int32_T **in_ngbfs,
			      emxArray_boolean_T **in_vtags,
			      emxArray_boolean_T **in_ftags,
			      int32_T *in_nverts,
			      int32_T *in_nfaces);
/*!
 * \brief Make all submeshes to be clean (no overlapping triangles).
 * This function uses tris_pinfo and nb_proc. In the end, nb_proc is rebuilt
 * and tris_pinfo is freed.
 * \param mesh The submeshes to be cleaned.
 */
void hpCleanMeshByPinfo(hiPropMesh* mesh);


/*!
 * \brief Build a n-ring ghost neighborhood on the current hiProp mesh
 * \detail The input mesh could have ghost points and triangles. After building
 * the n-ring ghost neighborhood, the neighbor processor information might need
 * to be updated.
 * \param mesh pointer to hiProp Mesh
 * \param num_ring number of rings needed to be built
 */
extern void hpBuildNRingGhost(hiPropMesh *mesh, const real_T num_ring);

extern void hpBuidBoundingBoxGhost(hiPropMesh *mesh, const emxArray_real_T *nb_box);

/*!
 * \brief Collect the n-ring neighborhood for a list of points
 * \detail Before calling this function, mesh has to have the opposite half edge
 * and incident half edge data. out_ps and out_tris don't need to be allocated
 * before calling this function
 * \param mesh pointer to a hiProp Mesh
 * \param in_psid list of point ids on which the n-ring neighborhood need to be
 * built
 * \param num_ring number of rings need to be built
 * \param out_ps address of the pointer to the output point ids
 * \param out_tris address of the pointer to the output triangle ids
 */
extern void hpCollectNRingTris(const hiPropMesh *mesh,
			       const emxArray_int32_T *in_psid,
			       const real_T num_ring,
			       emxArray_int32_T **out_ps,
			       emxArray_int32_T **out_tris);

/*!
 * \brief This function is a subfunction for hpBuildNRingGhost.
 * It collect all the overlay points for all neighboring processors
 * \detail The function collect all the overlaying point between the current
 * processor and it's neighboring processors and store in out_psid. The array
 * for pointers to the output point ids out_psid need to be allocated before
 * the function being called. The array themselves do not need to be allocated.
 * out_psid[i] is the pointer to the overlaying point array between the current
 * processor and processor mesh->nb_proc->data[i]
 * \param mesh pointer to a hiProp Mesh
 * \param out_psid array of pointers to the overlaying point ids for neighboring
 * processors.
 */
extern void hpCollectAllSharedPs(const hiPropMesh *mesh, emxArray_int32_T **out_psid);

extern void hpWriteUnstrMeshWithPInfo(const char *name, const hiPropMesh *mesh);

extern void hpDebugOutput(const hiPropMesh *mesh,
			  const emxArray_int32_T *debug_ps,
			  const emxArray_int32_T *debug_tris,
			  char *debug_file_name);

extern void hpBuildGhostPsTrisForSend(const hiPropMesh *mesh,
				      const int nb_proc_index,
				      const real_T num_ring,
				      emxArray_int32_T *psid_proc,
				      emxArray_int32_T **ps_ring_proc,
				      emxArray_int32_T **tris_ring_proc,
				      emxArray_real_T **buffer_ps,
				      emxArray_int32_T **buffer_tris);

extern void hpBuildGhostPsTrisPInfoForSend(const hiPropMesh *mesh,
					   const int nb_proc_index,
					   emxArray_int32_T *ps_ring_proc,
					   emxArray_int32_T *tris_ring_proc,
					   int **buffer_ps_pinfo_tag,
					   int **buffer_ps_pinfo_lindex,
					   int **buffer_ps_pinfo_proc,
					   int **buffer_tris_pinfo_tag,
					   int **buffer_tris_pinfo_lindex,
					   int **buffer_tris_pinfo_proc);

extern void hpAttachNRingGhostWithPInfo(hiPropMesh *mesh,
					const int rcv_id,
					emxArray_real_T *bps,
					emxArray_int32_T *btris,
					int *ppinfot,
					int *ppinfol,
					int *ppinfop,
					int *tpinfot,
					int *tpinfol,
					int *tpinfop);

extern void hpUpdatePInfo(hiPropMesh *mesh);

extern void hpUpdateMasterPInfo(hiPropMesh *mesh);
extern void hpUpdateAllPInfoFromMaster(hiPropMesh *mesh);

extern void hpUpdateNbWithPInfo(hiPropMesh *mesh);

extern void hpAddProcInfoForGhostPsTris(hiPropMesh *mesh,
					const int nb_proc_index,
					emxArray_int32_T *ps_ring_proc,
					emxArray_int32_T *tris_ring_proc);

extern void hpCollectAllGhostPs(hiPropMesh *mesh,
			 	const int nbp_index,
				int *sizep,
				int **ppinfol);

extern void hpCollectAllGhostTris(hiPropMesh *mesh,
			 	  const int nbp_index,
				  int *sizet,
				  int **tpinfol);


extern void hpMergeOverlayPsPInfo(hiPropMesh *mesh,
			   	  const int rcv_id,
				  int nump,
				  int *ppinfol);

extern void hpMergeOverlayTrisPInfo(hiPropMesh *mesh,
			     	    const int rcv_id,
				    int numt,
				    int *tpinfol);

extern void hpCollectAllOverlayPs(hiPropMesh *mesh,
				  const int nbp_index,
				  int *sizep,
				  int **ppinfot,
				  int **ppinfol,
				  int **ppinfop);

extern void hpCollectAllOverlayTris(hiPropMesh *mesh,
				    const int nbp_index,
				    int *sizet,
				    int **tpinfot,
				    int **tpinfol,
				    int **tpinfop);


extern void hpMergeGhostPsPInfo(hiPropMesh *mesh,
			   	const int rcv_id,
				int nump,
				int *ppinfot,
				int *ppinfol,
				int *ppinfop);

extern void hpMergeGhostTrisPInfo(hiPropMesh *mesh,
			     	  const int rcv_id,
				  int numt,
				  int *tpinfot,
				  int *tpinfol,
				  int *tpinfop);


#endif
