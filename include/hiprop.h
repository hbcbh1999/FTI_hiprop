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
/* #include "metis.h" */

/*!
 * \brief Parallel information element for each point/triangle in hiPropMesh
 */

typedef struct hpPInfoNode
{
    int pindex;		/*!< processor index for local nb_proc list, different pindex could have same processor ID */

    int8_T shift[3];	/*!< shifting for periodic boundary, could be 0, 1 and 2*/
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
    emxArray_real_T *ps;		/*!< point positions, # of points = num_ps */
    emxArray_int32_T *tris;		/*!< triangles, # of triangles = num_tris */
    emxArray_real_T *nor;		/*!< point normals, size num_ps */
    emxArray_real_T *curv;		/*!< point main curvatures, size num_ps */

    emxArray_int32_T *nb_proc;		/*!< neighbour processor list */
    emxArray_int8_T **nb_proc_shift;	/*!< neighbour processor shift list */

    real_T domain_len[3];		/*!< domain size in x,y,z */
    boolean_T has_periodic_boundary[3];	/*!< flag for periodic boundary in x,y,z */

    emxArray_int32_T *part_bdry;	/*!< partition boundary points*/
    emxArray_int32_T *ps_type;		/*!< point type, 0 INTERIOR, 1 OVERLAY, 2 GHOST, size num_ps */

    hpPInfoList *ps_pinfo;		/*!< parallel information for points */
    hpPInfoList *tris_pinfo;		/*!< parallel information for tris */
    
    emxArray_int32_T *opphe;		/*!< opposite half edge */
    emxArray_int32_T *inhe;		/*!< incident half edge */
    emxArray_real_T *est_nor;		/*!< estimated normal, given by tri normal average, size num_ps */

    emxArray_int32_T **ps_send_index;
    emxArray_int32_T **ps_recv_index;

    int32_T nps_clean;			/*!< number of points for the clean mesh (with no overlapping triangles) */
    int32_T ntris_clean;		/*!< number of tris for the clean mesh (with no overlapping triangles) */
    int32_T npspi_clean;		/*!< number of ps pinfo for the clean mesh (with no overlapping triangles) */
    boolean_T is_clean;			/*!< flag to denote whether current mesh is clean,
					     0 with overlapping triangles, 1 without overlapping triangles */

    
} hiPropMesh;


EXTERN_C void hpInitDomainBoundaryInfo(hiPropMesh *pmesh);

/*!
 * \brief Initialize a hiProp mesh and set the initial pointer to be NULL
 * \param pmesh Address of the hiProp mesh pointer
 */
EXTERN_C void hpInitMesh(hiPropMesh **pmesh);
/*!
 * \brief Free a hiProp mesh update info and set the pointer to be NULL
 * \param pmesh pointer to hiProp mesh
 */
EXTERN_C void hpFreeMeshUpdateInfo(hiPropMesh *pmesh);
/*!
 * \brief Free a hiProp mesh parallel info and set the pointer to be NULL
 * \param pmesh pointer to hiProp mesh
 */
EXTERN_C void hpFreeMeshParallelInfo(hiPropMesh *pmesh);

/*!
 * \brief Free a hiProp mesh basic info and set the pointer to be NULL
 * \param pmesh pointer to hiProp mesh
 */
EXTERN_C void hpFreeMeshBasicInfo(hiPropMesh *pmesh);

/*!
 * \brief Free a hiProp mesh augment info and set the pointer to be NULL
 * \param pmesh pointer to hiProp mesh
 */
EXTERN_C void hpFreeMeshAugmentInfo(hiPropMesh *pmesh);
/*!
 * \brief Free the data of a hiProp mesh 
 * \param pmesh hiProp mesh pointer
 */
EXTERN_C void hpFreeMesh(hiPropMesh *pmesh);

/*!
 * \brief Delete a hiProp mesh and set the pointer to NULL
 * \param pmesh address of the hiProp mesh pointer
 */
EXTERN_C void hpDeleteMesh(hiPropMesh **pmesh);

/*!
 * \brief Delete a hiProp parallel info list and set the pointer to NULL
 * \param plist address of the hpPInfoList
 */
EXTERN_C void hpDeletePInfoList(hpPInfoList **plist);
/*!
 * Read an ascii triangular vtk file with data type POLYGON.
 * \param name input file name
 * \param mesh mesh for storing the data read from file
 */
EXTERN_C int hpReadPolyMeshVtk3d(const char *name, hiPropMesh *mesh);
/*!
 * Write an ascii triangular vtk file with data type POLYGON.
 * \param name output file name
 * \param mesh mesh for output
 */
EXTERN_C int hpWritePolyMeshVtk3d(const char *name, hiPropMesh *mesh);
/*!
 * Read an ascii triangular vtk file with data type UNSTURCTURED_GRID.
 * \param name input file name
 * \param mesh mesh for storing the data read from file
 */
EXTERN_C int hpReadUnstrMeshVtk3d(const char *name, hiPropMesh *mesh);
/*!
 * \brief Write an ascii triangular vtk file with data type UNSTRUCTURED_GRID.
 * \param name output file name
 * \param mesh mesh for output
 */
EXTERN_C int hpWriteUnstrMeshVtk3d(const char *name, hiPropMesh *mesh);

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
EXTERN_C int hpMetisPartMesh(hiPropMesh *mesh, const int nparts, int **tri_part, int **pt_part);

/*!
 * \brief Distribute the mesh according to tri_part array got in hpMetisPartMesh,
 * call hpConstrPInfoFromGlobalLocalInfo() inside, to set parallel info
 * \param root root of the communication, it should contain in_mesh and tri_part info
 * \param in_mesh the input mesh to be partitioned
 * \param mesh the output mesh after partition
 * \param tri_part the triangle partition info generated by hpMetisPartMesh
 * \param tag tag of the communication
 * \param ps_globalid The global id of each point on each submesh, memory allocated inside the function
 * \param tri_globalid The global id of each triangle on each submesh, memory allocated inside the function
 */
EXTERN_C int hpDistMesh(int root, hiPropMesh *in_mesh, hiPropMesh *mesh, int *tri_part, int tag,
       	emxArray_int32_T **ps_globalid, emxArray_int32_T **tri_globalid);

/*!
 * \brief Construct parallel information for points assuming no overlapping triangles
 * \param mesh the submesh to construct the parallel info
 * \param g2lindex a (num_proc*num_total_points) matrix,
 * g2lindex[i][I1dm(j)] is the local index of point j on proc i
 * \param l2gindex an array of length equal to the number of points in the submesh
 * l2gindex[I1dm(i)] is the global index of the i-th point on the submesh
 * \param rank the rank of the current proc
 */
EXTERN_C void hpConstrPInfoFromGlobalLocalInfo(hiPropMesh *mesh,
	int** g2lindex, int* l2gindex, int rank);

/*!
 * \brief Get the neighboring processor ID and fill the nb_proc list from the
 * mesh points
 * \param mesh parallel hiPropMesh with overlapping points and triangles
 */
EXTERN_C void hpGetNbProcListAuto(hiPropMesh *mesh);

/*!
 * \brief Get the neighboring processor ID and fill the nb_proc list from the
 * user given neighboring processor list
 * \param mesh parallel hiPropMesh with overlapping points and triangles
 * \param num_nb_proc number of neighboring processor
 * \param in_nb_proc array of neighboring processors with length num_nb_proc
 */
EXTERN_C void hpGetNbProcListFromInput(hiPropMesh *mesh,
	const int in_num_nbp,
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
EXTERN_C void hpInitPInfo(hiPropMesh *mesh);

/*!
 * \brief Build the parallel information for submeshes with no overlapping
 * triangles (only with overlapping points)
 * \detail Need a hpInitPInfo before this function
 * \param mesh hiPropMesh mesh with no overlapping triangles
 */
EXTERN_C void hpBuildPInfoNoOverlappingTris(hiPropMesh *mesh);

/*!
 * \brief Build the parallel information for submeshes with overlapping triangles 
 * \detail Need a hpInitPInfo before this function
 * \param mesh hiPropMesh mesh with overlapping triangles
 */
EXTERN_C void hpBuildPInfoWithOverlappingTris(hiPropMesh *mesh);

/*!
 * \brief Utility function for automatically increase the parallel info list
 * when full
 * \detail When all the pre-allocated memory for this list is used up, the list
 * automatically increase itself by 10% to include more elements
 * \param pinfo A parallel information list
 */
EXTERN_C void hpEnsurePInfoCapacity(hpPInfoList *pinfo);

/*!
 * \brief Build the partition boundary information for a "clean" hiProp mesh
 * (with not overlapping triangles).
 * \detail Before calling this function, a parallel hiProp mesh with no
 * overlapping triangles and correct parallel information is required. The
 * result is output to the boolean array mesh->part_bdry. If
 * mesh->part_bdry->data[I1dm(i)] = 1, it means point i is a partition boundary
 * for the submesh on the current processor.
 * \param mesh pointer to the hiProp mesh.
 */
EXTERN_C void hpBuildPartitionBoundary(hiPropMesh *mesh);

/*!
 * \brief Build the point type information for a hiProp mesh
 * \detail Before calling this function, a parallel hiProp mesh with parallel
 * info is required. The result is output to the boolean array mesh->ps_type.
 * Point type = 0: INTERIOR point, only exists to current processor.
 * Point type = 1: OVERLAY point, owned by current processor and exists on other
 * processors.
 * Point type = 2: GHOST point, exists on current processor but owned by another
 * processor.
 * \param mesh pointer to the hiProp mesh.
 */
EXTERN_C void hpBuildPsType(hiPropMesh *mesh);

/*!
 * \brief Build the parallel update information for submesh 
 * \detail This uses the ps_pinfo,
 * so hpBuildPInfoNoOverlappingTris or hpBuildPInfoWithOverlappingTris or hpConstrPInfoFromGlobalLocalInfo
 * should be called before this function
 * \param mesh The submesh to build the parallel update info
 */
EXTERN_C void hpBuildPUpdateInfo(hiPropMesh *mesh);

/*!
 * \brief Wrapper for building the opposite half edge structure for triangular
 * mesh
 * \detail Fill the mesh->opphe data
 * \param mesh pointer to the hiProp mesh
 */
EXTERN_C void hpBuildOppositeHalfEdge(hiPropMesh *mesh);

/*!
 * \brief Wrapper for building the incident half edge structure for triangular
 * mesh
 * \detail Fill the mesh->inhe data
 * \param mesh pointer to the hiProp mesh
 */
EXTERN_C void hpBuildIncidentHalfEdge(hiPropMesh *mesh);

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
EXTERN_C void hpObtainNRingTris(const hiPropMesh *mesh,
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
 * \brief Make all submeshes to be clean (no overlapping triangles) and update
 * the parallel information.
 * \detail This function uses tris_pinfo and nb_proc. In the end, nb_proc is rebuilt
 * and tris_pinfo is freed.
 * \param mesh pointer to the hiProp mesh with overlapping triangles across
 * different processors.
 */
EXTERN_C void hpCleanMeshByPinfo(hiPropMesh* mesh);


/*!
 * \brief Build a n-ring ghost neighborhood on the current hiProp mesh
 * \detail The input mesh could have ghost points and triangles. Correct
 * parallel information is needed for both points and triangles before calling
 * this function. The parallel information would also be updated after this
 * function is called. At the mean time, the neighboring processor information
 * is checked and might also need to be updated.
 * \param mesh pointer to hiProp Mesh
 * \param num_ring number of rings needed to be built
 */
EXTERN_C void hpBuildNRingGhost(hiPropMesh *mesh, const real_T num_ring);


/*!
 * \brief Build ghost neighborhood on the current hiProp mesh based on the
 * bounding boxes
 * \detail The bounding box for each processor could overlap with each other.
 * Correct parallel information is needed for both points and triangles before
 * calling this function. The parallel information would also be updated after 
 * this function is called. At the mean time, the neighboring processor
 * information is checked and might also need to be updated.
 * \param mesh pointer to hiProp mesh
 * \param bd_box The bounding box for current processor, stored in order:
 * x_low, x_upper, y_low, y_upper, z_low, z_upper.
 */
EXTERN_C void hpBuildBoundingBoxGhost(hiPropMesh *mesh, const double *bd_box);


/*!
 * \brief Send ghost ps and tris to neighbor processors and receive ghost ps and
 * tris from neighbor processors with parallel information
 * \detail This function works as a subfunction for 3 types of building ghost
 * ps/tris. Before calling this function, for each neighbor processor, we need
 * to pick up the points and triangle that we want to send and put the local
 * indices into ps_ring_proc and tris_ring_proc. For conveniece we also put the
 * exact point positions and connective of the ghost mesh that we want to send
 * into buffer_ps and buffer_tris. There are three main steps in the function:
 * 1. Before communication, update the ps_pinfo and tris_pinfo partially. For a
 * point/triangle which does not exist on the neighbor processor but is to be
 * sent, we don't know the exact local index of the point/triangle on the new
 * processor, thus we could only partially construct the paralle information.
 * 2. Build up the parallel information for send.
 * 3. Send the buffer_ps, buffer_tris and the corresponding parallel information
 * to the neighbor processors.
 * 4. Receive the ghost points and triangles with parallel information coming
 * from the neighboring processors and add the ghost points and triangles to the
 * current mesh.
 * 5. Update the parallel information across differrent processors to make it
 * consistent.
 * 6. Update the nb_proc as the neighbor processors could be changed after
 * building ghost points and triangles.
 * \param mesh pointer to a hiProp mesh
 * \param ps_ring_proc local index of the ghost points that is to be sent to the
 * neighboring processors
 * \param tris_ring_proc local index of the ghost triangles that is to be sent
 * to the neighboring processors 
 * \param buffer_ps point positions of the ghost points that is to be sent to
 * the neighboring processors
 * \param buffer_tris triangle connectivity of the ghost triangles that is to be
 * sent to the neighboring processors. The elements in buffer_tris are index in
 * buffer_ps rather than ps_ring_proc. 
 */
EXTERN_C void hpCommPsTrisWithPInfo(hiPropMesh *mesh,
				    emxArray_int32_T **ps_ring_proc,
				    emxArray_int32_T **tris_ring_proc,
				    emxArray_int8_T **ps_shift_ring_proc,
				    emxArray_int8_T **tris_shift_ring_proc,
				    emxArray_real_T **buffer_ps,
				    emxArray_int32_T **buffer_tris);
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
EXTERN_C void hpCollectNRingTris(const hiPropMesh *mesh,
				 const int nb_proc_index,
				 const emxArray_int32_T *in_psid,
				 const emxArray_int8_T *in_ps_shift,
				 const real_T num_ring,
				 emxArray_int32_T **out_ps,
				 emxArray_int32_T **out_tris,
				 emxArray_int8_T **out_ps_shift,
				 emxArray_int8_T **out_tris_shift,
				 emxArray_int32_T **out_tris_buffer);

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
EXTERN_C void hpCollectAllSharedPs(const hiPropMesh *mesh, emxArray_int32_T **out_psid, emxArray_int8_T **out_ps_shift);

EXTERN_C void hpWriteUnstrMeshWithPInfo(const char *name, const hiPropMesh *mesh);

EXTERN_C void hpDebugOutput(const hiPropMesh *mesh,
			  const emxArray_int32_T *debug_ps,
			  const emxArray_int32_T *debug_tris,
			  char *debug_file_name);

EXTERN_C void hpBuildGhostPsTrisForSend(const hiPropMesh *mesh,
				      const int nb_proc_index,
				      const real_T num_ring,
				      emxArray_int32_T *psid_proc,
				      emxArray_int8_T *ps_shift_proc,
				      emxArray_int32_T **ps_ring_proc,
				      emxArray_int32_T **tris_ring_proc,
				      emxArray_int8_T **ps_shift_ring_proc,
				      emxArray_int8_T **tris_shift_ring_proc,
				      emxArray_real_T **buffer_ps,
				      emxArray_int32_T **buffer_tris);

EXTERN_C void hpBuildBdboxGhostPsTrisForSend(const hiPropMesh *mesh,
					   const int nb_proc_index,
					   const double *bd_box,
					   emxArray_int32_T **ps_ring_proc,
					   emxArray_int32_T **tris_ring_proc,
					   emxArray_real_T **buffer_ps,
					   emxArray_int32_T **buffer_tris);

EXTERN_C void hpBuildGhostPsTrisPInfoForSend(const hiPropMesh *mesh,
					   const int nb_proc_index,
					   emxArray_int32_T *ps_ring_proc,
					   emxArray_int32_T *tris_ring_proc,
					   int **buffer_ps_pinfo_tag,
					   int **buffer_ps_pinfo_lindex,
					   int **buffer_ps_pinfo_proc,
					   int8_T **buffer_ps_pinfo_shift,
					   int **buffer_tris_pinfo_tag,
					   int **buffer_tris_pinfo_lindex,
					   int **buffer_tris_pinfo_proc,
					   int8_T **buffer_tris_pinfo_shift);

EXTERN_C void hpAttachNRingGhostWithPInfo(hiPropMesh *mesh,
					const int rcv_id,
					emxArray_real_T *bps,
					emxArray_int32_T *btris,
					int *ppinfot,
					int *ppinfol,
					int *ppinfop,
					int8_T *ppinfos,
					int *tpinfot,
					int *tpinfol,
					int *tpinfop,
					int8_T *tpinfos);

EXTERN_C void hpUpdatePInfo(hiPropMesh *mesh);

EXTERN_C void hpUpdateMasterPInfo(hiPropMesh *mesh);
EXTERN_C void hpUpdateAllPInfoFromMaster(hiPropMesh *mesh);

EXTERN_C void hpUpdateNbWithPInfo(hiPropMesh *mesh);

EXTERN_C void hpAddProcInfoForGhostPsTris(hiPropMesh *mesh,
					const int nb_proc_index,
					emxArray_int32_T *ps_ring_proc,
					emxArray_int32_T *tris_ring_proc,
					emxArray_int8_T *ps_shift_ring_proc,
					emxArray_int8_T *tris_shift_ring_proc);

EXTERN_C void hpCollectAllGhostPs(hiPropMesh *mesh,
			 	const int nbp_index,
				int *size_send,
				int **ppinfol);

EXTERN_C void hpCollectAllGhostTris(hiPropMesh *mesh,
			 	  const int nbp_index,
				  int *size_send,
				  int **tpinfol);


EXTERN_C void hpMergeOverlayPsPInfo(hiPropMesh *mesh,
			   	  const int rcv_id,
				  int nump,
				  int *ppinfol);

EXTERN_C void hpMergeOverlayTrisPInfo(hiPropMesh *mesh,
			     	    const int rcv_id,
				    int numt,
				    int *tpinfol);

EXTERN_C void hpCollectAllOverlayPs(hiPropMesh *mesh,
				  const int nbp_index,
				  int *size_send,
				  int **ppinfot,
				  int **ppinfol,
				  int **ppinfop);

EXTERN_C void hpCollectAllOverlayTris(hiPropMesh *mesh,
				    const int nbp_index,
				    int *size_send,
				    int **tpinfot,
				    int **tpinfol,
				    int **tpinfop);


EXTERN_C void hpMergeGhostPsPInfo(hiPropMesh *mesh,
			   	const int rcv_id,
				int nump,
				int *ppinfot,
				int *ppinfol,
				int *ppinfop);

EXTERN_C void hpMergeGhostTrisPInfo(hiPropMesh *mesh,
			     	  const int rcv_id,
				  int numt,
				  int *tpinfot,
				  int *tpinfol,
				  int *tpinfop);
/*!
 * \brief Calculates the high-order normal and curvature
 * \detail The unit normal of each point output to mesh->nor, the 2 main
 * curvatures output to mesh->curv. Before calling this function, the adaptive
 * n-ring neighborhood should be construced based on the degree of fitting the
 * function needs. The minimun requirement of # of rings = (degree + 2)/2.
 * \param mesh pointer to a hiProp Mesh
 * \param in_degree the order of accuracy 
 */
EXTERN_C void hpComputeDiffops(hiPropMesh *mesh, int32_T in_degree);

/*!
 * \brief Calculates the estimated normal by averaging face normals 
 * \detail The result is saved in est_nor. At least one-ring ghost is needed
 * before calling this function.
 * \param mesh pointer to a hiProp Mesh
 */
EXTERN_C void hpComputeEstimatedNormal(hiPropMesh *mesh);

/*!
 * \brief Update the estimated normal of the ghost points from master
 * processors.
 * \param mesh pointer to a hiProp mesh
 */
EXTERN_C void hpUpdateEstimatedNormal(hiPropMesh *mesh);

/*!
 * \brief Update the calculated normal and curvature of the ghost points from
 * master processors.
 * \param mesh pointer to a hiProp mesh
 */
EXTERN_C void hpUpdateNorCurv(hiPropMesh *mesh);

/*!
 * \brief Build the adaptive ghost for partition boundary with different ring
 * size
 * \detail The input ring size of the function could be inconsistent between
 * different processors. The first step of the function is to do a parallel
 * reduction on the ring size of each partiontion boundary point. The second
 * step is to build ghost based on the ring size for each partition boundary
 * point.
 * \param mesh mesh pointer to a hiProp mesh
 * \param ring_size ring size for each partition boundary point, size of the
 * array = size of the part_bdry array
 */
EXTERN_C void hpBuildPartBdryGhost(hiPropMesh *mesh, emxArray_real_T *ring_size);

/*!
 * \brief Reduce the ghost ring size on the part_bdry to make sure the same
 * point on different processors would have the same ring of neighborhood
 * \param mesh mesh pointer to a hiProp mesh
 * \param ring_size input and output ring size for each partition boundary
 * point
 */
EXTERN_C void hpReducePartBdryGhostRingSize(hiPropMesh *mesh, emxArray_real_T *ring_size);

EXTERN_C void hpBuildPartBdryGhostPsTrisForSend(const hiPropMesh *mesh,
						const int nb_proc_index,
						const emxArray_real_T *num_ring,
						emxArray_int32_T **ps_ring_proc,
						emxArray_int32_T **tris_ring_proc,
						emxArray_real_T **buffer_ps,
						emxArray_int32_T **buffer_tris);

EXTERN_C void hpCollectPartBdryNRingTris(const hiPropMesh *mesh,
					 const int nb_proc_index,
					 const emxArray_real_T *num_ring,
					 emxArray_int32_T **out_ps,
					 emxArray_int32_T **out_tris);

EXTERN_C void hpUpdateGhostPointData_int32_T(hiPropMesh *mesh, emxArray_int32_T *array);

EXTERN_C void hpUpdateGhostPointData_real_T(hiPropMesh *mesh, emxArray_real_T *array);

EXTERN_C void hpUpdateGhostPointData_boolean_T(hiPropMesh *mesh, emxArray_boolean_T *array);

EXTERN_C void hpAdaptiveBuildGhost(hiPropMesh *mesh, const int32_T in_degree);

/*!
 * \brief Function for redistribute the points on the high order surface in
 * parallel.
 * \param mesh Mesh pointer to hiprop mesh
 * \param in_degree The degree of fitting for high order projection
 * \param method Projection method needed to be used in fitting, could be "CMF"
 * or "WALF"
 */

EXTERN_C void hpMeshSmoothing(hiPropMesh *mesh, int32_T in_degree, const char* method);
/*!
 * \brief Small function to print the pinfo of points and triangles to stdout
 * used for debugging
 * \param mesh mesh pointer to a hiProp mesh
 */
EXTERN_C void hpPrint_pinfo(hiPropMesh *mesh);

EXTERN_C void hpDebugParallelToSerialOutput(hiPropMesh *mesh, emxArray_real_T *array, const char *outname);
#endif
