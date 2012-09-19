/*!
 * \file hiprop.h
 * \brief hiProp functions
 *
 * \author Yijie Zhou
 * \date 2012.09.16
 */



#ifndef __HIPROP_H__
#define __HIPROP_H__


#include "stdafx.h"

/*!
 * \brief hiProp Mesh data structure
 */
typedef struct hiPropMesh
{
    emxArray_real_T *ps;	/*!< point positions */
    emxArray_int32_T *tris;	/*!< triangles */
    emxArray_real_T *nor;	/*!< point normals */

} hiPropMesh;


/*!
 * \brief Initialize a hiProp mesh and set the initial pointer to be NULL
 * \param pmesh Address of the hiProp mesh pointer
 */
extern void hpInitMesh(hiPropMesh **pmesh);

/*!
 * \brief Free a hiProp mesh and set the pointer to be NULL
 * \param pmesh Address of the hiProp mesh pointer
 */
extern void hpFreeMesh(hiPropMesh **pmesh);

/*!
 * Read an ascii triangular vtk file with data type POLYGON.
 * \param name input file name
 * \param mesh mesh for storing the data read from file
 */
extern int hpReadPolyMeshVtk3d(const char* name, hiPropMesh* mesh);
/*!
 * Write an ascii triangular vtk file with data type POLYGON.
 * \param name output file name
 * \param mesh mesh for output
 */
extern int hpWritePolyMeshVtk3d(const char* name, hiPropMesh* mesh);
/*!
 * Read an ascii triangular vtk file with data type UNSTURCTURED_GRID.
 * \param name input file name
 * \param mesh mesh for storing the data read from file
 */
extern int hpReadUnstrMeshVtk3d(const char* name, hiPropMesh* mesh);
/*!
 * Write an ascii triangular vtk file with data type UNSTRUCTURED_GRID.
 * \param name output file name
 * \param mesh mesh for output
 */
extern int hpWriteUnstrMeshVtk3d(const char* name, hiPropMesh* mesh);



#endif
