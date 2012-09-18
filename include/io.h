/*!
 * \file io.h
 * \brief I/O functions for mesh data in hiProp library
 *
 * \detail Currently support reading and writing ascii triangular mesh
 * for POLYGON and UNSTRUCTURED_GRID. 
 *
 * \author Chenzhe Diao, Yijie Zhou
 *
 * \date 2012.09.15
 */




#ifndef __IO_H__
#define __IO_H__

#include "stdafx.h"

/*!
 * Locate the current cursor after the searching string, if string not found,
 * return 0.
 * \param file file pointer
 * \param in_string string for search
 */
extern int FindString(FILE* file, const char* in_string);
/*!
 * Read an ascii triangular vtk file with data type POLYGON.
 * \param name input file name
 * \param mesh mesh for storing the data read from file
 */
extern int ReadPolyMeshVtk3d(const char* name, hiPropMesh* mesh);
/*!
 * Write an ascii triangular vtk file with data type POLYGON.
 * \param name output file name
 * \param mesh mesh for output
 */
extern int WritePolyMeshVtk3d(const char* name, hiPropMesh* mesh);
/*!
 * Read an ascii triangular vtk file with data type UNSTURCTURED_GRID.
 * \param name input file name
 * \param mesh mesh for storing the data read from file
 */
extern int ReadUnstrMeshVtk3d(const char* name, hiPropMesh* mesh);
/*!
 * Write an ascii triangular vtk file with data type UNSTRUCTURED_GRID.
 * \param name output file name
 * \param mesh mesh for output
 */
extern int WriteUnstrMeshVtk3d(const char* name, hiPropMesh* mesh);

#endif
