#ifndef __IO_H__
#define __IO_H__

#include "stdafx.h"

extern int FindString(FILE* file, const char* string);
extern int ReadMeshVtk3d(FILE* file, hiPropMesh* mesh);
extern int WriteMeshVtk3d(const char* name, hiPropMesh* mesh);

#endif
