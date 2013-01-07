/*!
 * \file datatypes.h
 * \brief HiProp Project Basic Array Types
 *
 * \detail Array types consistent with C code generated by codegen from MATLAB code. Use
 * one dimensional array with supplementary information to represent high
 * dimensional arrays. Column-major order is used to be consistent with fortran
 * and MATLAB. In each dimension, array index starts from 1 which is consistent to
 * fortran and MATLAB.
 * 
 * Each array has 
 * 
 * data : place to store elements
 * 
 * size : place to store dimension size
 * 
 * allocatedSize: allocated size, equals to the number of elements in data
 * 
 * numDimensions: number of dimensions
 * 
 * canFreeData : flag about whether the data could be freed
 *
 * \author Yijie Zhou
 * \date 2012.08.22
 */

#ifndef __DATA_TYPE_H__
#define __DATA_TYPE_H__

#include "rtwtypes.h"

/* Type Definitions */

#ifndef struct_emxArray_int8_T
#define struct_emxArray_int8_T

/*!
 * \brief 8bit int array
 */
typedef struct emxArray_int8_T
{
  int8_T *data;			/*!< column-major order 8bit int array */
  int32_T *size;		/*!< array size in each dimension */
  int32_T allocatedSize;	/*!< allocated size */
  int32_T numDimensions;	/*!< number of dimensions */
  boolean_T canFreeData;	/*!< whether data could be freed */
} emxArray_int8_T;

#endif

#ifndef struct_emxArray_int32_T
#define struct_emxArray_int32_T
/*!
 * \brief 32bit int array
 */
typedef struct emxArray_int32_T
{
    int32_T *data;		/*!< column-major order 32bit int array */
    int32_T *size;		/*!< array size in each dimension */
    int32_T allocatedSize;	/*!< allocated size */
    int32_T numDimensions;	/*!< number of dimensions */
    boolean_T canFreeData;	/*!< whether data could be freed */
} emxArray_int32_T;
#endif


#ifndef struct_emxArray_real_T
#define struct_emxArray_real_T
/*!
 * \brief 64bit double array
 */
typedef struct emxArray_real_T
{
    real_T *data;		/*!< column-major order 64bit double array */
    int32_T *size;		/*!< array size in each dimension */
    int32_T allocatedSize;	/*!< allocated size */
    int32_T numDimensions;	/*!< number of dimensions */
    boolean_T canFreeData;	/*!< whether data could be freed */
} emxArray_real_T;
#endif

#ifndef struct_emxArray__common
#define struct_emxArray__common
/*!
 * \brief unkown type array
 *
 * This common array could be used to point to any type of array
 */
typedef struct emxArray__common
{
    void *data;			/*!< column-major order unkown type array */
    int32_T *size;		/*!< array size in each dimension */
    int32_T allocatedSize;	/*!< allocated size */
    int32_T numDimensions;	/*!< number of dimensions */
    boolean_T canFreeData;	/*!< whether data could be freed */
} emxArray__common;
#endif

#ifndef struct_emxArray_boolean_T
#define struct_emxArray_boolean_T
/*!
 * \brief 8bit unsigned char array
 */
typedef struct emxArray_boolean_T
{
    boolean_T *data;		/*!< column-major order 8bit unsigned char array */
    int32_T *size;		/*!< array size in each dimension */
    int32_T allocatedSize;	/*!< allocated size */
    int32_T numDimensions;	/*!< number of dimensions */
    boolean_T canFreeData;	/*!< whether data could be freed */
} emxArray_boolean_T;


#endif
#endif
