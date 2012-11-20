/*!
 * \file emx_util.h
 * \brief basic utility functions for memory generated by codegen
 *
 * \detail 
 * Memory: Utility functions for allocating and deallocating arrays and
 * mesh defined in datatype.h
 * Array part consistent with C code generated by codegen from MATLAB code.
 *
 * \author Yijie Zhou 
 *
 * \date 2012.10.01
 */

#ifndef __EMX_UTIL_H__
#define __EMX_UTIL_H__

#include "stdafx.h"

/*!
 * \brief Initialization of a boolean array
 * 
 * Does not allocate memory for data
 * \param pEmxArray address of a boolean array pointer
 * \param numDimensions number of dimensions
 */
EXTERN_C void emxInit_boolean_T(emxArray_boolean_T **pEmxArray, int32_T numDimensions);

/*!
 * \brief Initialization of a 32bit int array
 * 
 * Does not allocate memory for data
 * \param pEmxArray address of a boolean array pointer
 * \param numDimensions number of dimensions
 */
EXTERN_C void emxInit_int32_T(emxArray_int32_T **pEmxArray, int32_T numDimensions);
/*!
 * \brief Initialization of a 64bit double array
 * 
 * Does not allocate memory for data
 * \param pEmxArray address of a boolean array pointer
 * \param numDimensions number of dimensions
 */
EXTERN_C void emxInit_real_T(emxArray_real_T **pEmxArray, int32_T numDimensions);

/*!
 * \brief Create a N-dimensional 32bit int array with initial value 0
 * \param numDimensions number of dimensions
 * \param size size for each dimention
 * \return a pointer to the generated array
 */
EXTERN_C emxArray_int32_T *emxCreateND_int32_T(int32_T numDimensions, int32_T *size);

/*!
 * \brief Create a N-dimensional 64bit double array with initial value 0
 * \param numDimensions number of dimensions
 * \param size size for each dimention
 * \return a pointer to the generated array
 */
EXTERN_C emxArray_real_T *emxCreateND_real_T(int32_T numDimensions, int32_T *size);

/*!
 * \brief Create a N-dimensional 8bit boolean array with initial value FALSE
 * \param numDimensions number of dimensions
 * \param size size for each dimention
 * \return a pointer to the generated array
 */
EXTERN_C emxArray_boolean_T *emxCreateND_boolean_T(int32_T numDimensions, int32_T *size);

/*!
 * \brief Create a 2D 32bit int array with initial value 0
 * \param rows number of rows
 * \param cols number of columns
 * \return a pointer to the generated array
 */
EXTERN_C emxArray_int32_T *emxCreate_int32_T(int32_T rows, int32_T cols);

/*!
 * \brief Create a 2D 64bit double array with initial value 0
 * \param rows number of rows
 * \param cols number of columns
 * \return a pointer to the generated array
 */
EXTERN_C emxArray_real_T *emxCreate_real_T(int32_T rows, int32_T cols);

/*!
 * \brief Create a 2D 8bit boolean array with initial value FALSE
 * \param rows number of rows
 * \param cols number of columns
 * \return a pointer to the generated array
 */
EXTERN_C emxArray_boolean_T *emxCreate_boolean_T(int32_T rows, int32_T cols);

/*!
 * \brief Create a wrapper for an N-dimensional 32bit int array 
 *
 * The data being wrapped could not be freed by calling #emxFree_int32_T
 * \param data data stored in a 1D array
 * \param numDimensions number of dimensions
 * \param size size for each dimension
 * \return a pointer to the generated array
 */
EXTERN_C emxArray_int32_T *emxCreateWrapperND_int32_T(int32_T *data, int32_T numDimensions, int32_T *size);

/*!
 * \brief Create a wrapper for an N-dimensional 64bit double array 
 *
 * The data being wrapped could not be freed by calling #emxFree_real_T
 * \param data data stored in a 1D array
 * \param numDimensions number of dimensions
 * \param size size for each dimension
 * \return a pointer to the generated array
 */
EXTERN_C emxArray_real_T *emxCreateWrapperND_real_T(real_T *data, int32_T numDimensions, int32_T *size);

/*!
 * \brief Create a wrapper for an N-dimensional 8bit boolean array 
 *
 * The data being wrapped could not be freed by calling #emxFree_boolean_T
 * \param data data stored in a 1D array
 * \param numDimensions number of dimensions
 * \param size size for each dimension
 * \return a pointer to the generated array
 */
EXTERN_C emxArray_boolean_T *emxCreateWrapperND_boolean_T(boolean_T *data, int32_T numDimensions, int32_T *size);

/*!
 * \brief Create a wrapper for an 2D 32bit int array 
 *
 * The data being wrapped could not be freed by calling #emxFree_int32_T
 * \param data data stored in a 1D array
 * \param rows number of rows
 * \param cols number of columns
 * \return a pointer to the generated array
 */
EXTERN_C emxArray_int32_T *emxCreateWrapper_int32_T(int32_T *data, int32_T rows, int32_T cols);

/*!
 * \brief Create a wrapper for an 2D 64bit double array 
 *
 * The data being wrapped could not be freed by calling #emxFree_real_T
 * \param data data stored in a 1D array
 * \param rows number of rows
 * \param cols number of columns
 * \return a pointer to the generated array
 */
EXTERN_C emxArray_real_T *emxCreateWrapper_real_T(real_T *data, int32_T rows, int32_T cols);

/*!
 * \brief Create a wrapper for an 2D 8bit boolean array 
 *
 * The data being wrapped could not be freed by calling #emxFree_boolean_T
 * \param data data stored in a 1D array
 * \param rows number of rows
 * \param cols number of columns
 * \return a pointer to the generated array
 */
EXTERN_C emxArray_boolean_T *emxCreateWrapper_boolean_T(boolean_T *data, int32_T rows, int32_T cols);

/*!
 * \brief Deallocate the memory of a 8bit boolean array
 *
 * \param pEmxArray address of the array being deallocated
 */
EXTERN_C void emxFree_boolean_T(emxArray_boolean_T **pEmxArray);

/*!
 * \brief Deallocate the memory of a 32bit int array
 *
 * \param pEmxArray address of the array being deallocated
 */
EXTERN_C void emxFree_int32_T(emxArray_int32_T **pEmxArray);

/*!
 * \brief Deallocate the memory of a 64bit double array
 *
 * \param pEmxArray address of the array being deallocated
 */
EXTERN_C void emxFree_real_T(emxArray_real_T **pEmxArray);

/*!
 * \brief Function for automatically increase the array generated by codgen
 * 
 * \param emxArray pointer to common type array
 * \param oldNumel old number of elements
 * \param elementSize size of each element
 */
EXTERN_C void emxEnsureCapacity(emxArray__common *emxArray, int32_T oldNumel, int32_T elementSize);

EXTERN_C void emxDestroyArray_int32_T(emxArray_int32_T *emxArray);
EXTERN_C void emxDestroyArray_real_T(emxArray_real_T *emxArray);
EXTERN_C void emxDestroyArray_boolean_T(emxArray_boolean_T *emxArray);



#endif
