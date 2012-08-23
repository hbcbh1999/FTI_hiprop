/*!
 * \file rtwtypes.h
 * \brief HiProp Project Common Types
 *
 * \detail HiProp project common types, consistent with types generated from codegen in
 * matlab.
 *
 * \author Yijie Zhou
 * \date 2012.08.22
 */

#ifndef __RTWTYPES_H__
#define __RTWTYPES_H__
#ifndef TRUE
# define TRUE (1U)
#endif
#ifndef FALSE
# define FALSE (0U)
#endif
#ifndef __TMWTYPES__
#define __TMWTYPES__

#include <limits.h>

/*=======================================================================* 
 * Target hardware information
 *   Device type: Generic->MATLAB Host Computer
 *   Number of bits:     char:   8    short:   16    int:  32
 *                       long:  64      native word size:  64
 *   Byte ordering: LittleEndian
 *   Signed integer division rounds to: Zero
 *   Shift right on a signed integer as arithmetic shift: on
 *=======================================================================*/

/*=======================================================================* 
 * Fixed width word size data types:                                     * 
 *   int8_T, int16_T, int32_T     - signed 8, 16, or 32 bit integers     * 
 *   uint8_T, uint16_T, uint32_T  - unsigned 8, 16, or 32 bit integers   * 
 *   real32_T, real64_T           - 32 and 64 bit floating point numbers * 
 *=======================================================================*/

typedef signed char int8_T;		/*!< 8bit signed char */
typedef unsigned char uint8_T;		/*!< 8bit unsigned char */
typedef short int16_T;			/*!< 16bit short */
typedef unsigned short uint16_T;	/*!< 16bit unsigned short */
typedef int int32_T;			/*!< 32bit int */
typedef unsigned int uint32_T;		/*!< 32bit unsigned int */
typedef long int64_T;			/*!< 64bit long */
typedef unsigned long uint64_T;		/*!< 64bit unsigned long */
typedef float real32_T;			/*!< 32bit float */
typedef double real64_T;		/*!< 64bit double */

/*===========================================================================* 
 * Generic type definitions: real_T, time_T, boolean_T, int_T, uint_T,       * 
 *                           ulong_T, char_T and byte_T.                     * 
 *===========================================================================*/

typedef double real_T;			/*!< 64bit double */
typedef double time_T;			/*!< 64bit double */
typedef unsigned char boolean_T;	/*!< 8bit unsigned char */
typedef int int_T;			/*!< 32bit int */
typedef unsigned uint_T;		/*!< 32bit unsigned int */
typedef unsigned long ulong_T;		/*!< 64bit unsigned long */
typedef char char_T;			/*!< 8bit signed char */
typedef char_T byte_T;			/*!< 8bit signed char */

/*===========================================================================* 
 * Complex number type definitions                                           * 
 *===========================================================================*/
#define CREAL_T

/*!
 * \brief 32bit float complex number
 */
   typedef struct {  
     real32_T re;  
     real32_T im;  
   } creal32_T;  
/*!
 * \brief 64bit double complex number
 */
   typedef struct {  
     real64_T re;  
     real64_T im;  
   } creal64_T;  
/*!
 * \brief 64bit double complex number
 */
   typedef struct {  
     real_T re;  
     real_T im;  
   } creal_T;  
/*!
 * \brief 8bit signed char complex number
 */
   typedef struct {  
     int8_T re;  
     int8_T im;  
   } cint8_T;  
/*!
 * \brief 8bit unsigned char complex number
 */
   typedef struct {  
     uint8_T re;  
     uint8_T im;  
   } cuint8_T;  
/*!
 * \brief 16bit int complex number
 */
   typedef struct {  
     int16_T re;  
     int16_T im;  
   } cint16_T;  
/*!
 * \brief 16bit unsigned int complex number
 */
   typedef struct {  
     uint16_T re;  
     uint16_T im;  
   } cuint16_T;  
/*!
 * \brief 32bit int complex number
 */
   typedef struct {  
     int32_T re;  
     int32_T im;  
   } cint32_T;  
/*!
 * \brief 32bit unsigned int complex number
 */
   typedef struct {  
     uint32_T re;  
     uint32_T im;  
   } cuint32_T;  
/*!
 * \brief 64bit long complex number
 */
   typedef struct {  
     int64_T re;  
     int64_T im;  
   } cint64_T;  
/*!
 * \brief 64bit unsigned long complex number
 */
   typedef struct {  
     uint64_T re;  
     uint64_T im;  
   } cuint64_T;  


/*=======================================================================* 
 * Min and Max:                                                          * 
 *   int8_T, int16_T, int32_T     - signed 8, 16, or 32 bit integers     * 
 *   uint8_T, uint16_T, uint32_T  - unsigned 8, 16, or 32 bit integers   * 
 *=======================================================================*/

#define MAX_int8_T  	((int8_T)(127))				/*!< Max number for char */
#define MIN_int8_T  	((int8_T)(-128))			/*!< Min number for char */
#define MAX_uint8_T 	((uint8_T)(255))			/*!< Max number for unsigned char */
#define MIN_uint8_T 	((uint8_T)(0))				/*!< Min number for unsigned char */
#define MAX_int16_T 	((int16_T)(32767))			/*!< Max number for short */
#define MIN_int16_T 	((int16_T)(-32768))			/*!< Min number for short */
#define MAX_uint16_T	((uint16_T)(65535))			/*!< Max number for unsigned short */
#define MIN_uint16_T	((uint16_T)(0))				/*!< Min number for unsigned short */
#define MAX_int32_T 	((int32_T)(2147483647))			/*!< Max number for int */
#define MIN_int32_T 	((int32_T)(-2147483647-1))		/*!< Min number for int */
#define MAX_uint32_T	((uint32_T)(0xFFFFFFFFU))		/*!< Max number for unsigned int */
#define MIN_uint32_T	((uint32_T)(0))				/*!< Min number for unsigned int */
#define MAX_int64_T	((int64_T)(9223372036854775807L))	/*!< Max number for long */
#define MIN_int64_T	((int64_T)(-9223372036854775807L-1L))	/*!< Min mumber for long */
#define MAX_uint64_T	((uint64_T)(0xFFFFFFFFFFFFFFFFUL))	/*!< Max number for unsigned long */
#define MIN_uint64_T	((uint64_T)(0UL))			/*!< Min number for unsigned long */

/* Logical type definitions */
#if !defined(__cplusplus) && !defined(__true_false_are_keywords)
#  ifndef false
#   define false (0U)
#  endif
#  ifndef true
#   define true (1U)
#  endif
#endif

/*
 * MATLAB for code generation assumes the code is compiled on a target using a 2's compliment representation
 * for signed integer values.
 */
#if ((SCHAR_MIN + 1) != -SCHAR_MAX)
#error "This code must be compiled using a 2's complement representation for signed integer values"
#endif

/*
 * Maximum length of a MATLAB identifier (function/variable)
 * including the null-termination character. Referenced by
 * rt_logging.c and rt_matrx.c.
 */
#define TMW_NAME_LENGTH_MAX	64		/*!< Maximum length of a MATLAB identifier */

#endif
#endif
