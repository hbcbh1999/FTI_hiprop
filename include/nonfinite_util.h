#ifndef __NONFINITE_UTIL_H__
#define __NONFINITE_UTIL_H__


#include "stdafx.h"
#include <stddef.h>
#include "rtwtypes.h"

EXTERN_C real_T rtInf;
EXTERN_C real_T rtMinusInf;
EXTERN_C real_T rtNaN;
EXTERN_C real32_T rtInfF;
EXTERN_C real32_T rtMinusInfF;
EXTERN_C real32_T rtNaNF;
EXTERN_C void rt_InitInfAndNaN(size_t realSize);
EXTERN_C boolean_T rtIsInf(real_T value);
EXTERN_C boolean_T rtIsInfF(real32_T value);
EXTERN_C boolean_T rtIsNaN(real_T value);
EXTERN_C boolean_T rtIsNaNF(real32_T value);

typedef struct {
  struct {
    uint32_T wordH;
    uint32_T wordL;
  } words;
} BigEndianIEEEDouble;

typedef struct {
  struct {
    uint32_T wordL;
    uint32_T wordH;
  } words;
} LittleEndianIEEEDouble;

typedef struct {
  union {
    real32_T wordLreal;
    uint32_T wordLuint;
  } wordL;
} IEEESingle;

EXTERN_C real_T rtGetNaN(void);
EXTERN_C real32_T rtGetNaNF(void);
EXTERN_C real_T rtGetInf(void);
EXTERN_C real32_T rtGetInfF(void);
EXTERN_C real_T rtGetMinusInf(void);
EXTERN_C real32_T rtGetMinusInfF(void);

#endif
