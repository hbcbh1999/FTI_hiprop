/*!
 * \file memutil.c
 * \brief Implementation of memutil.h
 *
 * \author Yijie Zhou
 * \date 2012.08.23
 */
#include "memutil.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void emxInit_boolean_T(emxArray_boolean_T **pEmxArray, int32_T numDimensions)
{
  emxArray_boolean_T *emxArray;
  int32_T loop_ub;
  int32_T i;
  *pEmxArray = (emxArray_boolean_T *)malloc(sizeof(emxArray_boolean_T));
  emxArray = *pEmxArray;
  emxArray->data = (boolean_T *)NULL;
  emxArray->numDimensions = numDimensions;
  emxArray->size = (int32_T *)malloc((uint32_T)(sizeof(int32_T) * numDimensions));
  emxArray->allocatedSize = 0;
  emxArray->canFreeData = TRUE;
  loop_ub = numDimensions - 1;
  for (i = 0; i <= loop_ub; i++) {
    emxArray->size[i] = 0;
  }
}

void emxInit_int32_T(emxArray_int32_T **pEmxArray, int32_T numDimensions)
{
  emxArray_int32_T *emxArray;
  int32_T loop_ub;
  int32_T i;
  *pEmxArray = (emxArray_int32_T *)malloc(sizeof(emxArray_int32_T));
  emxArray = *pEmxArray;
  emxArray->data = (int32_T *)NULL;
  emxArray->numDimensions = numDimensions;
  emxArray->size = (int32_T *)malloc((uint32_T)(sizeof(int32_T) * numDimensions));
  emxArray->allocatedSize = 0;
  emxArray->canFreeData = TRUE;
  loop_ub = numDimensions - 1;
  for (i = 0; i <= loop_ub; i++) {
    emxArray->size[i] = 0;
  }
}

void emxInit_real_T(emxArray_real_T **pEmxArray, int32_T numDimensions)
{
  emxArray_real_T *emxArray;
  int32_T loop_ub;
  int32_T i;
  *pEmxArray = (emxArray_real_T *)malloc(sizeof(emxArray_real_T));
  emxArray = *pEmxArray;
  emxArray->data = (real_T *)NULL;
  emxArray->numDimensions = numDimensions;
  emxArray->size = (int32_T *)malloc((uint32_T)(sizeof(int32_T) * numDimensions));
  emxArray->allocatedSize = 0;
  emxArray->canFreeData = TRUE;
  loop_ub = numDimensions - 1;
  for (i = 0; i <= loop_ub; i++) {
    emxArray->size[i] = 0;
  }
}

emxArray_int32_T *emxCreateND_int32_T(int32_T numDimensions, int32_T *size)
{
  emxArray_int32_T *emx;
  int32_T numEl;
  int32_T loop_ub;
  int32_T i;
  emxInit_int32_T(&emx, numDimensions);
  numEl = 1;
  loop_ub = numDimensions - 1;
  for (i = 0; i <= loop_ub; i++) {
    numEl *= size[i];
    emx->size[i] = size[i];
  }

  emx->data = (int32_T *)calloc((uint32_T)numEl, sizeof(int32_T));
  emx->numDimensions = numDimensions;
  emx->allocatedSize = numEl;
  return emx;
}

emxArray_real_T *emxCreateND_real_T(int32_T numDimensions, int32_T *size)
{
  emxArray_real_T *emx;
  int32_T numEl;
  int32_T loop_ub;
  int32_T i;
  emxInit_real_T(&emx, numDimensions);
  numEl = 1;
  loop_ub = numDimensions - 1;
  for (i = 0; i <= loop_ub; i++) {
    numEl *= size[i];
    emx->size[i] = size[i];
  }

  emx->data = (real_T *)calloc((uint32_T)numEl, sizeof(real_T));
  emx->numDimensions = numDimensions;
  emx->allocatedSize = numEl;
  return emx;
}

emxArray_boolean_T *emxCreateND_boolean_T(int32_T numDimensions, int32_T *size)
{
  emxArray_boolean_T *emx;
  int32_T numEl;
  int32_T loop_ub;
  int32_T i;
  emxInit_boolean_T(&emx, numDimensions);
  numEl = 1;
  loop_ub = numDimensions - 1;
  for (i = 0; i <= loop_ub; i++) {
    numEl *= size[i];
    emx->size[i] = size[i];
  }

  emx->data = (boolean_T *)calloc((uint32_T)numEl, sizeof(boolean_T));
  emx->numDimensions = numDimensions;
  emx->allocatedSize = numEl;
  return emx;
}

emxArray_int32_T *emxCreateWrapperND_int32_T(int32_T *data, int32_T numDimensions, int32_T *size)
{
  emxArray_int32_T *emx;
  int32_T numEl;
  int32_T loop_ub;
  int32_T i;
  emxInit_int32_T(&emx, numDimensions);
  numEl = 1;
  loop_ub = numDimensions - 1;
  for (i = 0; i <= loop_ub; i++) {
    numEl *= size[i];
    emx->size[i] = size[i];
  }

  emx->data = data;
  emx->numDimensions = numDimensions;
  emx->allocatedSize = numEl;
  emx->canFreeData = FALSE;
  return emx;
}

emxArray_real_T *emxCreateWrapperND_real_T(real_T *data, int32_T numDimensions, int32_T *size)
{
  emxArray_real_T *emx;
  int32_T numEl;
  int32_T loop_ub;
  int32_T i;
  emxInit_real_T(&emx, numDimensions);
  numEl = 1;
  loop_ub = numDimensions - 1;
  for (i = 0; i <= loop_ub; i++) {
    numEl *= size[i];
    emx->size[i] = size[i];
  }

  emx->data = data;
  emx->numDimensions = numDimensions;
  emx->allocatedSize = numEl;
  emx->canFreeData = FALSE;
  return emx;
}

emxArray_boolean_T *emxCreateWrapperND_boolean_T(boolean_T *data, int32_T numDimensions, int32_T *size)
{
  emxArray_boolean_T *emx;
  int32_T numEl;
  int32_T loop_ub;
  int32_T i;
  emxInit_boolean_T(&emx, numDimensions);
  numEl = 1;
  loop_ub = numDimensions - 1;
  for (i = 0; i <= loop_ub; i++) {
    numEl *= size[i];
    emx->size[i] = size[i];
  }

  emx->data = data;
  emx->numDimensions = numDimensions;
  emx->allocatedSize = numEl;
  emx->canFreeData = FALSE;
  return emx;
}

emxArray_int32_T *emxCreateWrapper_int32_T(int32_T *data, int32_T rows, int32_T cols)
{
  emxArray_int32_T *emx;
  int32_T size[2];
  int32_T numEl;
  int32_T i;
  size[0] = rows;
  size[1] = cols;
  emxInit_int32_T(&emx, 2);
  numEl = 1;
  for (i = 0; i < 2; i++) {
    numEl *= size[i];
    emx->size[i] = size[i];
  }

  emx->data = data;
  emx->numDimensions = 2;
  emx->allocatedSize = numEl;
  emx->canFreeData = FALSE;
  return emx;
}

emxArray_real_T *emxCreateWrapper_real_T(real_T *data, int32_T rows, int32_T cols)
{
  emxArray_real_T *emx;
  int32_T size[2];
  int32_T numEl;
  int32_T i;
  size[0] = rows;
  size[1] = cols;
  emxInit_real_T(&emx, 2);
  numEl = 1;
  for (i = 0; i < 2; i++) {
    numEl *= size[i];
    emx->size[i] = size[i];
  }

  emx->data = data;
  emx->numDimensions = 2;
  emx->allocatedSize = numEl;
  emx->canFreeData = FALSE;
  return emx;
}

emxArray_boolean_T *emxCreateWrapper_boolean_T(boolean_T *data, int32_T rows, int32_T cols)
{
  emxArray_boolean_T *emx;
  int32_T size[2];
  int32_T numEl;
  int32_T i;
  size[0] = rows;
  size[1] = cols;
  emxInit_boolean_T(&emx, 2);
  numEl = 1;
  for (i = 0; i < 2; i++) {
    numEl *= size[i];
    emx->size[i] = size[i];
  }

  emx->data = data;
  emx->numDimensions = 2;
  emx->allocatedSize = numEl;
  emx->canFreeData = FALSE;
  return emx;
}


emxArray_int32_T *emxCreate_int32_T(int32_T rows, int32_T cols)
{
  emxArray_int32_T *emx;
  int32_T size[2];
  int32_T numEl;
  int32_T i;
  size[0] = rows;
  size[1] = cols;
  emxInit_int32_T(&emx, 2);
  numEl = 1;
  for (i = 0; i < 2; i++) {
    numEl *= size[i];
    emx->size[i] = size[i];
  }

  emx->data = (int32_T *)calloc((uint32_T)numEl, sizeof(int32_T));
  emx->numDimensions = 2;
  emx->allocatedSize = numEl;
  return emx;
}

emxArray_real_T *emxCreate_real_T(int32_T rows, int32_T cols)
{
  emxArray_real_T *emx;
  int32_T size[2];
  int32_T numEl;
  int32_T i;
  size[0] = rows;
  size[1] = cols;
  emxInit_real_T(&emx, 2);
  numEl = 1;
  for (i = 0; i < 2; i++) {
    numEl *= size[i];
    emx->size[i] = size[i];
  }

  emx->data = (real_T *)calloc((uint32_T)numEl, sizeof(real_T));
  emx->numDimensions = 2;
  emx->allocatedSize = numEl;
  return emx;
}

emxArray_boolean_T *emxCreate_boolean_T(int32_T rows, int32_T cols)
{
  emxArray_boolean_T *emx;
  int32_T size[2];
  int32_T numEl;
  int32_T i;
  size[0] = rows;
  size[1] = cols;
  emxInit_boolean_T(&emx, 2);
  numEl = 1;
  for (i = 0; i < 2; i++) {
    numEl *= size[i];
    emx->size[i] = size[i];
  }

  emx->data = (boolean_T *)calloc((uint32_T)numEl, sizeof(boolean_T));
  emx->numDimensions = 2;
  emx->allocatedSize = numEl;
  return emx;
}

void emxFree_boolean_T(emxArray_boolean_T **pEmxArray)
{
  if (*pEmxArray != (emxArray_boolean_T *)NULL) {
    if ((*pEmxArray)->canFreeData) {
      free((void *)(*pEmxArray)->data);
    }

    free((void *)(*pEmxArray)->size);
    free((void *)*pEmxArray);
    *pEmxArray = (emxArray_boolean_T *)NULL;
  }
}

void emxFree_int32_T(emxArray_int32_T **pEmxArray)
{
  if (*pEmxArray != (emxArray_int32_T *)NULL) {
    if ((*pEmxArray)->canFreeData) {
      free((void *)(*pEmxArray)->data);
    }

    free((void *)(*pEmxArray)->size);
    free((void *)*pEmxArray);
    *pEmxArray = (emxArray_int32_T *)NULL;
  }
}

void emxFree_real_T(emxArray_real_T **pEmxArray)
{
  if (*pEmxArray != (emxArray_real_T *)NULL) {
    if ((*pEmxArray)->canFreeData) {
      free((void *)(*pEmxArray)->data);
    }

    free((void *)(*pEmxArray)->size);
    free((void *)*pEmxArray);
    *pEmxArray = (emxArray_real_T *)NULL;
  }
}

void emxDestroyArray_int32_T(emxArray_int32_T *emxArray)
{
  emxFree_int32_T(&emxArray);
}

void emxDestroyArray_real_T(emxArray_real_T *emxArray)
{
  emxFree_real_T(&emxArray);
}

void emxDestroyArray_boolean_T(emxArray_boolean_T *emxArray)
{
    emxFree_boolean_T(&emxArray);
}

void emxEnsureCapacity(emxArray__common *emxArray, int32_T oldNumel, int32_T elementSize)
{
  int32_T newNumel;
  int32_T loop_ub;
  int32_T i;
  void *newData;
  newNumel = 1;
  loop_ub = emxArray->numDimensions - 1;
  for (i = 0; i <= loop_ub; i++) {
    newNumel *= emxArray->size[i];
  }

  if (newNumel > emxArray->allocatedSize) {
    if (emxArray->allocatedSize) {
      loop_ub = emxArray->allocatedSize;
      if (loop_ub < 16) {
        loop_ub = 16;
       }

      while (loop_ub < newNumel) {
        loop_ub <<= 1;
      }
    } else {
      loop_ub = newNumel;
    }

    newData = calloc((uint32_T)loop_ub, (uint32_T)elementSize);
    if (emxArray->data != NULL) {
      memcpy(newData, emxArray->data, (uint32_T)(elementSize * oldNumel));
      if (emxArray->canFreeData) {
        free(emxArray->data);
      }
    }

    emxArray->data = newData;
    emxArray->allocatedSize = loop_ub;
    emxArray->canFreeData = TRUE;
  }
}

void addColumnToArray_common(emxArray__common *emxArray, int32_T numCol, uint32_T elementSize)
{
    void *newData;

    int numElemOld = emxArray->size[0] * emxArray->size[1];
    int numElemNew = emxArray->size[0] * (emxArray->size[1] + numCol);

    emxArray->size[1] += numCol;

    newData = calloc((uint32_T)numElemNew, elementSize);

    if (emxArray->data != NULL) {
	memcpy(newData, emxArray->data, (uint32_T)(elementSize * numElemOld));
	free(emxArray->data);
    }
    emxArray->data = newData;
    emxArray->allocatedSize = numElemNew;
}

void addColumnToArray_int32_T(emxArray_int32_T *emxArray, int32_T numCol)
{
    addColumnToArray_common( (emxArray__common *)emxArray, numCol, sizeof(int32_T) );
    emxArray->data = (int32_T *) emxArray->data;
}
void addColumnToArray_real_T(emxArray_real_T *emxArray, int32_T numCol)
{
    addColumnToArray_common( (emxArray__common *)emxArray, numCol, sizeof(real_T) );
    emxArray->data = (real_T *) emxArray->data;
}
void addColumnToArray_boolean_T(emxArray_boolean_T *emxArray, int32_T numCol)
{
    addColumnToArray_common( (emxArray__common *)emxArray, numCol, sizeof(boolean_T) );
    emxArray->data = (boolean_T *) emxArray->data;
}

void addRowToArray_int32_T(emxArray_int32_T *emxArray, int32_T numRow)
{
    int i;
    void *newData;

    int numRowOld = emxArray->size[0];
    int numRowNew = emxArray->size[0] + numRow;

    int numColOld, numColNew;
    numColNew = numColOld = emxArray->size[1];

    emxArray->size[0] = numRowNew;

    int numElemNew = numRowNew*numColNew;

    newData = calloc((uint32_T)numElemNew, sizeof(int32_T));

    if (emxArray->data != NULL)
    {
	for (i = 0; i < numColNew; i++)
	    memcpy((int32_T *)newData + i*numRowNew, emxArray->data + i*numRowOld, (uint32_T)(sizeof(int32_T) * numRowOld));
	free(emxArray->data);
    }

    emxArray->data = (int32_T *) newData;
    emxArray->allocatedSize = numElemNew;
}
void addRowToArray_real_T(emxArray_real_T *emxArray, int32_T numRow)
{
    int i;
    void *newData;

    int numRowOld = emxArray->size[0];
    int numRowNew = emxArray->size[0] + numRow;

    int numColOld, numColNew;
    numColNew = numColOld = emxArray->size[1];

    emxArray->size[0] = numRowNew;

    int numElemNew = numRowNew*numColNew;

    newData = calloc((uint32_T)numElemNew, sizeof(real_T));

    if (emxArray->data != NULL)
    {
	for (i = 0; i < numColNew; i++)
	    memcpy((real_T *)newData + i*numRowNew, emxArray->data + i*numRowOld, (uint32_T)(sizeof(real_T) * numRowOld));
	free(emxArray->data);
    }

    emxArray->data = (real_T *) newData;
    emxArray->allocatedSize = numElemNew;
}
void addRowToArray_boolean_T(emxArray_boolean_T *emxArray, int32_T numRow)
{
    int i;
    void *newData;

    int numRowOld = emxArray->size[0];
    int numRowNew = emxArray->size[0] + numRow;

    int numColOld, numColNew;
    numColNew = numColOld = emxArray->size[1];

    emxArray->size[0] = numRowNew;

    int numElemNew = numRowNew*numColNew;

    newData = calloc((uint32_T)numElemNew, sizeof(boolean_T));

    if (emxArray->data != NULL)
    {
	for (i = 0; i < numColNew; i++)
	    memcpy((boolean_T *)newData + i*numRowNew, emxArray->data + i*numRowOld, (uint32_T)(sizeof(boolean_T) * numRowOld));
	free(emxArray->data);
    }

    emxArray->data = (boolean_T *) newData;
    emxArray->allocatedSize = numElemNew;
}


void printArray_int32_T(const emxArray_int32_T *emxArray)
{
    int i,j;
    for (i = 1; i <= emxArray->size[0]; i++)
    {
	for (j = 1; j <= emxArray->size[1]; j++)
	    printf("%3d ",emxArray->data[I2dm(i,j,emxArray->size)]);
	printf("\n");
    }
}

void printArray_real_T(const emxArray_real_T *emxArray)
{
    int i,j;
    for (i = 1; i <= emxArray->size[0]; i++)
    {
	for (j = 1; j <= emxArray->size[1]; j++)
	    printf("%10.8g ",emxArray->data[I2dm(i,j,emxArray->size)]);
	printf("\n");
    }
}
void printArray_boolean_T(const emxArray_boolean_T *emxArray)
{
    int i,j;
    for (i = 1; i <= emxArray->size[0]; i++)
    {
	for (j = 1; j <= emxArray->size[1]; j++)
	    printf("%u ",emxArray->data[I2dm(i,j,emxArray->size)]);
	printf("\n");
    }
}

