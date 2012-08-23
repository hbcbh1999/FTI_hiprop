/*!
 * \file test.c
 * \brief Test file of hi-prop library
 *
 * \author Yijie Zhou
 * \date 2012.08.23
 */

#include <stdio.h>
#include "memutil.h"

int main()
{

    printf("\n Welcome to the test of Hi-Prop Library \n");

    int i,j,k;

    k = 1;
    emxArray_int32_T *test = emxCreate_int32_T(4, 3);

    for (i = 1; i <= 4; i++)
	for (j = 1; j <=3; j++)
	    test->data[I2dm(i,j,test->size)] = k++;
	    
    printArray_int32_T(test);

    addColumnToArray_int32_T(test, 2);

    printArray_int32_T(test);

    addRowToArray_int32_T(test, 4);

    printArray_int32_T(test);

    emxFree_int32_T(&test);



    return 0;
}
