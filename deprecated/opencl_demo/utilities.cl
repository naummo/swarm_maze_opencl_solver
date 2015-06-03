#include "utilities.h"

/* Function implementations */
void atomic_add_local(volatile __local float* source, float* operand) {
    /* Serialized atomic addition of floats.
    AWOOGA! Find copyrights.
    http://simpleopencl.blogspot.co.uk/2013/05/atomic-operations-and-floats-in-opencl.html
    */
    int_and_float newVal;
    int_and_float prevVal;
    int i;

    for (i = 0; i < DIMENSIONS; i++)
        do {
            prevVal.floatVal = source[i];
            newVal.floatVal = prevVal.floatVal + operand[i];
        } while (atomic_cmpxchg((volatile __local unsigned int *)&source[i], 
                                prevVal.intVal, newVal.intVal) != prevVal.intVal);
}

void atomic_add_global(volatile __global float* source, float* operand) {
    /* Serialized atomic addition of floats.
    AWOOGA! Find copyrights.
    http://simpleopencl.blogspot.co.uk/2013/05/atomic-operations-and-floats-in-opencl.html
    */

    int_and_float newVal;
    int_and_float prevVal;
    int i;

    for (i = 0; i < DIMENSIONS; i++)
        do {
            prevVal.floatVal = source[i];
            newVal.floatVal = prevVal.floatVal + operand[i];
        } while (atomic_cmpxchg((volatile __global unsigned int *)&source[i], 
                                prevVal.intVal, newVal.intVal) != prevVal.intVal);
}

void atomic_add_local_v(volatile __local float* source, FLOATN* operand) {
    float operand_array[DIMENSIONS];
    VSTOREN(*operand, operand_array);
    atomic_add_local(source, operand_array);
}

void atomic_add_global_v(volatile __global float* source, FLOATN* operand) {
    float operand_array[DIMENSIONS];
    VSTOREN(*operand, operand_array);
    atomic_add_global(source, operand_array);
}

void subtract_arrays(float result[], __constant float v1[], __constant float v2[]) {
    int i;
    for (i = 0; i < DIMENSIONS; i++)
        result[i] = v1[i] - v2[i];
}

void fetch_local(__local float* src, float* dest, int len) {
    int i;
    for (i = 0; i < len; i++)
        dest[i] = src[i];
}

/*void fetch_const(__constant float* src, float* dest, int len) {
    int i;
    for (i = 0; i < len; i++)
        dest[i] = src[i];
}*/

void fetch_global(__global float* src, float* dest, int len) {
    int i;
    for (i = 0; i < len; i++)
        dest[i] = src[i];
}