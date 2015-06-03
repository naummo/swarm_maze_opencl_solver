#include "utilities.h"

/* Function implementations */
void atomic_add_local_float(volatile __local float* source, float* operand) {
    /* Serialized atomic addition of floats.
    TODO: Add copyrights.
    http://simpleopencl.blogspot.co.uk/2013/05/atomic-operations-and-floats-in-opencl.html
    */
    uint_and_float newVal;
    uint_and_float prevVal;

    do {
        prevVal.floatVal = *source;
        newVal.floatVal = prevVal.floatVal + *operand;
    } while (atomic_cmpxchg((volatile __local unsigned int *)source, 
                            prevVal.uintVal, newVal.uintVal) != prevVal.uintVal);
}


void atomic_add_global_float(volatile __global float* source, float* operand) {
    /* Serialized atomic addition of floats.
    TODO: Add copyrights.
    http://simpleopencl.blogspot.co.uk/2013/05/atomic-operations-and-floats-in-opencl.html
    */
    uint_and_float newVal;
    uint_and_float prevVal;

    do {
        prevVal.floatVal = *source;
        newVal.floatVal = prevVal.floatVal + *operand;
    } while (atomic_cmpxchg((volatile __global unsigned int *)source, 
                            prevVal.uintVal, newVal.uintVal) != prevVal.uintVal);
}


void atomic_add_local_arr(volatile __local float* source, float* operand) {
    for (int i = 0; i < DIMENSIONS; i++)
        atomic_add_local_float(&source[i], &operand[i]);
}


void atomic_add_global_arr(volatile __global float* source, float* operand) {
    for (int i = 0; i < DIMENSIONS; i++)
        atomic_add_global_float(&source[i], &operand[i]);
}


void atomic_add_local_v(volatile __local float* source, FLOATN* operand) {
    float operand_array[DIMENSIONS];
    VSTOREN(*operand, operand_array);
    atomic_add_local_arr(source, operand_array);
}


void atomic_add_global_v(volatile __global float* source, FLOATN* operand) {
    float operand_array[DIMENSIONS];
    VSTOREN(*operand, operand_array);
    atomic_add_global_arr(source, operand_array);
}


void subtract_arrays(float result[], __constant float v1[], __constant float v2[]) {
    for (int i = 0; i < DIMENSIONS; i++)
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
    for (int i = 0; i < len; i++)
        dest[i] = src[i];
}

int float_to_int(float f) {
    return (int)(f / (float)ERROR_EXPONENT);
}

float int_to_float(int i) {
    return (float)i * (float)ERROR_EXPONENT;
}
