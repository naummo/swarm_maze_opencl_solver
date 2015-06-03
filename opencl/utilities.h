/* Custom memory types */
#ifndef TYPES_DECLARED
#define TYPES_DECLARED

typedef union {
    FLOATN v;
    float a[DIMENSIONS];
} v_and_a; //vector_and_array

typedef union {
    unsigned int uintVal;
    float floatVal;
} uint_and_float;

// We are not using OpenCL bools as their size is undefined and they
// cannot be passed as arguments to kernels.
typedef uchar boolean;

#endif

/* Prototypes */
void atomic_add_local_float(volatile __local float* source, float* operand);

void atomic_add_global_float(volatile __global float* source, float* operand);

void atomic_add_local_arr(volatile __local float* source, float* operand);

void atomic_add_global_arr(volatile __global float* source, float* operand);

void atomic_add_local_v(volatile __local float* source, FLOATN* operand);

void atomic_add_global_v(volatile __global float* source, FLOATN* operand);

void fetch_local(__local float* src, float* dest, int len);

//void fetch_const(__constant float* src, float* dest, int len);

void fetch_global(__global float* src, float* dest, int len);

int float_to_int(float f);

float int_to_float(int i);
