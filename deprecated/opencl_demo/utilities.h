/* Custom memory types */
#ifndef V_AND_A_DECLARED
#define V_AND_A_DECLARED

typedef union {
    FLOATN v;
    float a[DIMENSIONS];
} v_and_a; //vector_and_array

#endif

#ifndef INT_AND_FLOAT_DECLARED
#define INT_AND_FLOAT_DECLARED

typedef union {
    unsigned int intVal;
    float floatVal;
} int_and_float;

#endif

/* Prototypes */
void atomic_add_local(volatile __local float* source, float* operand);

void atomic_add_global(volatile __global float* source, float* operand);

void atomic_add_local_v(volatile __local float* source, FLOATN* operand);

void atomic_add_global_v(volatile __global float* source, FLOATN* operand);

void fetch_local(__local float* src, float* dest, int len);

//void fetch_const(__constant float* src, float* dest, int len);

void fetch_global(__global float* src, float* dest, int len);