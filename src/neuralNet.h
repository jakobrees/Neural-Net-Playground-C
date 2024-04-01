#ifndef NEURALNET_H
#define NEURALNET_H

#include<stdio.h>
#include<math.h>
#include<stdlib.h>
#include<string.h>

// structs & typedefs

typedef struct {
    int input_num;
    int size;
    int *layer_size;
    float ***weights;
    float **biases;

    float (*mainActFunc)(float);    // input is logits
    float (*mainActDeri)(float);    // input is activation

    void (*outActFunc)(float*, int);
    float (*lossFunc)(float*, int, int);
    void (*lossFuncDeri)(float*, float*, int, int);
} NeuralNet;

typedef NeuralNet *NeuralNetPtr;


// functions
void alloc_nn(NeuralNetPtr);
void clear_nn(NeuralNetPtr);
void init_rand_nn(NeuralNetPtr, unsigned int);
void he_init_rand_nn(NeuralNetPtr, unsigned int);
float* run_nn(NeuralNetPtr, unsigned char *);

#endif //NEURALNET_H