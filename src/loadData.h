#ifndef LOADDATA_H
#define LOADDATA_H

#include<stdio.h>
#include<stdlib.h>

#include"neuralNet.h"

// structs & typedefs
typedef struct {
    int count;
    unsigned char **data;
} ImageData;

typedef struct {
    int count;
    unsigned char *data;
} LabelData;

typedef ImageData *ImageDataPtr;
typedef LabelData *LabelDataPtr;


// functions
void load_all_labels(LabelDataPtr*, FILE *);
void load_all_images(ImageDataPtr*, FILE *);
void store_nn(NeuralNetPtr, FILE*);
void load_nn(NeuralNetPtr, FILE *);

#endif //LOADDATA_H