#ifndef BACKPROP_H
#define BACKPROP_H

#include"loadData.h"
#include"neuralNet.h"

typedef struct {
    int index, done, correct, epochs;
    ImageDataPtr images;
    LabelDataPtr labels;
    NeuralNetPtr net;
    float ***weightGradient, **biasGradient, **activations, **deriv;
} TrainEnvi;

typedef TrainEnvi *TrainEnviPtr;

void alloc_te(TrainEnviPtr*, ImageDataPtr, LabelDataPtr, NeuralNetPtr);
void clear_te(TrainEnviPtr);
void update_nn(TrainEnviPtr, float);
void back_prop(TrainEnviPtr, int);
void forward_prop(TrainEnviPtr);     // Helper
float avg_cost(TrainEnviPtr);
void gradient_norm_clip(TrainEnviPtr, float);

#endif //BACKPROP_H