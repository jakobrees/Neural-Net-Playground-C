#include"neuralNet.h"

#define PIXEL_SCALE(x) (((float) (x)) / 255.0f)


// ------------------------    Function Declarations    ------------------------


float relu (float);
float relu_deri (float);
float leaky_relu (float);
float leaky_relu_deri (float);
float sigmoid (float);
float sigmoid_deri (float);
float tanhf_deri (float);
void softmax (float*, int);
void cross_entropy_deri_with_softmax (float*, float*, int, int);
float cross_entropy (float*, int, int);
float mean_squared_error (float*, int, int);


// ------------------------    Main NN Struct Functions    ------------------------


/**
 * Assumes that the struct space is already allocated and that, 
 * input_num, size, *layer_size, are all already specified. 
 * Allocates space for weigths and biases. **The activation 
 * fucntions of the neural net must be changed manually here.**
*/
void alloc_nn(NeuralNetPtr nnPtr) {

    // FUNCTION POINTER ASSIGNMENT IS IN HERE AND MAY BE CHANGED        <-------
    nnPtr->mainActFunc = &sigmoid;
    nnPtr->mainActDeri = &sigmoid_deri;
    nnPtr->outActFunc = &softmax;
    nnPtr->lossFunc = &cross_entropy;
    nnPtr->lossFuncDeri = &cross_entropy_deri_with_softmax;
    // FUNCTION POINTER ASSIGNMENT IS IN HERE AND MAY BE CHANGED        <-------
    
    // for each layer, alloc bias and weight space
    nnPtr->weights = (float***) malloc(nnPtr->size * sizeof(float**));
    if (nnPtr->weights == NULL) {
        fprintf(stderr, "Error allocating weight layer pointer.");
        exit(1);
    }

    nnPtr->biases = (float**) malloc(nnPtr->size * sizeof(float*));
    if (nnPtr->biases == NULL) {
        fprintf(stderr, "Error allocating baies layer pointer.");
        exit(1);
    }


    // allocate internal arrays for bias and weights
    for (int l = 0; l < nnPtr->size; l++) {

        // for the rows
        nnPtr->biases[l] = (float*) malloc(nnPtr->layer_size[l] * sizeof(float));
        if (nnPtr->biases[l] == NULL) {
            fprintf(stderr, "Error allocating baies row pointer.");
            exit(1);
        }
        nnPtr->weights[l] = (float**) malloc(nnPtr->layer_size[l] * sizeof(float*));
        if (nnPtr->weights[l] == NULL) {
            fprintf(stderr, "Error allocating weight row pointer.");
            exit(1);
        }

        // for the cols
        for (int r = 0; r < nnPtr->layer_size[l]; r++) {

            if (l == 0) nnPtr->weights[l][r] = (float*) malloc(nnPtr->input_num * sizeof(float));
            else nnPtr->weights[l][r] = (float*) malloc(nnPtr->layer_size[l-1] * sizeof(float));
            if (nnPtr->weights[l][r] == NULL) {
                fprintf(stderr, "Error allocating weight column pointer.");
                exit(1);
            }
        }
    }
}


/**
 * Only clears the weights and biases of the Neural net.
 * The struct itself is not deallocated.
*/
void clear_nn(NeuralNetPtr nnPtr) {

    for (int l = 0; l < nnPtr->size; l++) {
        free(nnPtr->biases[l]);
        for (int r = 0; r < nnPtr->layer_size[l]; r++) {
            free(nnPtr->weights[l][r]);
        }
        free(nnPtr->weights[l]);
    }

    free(nnPtr->biases);
    free(nnPtr->weights);
}


/**
 * This uses default random weight and bias initialization.
 * The values generated are in range (0,1].
*/
void init_rand_nn(NeuralNetPtr nnPtr, unsigned int seed) {

    // for consistency
    srand(seed);

    // give all weights and biases random values
    for (int l = 0; l < nnPtr->size; l++) {
        for (int r = 0; r < nnPtr->layer_size[l]; r++) {

            // random value for bias
            nnPtr->biases[l][r] = ((float)rand() + 1) / ((float)RAND_MAX + 1);

            // random values for all weights in this row
            if (l == 0) {
                for (int c = 0; c < nnPtr->input_num; c++) 
                    nnPtr->weights[l][r][c] = ((float)rand() + 1) / ((float)RAND_MAX + 1);
            } else {
                for (int c = 0; c < nnPtr->layer_size[l-1]; c++)
                    nnPtr->weights[l][r][c] = ((float)rand() + 1) / ((float)RAND_MAX + 1);
            }
        }
    }
}


/**
 * Uses He weight and bias initlialization. This is intended to 
 * be used for ReLU activation function based neural nets.
 * The values generated are in range (-sqrt(6/n_in),sqrt(6/n_in)).
*/
void he_init_rand_nn(NeuralNetPtr nnPtr, unsigned int seed) {

    // for consistency
    srand(seed);

    // claculate the uniform distribution range
    float range = sqrtf(6.0 / (float)nnPtr->input_num);

    // give all weights and biases random values
    for (int l = 0; l < nnPtr->size; l++) {
        for (int r = 0; r < nnPtr->layer_size[l]; r++) {

            // random value for bias
            nnPtr->biases[l][r] = ((float)rand() / (float)RAND_MAX) * (2 * range) - range;

            // random values for all weights in this row
            if (l == 0) {
                for (int c = 0; c < nnPtr->input_num; c++) 
                    nnPtr->weights[l][r][c] = ((float)rand() / (float)RAND_MAX) * (2 * range) - range;
            } else {
                for (int c = 0; c < nnPtr->layer_size[l-1]; c++)
                    nnPtr->weights[l][r][c] = ((float)rand() / (float)RAND_MAX) * (2 * range) - range;
            }
        }
    }
}


/**
 * Takes filled neural net with image pointer, returns output array.
 * runner must free output array.
*/
float* run_nn(NeuralNetPtr nnPtr, unsigned char * image) {

    // allocate space for the calculations
    float **activations = (float **) malloc(nnPtr->size * sizeof(float*));
    if (activations == NULL) {
        fprintf(stderr, "Error allocating calculation pointer space.");
        return NULL;
    }

    // begin forward propagation
    for (int l = 0; l < nnPtr->size; l++) {

        // alocate actual arrays for activations to be stored
        activations[l] = (float *) calloc(nnPtr->layer_size[l], sizeof(float));
        if (activations[l] == NULL) {
            fprintf(stderr, "Error allocating calculation space.");
            // free calculation space
            for (int layer = 0; layer < l; layer++) {
                free(activations[layer]);
            }
            free(activations);
            return NULL;
        }

        // now calc logits
        for (int r = 0; r < nnPtr->layer_size[l]; r++) {

            // add bias
            activations[l][r] += nnPtr->biases[l][r];

            // first layer weight products & activation
            if (l == 0) {
                for (int c = 0; c < nnPtr->input_num; c++) {
                    activations[l][r] += nnPtr->weights[l][r][c] * PIXEL_SCALE(image[c]);
                }
                // we have logits, pass through activation function
                activations[l][r] = nnPtr->mainActFunc(activations[l][r]);
            } 

            // last layer weight products & activation
            else if (l == nnPtr->size - 1) {
                for (int c = 0; c < nnPtr->layer_size[l-1]; c++) {
                    activations[l][r] += nnPtr->weights[l][r][c] * activations[l-1][c];
                }
                // we wait for all logits since some output act func need all logits
            }

            // hidden layer products & activation
            else {
                for (int c = 0; c < nnPtr->layer_size[l-1]; c++) {
                    activations[l][r] += nnPtr->weights[l][r][c] * activations[l-1][c];
                }
                // we have logits, pass through output activation function
                activations[l][r] = nnPtr->mainActFunc(activations[l][r]);
            }
        }
    }

    // apply output activation function after all output logits have been calculated
    nnPtr->outActFunc(activations[nnPtr->size-1], nnPtr->layer_size[nnPtr->size-1]);

    // create return array and free calc space.
    float *retArr = (float *) malloc(nnPtr->layer_size[nnPtr->size-1] * sizeof(float));
    if (retArr == NULL) {
        fprintf(stderr, "Error allocating return array.");
    } else {
        memcpy(retArr, activations[nnPtr->size-1], nnPtr->layer_size[nnPtr->size-1] * sizeof(float));
    }

    // free calculation space
    for (int l = 0; l < nnPtr->size; l++) {
        free(activations[l]);
    }
    free(activations);

    // return outputs
    return retArr;
}


// ------------------------    Main Activation Functions    ------------------------


/**
 * Input should be the logits of relevant node.
*/
float relu (float x) {
    return x > 0 ? x : 0.0f;
}

/**
 * Input should be the activations of relevant node.
*/
float relu_deri (float x) {
    return x > 0 ? 1.0f : 0.0f;
}

/**
 * Input should be the logits of relevant node.
*/
float leaky_relu (float x) {
    return x > 0 ? x : 0.01f * x;
}

/**
 * Input should be the activation of relevant node.
*/
float leaky_relu_deri (float x) {
    return x > 0 ? 1.0f : 0.01f;
}

/**
 * Input should be the logits of relevant node.
*/
float sigmoid (float x) {
    return 1.0f / (1.0f + expf( -1 * x)); 
}

/**
 * Input should be the activation of relevant node.
*/
float sigmoid_deri (float x) {
    return x * (1.0f - x);
}

/**
 * Input should be the activation of relevant node.
 * There is no activation function since <math.h> defines it.
*/
float tanhf_deri (float x) {
    return 1.0f - x * x;
}


// -----------------------    Output Activation Functions    -----------------------


/**
 * Input should be list of output floats.
*/
void softmax (float* outputs, int size) {
    float denom = 0;
    for (int i = 0; i < size; i++) {
        outputs[i] = expf(outputs[i]);
        denom += outputs[i];
    }
    for (int i = 0; i < size; i++) {
        outputs[i] /= denom;
    }
}


// -----------------------------    Loss Functions    -----------------------------


/**
 * The output float values should have the softmax activation function applied.
 * The derivatives are stored in the derivatives float array provided 
*/
void cross_entropy_deri_with_softmax (float* outputs, float* derivatives, int label, int size) {
    int target;
    for (int i = 0; i < size; i++) {
        target = (i == label) ? 1 : 0;
        derivatives[i] = outputs[i] - target;
    }
}

/**
 * The output float values should have the activation func applied.
 * Loss is returned as float.
*/
float cross_entropy (float* outputs, int label, int size) {
    float cost = 0.0f;
    for (int i = 0; i < size; i++) {
        int target = (i == label) ? 1 : 0;
        cost += -1 * target * logf(outputs[i]);
    }
    return cost;
}

/**
 * The output float values should have the activation func applied.
 * Loss is returned as float.
*/
float mean_squared_error (float* outputs, int label, int size) {
    float cost = 0.0f;
    for (int i = 0; i < size; i++) {
        int target = (i == label) ? 1 : 0;
        cost += (outputs[i] - target) * (outputs[i] - target);
    }

    return cost;
}