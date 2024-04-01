#include"backProp.h"

#define PIXEL_SCALE(x) (((float) (x)) / 255.0f)


/**
 * This function COMPLETELY creates a new training environment.
 * Nothing is needed except relevant pointers.
*/
void alloc_te(TrainEnviPtr* tePtr, ImageDataPtr imagePtr, LabelDataPtr labelPtr, NeuralNetPtr nnPtr) {

    // allocate space for te, make easier reference pointer
    *tePtr = (TrainEnviPtr) malloc(sizeof(TrainEnvi));
    TrainEnviPtr tempPtr = *tePtr;
    if (tempPtr == NULL) {
        fprintf(stderr, "Error allocating TE struct.");
        exit(1);
    }

    tempPtr->index = 0;
    tempPtr->done = 0;
    tempPtr->correct = 0;
    tempPtr->epochs = 0;
    tempPtr->images = imagePtr;
    tempPtr->labels = labelPtr;
    tempPtr->net = nnPtr;

    // Now allocate reusable calculation space for backProp
    tempPtr->activations = (float **) malloc(nnPtr->size * sizeof(float*));
    if (tempPtr->activations == NULL) {
        fprintf(stderr, "Error allocating TE activation pointer space.");
        exit(1);
    }
    tempPtr->deriv = (float **) malloc(nnPtr->size * sizeof(float*));
    if (tempPtr->deriv == NULL) {
        fprintf(stderr, "Error allocating TE deriv pointer space.");
        exit(1);
    }
    tempPtr->biasGradient = (float **) malloc(nnPtr->size * sizeof(float*));
    if (tempPtr->biasGradient == NULL) {
        fprintf(stderr, "Error allocating TE bias  gradient pointer space.");
        exit(1);
    }
    tempPtr->weightGradient = (float ***) malloc(nnPtr->size * sizeof(float**));
    if (tempPtr->weightGradient == NULL) {
        fprintf(stderr, "Error allocating TE weight gradient pointer space.");
        exit(1);
    }

    // alloc inner array space
    for (int l = 0; l < nnPtr->size; l++) {

        // alloc space for rows
        tempPtr->activations[l] = (float *) malloc(nnPtr->layer_size[l] * sizeof(float));
        if (tempPtr->activations[l] == NULL) {
            fprintf(stderr, "Error allocating TE activation pointer space.");
            exit(1);
        }
        tempPtr->deriv[l] = (float *) malloc(nnPtr->layer_size[l] * sizeof(float));
        if (tempPtr->deriv[l] == NULL) {
            fprintf(stderr, "Error allocating TE deriv pointer space.");
            exit(1);
        }
        tempPtr->biasGradient[l] = (float *) malloc(nnPtr->layer_size[l] * sizeof(float));
        if (tempPtr->biasGradient[l] == NULL) {
            fprintf(stderr, "Error allocating TE bias  gradient pointer space.");
            exit(1);
        }
        tempPtr->weightGradient[l] = (float **) malloc(nnPtr->layer_size[l] * sizeof(float*));
        if (tempPtr->weightGradient[l] == NULL) {
            fprintf(stderr, "Error allocating TE weight gradient pointer space.");
            exit(1);
        }

        // alloc space for cols
        int rep = (l == 0) ? nnPtr->input_num : nnPtr->layer_size[l-1];

        for (int r = 0; r < nnPtr->layer_size[l]; r++) {
            tempPtr->weightGradient[l][r] = (float *) malloc(rep * sizeof(float));
            if (tempPtr->weightGradient[l][r] == NULL) {
                fprintf(stderr, "Error allocating weight gradient col space.");
                exit(1);
            }
        }
    }
}


/**
 * This function clears all arrays added on top.
 * I.E. not the image, label, and NeuralNet structs.
*/
void clear_te(TrainEnviPtr tePtr) {

    // deallocate all the imbedded spaces
    for (int l = 0; l < tePtr->net->size; l++) {
        for (int r = 0; r < tePtr->net->layer_size[l]; r++) {
            free(tePtr->weightGradient[l][r]);
        }
        free(tePtr->weightGradient[l]); 
        free(tePtr->biasGradient[l]);
        free(tePtr->activations[l]);
        free(tePtr->deriv[l]);
    }

    // dealocate pointer arrays & other
    free(tePtr->weightGradient);
    free(tePtr->biasGradient);
    free(tePtr->activations);
    free(tePtr->deriv);
    free(tePtr);
}


/**
 * Helper functon that stores the activations in the parameterized TrainEnviPtr.
*/
void forward_prop(TrainEnviPtr tePtr) {
    
    // begin forward propagation
    for (int l = 0; l < tePtr->net->size; l++) {

        // now calc logits
        for (int r = 0; r < tePtr->net->layer_size[l]; r++) {

            // add bias, and clear the previous activations
            tePtr->activations[l][r] = tePtr->net->biases[l][r];

            // first layer weight products & activation
            if (l == 0) {
                for (int c = 0; c < tePtr->net->input_num; c++) {
                    tePtr->activations[l][r] += 
                        tePtr->net->weights[l][r][c] * PIXEL_SCALE(tePtr->images->data[tePtr->index][c]);
                }
                // we have logits, pass through activation function
                tePtr->activations[l][r] = tePtr->net->mainActFunc(tePtr->activations[l][r]);
            } 

            // last layer weight products & activation
            else if (l == tePtr->net->size - 1) {
                for (int c = 0; c < tePtr->net->layer_size[l-1]; c++) {
                    tePtr->activations[l][r] += tePtr->net->weights[l][r][c] * tePtr->activations[l-1][c];
                }
                // we wait for all logits since some output act func need all logits
            }

            // hidden layer products & activation
            else {
                for (int c = 0; c < tePtr->net->layer_size[l-1]; c++) {
                    tePtr->activations[l][r] += tePtr->net->weights[l][r][c] * tePtr->activations[l-1][c];
                }
                // we have logits, pass through output activation function
                tePtr->activations[l][r] = tePtr->net->mainActFunc(tePtr->activations[l][r]);
            }
        }
    }

    // apply output activation function after all output logits have been calculated
    tePtr->net->outActFunc(tePtr->activations[tePtr->net->size-1], tePtr->net->layer_size[tePtr->net->size-1]);
}


/**
 * Gradient averages are stored int the gradeient arrays of the fucntion.
*/
void back_prop(TrainEnviPtr tePtr, int batch) {

    // clear the gradient arrays
    for (int l = 0; l < tePtr->net->size; l++) {
        // bias gradient memset to 0
        memset(tePtr->biasGradient[l], 0, tePtr->net->layer_size[l] * sizeof(float));

        // weight gradient memset to 0
        int rep = (l == 0) ? tePtr->net->input_num : tePtr->net->layer_size[l-1];
        for (int r = 0; r < tePtr->net->layer_size[l]; r++) {
            memset(tePtr->weightGradient[l][r], 0, rep * sizeof(float));
        }
    }
    
    // make sure we are getting valid data from labels and images
    if (tePtr->index >= tePtr->images->count) {
        tePtr->index = 0;
        tePtr->epochs += 1;
    }
    
    // for simply accuracy indication
    tePtr->done = batch;
    tePtr->correct = 0;

    
    // begin loop for batches
    for (int step = 0; step < batch; step++) {
        
        // do forward propagation pass over image
        forward_prop(tePtr);
        
        // was hypothesis certain above 50%? Yes: add 1 to correct
        if (tePtr->activations[tePtr->net->size -1][tePtr->labels->data[tePtr->index]] >= 0.5f) {
            tePtr->correct += 1;
        }
        
        // step 1: get deriv of output wrt logits
        int l = tePtr->net->size -1;
        tePtr->net->lossFuncDeri(
            tePtr->activations[l],
            tePtr->deriv[l],
            tePtr->labels->data[tePtr->index],
            tePtr->net->layer_size[l]
        );
        
        // step 2: over act of last layer * [deriv] = weight
        for (int r = 0; r < tePtr->net->layer_size[l]; r++) {
            
            // the bias der wrt logits is 1, so same gradient as deriv
            tePtr->biasGradient[l][r] += tePtr->deriv[l][r];

            // the weight bias is corresponding activation of node * der wrt logits
            for (int c = 0; c < tePtr->net->layer_size[l-1]; c++) {
                tePtr->weightGradient[l][r][c] += tePtr->activations[l-1][c] * tePtr->deriv[l][r];

                // step 3: [deriv] * weight += into [deriv-1] of next layer
                tePtr->deriv[l-1][c] += tePtr->net->weights[l][r][c] * tePtr->deriv[l][r];
            }
        }
        
        
        // now for the front most l-1 layers do backprop
        for (l -= 1; l >= 0; l--) {
            for (int r = 0; r < tePtr->net->layer_size[l]; r++) {

                // step 1: (deri of act wrt logits) * [cost deriv wrt act] = bias & (der wrt logits -> deri)
                tePtr->deriv[l][r] *= tePtr->net->mainActDeri(tePtr->activations[l][r]);
                tePtr->biasGradient[l][r] += tePtr->deriv[l][r];
                
                // step 2: over act of layer l-1 * (deriv wrt logits) = weight
                if (l != 0) {
                    for (int c = 0; c < tePtr->net->layer_size[l-1]; c++) {
                        tePtr->weightGradient[l][r][c] += tePtr->activations[l-1][c] * tePtr->deriv[l][r];

                        // step 3: [deriv] * weight += into [deriv-1] of next layer
                        tePtr->deriv[l-1][c] += tePtr->net->weights[l][r][c] * tePtr->deriv[l][r];
                    }
                } 
        
                // step 2.1: over act of layer l-1 * (deriv wrt logits) = weight
                else {
                    for (int c = 0; c < tePtr->net->input_num; c++) {
                        tePtr->weightGradient[l][r][c] += 
                            PIXEL_SCALE(tePtr->images->data[tePtr->index][c]) * tePtr->deriv[l][r];
                    }
                } 
            }
        }
        
        // upadate the index
        if (tePtr->index >= tePtr->images->count - 1) {
            tePtr->index = 0;
            tePtr->epochs += 1;
        } else {
            tePtr->index += 1;
        }
        
    }


    // average the gradients!
    for (int l = 0; l < tePtr->net->size; l++) {
        for (int r = 0; r < tePtr->net->layer_size[l]; r++) {

            tePtr->biasGradient[l][r] /= (float) batch;

            int rep = (l == 0) ? tePtr->net->input_num : tePtr->net->layer_size[l-1];
            for (int c = 0; c < rep; c++) {
                tePtr->weightGradient[l][r][c] /= (float) batch;
            }
        }
    }
    
}


/**
 * Adds up all gradient values (bias and weigths), then normalizes
 * then according to the provided threshold. 
 * May help with gradient explosion. (likely not numerically stable)
*/
void gradient_norm_clip(TrainEnviPtr tePtr, float threshold) {

    // calculate euclidean magnitude
    float sum = 0, norm = 0;
    for (int l = 0; l < tePtr->net->size; l++) {
        for (int r = 0; r < tePtr->net->layer_size[l]; r++) {
            sum += tePtr->biasGradient[l][r] * tePtr->biasGradient[l][r];

            int rep = (l == 0) ? tePtr->net->input_num : tePtr->net->layer_size[l-1];
            for (int c = 0; c < rep; c++) {
                sum += tePtr->weightGradient[l][r][c] * tePtr->weightGradient[l][r][c];
            }
        }
    }
    sum = sqrtf(sum);

    // clip according to norm if over threshold
    if (sum >= threshold) { 
        norm = threshold/sum;
        for (int l = 0; l < tePtr->net->size; l++) {
            for (int r = 0; r < tePtr->net->layer_size[l]; r++) {

                // update biases
                tePtr->biasGradient[l][r] *= norm;

                int rep = (l == 0) ? tePtr->net->input_num : tePtr->net->layer_size[l-1];
                for (int c = 0; c < rep; c++) {
                    // update weights
                    tePtr->weightGradient[l][r][c] *= norm;
                }
            }
        }
    }
}


/**
 * Updates net weigths and biases according to step size.
*/ 
void update_nn(TrainEnviPtr tePtr, float stepSize) {

    // applies already averaged gradients
    for(int l = 0; l < tePtr->net->size; l++) {
        for(int r = 0; r < tePtr->net->layer_size[l]; r++) {
            
            // apply gradient to biases
            tePtr->net->biases[l][r] -= stepSize * tePtr->biasGradient[l][r];

            // apply gradient to weights
            int rep = (l == 0) ? tePtr->net->input_num : tePtr->net->layer_size[l-1];
            for (int c = 0; c < rep; c++) 
                tePtr->net->weights[l][r][c] -= stepSize * tePtr->weightGradient[l][r][c];
        }
    }
}


/**
 * Calcs average cost over 256 images.
 * Also stores #/256 guessed w/ >50% certainty.
*/
float avg_cost(TrainEnviPtr tePtr) {

    // reset trackers
    tePtr->done = 256;
    tePtr->correct = 0;

    // rand index in training data for test
    int start_index = tePtr->index;
    tePtr->index = rand() / (RAND_MAX / (tePtr->images->count -256) + 1);
    float cost = 0;

    // now do tests
    for (int i = 0; i < 256; i++) {

        forward_prop(tePtr);

        // was hypothesis certain above 50%?
        if (tePtr->activations[tePtr->net->size -1][tePtr->labels->data[tePtr->index]] >= 0.5f) {
            tePtr->correct += 1;
        }

        // add cost
        cost += tePtr->net->lossFunc(
            tePtr->activations[tePtr->net->size -1], 
            tePtr->labels->data[tePtr->index],
            tePtr->net->layer_size[tePtr->net->size -1]
        );
        tePtr->index += 1;
    }

    // average cost
    tePtr->index = start_index;
    return (cost / 256.0);
}