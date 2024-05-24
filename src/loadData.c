#include"loadData.h"


/**
 * This function changes based on the architecture that the program runs on. 
 * The MNIST data set is in high-endian format and must be converted to be
 * read in low-endian processors.
*/
int32_t map_int32(int32_t in) {
    #if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__ 
        return (
            ((in & 0xFF000000) >> 24) |
            ((in & 0x00FF0000) >>  8) |
            ((in & 0x0000FF00) <<  8) |
            ((in & 0x000000FF) << 24)
        );
    #else
        return in;
    #endif
}


/**
 * This function reads in the MNIST label data, specifically fitted 
 * for data from "http://yann.lecun.com/exdb/mnist/". The exact format 
 * may be found there.
*/
void load_all_labels(LabelDataPtr* lPtr, FILE *file) {

    // allocate space for LabelData struct
    *lPtr = (LabelDataPtr) malloc(sizeof(LabelData));
    LabelDataPtr tempPtr = *lPtr;

    if (tempPtr == NULL) {
        fprintf(stderr, "Error allocating struct for labels.");
        exit(1);
    }
    

    // start reading in format data
    int magic_number;
    if ( 1 != fread(&magic_number, sizeof(magic_number), 1, file)) {
        fprintf(stderr, "Error reading file header.");
        exit(1);
    }
    magic_number = map_int32(magic_number);

    if ( 1 != fread(&tempPtr->count, sizeof(magic_number), 1, file)) {
        fprintf(stderr, "Error reading dimension 1.");
        exit(1);
    }
    tempPtr->count = map_int32(tempPtr->count);


    // allocatew space for labels
    tempPtr->data = (unsigned char *) malloc(tempPtr->count * sizeof(unsigned char));

    if (tempPtr->data == NULL) {
        fprintf(stderr, "Error allocating space for labels.");
        exit(1);
    }

    // now read the Labels (unsigned byte 0-9)
    for (int i = 0; i < tempPtr->count; i++) { 
        fread(tempPtr->data + i, sizeof(unsigned char), 1, file);
    }
}


/**
 * This function reads in the MNIST image data, specifically fitted 
 * for data from "http://yann.lecun.com/exdb/mnist/". The exact format 
 * may be found there.
*/
void load_all_images(ImageDataPtr* iPtr, FILE *file) {
    
    // allocate space for ImageData struct
    *iPtr = (ImageDataPtr) malloc(sizeof(ImageData)); 
    ImageDataPtr tempPtr = *iPtr;

    if (tempPtr == NULL) {
        fprintf(stderr, "Error mallocating struct for images.");
        exit(1);
    }
    

    // start reading in data
    int magic_number, height, width;
    if (1 != fread(&magic_number, sizeof(magic_number), 1, file)) {
        fprintf(stderr, "Error reading file header.");
        exit(1);
    }
    magic_number = map_int32(magic_number);

    if (1 != fread(&tempPtr->count, sizeof(int), 1, file)) {
        fprintf(stderr, "Error reading dimension 1 (image count).");
        exit(1);
    }
    tempPtr->count = map_int32(tempPtr->count);

    if (1 != fread(&height, sizeof(int), 1, file)) {
        fprintf(stderr, "Error reading dimension 2 (image height).");
        exit(1);
    }
    height = map_int32(height);
    
    if (1 != fread(&width, sizeof(int), 1, file)) {
        fprintf(stderr, "Error reading dimension 3 (image width).");
        exit(1);
    }
    width = map_int32(width);


    // allocate space for images
    int length = height * width;

    tempPtr->data = (unsigned char **) malloc(tempPtr->count * sizeof(unsigned char *));
    if (tempPtr->data == NULL) {
        fprintf(stderr, "Error allocating space for image pointers.");
        exit(1);
    }
    
    //now read the Images (unsigned byte 0-255)
    for (int i = 0; i < tempPtr->count; i++) {

        tempPtr->data[i] = (unsigned char *) malloc(length * sizeof(unsigned char));
        if (tempPtr->data[i] == NULL) {
            fprintf(stderr, "Error allocating space for image pixel values.");
            exit(1);
        }

        fread(tempPtr->data[i], sizeof(unsigned char), length, file);
    }
}


/**
 * This stores a neural net in a custom format for the "neuralNet.c"
 * file included in this folder. It will not exit, because it may be
 * backing up a net currently training.
*/
void store_nn(NeuralNetPtr nnPtr, FILE* file) {

    // Write input_num, nn size, layer size.
    if (1 != fwrite(&nnPtr->input_num, sizeof(int), 1, file)) {
        fprintf(stderr, "Error writing input_num.");
        return; // don't exit while training.
    }
    if (1 != fwrite(&nnPtr->size, sizeof(int), 1, file)) {
        fprintf(stderr, "Error writing size.");
        return; // don't exit while training.
    }
    if (nnPtr->size != fwrite(nnPtr->layer_size, sizeof(int), nnPtr->size, file)) {
        fprintf(stderr, "Error writing layer_size.");
        return; // don't exit while training.
    }

    
    // weight matricies
    for (int l = 0; l < nnPtr->size; l++) {             
        for (int r = 0; r < nnPtr->layer_size[l]; r++) {

            if (l == 0) {
                if (nnPtr->input_num != fwrite(nnPtr->weights[l][r], sizeof(float), nnPtr->input_num, file)) {
                    fprintf(stderr, "Error writing weights l = 0.");
                    return; // don't exit while training.
                }
            
            } else if (nnPtr->layer_size[l-1] != fwrite(nnPtr->weights[l][r], sizeof(float), nnPtr->layer_size[l-1], file)) {
                fprintf(stderr, "Error writing weights l != 0.");
                return; // don't exit while training.

            }
        }
    }

    // bias vectors
    for (int l = 0; l < nnPtr->size; l++) {
        if (nnPtr->layer_size[l] != fwrite(nnPtr->biases[l], sizeof(float), nnPtr->layer_size[l], file)) {
            fprintf(stderr, "Error writing biases.");
            return; // don't exit while training.
        }
    }
}


/**
 * This reads in a neural net in a custom format for the "neuralNet.c"
 * file included in this folder. Does not require alloc_nn, only valid pointer.
*/
void load_nn(NeuralNetPtr nnPtr, FILE * file) {

    // Read input_num, nn size, layer size.
    if (1 != fread(&nnPtr->input_num, sizeof(int), 1, file)) {
        fprintf(stderr, "Error reading input_num.");
        exit(1);
    }
    if (1 != fread(&nnPtr->size, sizeof(int), 1, file)) {
        fprintf(stderr, "Error reading size.");
        exit(1);
    }
    nnPtr->layer_size = (int *) malloc(nnPtr->size * sizeof(int));
    if (nnPtr->layer_size == NULL) {
        fprintf(stderr, "Error allocating space for layer_size.");
        exit(1);
    }
    if (nnPtr->size != fread(nnPtr->layer_size, sizeof(int), nnPtr->size, file)) {
        fprintf(stderr, "Error reading layer_size.");
        exit(1);
    }

    // space for weight and biases matricies/vectors
    alloc_nn(nnPtr);
    
    // weight matricies
    for (int l = 0; l < nnPtr->size; l++) {
        for (int r = 0; r < nnPtr->layer_size[l]; r++) {

            if (l == 0) {
                if (nnPtr->input_num != fread(nnPtr->weights[l][r], sizeof(float), nnPtr->input_num, file)) {
                    fprintf(stderr, "Error reading weights l = 0.");
                    exit(1);
                }
            } else {
                if (nnPtr->layer_size[l-1] != fread(nnPtr->weights[l][r], sizeof(float), nnPtr->layer_size[l-1], file)) {
                    fprintf(stderr, "Error reading weights l != 0.");
                    exit(1);
                }
            }
        }
    }

    // bias vectors
    for (int l = 0; l < nnPtr->size; l++) {
        if (nnPtr->layer_size[l] != fread(nnPtr->biases[l], sizeof(float), nnPtr->layer_size[l], file)) {
            fprintf(stderr, "Error reading biases.");
            exit(1);
        }
    }
}
