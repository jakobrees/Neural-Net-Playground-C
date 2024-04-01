#include"backProp.h"
#include"neuralNet.h"
#include"loadData.h"


// print basic ascii of image
void printAscii (int row, int col, unsigned char *arr) {
    // birghtness scale: 
    // "$@B%%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\\|()1{}[]?-_+~<>i!lI;:,\"^`\'."
    
    char brightness[] = " -*@"; // these are just easy to read

    int i, c;
    for (i = 0; i < (row*col);) {
        for (c = 0; c < col; c++) {
            //printf("%f -> %d\n", arr[i+c], (int)(255.0 * arr[i+c]) >> 6);
            printf("%c ", brightness[((int)arr[i+c]) >> 6]);
        }
        i += c;
        printf("\n");
    }
    printf("\n");
}

int main () {

    // load and prepare nn
    NeuralNetPtr nNet = (NeuralNetPtr) malloc(sizeof(NeuralNet));
    nNet->size = 3;
    nNet->input_num = 28*28;
    nNet->layer_size = (int*) malloc(nNet->size * sizeof(int));
    nNet->layer_size[0] = 32, nNet->layer_size[1] = 32, nNet->layer_size[2] = 10;
    alloc_nn(nNet);
    //he_init_rand_nn(nNet, 111111u);
    FILE *file;
    file = fopen("data/models/model 32x32x10 sigmoid/mode1", "rb");
    load_nn(nNet, file);
    fclose(file);

    // load images and labels
    file = fopen("data/test/t10k-images-idx3-ubyte","rb");
    ImageDataPtr images;
    load_all_images(&images, file);
    fclose(file);

    file = fopen("data/test/t10k-labels-idx1-ubyte","rb");
    LabelDataPtr labels;
    load_all_labels(&labels, file);
    fclose(file);


    // now average accuracy
    int correct = 0;
    
    for (int i = 0; i < images->count; i++) {
        float * output = run_nn(nNet, images->data[i]);

        // was hypothesis certain above 50%?
        if (output[labels->data[i]] >= 0.5f) {
            correct ++;
        }

        free(output);
    }

    printf("Accuracy: %%%.2f", 100 * (float)correct/ images->count);

    for (int i = 10; i < 20; i++) {
        printAscii(28, 28, images->data[i]);
        float * output = run_nn(nNet, images->data[i]);
        for (int out = 0; out < 10; out++) {
            printf("\nP(%d) = %f", out, output[out]);

        }
        free(output);
        printf("\n Correct Answer: %d\n", labels->data[i]);
    }


}