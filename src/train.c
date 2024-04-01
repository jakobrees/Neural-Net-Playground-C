#include<time.h>

#include"backProp.h"
#include"neuralNet.h"
#include"loadData.h"


int main () {

    FILE *file;

    // create and prepare nn
    NeuralNetPtr nNet = (NeuralNetPtr) malloc(sizeof(NeuralNet));
    nNet->size = 3;
    nNet->input_num = 28*28;
    nNet->layer_size = (int*) malloc(nNet->size * sizeof(int));
    nNet->layer_size[0] = 16, nNet->layer_size[1] = 16, nNet->layer_size[2] = 10;
    alloc_nn(nNet);
    //he_init_rand_nn(nNet, 111111u);
    file = fopen("data/models/model 16x16x10 sigmoid/succ1", "rb");
    load_nn(nNet, file);
    fclose(file);
    

    // load images and labels
    file = fopen("data/train/train-images-idx3-ubyte","rb");
    ImageDataPtr images;
    load_all_images(&images, file);
    fclose(file);

    file = fopen("data/train/train-labels-idx1-ubyte","rb");
    LabelDataPtr labels;
    load_all_labels(&labels, file);
    fclose(file);

    // create training struct for mem efficiency
    TrainEnviPtr tEnvi;
    alloc_te(&tEnvi, images, labels, nNet);

    
    // START TRAINING PROCESS
    float lRate = 0.025f, avgCost = avg_cost(tEnvi), accuracy = 0, max = 0;
    int curEpoch = 0, dummy = 1, batchSize = 32;
    clock_t start = clock();
    while( curEpoch == tEnvi->epochs || avgCost >= 0.005) {

        // check for new epoch
        if (curEpoch != tEnvi->epochs) { 

            // update epoch
            curEpoch = tEnvi->epochs;
            avgCost = avg_cost(tEnvi);
            accuracy = (float)tEnvi->correct/(float)tEnvi->done;

            // print log of progress
            printf("New epoch! Time: %.2lf seconds\n", ((double) (clock() - start)) / CLOCKS_PER_SEC);
            start = clock();
            printf(
                "    Epoch: %d, lRate: %f, Avg Cost: %f, Correct: %%%.2f\n",
                curEpoch, lRate, avgCost, 100 * accuracy
            );

            // back up every epoch ponly if accuracy is better
            if (accuracy >= max) {
                max = accuracy;
                file = fopen("data/models/model 16x16x10 sigmoid/backups/model2","wb");
                if (file == NULL) {
                    fprintf(stderr, "Couldnt write to file.");
                } else {
                    store_nn(tEnvi->net, file);
                    fclose(file);
                }
            }

            
            // reduce learning rate based on performance
            if (dummy == 1 && avgCost <= 0.15) {
                // decay learning rate
                lRate *= 0.5f;
                dummy = 2;
            } else if (dummy == 2 && avgCost <= 0.11) {
                // decay learning rate
                lRate *= 0.5f;
                batchSize = 64;
                dummy = 0;
            }
        }

        // do training batch regiment
        back_prop(tEnvi, batchSize);
        update_nn(tEnvi, lRate);

        /*printf(
                "    Epoch: %d, Index: %d, Avg Cost: %f, Correct: %%%.2f\n",
                curEpoch, tEnvi->index, avg_cost(tEnvi), 100 * (float)tEnvi->correct/(float)tEnvi->done
            );*/
    }

    // store
    file = fopen("data/models/model 16x16x10 sigmoid/backups/model2","wb");
    if (file == NULL) {
        fprintf(stderr, "Exit model not stored, err.");
    } else {
        store_nn(tEnvi->net, file);
        fclose(file);
    }
    
    clear_nn(nNet);
    clear_te(tEnvi);

    return 0;
}