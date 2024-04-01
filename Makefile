all:
	gcc src/test.c src/backProp.c src/loadData.c src/neuralNet.c -lm -o test

