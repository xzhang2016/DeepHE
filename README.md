# DeepHE
DeepHE is a deep learning framework for essential gene prediction based on sequence data and PPI network.

# Requirements
The code is based on python 3. It depends on several python packages, such as numpy, keras, scikit-learn, networkx, pandas, tensorflow or theano, node2vec, gensim.

# Usage
For network features, please follow the instructions of [node2vec](https://github.com/aditya-grover/node2vec)(or [here](https://github.com/eliorc/node2vec)
). 

Command line usage:
```
$main.py [--expName EXPNAME] [--fold FOLD] [--embedF EMBEDF]
         [--data_dir DATA_DIR] [--trainProp TRAINPROP] [--repeat REPEAT]
         [--result_dir RESULT_DIR] [--numHiddenLayer NUMHIDDENLAYER]
```
For more information about the parameters, you can type:
```
$python main.py --help
```
or
```
$python main.py -h
```

