# GPU-Accelerated Label Propagation for Graph Clustering

An implementation of gpu-accelerated label propagation for graph clustering, described in the
following papers:

* A longer journal version, under review.

* Yusuke Kozawa, Toshiyuki Amagasa, and Hiroyuki Kitagawa,
  "GPU-Accelerated Graph Clustering via Parallel Label Propagation,"
  CIKM 2017, pp. 567--576.


## Compilation

Go to `label_propagation` and just run the `make` command.  You may want to modify `Makefile`
according to the GPU's compute capability.


## Usage

You can run the program as follows:
```
galp [options] method policy data [test]
```

* `method`: specify the method to be used.
  * 0: Use data-parallel primitives
  * 1: Use lock-free hash tables

* `policy`: specifiy how to process the data.
  * 0: GPU in-core
  * 1: out-of-core without overlap
  * 2: out-of-core with overlap
  * 3: CPU--GPU hybrid
  * 4: depened on the method:
    * method=0: load-imbalanced in-core
    * method=1: multi-GPU, out-of-core with overlap
  * 5: multi-GPU incore (only for method 1)

* `data`: the graph data file.
  * The available graph formats are described below.

* `test`: the ground-truth file.
  * Some accuracy measures are computed if supplied.

### Options

* `-b n`: set the buffer size to 2^n.
  * The default is 24.

* `-i n`: set the number of iterations.
  * The default is 10.

* `-g n`: set the number of GPUs.
  * The default is 1.


### Graph formats

Basically, the program can handle two kinds of formats: text (`.txt` files) and binary (`.bin`
files).  The text format is probably easier to prepare, but the data loading is slow.  The binary
format can be loaded much faster.

#### Text

The text format is a list of edges like `datasets/sample.txt`:
```
0 1
0 2
1 2
...
```

Several assumptions of the text format is the following:
* The vertices are numbered from zero to n - 1, where n is the number of vertices.
* Edges exist only for vertices i and j such that i < j.

You can "normalize" an edge-list data (e.g., the [SNAP](https://snap.stanford.edu/data/index.html)
datasets) by using `utils/normalize.py`.


#### Binary

The binary format is a list of pairs of 32-bit integers that denote edges, including edges of both
directions in this case.  For example, the above sample is represented in binary as:
```
0 1 0 2 1 0 ...
```
You can generate this format by using `utils/edgelist2bin.cpp`.  Since this program also
does the "normalization," you can directly apply it to the SNAP dataset (e.g., `./edgelist2bin
com-friendster.ungraph.txt`).
