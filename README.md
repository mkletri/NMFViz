NMFViz
======

This simple program proposes a visualization of what happens while training a NMF as in 
*Algorithms for non-negative matrix factorization* from Lee & Seung 
[(NIPS 2011)](http://papers.nips.cc/paper/1861-algorithms-for-non-negative-matrix-factorization.pdf).
It does so using the [Extended Yale B dataset](http://vision.ucsd.edu/~leekc/ExtYaleDatabase/ExtYaleB.html)

It works for me (tm) on OSX with Python 3.4.3 and these libraries:

* matplotlib==1.4.3
* numpy==1.9.2
* simplejson==3.8.0

Please do contact me if you have any trouble using this tool.

How-to
------

1. Use the get_yale.sh bash script to get the data, put them all in the same folder (and remove *Ambient.pgm files*)
1. Use the nmf_viz.py python script to visualize the training procedure.

### Parameters of viz.py
The python script viz.py reads its configuration from the conf.json file. Parameters are :

1. data = Information on the data 
    1. path: The path to the directory storing all images (CroppedYale/ if you successfully used the get_yale.sh script)
    1. number: Number of images to use (they are 2409 that are successfully loaded by the script from cropped Yale)
1. nmf = Describes the NMF with:
    1. type: the name of the loss to use ("euclidean" or "divergence")
    1. components: the number of components to use 
    1. iterations: the number of training iterations


