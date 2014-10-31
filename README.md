NMFViz
======

This simple program proposes a visualization of what happens while training a NMF as in *Algorithms for non-negative matrix factorization* from Lee & Seung (NIPS 2011).
It does so using the [Extended Yale B dataset](http://vision.ucsd.edu/~leekc/ExtYaleDatabase/ExtYaleB.html)

It works for me (tm) on Ubuntu (14.10), using python 2.7.
Please contact me if you have any trouble.

How-to
------

1. Use the get_yale.sh bash script to get the data, put them all in the same folder (and remove *Ambient.pgm files)
2. Use the viz.py python script to visualize the training procedure.

### Parameters of viz.py
The python script viz.py expects 4 parameters, in order:

1. The path to the folder storing all images (should be CroppedYale/ if you successfully used the get_yale.sh script)
2. The type of loss to use: either "euclidean" or "divergence" 
3. The number of components. 10 is good start.
4. The number of iterations. 50 is OK.


