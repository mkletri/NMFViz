#!/bin/bash

echo "Retrieving data from http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz ..."
wget http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
echo "Extract files from tarball ..."
tar xvzf cifar-10-python.tar.gz
echo "Converting so Python 3 can read it ..."
python2 cifar10_to_p3.py cifar-10-batches-py
exit;