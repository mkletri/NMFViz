#!/bin/bash
echo "Downloading data from http://cbcl.mit.edu/projects/cbcl/software-datasets/faces.tar.gz ..."
wget http://cbcl.mit.edu/projects/cbcl/software-datasets/faces.tar.gz
echo "Creating CBCL directory"
mkdir -v CBCL
echo "Extract files from tarball"
tar -xvf faces.tar.gz -C CBCL/
