#!/bin/bash

echo "retrieving data from http://vision.ucsd.edu/extyaleb/CroppedYaleBZip/CroppedYale.zip"
wget http://vision.ucsd.edu/extyaleb/CroppedYaleBZip/CroppedYale.zip
echo "unzip CroppedYale.zip"
unzip CroppedYale.zip
echo "moving files to CroppedYale/ folder"
mkdir CroppedYale/imgs
mv CroppedYale/yaleB*/*.pgm CroppedYale/imgs/
rm -rf CroppedYale/yaleB*
mv CroppedYale/imgs/* CroppedYale/
rm -rf CroppedYale/imgs
rm CroppedYale/*Ambient*
exit;