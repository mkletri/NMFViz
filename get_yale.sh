#!/bin/bash

echo "Retrieving data from http://vision.ucsd.edu/extyaleb/CroppedYaleBZip/CroppedYale.zip ..."
wget http://vision.ucsd.edu/extyaleb/CroppedYaleBZip/CroppedYale.zip
echo "Unzip CroppedYale.zip ..."
unzip CroppedYale.zip
echo "Moving files to CroppedYale/ folder and formatting names ..."
mkdir CroppedYale/imgs
mv CroppedYale/yaleB*/*.pgm CroppedYale/imgs/
rm -rf CroppedYale/yaleB*
mv CroppedYale/imgs/* CroppedYale/
rm -rf CroppedYale/imgs
rm CroppedYale/*Ambient*
exit;