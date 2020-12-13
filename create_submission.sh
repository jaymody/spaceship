#!/bin/bash

mkdir -p submission/examples
mkdir -p submission/models/clf
mkdir -p submission/models/reg

cp *.py submission
cp README.md submission

cp models/clf/best_model.hdf5 submission/models/clf/best_model.hdf5
cp models/clf/summary.txt submission/models/clf/summary.txt
cp models/reg/best_model.hdf5 submission/models/reg/best_model.hdf5
cp models/reg/summary.txt submission/models/reg/summary.txt

cp -r examples submission/examples

echo -e "Classification Model\n--------------------" > submission/summary.txt
cat models/clf/summary.txt >> submission/summary.txt
echo -e "\n\n\n\nRegression Model\n----------------" >> submission/summary.txt
cat models/reg/summary.txt >> submission/summary.txt

zip -vr Jay_Mody_CV_Take_Home.zip submission/ -x "*.DS_Store"
