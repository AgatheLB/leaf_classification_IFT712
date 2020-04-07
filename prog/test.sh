#!/bin/bash

# Test script for IFT712 - Classifieur de feuilles
# GGS - 7/04/2020

RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m'
log_file='log_test.txt'

if [ -f "$log_file" ]
then
    rm $log_file
fi

for method in 'MLP' 'regression' 'SVM' 'randomforest' 'naive_bayes' 'linear_discriminant_analysis' 'all'
do
   printf "${GREEN}"
   printf "=== ${method} ===\n" | tee -a $log_file
   printf "${NC}"

   python3 ./main.py --method $method | tee -a $log_file
   printf "\n\n" | tee -a $log_file
done


