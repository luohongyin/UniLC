#!/bin/bash

wget https://people.csail.mit.edu/hyluo/data/ulsc_data.tar.gz

tar -xzvf ulsc_data.tar.gz
rm ulsc_data.tar.gz

echo "ULSC data downloaded."