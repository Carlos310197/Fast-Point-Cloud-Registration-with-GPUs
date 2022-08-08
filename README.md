# Fast LiDAR Data Registration with GPUs 

This is an implementation of the ICP registration algorithm using CUDA C.

# How to run the code

```
cd ~/thesis
mkdir build && cd build
cmake ..
make
```
# Datasets

3 datasets were used for testing the algorithm:

+ Synthetic Data: Given by the function $z=x^2-y^2$
![Syntethic Point Cloud](/images/synthetic_data.jpg)

+ Stanford Bunny Dataset:
![Bunny Point Cloud](/images/bunny_data.jpg)

+ Real Point Cloud of a hall:
![Real Point Cloud data](/images/hall_data.jpg)

# Results

Time Complexity of serial and parallelized ICP algorithm
![Time Complexity](/images/matching_optimization.pdf)

Optimization of the matching step using the GPU
![Matching Optimization](/images/matching_optimization.pdf)