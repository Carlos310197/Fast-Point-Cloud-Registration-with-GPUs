# Fast LiDAR Data Registration with GPUs 

This is an implementation of the ICP registration algorithm using CUDA C.

# How to run the code

```
cd ~/thesis
mkdir build && cd build
cmake ..
make
./icp_standard
```
# Datasets

3 datasets were used for testing the algorithm:

### Synthetic Data: Given by the function $z=x^2-y^2$
![Syntethic Point Cloud](/images/synthetic_data.jpg =20x)

### Stanford Bunny Dataset:
![Bunny Point Cloud](/images/bunny_data.jpg =20x)

### Real Point Cloud of a hall:
![Real Point Cloud data](/images/hall_data.jpg =20x)

# Results

### Time Complexity of serial and parallelized ICP algorithm
![Real Point Cloud data](/images/time_complexity.jpg)

### Optimization of the matching step using the GPU
![Real Point Cloud data](/images/matching_optimization.jpg)