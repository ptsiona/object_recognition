object_recognition
==================

Simple object recognition code

Compilation:
- mkdir build
- cd build
- cmake .. 
- make

Usage example:
object_recognition --step 1 --dataFile data.txt --scene cloud.pcd

--step: specify how many model views to skip at each iteration (e.g. if it is 2 it takes one model each 2 and so on)
--dir: specify the directory where to find the data.txt file of a specific model
--scene: specify the .pcd point cloud where the model has to be identified