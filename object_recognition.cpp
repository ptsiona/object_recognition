#include <iostream>
#include <fstream>

#include <Eigen/Geometry>

#include <pcl/console/parse.h>
#include <pcl/visualization/pcl_visualizer.h>

#include "pcl_high_level_ctl.hpp"
#include "recognizer.hpp"

bool readDataFileRow(std::string &name, Eigen::Isometry3f &transform, std::ifstream &is);

int main(int argc, char **argv) { 
  if(argc == 1) {
    std::cout << "Usage example: " << std::endl;
    std::cout << "object_recognition --step 1 --dataFile data.txt --scene cloud.pcd" << std::endl;
    std::cout << "\t--step: specify how many model views to skip at each iteration" << std::endl;
    std::cout << "\t--dir: specify the directory where to find the data.txt file of a specific model" << std::endl;
    std::cout << "\t--scene: specify the .pcd point cloud where the model has to be identified" << std::endl;
    return 0;
  }

  // Input handling
  int step = 1;
  std::string dir = "dir";
  std::string sceneFile = "";  
  pcl::console::parse_argument(argc, argv, "--step", step);
  pcl::console::parse_argument(argc, argv, "--dir", dir);
  pcl::console::parse_argument(argc, argv, "--scene", sceneFile);
  
  std::ifstream isDataFile((dir + "/data.txt").c_str());
  if(!isDataFile) {
    std::cerr << "Impossible to open input data file " << (dir + "/data.txt") << "... quitting!";
    return 0;
  }
   
  int count = 0;
    
  float best_score = std::numeric_limits<float>::max();
  Eigen::Matrix4f final_transform = Eigen::Matrix4f::Identity();
  pcl::PointCloud<PointType>::Ptr final_cloud(new pcl::PointCloud<PointType>());
  PCLHighLevelCtl scene(sceneFile);

  // Cycle over the model views
  while(isDataFile.good()) {
    // Get current model view 
    std::string namePrefix = "";
    Eigen::Isometry3f transform = Eigen::Isometry3f::Identity();
    transform.matrix().row(3) << 0.0f, 0.0f, 0.0f, 1.0f;    
    if(!readDataFileRow(namePrefix, transform, isDataFile)) {
      continue;
    }    
    if(count++ % step != 0) {
      continue;
    }
    
    std::cout << "Processing view with prefix " << dir + "/" + namePrefix << " and camera transform: " << std::endl 
	      << transform.matrix() << std::endl;

    // Instantiate high level controller for point cloud processing
    PCLHighLevelCtl model(dir + "/" + namePrefix + ".pcd", transform);
    
    // Compute model normals by using pcl function NormalEstimationOMP
    model.computeNormals();

    // Compute model keypoints by using pcl function uniform_sampling
    pcl::PointCloud<PointType>::Ptr model_keypoints = model.keypointsExtraction();

    // Compute scene segmentation and clusterization by using pcl functions SACSegmentation and 
    // EuclideanClusterExtraction
    std::vector<pcl::PointCloud<PointType>::Ptr> scene_clusters = scene.segment();
    		
    // Iterate over clusters
    for(std::vector<pcl::PointCloud<PointType>::Ptr>::iterator i = scene_clusters.begin(); i != scene_clusters.end(); ++i) {
      // Compute cluster normals and keypoints
      PCLHighLevelCtl cluster(*i);
      cluster.computeNormals();
      pcl::PointCloud<PointType>::Ptr cluster_keypoints = cluster.keypointsExtraction();
	
      //Recognizer object initialization
      Recognizer rec;
    	
      // Parameter search
      for(float search_radius = 0.01; search_radius < 0.03; search_radius += 0.005) {
	// Compute FPFH descriptors for the cluster and the model by using pcl function FPFHEstimationOMP
	pcl::PointCloud<DescriptorType>::Ptr model_descriptors = model.computeFPFHDescriptors(search_radius);
	pcl::PointCloud<DescriptorType>::Ptr cluster_descriptors = cluster.computeFPFHDescriptors(search_radius);
 
	// Compute correspondences between the computed descriptors by using pcl function KdTreeFLANN
	rec.flann_matcher(model_descriptors, cluster_descriptors);
	
	// Compute alignment between the model view and the current cluster by using pcl 
	// function IterativeClosestPoint with initial guess found by using an other pcl function 
	// GeometricConsistencyGrouping
	Eigen::Matrix4f transform = rec.aligner(model_keypoints, cluster_keypoints);

	// Compute a score for the current transformation found with a custom scoring function, if the current
	// transformation is better than the previous best one update it
	if(rec.getLastScore() < best_score) {
	  best_score = rec.getLastScore();
	  std::cout << "x";
	  final_transform = transform;
	  final_cloud = model.getCloudPtr();
	}
	else {
	  std::cout << ".";
	}
	std::cout.flush();
      }
    }
    std::cout << std::endl;
  }
 
  // Visualize the final result
  std::cout << "Best transform: " << std::endl << final_transform << std::endl;
  pcl::transformPointCloud(*final_cloud, *final_cloud, final_transform);
  pcl::visualization::PCLVisualizer viewer("Result");
  pcl::visualization::PointCloudColorHandlerCustom<PointType> scene_color_handler(scene.getCloudPtr(), 255, 255, 255);
  viewer.addPointCloud(scene.getCloudPtr(), scene_color_handler , "scene");
  pcl::visualization::PointCloudColorHandlerCustom<PointType> final_cloud_color_handler(final_cloud, 255, 0, 0);
  viewer.addPointCloud(final_cloud, final_cloud_color_handler , "final");
  viewer.spin(); 

  return 0;
}

bool readDataFileRow(std::string &name, Eigen::Isometry3f &transform, std::ifstream &is) {
  char buffer[2048];
  is.getline(buffer, 2048);
  std::istringstream iss(buffer);
  if(iss.rdbuf()->in_avail() == 0) {
    return false;
  }
  iss >> name;
  iss >> 
    transform.matrix()(0, 0) >> transform.matrix()(0, 1) >> transform.matrix()(0, 2) >> transform.matrix()(0, 3) >>
    transform.matrix()(1, 0) >> transform.matrix()(1, 1) >> transform.matrix()(1, 2) >> transform.matrix()(1, 3) >>
    transform.matrix()(2, 0) >> transform.matrix()(2, 1) >> transform.matrix()(2, 2) >> transform.matrix()(2, 3) >>
    transform.matrix()(3, 0) >> transform.matrix()(3, 1) >> transform.matrix()(3, 2) >> transform.matrix()(3, 3);
  return true;
}
