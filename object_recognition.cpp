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
  // Compute scene segmentation and clusterization by using pcl functions SACSegmentation and 
  // EuclideanClusterExtraction
  std::vector<pcl::PointCloud<PointType>::Ptr> scene_clusters = scene.segment();

  // Show clusters
  pcl::visualization::PCLVisualizer viewer("Result");
  int z = 0;
  std::cout << "Num. clusters: " << scene_clusters.size() << std::endl;
  pcl::visualization::PointCloudColorHandlerCustom<PointType> scene_color_handler(scene.getCloudPtr(), 255, 255, 255);
  viewer.addPointCloud(scene.getCloudPtr(), scene_color_handler, "scene");
  for(std::vector<pcl::PointCloud<PointType>::Ptr>::iterator i = scene_clusters.begin(); i != scene_clusters.end(); ++i, ++z) {
    pcl::PointCloud<PointType>::Ptr &scene_cluster = *i;
    std::stringstream scene_cluster_str;
    scene_cluster_str << "scene_cloud" << z;
    pcl::visualization::PointCloudColorHandlerCustom<PointType> color_handler(scene_cluster, rand() % 256, rand() % 256, rand() % 256);
    viewer.addPointCloud(scene_cluster, color_handler, scene_cluster_str.str());
  } 
  
  // Cycle over the model views
  Eigen::Matrix4f camera_transform = Eigen::Matrix4f::Identity();
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
    
    // Compute model RGB average
    PointRGB model_avg = model.averageRGB();

    // Compute model keypoints by using pcl function uniform_sampling
    pcl::PointCloud<PointType>::Ptr model_keypoints = model.keypointsExtraction();
    
    // Iterate over clusters
    for(std::vector<pcl::PointCloud<PointType>::Ptr>::iterator i = scene_clusters.begin(); i != scene_clusters.end(); ++i) {
      PCLHighLevelCtl cluster(*i);
		        		
      // Compute cluster RGB average
      PointRGB cluster_avg = cluster.averageRGB();
		  
      // If color average doesn't match, skip the cluster
      if(abs(model_avg.r - cluster_avg.r) > 25 || abs(model_avg.g - cluster_avg.g) > 25 || abs(model_avg.b - cluster_avg.b) > 25)
	continue; 
		  	
      //std::cout << "model_avg " << model_avg.r << " " << model_avg.g << " " << model_avg.b << std::endl;
      //std::cout << "cluster_avg " << cluster_avg.r << " " << cluster_avg.g << " " << cluster_avg.b << std::endl;
    	
      // Compute cluster normals and keypoints
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
	Eigen::Matrix4f relative_transform = rec.aligner(model_keypoints, cluster_keypoints);

	// Compute a score for the current transformation found with a custom scoring function, if the current
	// transformation is better than the previous best one update it
	if(rec.getLastScore() < best_score) {
	  camera_transform = transform.matrix();
	  best_score = rec.getLastScore();
	  std::cout << "x";
	  final_transform = relative_transform;
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
 
  if(final_transform == Eigen::Matrix4f::Identity()) {
    std::cout << "The searched object could not be found." << std::endl;
  }
  else {
    std::cout << "Object found! Best transform: " << std::endl << (final_transform * camera_transform) * camera_transform << std::endl;
    pcl::transformPointCloud(*final_cloud, *final_cloud, final_transform);
  }
  Eigen::Matrix4f desperation = Eigen::Matrix4f::Identity();
  desperation = (final_transform * camera_transform) * camera_transform;
  Eigen::Vector4f origin_point(0.0f, 0.0f, 0.0f, 1.0f);
  pcl::PointCloud<PointType>::Ptr origin(new pcl::PointCloud<PointType>());
  origin_point = desperation * origin_point;
  PointType op;
  op.x = origin_point.x();
  op.y = origin_point.y();
  op.z = origin_point.z();
  op.r = 0;
  op.g = 255;
  op.b = 0;
  std::cout << "Origin point: " << origin_point.transpose() << std::endl;
  origin->points.push_back(op);
  std::cout << "op: " << origin->points[0] << std::endl;
  origin->width = 1;
  origin->height = 1;
	
  pcl::io::savePCDFileASCII("origin.pcd", *origin);
  pcl::io::savePCDFileASCII("scene.pcd", *scene.getCloudPtr());
  pcl::io::savePCDFileASCII("result.pcd", *final_cloud);
	
  // Visualize the final result
  pcl::visualization::PointCloudColorHandlerCustom<PointType> final_cloud_color_handler(final_cloud, 255, 0, 0);
  viewer.addPointCloud(final_cloud, final_cloud_color_handler, "final");
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
