#include <iostream>
#include <fstream>

#include <Eigen/Geometry>

#include <pcl/console/parse.h>

#include "pcl_high_level_ctl.hpp"
#include "recognizer.hpp"

bool readDataFileRow(std::string &name, Eigen::Isometry3f &transform, std::ifstream &is);

int main(int argc, char **argv) {
  if(argc == 1) {
    std::cout << "Usage example: " << std::endl;
    std::cout << "object_recognition --step 1 --dataFile data.txt --scene cloud.pcd" << std::endl;
    std::cout << "\t--step: specify how many model views to skip at each iteration" << std::endl;
    std::cout << "\t--dataFile: specify the filename where the program can find model views information" << std::endl;
    std::cout << "\t--scene: specify the .pcd point cloud where the model has to be identified" << std::endl;
    return 0;
  }

  // Input handling
  int step = 1;
  std::string dataFile = "data.txt";
  std::string sceneFile = "";  
  pcl::console::parse_argument(argc, argv, "--step", step);
  pcl::console::parse_argument(argc, argv, "--dataFile", dataFile);
  pcl::console::parse_argument(argc, argv, "--scene", sceneFile);
  
  std::ifstream isDataFile(dataFile.c_str());
  if(!isDataFile) {
    std::cerr << "Impossible to open input data file " << dataFile << "... quitting!";
    return 0;
  }
   
  int count = 0;
    
  float best_score = std::numeric_limits<float>::max();
  Eigen::Matrix4f final_transform = Eigen::Matrix4f::Identity();
  pcl::PointCloud<PointType>::Ptr final_cloud(new pcl::PointCloud<PointType>());
  
  while(isDataFile.good()) {
    // Get current view data
    std::string namePrefix = "";
    Eigen::Isometry3f transform = Eigen::Isometry3f::Identity();
    transform.matrix().row(3) << 0.0f, 0.0f, 0.0f, 1.0f;
    
    if(!readDataFileRow(namePrefix, transform, isDataFile)) {
      continue;
    }
    
    if(count++ % step != 0) {
      continue;
    }
    
    std::cout << "Processing view with prefix " << namePrefix << " and camera transform: " << std::endl 
	      << transform.matrix() << std::endl;

    // Load PCD files
    PCLHighLevelCtl *scene, *model, *cluster;
    scene = new PCLHighLevelCtl(sceneFile);
    model = new PCLHighLevelCtl(namePrefix + ".pcd", transform);
    
    pcl::PointCloud<PointType>::Ptr model_cloud = model->getCloudPtr();
    // Compute model normals
    pcl::PointCloud<NormalType>::Ptr model_normals = model->computeNormals();

    // Compute model keypoints
    pcl::PointCloud<PointType>::Ptr model_keypoints = model->keypointsExtraction();

    // Compute scene clusters
    std::vector<pcl::PointCloud<PointType>::Ptr> scene_clusters = scene->segment();
    		
    // Iterate over clusters
    for(std::vector<pcl::PointCloud<PointType>::Ptr>::iterator i = scene_clusters.begin(); i != scene_clusters.end(); ++i) {
      // Compute cluster normals and keypoints
      cluster = new PCLHighLevelCtl(*i);
      pcl::PointCloud<NormalType>::Ptr cluster_normals = cluster->computeNormals();
      pcl::PointCloud<PointType>::Ptr cluster_keypoints = cluster->keypointsExtraction();
	
      //Recognizer initialization
      Recognizer* rec = new Recognizer();
    	
      // Parameter's search
      for(float search_radius = 0.01; search_radius < 0.03; search_radius += 0.005) {
	// Compute FPFH descriptors
	pcl::PointCloud<DescriptorType>::Ptr model_descriptors = model->computeFPFHDescriptors(search_radius);
	pcl::PointCloud<DescriptorType>::Ptr cluster_descriptors = cluster->computeFPFHDescriptors(search_radius);
 
	rec->flann_matcher(model_descriptors, cluster_descriptors);
	Eigen::Matrix4f transform = rec->aligner(model_keypoints, cluster_keypoints);

	if(rec->getLastScore() < best_score) {
	  best_score = rec->getLastScore();
	  std::cerr << "x";
	  final_transform = transform;
	  final_cloud = model->getCloudPtr();
	}
	else
	  std::cerr << ".";
      }
    }
    std::cout << std::endl;
  }
  
  std::cout << "Best transform: " << std::endl << final_transform << std::endl;
  pcl::transformPointCloud(*final_cloud, *final_cloud, final_transform);
  pcl::io::savePCDFileASCII("result.pcd", *final_cloud);
 
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
