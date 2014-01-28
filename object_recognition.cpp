#include <iostream>
#include <fstream>

#include <Eigen/Geometry>

#include <pcl/console/parse.h>

#include "pcl_high_level_ctl.hpp"
#include "recognizer.hpp"

bool readDataFileRow(std::string &name, Eigen::Isometry3f &transform, std::ifstream &is);

int main(int argc, char **argv) {
  // Input handling
  int step = 1;
  std::string dataFile = "data.txt";
  std::string sceneFile = "";
  pcl::console::parse_argument(argc, argv, "--step", step);
  pcl::console::parse_argument(argc, argv, "--dataFile", dataFile);
  pcl::console::parse_argument(argc, argv, "--scene", sceneFile);
  pcl::visualization::PCLVisualizer viewer("Normals");
  
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
    
    std::cout << "****************************************************************" << std::endl;
    std::cout << "Processing data with prefix " << namePrefix << " and transform: " << std::endl 
	      << transform.matrix() << std::endl;

    // Load PCD files
    PCLHighLevelCtl *scene, *model, *cluster;
    scene = new PCLHighLevelCtl(sceneFile);
    model = new PCLHighLevelCtl(namePrefix + ".pcd", transform);
    
    std::cout << "Model initial processing" << std::endl;
    pcl::PointCloud<PointType>::Ptr model_cloud = model->getCloudPtr();
    // Compute model normals
    pcl::PointCloud<NormalType>::Ptr model_normals = model->computeNormals();
    viewer.addPointCloudNormals<PointType, NormalType>(model_cloud, model_normals, 50, 0.02, "normals");
    // Compute model keypoints
    pcl::PointCloud<PointType>::Ptr model_keypoints = model->keypointsExtraction();

    std::cout << "Model segmentation" << std::endl;
    // Compute scene clusters
    std::vector<pcl::PointCloud<PointType>::Ptr> scene_clusters = scene->segment();
    		
    std::cout << "Starting loop over " << scene_clusters.size() << " clusters..." << std::endl;
    // Iterate over clusters
    int j = 0, k = 0;
    
    for(std::vector<pcl::PointCloud<PointType>::Ptr>::iterator i = scene_clusters.begin(); i != scene_clusters.end(); ++i, ++k) {
    	std::stringstream ss;
    	ss << "class_" << k << ".pcd";
    	pcl::io::savePCDFileASCII(ss.str(), **i);
    }
    
    for(std::vector<pcl::PointCloud<PointType>::Ptr>::iterator i = scene_clusters.begin(); i != scene_clusters.end(); ++i, ++j) {
    	std::cout << "Processing cluster " << j << std::endl;
    	// Compute cluster normals and keypoints
    	cluster = new PCLHighLevelCtl(*i);
    	pcl::PointCloud<NormalType>::Ptr cluster_normals = cluster->computeNormals();
    	pcl::PointCloud<PointType>::Ptr cluster_keypoints = cluster->keypointsExtraction();
	
	viewer.addPointCloudNormals<PointType, NormalType>(*i, cluster_normals, 50, 0.02, "normals2");
    	//viewer.spinOnce(2000);
    	viewer.removePointCloud("normals2");
    	int a;
    	//std::cin >> a;
    	//Recognizer initialization
    	Recognizer* rec = new Recognizer();
    	
    	std::cout << "Starting recognition" << std::endl;
    	// Parameter's search
    	for(float search_radius = 0.01; search_radius < 0.03; search_radius += 0.005) {
    		// Compute FPFH descriptors
		pcl::PointCloud<DescriptorType>::Ptr model_descriptors = model->computeFPFHDescriptors(search_radius);
    		pcl::PointCloud<DescriptorType>::Ptr cluster_descriptors = cluster->computeFPFHDescriptors(search_radius);
 
    		rec->flann_matcher(model_descriptors, cluster_descriptors);
    		Eigen::Matrix4f transform = rec->aligner(model_keypoints, cluster_keypoints);

    		if(rec->getLastScore() < best_score) {
    			best_score = rec->getLastScore();
			std::cout << "Found better score: " << best_score << std::endl;
    			final_transform = transform;
    			final_cloud = model->getCloudPtr();
    		}
    	}
    }
  }
  
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
