#include "pcl_high_level_ctl.hpp"

void PCLHighLevelCtl::init() {
  cloud = pcl::PointCloud<PointType>::Ptr(new pcl::PointCloud<PointType>());
  normals = pcl::PointCloud<NormalType>::Ptr(new pcl::PointCloud<NormalType>());
  keypoints = pcl::PointCloud<PointType>::Ptr(new pcl::PointCloud<PointType>());
}

PCLHighLevelCtl::PCLHighLevelCtl(std::string file) {
  this->init();
  pcl::io::loadPCDFile(file, *(this->cloud));
}

PCLHighLevelCtl::PCLHighLevelCtl(std::string file, Eigen::Isometry3f transform) {
  this->init();
  pcl::io::loadPCDFile(file, *(this->cloud));
  pcl::transformPointCloud(*(this->cloud), *(this->cloud), transform);
}

PCLHighLevelCtl::PCLHighLevelCtl(pcl::PointCloud<PointType>::Ptr cloud) {
  this->init();
  this->cloud = cloud;
}

pcl::PointCloud<NormalType>::Ptr PCLHighLevelCtl::computeNormals(float radius_search) {	
  pcl::NormalEstimationOMP<PointType, NormalType> norm_est;
  norm_est.setRadiusSearch(radius_search);
  norm_est.setInputCloud(this->cloud);
  norm_est.compute(*this->normals);
	
  return this->normals;
}

pcl::PointCloud<PointType>::Ptr PCLHighLevelCtl::keypointsExtraction(float radius_search) {
  pcl::UniformSampling<PointType> uniform_sampling;
  pcl::PointCloud<int> sampled_indices;
	
  uniform_sampling.setInputCloud(this->cloud);
  uniform_sampling.setRadiusSearch(radius_search);
  uniform_sampling.compute(sampled_indices);
	
  pcl::copyPointCloud(*(this->cloud), sampled_indices.points, *(this->keypoints));
		
  return this->keypoints;
}

pcl::PointCloud<DescriptorType>::Ptr PCLHighLevelCtl::computeFPFHDescriptors(float radius_search) {
  pcl::PointCloud<DescriptorType>::Ptr descriptors(new pcl::PointCloud<DescriptorType>());
  pcl::FPFHEstimationOMP<PointType, NormalType, DescriptorType> fpfh;
  pcl::search::KdTree<PointType>::Ptr tree(new pcl::search::KdTree<PointType>);
	
  if(!this->normals)
    this->computeNormals();
		
  if(!this->keypoints)
    this->keypointsExtraction();
	
  fpfh.setInputCloud(this->keypoints);
  fpfh.setRadiusSearch(radius_search);
  fpfh.setSearchSurface(this->cloud);
  fpfh.setInputNormals(this->normals);
  fpfh.setSearchMethod(tree);
  fpfh.compute(*descriptors);
	
  return descriptors;
}

std::vector<pcl::PointCloud<PointType>::Ptr> PCLHighLevelCtl::segment() {
  std::vector<pcl::PointCloud<PointType>::Ptr> clusters;
  segmentCloud(this->cloud, clusters);
	
  return clusters;
}

pcl::PointCloud<PointType>::Ptr PCLHighLevelCtl::getCloudPtr() {
  return this->cloud;
}

PointRGB PCLHighLevelCtl::averageRGB() {
	PointRGB average(0, 0, 0);
	int considered_points = 0;
	
	for(int i = 0; i < (this->cloud)->size(); i++) {
		if(((this->cloud)->points[i].r == 0 && (this->cloud)->points[i].g == 0 && (this->cloud)->points[i].b == 0) || (this->cloud)->points[i].z == 0)
			continue;
		
		average.r += (this->cloud)->points[i].r;
		average.g += (this->cloud)->points[i].g;
		average.b += (this->cloud)->points[i].b;
		considered_points++;
	}
	
	average.r = average.r/considered_points;
	average.g = average.g/considered_points;
	average.b = average.b/considered_points;

	return average;
}

