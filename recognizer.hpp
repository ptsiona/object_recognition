#pragma once

#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>

#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/kdtree/impl/kdtree_flann.hpp>
#include <pcl/filters/voxel_grid.h>

#include <pcl/correspondence.h>
#include <pcl/recognition/cg/correspondence_grouping.h>
#include <pcl/recognition/cg/geometric_consistency.h>
#include <pcl/recognition/cg/hough_3d.h>

#include <pcl/registration/icp.h>

#include <pcl/visualization/pcl_visualizer.h>

#include "typedefs.h"

class Recognizer {
	private:
		float gc_size;
		pcl::CorrespondencesPtr corrs;
		float last_score;
		pcl::visualization::PCLVisualizer *viewer;
		double getFitnessScore(pcl::PointCloud<PointType>::Ptr target_cloud, pcl::PointCloud<PointType>::Ptr input_transformed);
	public:
		Recognizer() { corrs = pcl::CorrespondencesPtr(new pcl::Correspondences()); viewer = new pcl::visualization::PCLVisualizer("Simple Object Recognition");}
		void setGcSize(float gc_size);
		pcl::CorrespondencesPtr flann_matcher(pcl::PointCloud<DescriptorType>::Ptr input_descriptors, pcl::PointCloud<DescriptorType>::Ptr target_descriptors, float match_thresh = 100.0f);
		Eigen::Matrix4f aligner(pcl::CorrespondencesPtr corrs, pcl::PointCloud<PointType>::Ptr source_keypoints, pcl::PointCloud<PointType>::Ptr target_keypoints, float cg_thresh = 5.0f);
		Eigen::Matrix4f aligner(pcl::PointCloud<PointType>::Ptr source_keypoints, pcl::PointCloud<PointType>::Ptr target_keypoints, float cg_thresh = 5.0f);
		float getLastScore();
};

