#pragma once

#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/common/transforms.h>

#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/shot_omp.h>
#include <pcl/features/board.h>
#include <pcl/features/fpfh_omp.h>

#include <pcl/keypoints/uniform_sampling.h>

#include <pcl/kdtree/kdtree_flann.h>

#include "segmentation.h"
#include "typedefs.h"

class PCLHighLevelCtl {
	private:
		pcl::PointCloud<PointType>::Ptr cloud;
		pcl::PointCloud<NormalType>::Ptr normals;
		pcl::PointCloud<PointType>::Ptr keypoints;
		
		void init();
	public:
		PCLHighLevelCtl(std::string file);
		PCLHighLevelCtl(std::string file, Eigen::Isometry3f transform);
		PCLHighLevelCtl(pcl::PointCloud<PointType>::Ptr cloud);
		pcl::PointCloud<NormalType>::Ptr computeNormals(float radius_search=0.01f);
		pcl::PointCloud<PointType>::Ptr keypointsExtraction(float radius_search=0.01f);
		pcl::PointCloud<DescriptorType>::Ptr computeFPFHDescriptors(float radius_search=0.01f);
		std::vector<pcl::PointCloud<PointType>::Ptr> segment();
		pcl::PointCloud<PointType>::Ptr getCloudPtr();
		PointRGB averageRGB();
};

