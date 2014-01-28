#include "segmentation.h"

void segmentCloud( pcl::PointCloud<PointType>::Ptr &src_cloud, 
                   std::vector< pcl::PointCloud<PointType>::Ptr > &dst_clouds )
{
  pcl::PointCloud<PointType>::Ptr cropped = pcl::PointCloud<PointType>::Ptr(new pcl::PointCloud<PointType>());
  
  // Create the filtering object: downsample the dataset using a leaf size of 1cm
  for(int i = 0; i < src_cloud->size(); ++i) {
    if(src_cloud->points[i].z < 1.5f && src_cloud->points[i].z > 0.5f && src_cloud->points[i].y < 2.0f && src_cloud->points[i].y > -2.0f && src_cloud->points[i].x < 2.0f && src_cloud->points[i].x > -2.0f)
      cropped->push_back(src_cloud->points[i]);
  }
  
  pcl::VoxelGrid<PointType> vg;
  pcl::PointCloud<PointType>::Ptr cloud_f (new pcl::PointCloud<PointType>), cloud_filtered (new pcl::PointCloud<PointType>);
  vg.setInputCloud (cropped);
  vg.setLeafSize (0.001f, 0.001f, 0.001f);
  vg.filter (*cloud_filtered);

  // Create the segmentation object for the planar model and set all the parameters
  pcl::SACSegmentation<PointType> seg;
  pcl::PointIndices::Ptr inliers (new pcl::PointIndices);
  pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients);
  pcl::PointCloud<PointType>::Ptr cloud_plane (new pcl::PointCloud<PointType> ());
  pcl::PCDWriter writer;
  seg.setOptimizeCoefficients (true);
  seg.setModelType (pcl::SACMODEL_PLANE);
  seg.setMethodType (pcl::SAC_RANSAC);
  seg.setMaxIterations (100);
  seg.setDistanceThreshold (0.01);

  int i=0, nr_points = (int) cloud_filtered->points.size ();
  while(cloud_filtered->points.size () > 0.3 * nr_points) {
    // Segment the largest planar component from the remaining cloud
    seg.setInputCloud (cloud_filtered);
    seg.segment (*inliers, *coefficients);
    if(inliers->indices.size () == 0) {
      break;
    }

    // Extract the planar inliers from the input cloud
    pcl::ExtractIndices<PointType> extract;
    extract.setInputCloud (cloud_filtered);
    extract.setIndices (inliers);
    extract.setNegative (false);

    // Get the points associated with the planar surface
    extract.filter (*cloud_plane);
    
    // Remove the planar inliers, extract the rest
    extract.setNegative (true);
    extract.filter (*cloud_f);
    *cloud_filtered = *cloud_f;
  }

  // Creating the KdTree object for the search method of the extraction
  pcl::search::KdTree<PointType>::Ptr tree (new pcl::search::KdTree<PointType>);
  tree->setInputCloud (cloud_filtered);

  std::vector<pcl::PointIndices> cluster_indices;
  pcl::EuclideanClusterExtraction<PointType> ec;
  ec.setClusterTolerance (0.01); // 2cm
  ec.setMinClusterSize (20);
  ec.setMaxClusterSize (25000);
  ec.setSearchMethod (tree);
  ec.setInputCloud (cloud_filtered);
  ec.extract (cluster_indices);
  
  dst_clouds.clear();
  for(std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin (); it != cluster_indices.end (); ++it) {
    pcl::PointCloud<PointType>::Ptr cloud_cluster (new pcl::PointCloud<PointType>);
    for(std::vector<int>::const_iterator pit = it->indices.begin (); pit != it->indices.end (); pit++)
      cloud_cluster->points.push_back (cloud_filtered->points[*pit]); //*
    cloud_cluster->width = cloud_cluster->points.size ();
    cloud_cluster->height = 1;
    cloud_cluster->is_dense = true;

    dst_clouds.push_back( cloud_cluster );
  }
};
