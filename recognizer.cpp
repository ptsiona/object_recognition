#include "recognizer.hpp"

Recognizer::Recognizer() { 
  corrs = pcl::CorrespondencesPtr(new pcl::Correspondences());
}

void Recognizer::setGcSize(float gc_size) {
  this->gc_size = gc_size;
}

double Recognizer::getFitnessScore(pcl::PointCloud<PointType>::Ptr target_cloud, pcl::PointCloud<PointType>::Ptr input_transformed) {
  pcl::KdTreeFLANN<PointType>::Ptr tree(new pcl::KdTreeFLANN<PointType>);
  double fitness_score = 0.0;
  
  std::vector<int> nn_indices (1);
  std::vector<float> nn_dists (1);
  // For each point in the source dataset
  // Initialize voxel grid filter object with the leaf size given by the user.
  pcl::VoxelGrid<PointType> sor;
  sor.setLeafSize(0.025, 0.025, 0.025);
  pcl::PointCloud<PointType>::Ptr filteredTarget(new pcl::PointCloud<PointType>);
  pcl::PointCloud<PointType>::Ptr filteredAligned(new pcl::PointCloud<PointType>);
  sor.setInputCloud(target_cloud);
  sor.filter(*filteredTarget);
  sor.setInputCloud(input_transformed);
  sor.filter(*filteredAligned);
  tree->setInputCloud(filteredTarget);
  int nr = 0;
  for (size_t i = 0; i < filteredAligned->points.size(); ++i) {
    Eigen::Vector4f p1 = Eigen::Vector4f(filteredAligned->points[i].x,
                                         filteredAligned->points[i].y,
                                         filteredAligned->points[i].z, 0);
    if(isnan(p1(0)) || isnan(p1(1)) || isnan(p1(2)) || isinf(p1(0)) || isinf(p1(1)) || isinf(p1(2)))
      continue;                                                                     
    
    // Find its nearest neighbor in the target
    tree->nearestKSearch(filteredAligned->points[i], 1, nn_indices, nn_dists);
    
    // Deal with occlusions (incomplete targets)   
    if (nn_dists[0] > std::numeric_limits<double>::max ())
      continue;
    
    Eigen::Vector4f p2 = Eigen::Vector4f(filteredTarget->points[nn_indices[0]].x,
                                         filteredTarget->points[nn_indices[0]].y,
                                         filteredTarget->points[nn_indices[0]].z, 0);
    // Calculate the fitness score
    fitness_score += fabs ((p1-p2).squaredNorm());
    nr++;  
  }
  
  if (nr > 0)
    return(fitness_score/nr);
  else
    return(std::numeric_limits<double>::max());
}

pcl::CorrespondencesPtr Recognizer::flann_matcher(pcl::PointCloud<DescriptorType>::Ptr input_descriptors, pcl::PointCloud<DescriptorType>::Ptr target_descriptors, float match_thresh) {
  pcl::KdTreeFLANN<DescriptorType> match_search;
  match_search.setInputCloud(input_descriptors);
	
  for(size_t i = 0; i < target_descriptors->size (); ++i) {
    std::vector<int> neigh_indices (1);
    std::vector<float> neigh_sqr_dists (1);
 
    for (int j = 0; j < 33; j++) { // for each bin
      if(pcl_isnan(target_descriptors->at(i).histogram[j]) || !pcl_isfinite(target_descriptors->at(i).histogram[j]))
	continue;
    }

    int found_neighs = match_search.nearestKSearch(target_descriptors->at(i), 1, neigh_indices, neigh_sqr_dists);

    if(found_neighs == 1 && neigh_sqr_dists[0] < match_thresh) { // add match only if the squared descriptor distance is less than 0.25 (SHOT descriptor distances are between 0 and 1 by design)
      pcl::Correspondence corr (neigh_indices[0], static_cast<int> (i), neigh_sqr_dists[0]);
      (this->corrs)->push_back(corr);
    }
  }
	
  return this->corrs;
}

Eigen::Matrix4f Recognizer::aligner(pcl::CorrespondencesPtr corrs, pcl::PointCloud<PointType>::Ptr source_keypoints, pcl::PointCloud<PointType>::Ptr target_keypoints, float cg_thresh) {
  pcl::GeometricConsistencyGrouping<PointType, PointType> gc_clusterer;
  std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > rototranslations;
  std::vector<pcl::Correspondences> clustered_corrs;
  Eigen::Matrix4f best_transform = Eigen::Matrix4f::Identity();
  float f_min = std::numeric_limits<float>::max();
	
  for(float cg_size = 0.01f; cg_size < 0.03; cg_size += 0.005f) {
    gc_clusterer.setGCSize(cg_size);
    gc_clusterer.setGCThreshold(cg_thresh);
    gc_clusterer.setInputCloud(source_keypoints);
    gc_clusterer.setSceneCloud(target_keypoints);
    gc_clusterer.setModelSceneCorrespondences(corrs);
    gc_clusterer.recognize(rototranslations, clustered_corrs);
		
    for(size_t i = 0; i < rototranslations.size(); ++i) {
      pcl::PointCloud<PointType>::Ptr rotated_source(new pcl::PointCloud<PointType>()), registered_source(new pcl::PointCloud<PointType>());
      pcl::transformPointCloud(*source_keypoints, *rotated_source, rototranslations[i]);
			
      pcl::IterativeClosestPoint<PointType,PointType> icp; 
      icp.setInputSource(rotated_source);
      icp.setInputTarget(target_keypoints);
      icp.setTransformationEpsilon(1e-10);
      icp.setMaxCorrespondenceDistance(0.05);
      icp.setMaximumIterations(100);

      icp.setEuclideanFitnessEpsilon(0.05);
      icp.align(*registered_source);
			
      pcl::PointCloud<PointType>::Ptr scoring_biggest, scoring_smallest;
      //Select the biggest cloud
      if(target_keypoints->size() < registered_source->size()) {
	scoring_biggest = registered_source;
	scoring_smallest = target_keypoints;
      }
      else {
	scoring_biggest = target_keypoints;
	scoring_smallest = registered_source;
      }
			
      float score = this->getFitnessScore(scoring_smallest, scoring_biggest);
			
      if(score < f_min) {
	f_min = score;
	best_transform = icp.getFinalTransformation()*rototranslations[i];
      }
    }
  }

  this->last_score = f_min;
  return best_transform;
}

Eigen::Matrix4f Recognizer::aligner(pcl::PointCloud<PointType>::Ptr source_keypoints, pcl::PointCloud<PointType>::Ptr target_keypoints, float cg_thresh) {
  return this->aligner(this->corrs, source_keypoints, target_keypoints, cg_thresh);
}

float Recognizer::getLastScore() {
  return this->last_score;
}



