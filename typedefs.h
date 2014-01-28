#pragma once

typedef pcl::PointXYZRGB PointType;
typedef pcl::Normal NormalType;
typedef pcl::FPFHSignature33 DescriptorType;

struct EIGEN_ALIGN16 PointRGB {
	int b;
	int g;
	int r;

	inline PointRGB () {}

	inline PointRGB (const int b, const int g, const int r) : b (b), g (g), r (r) {}

	EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};
