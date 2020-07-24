#ifndef POINTCLOUD_H_
#define POINTCLOUD_H_

#include "utils.h"    

struct retPointcloud {
    vector<Mat> ret;
    vector<Point2f> src;
    vector<Point2f> dst;
};

retPointcloud createPointClouds(Mat, Mat, std::vector<KeyPoint>,  std::vector<KeyPoint>, std::vector< std::vector<DMatch> >);


#endif