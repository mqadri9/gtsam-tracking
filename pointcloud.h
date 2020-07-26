#ifndef POINTCLOUD_H_
#define POINTCLOUD_H_

#include "utils.h"    

struct retPointcloud {
    vector<Mat> ret;
    vector<Point2f> src;
    vector<Point2f> dst;
};

struct retFiltering {
    vector<int> considered_poses;
    vector<retPointcloud> filteredOutput;
};




retPointcloud createPointClouds(Mat, Mat, std::vector<KeyPoint>,  std::vector<KeyPoint>, std::vector< std::vector<DMatch> >);
retFiltering filterImages(vector<string>);
float avgDepth(Mat);

#endif