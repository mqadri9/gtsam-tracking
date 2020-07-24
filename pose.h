#ifndef POSE_H_
#define POSE_H_

#include "utils.h"

struct retPose {
    Rot3 R;
    Point3 t;
};

retPose getPose(Mat, Mat, const std::string&);

#endif