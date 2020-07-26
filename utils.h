#ifndef UTILS_H_
#define UTILS_H_

// Camera observations of landmarks (i.e. pixel coordinates) will be stored as Point2 (x, y).
#include <gtsam/geometry/Point2.h>

// Each variable in the system (poses and landmarks) must be identified with a unique key.
// We can either use simple integer keys (1, 2, 3, ...) or symbols (X1, X2, L1).
// Here we will use Symbols
#include <gtsam/inference/Symbol.h>

// In GTSAM, measurement functions are represented as 'factors'. Several common factors
// have been provided with the library for solving robotics/SLAM/Bundle Adjustment problems.
// Here we will use Projection factors to model the camera's landmark observations.
// Also, we will initialize the robot at some location using a Prior factor.
#include <gtsam/slam/ProjectionFactor.h>

// We want to use iSAM to solve the structure-from-motion problem incrementally, so
// include iSAM here
#include <gtsam/nonlinear/NonlinearISAM.h>

// iSAM requires as input a set set of new factors to be added stored in a factor graph,
// and initial guesses for any new variables used in the added factors
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>

#include <gtsam/nonlinear/DoglegOptimizer.h>

#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <utility> 
#include <stdexcept> 
#include <sstream> 

#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "pointmatcher/IO.h"
#include <cassert>
#include <typeinfo>
#include <fstream>
//#include "SFMdata.h"
#include "Procrustes.h"
#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics/stats.hpp>
#include <boost/accumulators/statistics/mean.hpp>
#include <boost/accumulators/statistics/variance.hpp>


using namespace boost::accumulators;

using namespace std;
using namespace gtsam;
//#include "boost/filesystem.hpp"
using namespace cv;
using namespace cv::xfeatures2d;

//  DEFINE GLOBAL VARIABLES

static int THRESHOLD_NUMBER_MATCHES = 100;
static float RESIZE_FACTOR = 0.2;
static float focal_length = 2869.381763767118*RESIZE_FACTOR;
static float baseline = 0.089;
static float cx = 2120.291162136175*RESIZE_FACTOR;
static float cy = 1401.17755609316*RESIZE_FACTOR;
static string image_folder = "/mnt/c/Users/moham/OneDrive/Desktop/others/151_6_6/rect1";
static string data_folder = "/mnt/c/Users/moham/OneDrive/Desktop/others/151_6_6/disparities" ;

// FUNCTION HEADERS

vector<vector<float>> read_csv(std::string);
void printKeypointMapper(map<int, map<int, Point2f>>);
void printKeypointIndexer(map<string, int>);
string getKpKey(Point2f);
PointMatcher<float>::DataPoints create_datapoints(Mat);
void ShowBlackCircle( const cv::Mat&, cv::Point, int);


// TEMPLATED FUNCTIONS
template <typename T>
Mat matrix_transform(Mat M, Mat R, Mat Tr){
    Mat ret(M.size(), M.type());
    Scalar translation( Tr.at<T>(0), Tr.at<T>(1), Tr.at<T>(2));
    for(int i=0; i<M.rows; i++){
        Mat tmp = (((M.row(i)).reshape(1,3)).t())*R;
        ret.at<T>(i, 0) = tmp.at<T>(0) + Tr.at<T>(0);
        ret.at<T>(i, 1) = tmp.at<T>(1) + Tr.at<T>(1); 
        ret.at<T>(i, 2) = tmp.at<T>(2) + Tr.at<T>(2);
    }
    return ret;
}

template <typename T>
void save_vtk(Mat pointcloud, string path) {
    ostringstream os; 
    os << "x,y,z\n";
    for(int i = 0; i < pointcloud.rows; i++)
    {
       os << pointcloud.at<T>(i, 0) << "," << pointcloud.at<T>(i, 1) << "," << pointcloud.at<T>(i, 2) << "\n";
    } 
    string s = os.str();
    std::istringstream in_stream(s);
    auto points = PointMatcherIO<T>::loadCSV(in_stream);
    const typename PointMatcher<T>::DataPoints PC(points);
    PC.save(path);
}

#endif 