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

#include <vector>

#include <iostream>
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


void save_vtk(Mat pointcloud, string path) {
    ostringstream os; 
    os << "x,y,z\n";
    for(int i = 0; i < pointcloud.rows; i++)
    {
       os << pointcloud.at<float>(i, 0) << "," << pointcloud.at<float>(i, 1) << "," << pointcloud.at<float>(i, 2) << "\n";
    } 
    string s = os.str();
    std::istringstream in_stream(s);
    auto points = PointMatcherIO<float>::loadCSV(in_stream);
    const PointMatcher<float>::DataPoints PC(points);
    PC.save(path);
}


PointMatcher<float>::DataPoints create_datapoints(Mat pointcloud) {
    ostringstream os; 
    os << "x,y,z\n";
    for(int i = 0; i < pointcloud.rows; i++)
    {
       os << pointcloud.at<float>(i, 0) << "," << pointcloud.at<float>(i, 1) << "," << pointcloud.at<float>(i, 2) << "\n";
    } 
    string s = os.str();
    std::istringstream in_stream(s);
    auto points = PointMatcherIO<float>::loadCSV(in_stream);
    const PointMatcher<float>::DataPoints PC(points);
    return PC;
}


int main(int argc, char* argv[]) {

    float focal_length = 2869.381763767118;
    float baseline = 0.089;

    string image_folder = "/mnt/c/Users/moham/OneDrive/Desktop/others/133/rect1";
    string data_folder = "/mnt/c/Users/moham/OneDrive/Desktop/others/133/disparities" ;
    string arg1 = "/mnt/c/Users/moham/OneDrive/Desktop/gtsam/gtsam/3rdparty/libpointmatcher/examples/data/2D_twoBoxes.csv";
    string arg2 = "/mnt/c/Users/moham/OneDrive/Desktop/gtsam/gtsam/3rdparty/libpointmatcher/examples/data/2D_oneBox.csv";
    vector<string> frames;
    vector<Pose3> poses;
 
     // Define point matcher parameters  
    typedef PointMatcher<float> PM;
    typedef PM::DataPoints DP;
    
    for(int k=15; k <36; k++) {
        frames.push_back(to_string(k));
    }
    
    // Create SIFT detector and define parameters
    std::vector<KeyPoint> keypoints1, keypoints2;
    int minHessian = 400;
    Ptr<SURF> detector = SURF::create( minHessian );
    Mat descriptors1, descriptors2; 
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);
    std::vector< std::vector<DMatch> > knn_matches;
    vector<Mat> imgs;
    const float ratio_thresh = 0.9f;
    std::vector<DMatch> good_matches;


    // Create the default ICP algorithm
    PM::ICP icp;
        
    for(size_t i=0; i<frames.size(); ++i) {
        
        // Read the image at index i 
        string img_path = image_folder + "/frame" + frames[i] + ".jpg";
        cout << img_path <<"\n";
        Mat img = imread( samples::findFile( img_path ), IMREAD_GRAYSCALE );
        imgs.push_back(img);
        if (img.empty()) {
            cout << "Could not open or find the image!\n" << endl;
            return -1;
        }        
        
        // If it is the first image in the sequence, detect the keypoints and continue 
        // to next image
        cout << "ITERATION " + i << "\n";
        if (i==0) {
            detector->detectAndCompute( img, noArray(), keypoints2, descriptors2 );
            continue;
        }
        
        keypoints1 = keypoints2; 
        descriptors1 = descriptors2;         
        detector->detectAndCompute( img, noArray(), keypoints2, descriptors2 );
        //cout << (keypoints1.front()).pt << "\n";
        //cout << descriptors1.at<double>(0,0) << "\n";

        matcher->knnMatch( descriptors1, descriptors2, knn_matches, 2 );
        //-- Filter matches using the Lowe's ratio test
        
        img_path = data_folder + "/frame" + frames[i-1] + ".jpg";
        Mat disparity1 = imread( samples::findFile( img_path ), 0);

        img_path = data_folder + "/frame" + frames[i] + ".jpg";
        Mat disparity2 = imread( samples::findFile( img_path ), 0);
        
        //cout << (int)disparity1.at<uchar>(528, 650) << "\n";
        //namedWindow( "Display window", WINDOW_AUTOSIZE );// Create a window for display.
        //imshow( "Display window", disparity2 );                   // Show our image inside it.
        
        //waitKey(0);  
        vector<float> errors;
        vector<float> rawcloud1;
        vector<float> rawcloud2;
        for (size_t j = 0; j < knn_matches.size(); j++)
        {
           if (knn_matches[j][0].distance < ratio_thresh * knn_matches[j][1].distance)
            {   
                //cout << (keypoints1[j]).pt << "    " << (keypoints2[j]).pt<< "\n";
                 
                float x1 = (keypoints1[j]).pt.x;
                float y1 = (keypoints1[j]).pt.y;
                float d1 = disparity1.at<uchar>((int)y1,(int)x1);
                if(d1 == 0) {
                    continue;
                }
                float z1 = baseline*focal_length/d1; 
                x1 = x1*z1/focal_length;
                y1 = y1*z1/focal_length;
                
                
                float x2 = (keypoints2[j]).pt.x;
                float y2 = (keypoints2[j]).pt.y;
                float d2 = disparity2.at<uchar>((int)y2,(int)x2);
                if(d2 == 0) {
                    continue;
                }
                float z2 = baseline*focal_length/d2; 
                x2 = x2*z2/focal_length;
                y2 = y2*z2/focal_length;
                rawcloud1.push_back(x1);
                rawcloud1.push_back(y1);
                rawcloud1.push_back(z1);
                rawcloud2.push_back(x2);
                rawcloud2.push_back(y2);
                rawcloud2.push_back(z2);
                errors.push_back(abs(z1 - z2));
                //cout << x1 << " " << y1 << " " << z1 << "\n";
                //cout << x2 << " " << y2 << " " << z2 << "\n";
            }
        }
        accumulator_set<double, stats<tag::mean, tag::variance> > acc;
        for_each(errors.begin(), errors.end(), bind<void>(ref(acc), _1));       
        float mean = boost::accumulators::mean(acc);
        float std = sqrt(variance(acc));
        vector<float> pointcloud1;
        vector<float> pointcloud2;
        int j = 0;      
        for(size_t i=0; i < errors.size(); i++) {
            if (errors[i] < mean/100 ) {
                pointcloud1.push_back(rawcloud1[j]);
                pointcloud1.push_back(rawcloud1[j+1]);
                pointcloud1.push_back(rawcloud1[j+2]);
                pointcloud2.push_back(rawcloud2[j]);
                pointcloud2.push_back(rawcloud2[j+1]);
                pointcloud2.push_back(rawcloud2[j+2]);                
            }
            j = j + 3;
        }
        //pointcloud1 = rawcloud1;
        //pointcloud2 = rawcloud2;
        Mat m1 = Mat(pointcloud1.size()/3, 1, CV_32FC3);
        memcpy(m1.data, pointcloud1.data(), pointcloud1.size()*sizeof(float)); 
        
        Mat m2 = Mat(pointcloud2.size()/3, 1, CV_32FC3);
        memcpy(m2.data, pointcloud2.data(), pointcloud2.size()*sizeof(float)); 
        Procrustes proc(false, false);
        cout << m1.size() << endl;
        //cout << m1 << endl;
        proc.procrustes(m1, m2);
        cout << proc.error << endl;
        cout << proc.rotation << endl;
        cout << proc.translation << endl;
        Mat m2_transformed;
        cv::transform( m2, m2_transformed, proc.rotation );
        Scalar translation( proc.translation.at<float>(0), proc.translation.at<float>(1), proc.translation.at<float>(2));
        m2_transformed += translation;
        save_vtk(m2_transformed, "m2_transformed.vtk");
        save_vtk(m2, "m2.vtk");
        save_vtk(m1, "m1.vtk");
        save_vtk(proc.Yprime, "Yprime.vtk");
        
        
        const DP ref = create_datapoints(m1);
        const DP data = create_datapoints(m2);
        icp.setDefault();
        //cout << dm1 - dm2 << endl;  
        PM::TransformationParameters T = icp(ref, data);
        DP data_out(data);
        icp.transformations.apply(data_out, T);
        ref.save("test_ref.vtk");
        data.save("test_data_in.vtk");
        data_out.save("test_data_out.vtk");
        cout << "Final transformation:" << endl << T << endl;
    }
    
    
    //std::cout << argv[1] << "\n";
    // Load point clouds
    //auto tmp = DP::load(argv[1]);
    
    std::ifstream in_file(arg1.c_str());
    const char *text =
      "x,y,z\n"
      "1,1,1\n"
      "2,2,2\n"
      "3,3,3\n"
      "10,11,14\n"
      "9,1,3\n"
      "0,0,5\n"
      "8,4,1\n"
      "4,7,8\n"
      "3,2,9\n"
      "7,1,10\n"
      "1,0,9\n"
      "2,8,8\n"
      "4,4,2\n";
      
    //std::istringstream in_stream(text);
    //auto points = PointMatcherIO<float>::loadCSV(in_stream);
    
    //const DP ref(points);
    //const DP ref(DP::load(arg1));
    //const DP data(DP::load(arg2));
        
    // See the implementation of setDefault() to create a custom ICP algorithm
    //icp.setDefault();

    // Compute the transformation to express data in ref
    //PM::TransformationParameters T = icp(data, ref);

    // Transform data to express it in ref
    //DP data_out(data);
    //icp.transformations.apply(data_out, T);
    
    // Safe files to see the results
    //points.save("test_points.vtk");
    //ref.save("test_ref.vtk");
    //data.save("test_data_in.vtk");
    //data_out.save("test_data_out.vtk");
    //cout << "Final transformation:" << endl << T << endl;


   
    return 0;
}

