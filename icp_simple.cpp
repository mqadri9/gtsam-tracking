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


vector<vector<float>> read_csv(std::string filename){

    // Create an input filestream
    std::ifstream myFile(filename);
    vector<vector<float>> result;
    std::string line;
    // Make sure the file is open
    if(!myFile.is_open()) throw std::runtime_error("Could not open file");
    while(std::getline(myFile, line))
    {
        vector<float> tmp;
        std::stringstream ss(line);
        while( ss.good() )
        {
            string substr;
            getline( ss, substr, ',' );
            float d = boost::lexical_cast<float>(substr);
            tmp.push_back( d );
        }
        result.push_back(tmp);
    }
    return result;
}

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

void printKeypointMapper(map<int, map<int, Point2f>> mainMap)
{
  map<int, map<int, Point2f> >::iterator it;
  map<int, Point2f>::iterator inner_it;
  for ( it=mainMap.begin() ; it != mainMap.end(); it++ ) {
    cout << "\n\nNew element\n" << (*it).first << endl;

    for( inner_it=(*it).second.begin(); inner_it != (*it).second.end(); inner_it++)
      cout << (*inner_it).first << " => " << (*inner_it).second << endl;
  }
}

void printKeypointIndexer(map<string, int> mainMap)
{
  map<string, int>::iterator it;

  for ( it=mainMap.begin() ; it != mainMap.end(); it++ ) {
    cout << it->first << " => " << it->second << endl;
  }
}

string getKpKey(Point2f m){
    ostringstream key ;
    key << m.x << "," << m.y;
    return key.str();
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

void ShowBlackCircle( const cv::Mat & img, cv::Point cp, int radius )
{
    int t_out = 0;
    std::string win_name = "circle";
    cv::Scalar black( 255, 0, 0 );
    cv::circle( img, cp, radius, black );
    cv::imshow( win_name, img ); cv::waitKey( t_out );
}


int main(int argc, char* argv[]) {

    float RESIZE_FACTOR = 0.2;
    float focal_length = 2869.381763767118*RESIZE_FACTOR;
    float baseline = 0.089;
    float cx = 2120.291162136175*RESIZE_FACTOR;
    float cy = 1401.17755609316*RESIZE_FACTOR;
    
    // Keypoint Mapper is a map that takes the landmark i 
    // and returns a map which maps each image j to the coordinate
    // where landmark i appeared in image j
    map<int, map<int, Point2f>> KeypointMapper;
    map<string,int> prevKeypointIndexer;
    map<string,int> currKeypointIndexer;
    
    typedef PointMatcher<float> PM;
    typedef PM::DataPoints DP;    
    PM::ICP icp;
    
    string image_folder = "/mnt/c/Users/moham/OneDrive/Desktop/others/133/rect1";
    string data_folder = "/mnt/c/Users/moham/OneDrive/Desktop/others/133/disparities" ;
    string arg1 = "/mnt/c/Users/moham/OneDrive/Desktop/gtsam/gtsam/3rdparty/libpointmatcher/examples/data/2D_twoBoxes.csv";
    string arg2 = "/mnt/c/Users/moham/OneDrive/Desktop/gtsam/gtsam/3rdparty/libpointmatcher/examples/data/2D_oneBox.csv";
    vector<string> frames;
    vector<Pose3> poses;
    vector<Mat> disparities;
    
    Mat K = Mat::eye(3, 3, CV_64F);
    
    K.at<double>(0,0) = focal_length;
    K.at<double>(1,1) = focal_length;
    K.at<double>(0,2) = cx;
    K.at<double>(1,2) = cy;
    
    
    for(int k=15; k <35; k++) {
        frames.push_back(to_string(k));
    }
    
    // Create SIFT detector and define parameters
    std::vector<KeyPoint> keypoints1, keypoints2;
    int minHessian = 400;
    Ptr<SURF> detector = SURF::create( minHessian );
    Mat descriptors1, descriptors2; 
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);
    
    const float ratio_thresh = 0.9f;
        
    for(size_t i=0; i<frames.size(); ++i) {
        std::vector< std::vector<DMatch> > knn_matches;
        
        
        // Read the image at index i 
        string img_path = image_folder + "/frame" + frames[i] + ".jpg";
        cout << img_path <<"\n";
        Mat img = imread( samples::findFile( img_path ), IMREAD_GRAYSCALE );
        if (img.empty()) {
            cout << "Could not open or find the image!\n" << endl;
            return -1;
        }        
        
        // If it is the first image in the sequence, detect the keypoints and continue 
        // to next image
        if (i==0) {
            detector->detectAndCompute( img, noArray(), keypoints2, descriptors2 );
            Rot3 R(1, 0, 0, 0, 1, 0, 0, 0, 1);
            Point3 t;
            t(0) = 0;
            t(1) = 0;
            t(2) = 0;
            Pose3 pose(R, t);
            poses.push_back(pose);
            continue;
        }
        
        keypoints1 = keypoints2; 
        descriptors1 = descriptors2;         
        detector->detectAndCompute( img, noArray(), keypoints2, descriptors2 );

        matcher->knnMatch( descriptors1, descriptors2, knn_matches, 2 );
        //-- Filter matches using the Lowe's ratio test
        
        img_path = data_folder + "/frame" + frames[i-1] + ".jpg";
        Mat disparity1 = imread( samples::findFile( img_path ), 0);

        img_path = data_folder + "/frame" + frames[i] + ".jpg";
        Mat disparity2 = imread( samples::findFile( img_path ), 0);
        
        //if(i==0) {
        //    disparities.push_back(disparity1);
        //    disparities.push_back(disparity2);
       // }
        
        //cout << (int)disparity1.at<uchar>(528, 650) << "\n";
        //namedWindow( "Display window", WINDOW_AUTOSIZE );// Create a window for display.
        //imshow( "Display window", disparity2 );                   // Show our image inside it.
        
        //waitKey(0);  
        vector<float> errors;
        vector<float> rawcloud1;
        vector<float> rawcloud2;
        vector<Point2f> src, dst;

        //cout << i << endl;
        //cout << keypoints1[0].pt << " | " <<  keypoints1[1].pt << " | " << keypoints1[2].pt << " | " << endl;
        //cout << keypoints2[0].pt << " | " <<  keypoints2[1].pt << " | "  << keypoints2[2].pt << " | " << endl;
        //cout << descriptors1.at<float>(0) << " | " <<  descriptors1.at<float>(1) << " | " << descriptors1.at<float>(2) << " | " << endl;
        //cout << descriptors2.at<float>(0) << " | " <<  descriptors2.at<float>(1) << " | "  << descriptors2.at<float>(2) << " | " << endl;
        
        for (auto &m : knn_matches) {
            if(m[0].distance < 0.7*m[1].distance) {
                float x1 = keypoints1[m[0].queryIdx].pt.x;
                float y1 = keypoints1[m[0].queryIdx].pt.y;
                float d1 = disparity1.at<uchar>((int)y1,(int)x1);
                if(d1 == 0) {
                    continue;
                }
                float z1 = baseline*focal_length/d1; 
                x1 = (x1-cx)*z1/focal_length;
                y1 = (y1-cy)*z1/focal_length;
                
                
                float x2 = keypoints2[m[0].trainIdx].pt.x;
                float y2 = keypoints2[m[0].trainIdx].pt.y;
                float d2 = disparity2.at<uchar>((int)y2,(int)x2);
                if(d2 == 0) {
                    continue;
                }
                float z2 = baseline*focal_length/d2; 
                x2 = (x2-cx)*z2/focal_length;
                y2 = (y2-cy)*z2/focal_length;
                rawcloud1.push_back(x1);
                rawcloud1.push_back(y1);
                rawcloud1.push_back(z1);
                rawcloud2.push_back(x2);
                rawcloud2.push_back(y2);
                rawcloud2.push_back(z2);
                errors.push_back(abs(z1 - z2));
                
                // Save the keypoint indices where the threshold condition 
                // is met for image i-1 and i
                src.push_back(keypoints1[m[0].queryIdx].pt);
                dst.push_back(keypoints2[m[0].trainIdx].pt);
            }
        }
        accumulator_set<double, stats<tag::mean, tag::variance> > acc;
        for_each(errors.begin(), errors.end(), bind<void>(ref(acc), _1));       
        float mean = boost::accumulators::mean(acc);
        float std = sqrt(variance(acc));
        vector<float> pointcloud1;
        vector<float> pointcloud2;
        int j = 0;      
        for(size_t l=0; l < errors.size(); l++) {
            // Only select points that have depth error less than 1cm
            if (errors[l] < mean/100 ) {
                pointcloud1.push_back(rawcloud1[j]);
                pointcloud1.push_back(rawcloud1[j+1]);
                pointcloud1.push_back(rawcloud1[j+2]);
                pointcloud2.push_back(rawcloud2[j]);
                pointcloud2.push_back(rawcloud2[j+1]);
                pointcloud2.push_back(rawcloud2[j+2]);                
            }
            j = j + 3;
        }
        //cout << pointcloud1.size()/3 << endl;
        Mat m1 = Mat(pointcloud1.size()/3, 1, CV_32FC3);
        memcpy(m1.data, pointcloud1.data(), pointcloud1.size()*sizeof(float)); 
        
        Mat m2 = Mat(pointcloud2.size()/3, 1, CV_32FC3);
        memcpy(m2.data, pointcloud2.data(), pointcloud2.size()*sizeof(float)); 
        Procrustes proc(false, false);
        proc.procrustes(m1, m2);
        //cout << proc.error << endl;
        Mat Rr = proc.rotation ;
        Mat Tr = proc.translation;
        
        // UNCOMMENT CODE TO USE ICP
        /*
        const DP ref = create_datapoints(m1);
        const DP data = create_datapoints(m2);
        icp.setDefault();
        PM::TransformationParameters T = icp(ref, data);
        cout << "--------" << endl;   
        cout << T << endl;
        Rot3 R1(
            T(0,0),
            T(0,1),
            T(0,2),

            T(1,0),
            T(1,1),
            T(1,2),

            T(2,0),
            T(2,1),
            T(2,2) 
        );
        
        Point3 t1;
        t1(0) = T(0,3);
        t1(1) = T(1,3);
        t1(2) = T(2,3);        
        
        cout << R1 << endl;
        cout << t1 << endl;
        cout << "--------" << endl;
        */
        
        Rot3 R(
            Rr.at<float>(0,0),
            Rr.at<float>(0,1),
            Rr.at<float>(0,2),

            Rr.at<float>(1,0),
            Rr.at<float>(1,1),
            Rr.at<float>(1,2),

            Rr.at<float>(2,0),
            Rr.at<float>(2,1),
            Rr.at<float>(2,2)
        );

        Point3 t;

        t(0) = Tr.at<float>(0);
        t(1) = Tr.at<float>(1);
        t(2) = Tr.at<float>(2);

        Pose3 pose(R, t);
        poses.push_back(poses[i-1]*pose);

        //Mat m2_transformed = matrix_transform<float>(m2, Rr, Tr);

        //save_vtk<float>(m2_transformed, "m2_transformed.vtk");
        //save_vtk<float>(m2, "m2.vtk"); 
        //save_vtk<float>(m1, "m1.vtk"); 
        
        std::map<int,vector<Point2f>>::iterator it;
        std::map<string,int>::iterator itr;
        // Need to construct the landmarks array
        // Initialize and create gtsam graph here
        // i is the image index
        
        if(i == 1) {
            for(int l =0; l < dst.size(); l++) {
                // assign incremental landmark IDs for the first two images
                KeypointMapper[l].insert(make_pair(i-1, src[l]));
                KeypointMapper[l].insert(make_pair(i, dst[l]));
                prevKeypointIndexer[getKpKey(dst[l])] = l;
            }
            continue;
        }
        /* For each keypoint in the new image
           Check if the match at image i-1 already exists (data associated)
           If it does, get the of the corresponding landmark_id and populate KeypointMapper[landmark_id]
           If it does not, assign a new landmark_id and populate KeypointMapper[landmark_id]
        */
        for(int l =0; l < dst.size(); l++) {
            itr = prevKeypointIndexer.find(getKpKey(src[l]));
            int landmark_id;
            if ( itr != prevKeypointIndexer.end() ) {
                landmark_id = itr->second;
            }
            else{
                int largest_landmark_id = KeypointMapper.rbegin()->first;
                landmark_id = largest_landmark_id + 1;
            }
            KeypointMapper[landmark_id].insert(make_pair(i, dst[l]));
            currKeypointIndexer[getKpKey(dst[l])] = landmark_id;
        }
                
        prevKeypointIndexer.clear();
        prevKeypointIndexer = currKeypointIndexer;
    }

    // Run GTSAM bundle adjustment
    gtsam::Values result;
    Cal3_S2::shared_ptr Kgt(new Cal3_S2(focal_length, focal_length, 0 /* skew */, cx, cy));
    
    // Define the camera observation noise model
    auto noise = noiseModel::Isotropic::Sigma(2, 2.0);  // one pixel in u and v
    
    // Create a Factor Graph and Values to hold the new data
    NonlinearFactorGraph graph;
    auto poseNoise = noiseModel::Diagonal::Sigmas(
        (Vector(6) << Vector3::Constant(0.1), Vector3::Constant(0.3))
            .finished());  // 30cm std on x,y,z 0.1 rad on roll,pitch,yaw
    graph.addPrior(Symbol('x', 0), poses[0], poseNoise);  // add directly to graph        

    int N = frames.size();
    map<int, map<int, Point2f> > FilteredKeypointMapper;
    map<int, map<int, Point3f> > landmarks;
    map<int, map<int, Point2f> >::iterator it;
    int j = 0;
    for ( it=KeypointMapper.begin() ; it != KeypointMapper.end(); it++ ) {
        // Filter out keypoints that do not appear in at least N images
        if((it->second).size() < N) {
            continue;
        }
        FilteredKeypointMapper.insert(make_pair(j++, it->second));
    }
    
    for (size_t i = 0; i < poses.size(); ++i) {
        PinholeCamera<Cal3_S2> camera(poses[i], *Kgt);
        map<int, map<int, Point2f> >::iterator it;
        int j=0;
        for ( it=FilteredKeypointMapper.begin() ; it != FilteredKeypointMapper.end(); it++ ) {
            Point2f measurement_cv2 = (it->second).find(i)->second;
            Point2 measurement;
            measurement(0) = measurement_cv2.x;
            measurement(1) = measurement_cv2.y;            
            graph.emplace_shared<GenericProjectionFactor<Pose3, Point3, Cal3_S2> >(
              measurement, noise, Symbol('x', i), Symbol('l', j), Kgt);
            j++;
        }
    }
    
    if(N != frames.size())
        throw invalid_argument( "This only works if N=frames.size(). Update code" );

    string img_path = data_folder + "/frame" + frames[0] + ".jpg";
    Mat disparity = imread( samples::findFile( img_path ), 0);
            
    // map keys are ordered
    vector<Point3> landmarks3d; 
    for ( it=FilteredKeypointMapper.begin() ; it != FilteredKeypointMapper.end(); it++ ) {
        Point2f landmark = (it->second).find(0)->second;  
        float x = landmark.x;
        float y = landmark.y;
        float d = disparity.at<uchar>((int)y,(int)x);
        float z = baseline*focal_length/d;
        x = (x-cx)*z/focal_length;
        y = (y-cy)*z/focal_length;
        Point3 landmark3d;
        landmark3d(0) = x;
        landmark3d(1) = y;
        landmark3d(2) = z;
        landmarks3d.push_back(landmark3d);
    }
    
    // Because the structure-from-motion problem has a scale ambiguity, the
    // problem is still under-constrained Here we add a prior on the position of
    // the first landmark. This fixes the scale by indicating the distance between
    // the first camera and the first landmark. All other landmark positions are
    // interpreted using this scale.
    auto pointNoise = noiseModel::Isotropic::Sigma(3, 0.1);
    graph.addPrior(Symbol('l', 0), landmarks3d[0],
                  pointNoise);  // add directly to graph
    
    
    //graph.print("Factor Graph:\n");
     
    // Create the data structure to hold the initial estimate to the solution
      // Intentionally initialize the variables off from the ground truth
    Values initialEstimate;
    for (size_t i = 0; i < poses.size(); ++i) {
        initialEstimate.insert(
            Symbol('x', i), poses[i]);
    }
    for (size_t j = 0; j < landmarks3d.size(); ++j) {
        initialEstimate.insert<Point3>(Symbol('l', j), landmarks3d[j]);
    }
    //graph.print("Factor Graph:\n");
    //cout << "===================== POSES =========================" << endl;
    //for (size_t i = 0; i < poses.size(); ++i) {
    //        cout << poses[i] << endl;
    //}       

    //cout << "===================== LANDMARKS =========================" << endl;
    //for (size_t i = 0; i < landmarks3d.size(); ++i) {
    //       cout << landmarks3d[i] << endl;
    //}
    
    result = DoglegOptimizer(graph, initialEstimate).optimize();
    result.print("Final results:\n");
    cout << "initial error = " << graph.error(initialEstimate) << endl;
    cout << "final error = " << graph.error(result) << endl;
   
    //cout << poses.size() << endl;;
    //cout << landmarks3d.size() << endl;
    //cout << landmarks3d.size() << endl;
    

    // Test GTSAM output  
    {
        string img_path = image_folder + "/frame" + frames[0] + ".jpg";
        Mat img = imread( samples::findFile( img_path ));
        
        string filename =  "/mnt/c/Users/moham/OneDrive/Desktop/others/133/bbox/frame000015_FRUITLET.csv";
        vector<vector<float>> csv = read_csv(filename);
        img_path = data_folder + "/frame" + frames[0] + ".jpg";
        Mat disparity = imread( samples::findFile( img_path ), 0);
        
        for(int j = 0; j < csv.size(); j++) {
            cout << " | " << csv[j][0] << " | " << csv[j][1] << " | " << csv[j][2] << " | " << csv[j][3] << endl;
            float x1 = csv[j][0];
            float y1 = csv[j][1];
            float x2 = csv[j][2];
            float y2 = csv[j][3];
        }
        cv::Point2f pt1(csv[0][0], csv[0][1]);
        // and its bottom right corner.
        cv::Point2f  pt2(csv[0][2], csv[0][3]);
        // These two calls...
        //cv::rectangle(img, pt1, pt2, cv::Scalar(0, 255, 0));
        //namedWindow( "Display window", WINDOW_AUTOSIZE );// Create a window for display.
        //imshow( "Display window", img );                   // Show our image inside it.
        //waitKey(0);         
        
        float x = (csv[0][0]+csv[0][2])/2;
        float y = (csv[0][1]+csv[0][3])/2;
        
        //x = 470.246429;
        //y = 346.837891;
        
        cv::Point2f pt3(x, y);
        
        cout << "x= " << x << " | y= " << y << endl;
        float d = disparity.at<uchar>((int)y,(int)x);
        float z = baseline*focal_length/d;
        x = (x-cx)*z/focal_length;
        y = (y-cy)*z/focal_length;
        Point3 landmark3d;
        landmark3d(0) = x;
        landmark3d(1) = y;
        landmark3d(2) = z;
        int num = 15;      
        Pose3 P0 = result.at(Symbol('x', 0).key()).cast<Pose3>();
        Pose3 P = result.at(Symbol('x', num).key()).cast<Pose3>();
        cout << P0 << endl;
        //landmark3d = P0.rotation()*(landmark3d - P0.translation());
        cout << landmark3d << endl;
        PinholeCamera<Cal3_S2> camera(P, *Kgt);
        Point2 measurement = camera.project(landmark3d);
        Point2f mmm;
        mmm.x = measurement(0);
        mmm.y = measurement(1);
             
        cout << measurement << endl;
        ShowBlackCircle(img, pt3, 5);
        img_path = image_folder + "/frame" + frames[num] + ".jpg";
        img = imread( samples::findFile( img_path ));        
        
        ShowBlackCircle(img, mmm, 5);
    }

    return 0;
}

