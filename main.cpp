#include "utils.h" 
#include "pointcloud.h" 
#include "optimizer.h"
#include "test_sfm.h"
#include "pose.h"

int main(int argc, char* argv[]) {
    
    vector<int> considered_poses; 

    // Keypoint Mapper is a map that takes the landmark i 
    // and returns a map which maps each image j to the coordinate
    // where landmark i appeared in image j
    map<int, map<int, Point2f>> KeypointMapper;
    map<string,int> prevKeypointIndexer;
    map<string,int> currKeypointIndexer;
    std::map<int,vector<Point2f>>::iterator it;
    std::map<string,int>::iterator itr;
               
    vector<string> frames;
    vector<Pose3> poses;
    vector<Mat> disparities;
    retPointcloud s;
        
    for(int k=15; k <35; k++) {
        frames.push_back(to_string(k));
    }
    
    // Create SIFT detector and define parameters
    std::vector<KeyPoint> keypoints1, keypoints2;
    int minHessian = 400;
    Ptr<SURF> detector = SURF::create( minHessian );
    Mat descriptors1, descriptors2; 
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);
    
    const float ratio_thresh = 0.7f;
        
    for(size_t i=0; i<frames.size(); ++i) {
        std::vector< std::vector<DMatch> > knn_matches;
        
        
        // Read image at index i 
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
        
        img_path = data_folder + "/frame" + frames[i-1] + ".jpg";
        Mat disparity1 = imread( samples::findFile( img_path ), 0);

        img_path = data_folder + "/frame" + frames[i] + ".jpg";
        Mat disparity2 = imread( samples::findFile( img_path ), 0);
         
        retPointcloud s = createPointClouds(disparity1, disparity2, keypoints1, keypoints2, knn_matches);
        Mat m1 = s.ret[0];
        Mat m2 = s.ret[1];
        vector<Point2f> src = s.src;
        vector<Point2f> dst = s.dst;;
        
        if(m1.size().height < THRESHOLD_NUMBER_MATCHES) {
            if(considered_poses.empty()) continue;
            else break;
        }
        if(i==1){
            considered_poses.push_back(i-1);
        }
        considered_poses.push_back(i);
        
        retPose p = getPose(m1, m2, "procrustes");        
        Pose3 pose(p.R, p.t);
        poses.push_back(poses[i-1]*pose);        

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
    result = Optimize(KeypointMapper, Kgt, frames, considered_poses, poses);
    

    // Test GTSAM output  
    test_sfm(result, Kgt);

    return 0;
}

