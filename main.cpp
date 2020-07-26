#include "utils.h" 
#include "pointcloud.h" 
#include "optimizer.h"
#include "test_sfm.h"
#include "pose.h"
#include <boost/filesystem.hpp>
#include <boost/iterator/filter_iterator.hpp>
namespace fs = boost::filesystem;

int main(int argc, char* argv[]) {
    
    // Keypoint Mapper is a map that takes the landmark i 
    // and returns a map which maps each image j to the coordinate
    // where landmark i appeared in image j
    map<int, map<int, Point2f>> KeypointMapper;
    map<string,int> prevKeypointIndexer;
    map<string,int> currKeypointIndexer;
    std::map<int,vector<Point2f>>::iterator it;
    std::map<string,int>::iterator itr;
               
    vector<string> frames;
    vector<string> frames1;
    vector<Pose3> poses;
    vector<Mat> disparities;
    retPointcloud s;

    fs::path p(data_folder);
    fs::directory_iterator dir_first(p), dir_last;

    auto pred = [](const fs::directory_entry& p)
    {
        return fs::is_regular_file(p);
    };

    
    int count = std::distance(boost::make_filter_iterator(pred, dir_first, dir_last),
                      boost::make_filter_iterator(pred, dir_last, dir_last));

    for(int k=0; k <count; k++) {
        frames.push_back(to_string(k));
    }
    cout << frames.size() << endl;
    
    retFiltering filtered;
    filtered = filterImages(frames);
    /*
    for(int k=13; k <43; k++) {
        frames1.push_back(to_string(k));
    }
    
    
     
    retFiltering filtered1; 
    
    filtered1 = filterImages(frames1);
    
    retFiltering filtered2 = filtered;
    retFiltering filtered3 = filtered;
    retFiltering filtered4 = filtered;
    retFiltering filtered5 = filtered;
    
    //if (filtered.descriptors1 == filtered1.descriptors1) std::cout << "matched1" << std::endl;
    //if (filtered.descriptors2 == filtered1.descriptors2) std::cout << "matched2" << std::endl;
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::BRUTEFORCE);
    Ptr<DescriptorMatcher> matcher1 = DescriptorMatcher::create(DescriptorMatcher::BRUTEFORCE);
    Ptr<DescriptorMatcher> matcher2 = DescriptorMatcher::create(DescriptorMatcher::BRUTEFORCE);
    Ptr<DescriptorMatcher> matcher3 = DescriptorMatcher::create(DescriptorMatcher::BRUTEFORCE);
    Ptr<DescriptorMatcher> matcher4 = DescriptorMatcher::create(DescriptorMatcher::BRUTEFORCE);
    cv::Mat diff = filtered.descriptors1 != filtered1.descriptors1;
    // Equal if no elements disagree
    bool eq = cv::countNonZero(diff) == 0;
    cout << eq << endl;
    cv::Mat diff1 = filtered.descriptors2 != filtered1.descriptors2;
    // Equal if no elements disagree
    bool eq1 = cv::countNonZero(diff1) == 0;
    cout << eq1 << endl;   
    cout << "========================================================" << endl;
    std::vector< std::vector<DMatch> > knn_matches;
    std::vector< std::vector<DMatch> > knn_matches1;
    std::vector< std::vector<DMatch> > knn_matches2;
    std::vector< std::vector<DMatch> > knn_matches3;
    std::vector< std::vector<DMatch> > knn_matches4;
    
    matcher->knnMatch( filtered.descriptors1, filtered.descriptors2, knn_matches, 2 );
    cout << filtered1.descriptors1.row(0) << endl;
    cout << filtered1.descriptors2.row(1) << endl;
    cout << " | " << knn_matches[0][0].distance << " | " << knn_matches[0][0].trainIdx << " | " << knn_matches[0][0].queryIdx << " | " <<  knn_matches[0][0].imgIdx << endl;
    cout << " | " << knn_matches[0][1].distance << " | " << knn_matches[0][1].trainIdx << " | " << knn_matches[0][1].queryIdx << " | " <<  knn_matches[0][1].imgIdx << endl;                
    
    
    matcher1->knnMatch( filtered2.descriptors1, filtered2.descriptors2, knn_matches1, 2 );
    cout << filtered2.descriptors1.row(0) << endl;
    cout << filtered2.descriptors2.row(1) << endl;
    cout << " | " << knn_matches1[0][0].distance << " | " << knn_matches1[0][0].trainIdx << " | " << knn_matches1[0][0].queryIdx << " | " <<  knn_matches1[0][0].imgIdx << endl;
    cout << " | " << knn_matches1[0][1].distance << " | " << knn_matches1[0][1].trainIdx << " | " << knn_matches1[0][1].queryIdx << " | " <<  knn_matches1[0][1].imgIdx << endl;                
    
    
    matcher2->knnMatch( filtered3.descriptors1, filtered3.descriptors2, knn_matches2, 2 );
    cout << filtered3.descriptors1.row(0) << endl;
    cout << filtered3.descriptors2.row(1) << endl;
    cout << " | " << knn_matches2[0][0].distance << " | " << knn_matches2[0][0].trainIdx << " | " << knn_matches2[0][0].queryIdx << " | " <<  knn_matches2[0][0].imgIdx << endl;
    cout << " | " << knn_matches2[0][1].distance << " | " << knn_matches2[0][1].trainIdx << " | " << knn_matches2[0][1].queryIdx << " | " <<  knn_matches2[0][1].imgIdx << endl;                
    
    
    matcher3->knnMatch( filtered4.descriptors1, filtered4.descriptors2, knn_matches3, 2 );
    cout << filtered4.descriptors1.row(0) << endl;
    cout << filtered4.descriptors2.row(1) << endl;
    cout << " | " << knn_matches3[0][0].distance << " | " << knn_matches3[0][0].trainIdx << " | " << knn_matches3[0][0].queryIdx << " | " <<  knn_matches3[0][0].imgIdx << endl;
    cout << " | " << knn_matches3[0][1].distance << " | " << knn_matches3[0][1].trainIdx << " | " << knn_matches3[0][1].queryIdx << " | " <<  knn_matches3[0][1].imgIdx << endl;         

    
    matcher4->knnMatch( filtered5.descriptors1, filtered5.descriptors2, knn_matches4, 2 );
    cout << filtered1.descriptors1.row(0) << endl;
    cout << filtered1.descriptors2.row(1) << endl;
    cout << " | " << knn_matches4[0][0].distance << " | " << knn_matches4[0][0].trainIdx << " | " << knn_matches4[0][0].queryIdx << " | " <<  knn_matches4[0][0].imgIdx << endl;
    cout << " | " << knn_matches4[0][1].distance << " | " << knn_matches4[0][1].trainIdx << " | " << knn_matches4[0][1].queryIdx << " | " <<  knn_matches4[0][1].imgIdx << endl;         
    
               
    if (filtered.considered_poses == filtered1.considered_poses) std::cout << "matched3" << std::endl;
        
    
    cout << filtered.filteredOutput[0].src.size() << endl;
    cout << filtered1.filteredOutput[0].src.size() << endl;
    
    if (filtered.filteredOutput[0].src == filtered1.filteredOutput[0].src)
    {
        std::cout << "matched" << std::endl;
    }
    else{
        std::cout << "Not matched" << std::endl;
        cout << filtered.filteredOutput[0].src[0] << endl;
        cout << filtered1.filteredOutput[0].src[0] << endl;
    }
    */
    vector<int> considered_poses = filtered.considered_poses;
    vector<retPointcloud> filteredOutput = filtered.filteredOutput;
    
    //cout << "considered_poses " << considered_poses.size() << endl;
    for(int i=0; i< considered_poses.size(); ++i)
        cout << considered_poses[i] << endl;;
        
    for(size_t i=0; i<considered_poses.size(); ++i) {
        
        if(i == 0) {
            Rot3 R(1, 0, 0, 0, 1, 0, 0, 0, 1);
            Point3 t;
            t(0) = 0;
            t(1) = 0;
            t(2) = 0;
            Pose3 pose(R, t);
            poses.push_back(pose);
            continue;   
        }
        Mat m1 = filteredOutput[i].ret[0];
        Mat m2 = filteredOutput[i].ret[1];
        vector<Point2f> src = filteredOutput[i].src;
        vector<Point2f> dst = filteredOutput[i].dst;
        
        retPose p = getPose(m1, m2, "icp");        
        Pose3 pose(p.R, p.t);
        poses.push_back(poses.back()*pose);        

        // Need to construct the landmarks array
        // Initialize and create gtsam graph here
        // i is the image index
        if(poses.size() == 2) {
            //cout << "Initializing prevKeypointIndexer" << endl;
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
        if(poses.size() > 2) {
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
    }

    cout << "Populated maps" << endl;
    
    // Run GTSAM bundle adjustment
    gtsam::Values result;
    Cal3_S2::shared_ptr Kgt(new Cal3_S2(focal_length, focal_length, 0 /* skew */, cx, cy));
    result = Optimize(KeypointMapper, Kgt, frames, considered_poses, poses);
    

    // Test GTSAM output  
    test_sfm(result, Kgt, considered_poses);

    return 0;
}

