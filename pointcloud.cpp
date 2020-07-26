#include "pointcloud.h"


retFiltering filterImages(vector<string> frames)
{
    vector<int> tmp_considered;
    vector<int> considered_poses;
    // Create SIFT detector and define parameters
    std::vector<KeyPoint> keypoints1, keypoints2;
    int minHessian = 400;
    Ptr<SURF> detector = SURF::create( minHessian );
    Mat descriptors1, descriptors2; 
    Mat d1, d2;
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::BRUTEFORCE);
    //cout << "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++" << endl;
    const float ratio_thresh = 0.7f;
    vector<retPointcloud> filteredOutput;
    retFiltering rf;
    for(size_t i=0; i<frames.size(); ++i) {
        std::vector< std::vector<DMatch> > knn_matches;

        // Read image at index i 
        string img_path = image_folder + "/frame" + frames[i] + ".jpg";
        cout << img_path <<"\n";
        Mat img = imread( samples::findFile( img_path ), IMREAD_GRAYSCALE );
        if (img.empty()) {
            throw std::invalid_argument( "Unable to open image");
        }        
        
        // If it is the first image in the sequence, detect the keypoints and continue 
        // to next image
        if (i==0) {
            detector->detectAndCompute( img, noArray(), keypoints2, descriptors2 );
            continue;
        }
        keypoints1 = keypoints2; 
        descriptors1 = descriptors2;         
        detector->detectAndCompute( img, noArray(), keypoints2, descriptors2 );
        
        matcher->knnMatch( descriptors1, descriptors2, knn_matches, 2 );        
        //if(frames[i] == "15") {
        //    d1 = descriptors1;
        //    d2 = descriptors2;
            //cout << d1.row(0) << endl;
            //cout << d2.row(1) << endl;
            //cout << i << " | " << knn_matches[0][0].distance << " | " << knn_matches[0][0].trainIdx << " | " << knn_matches[0][0].queryIdx << " | " <<  knn_matches[0][0].imgIdx << endl;
            //cout << i << " | " << knn_matches[0][1].distance << " | " << knn_matches[0][1].trainIdx << " | " << knn_matches[0][1].queryIdx << " | " <<  knn_matches[0][1].imgIdx << endl;            
        //}
                       
        img_path = data_folder + "/frame" + frames[i-1] + ".jpg";
        Mat disparity1 = imread( samples::findFile( img_path ), 0);

        img_path = data_folder + "/frame" + frames[i] + ".jpg";
        Mat disparity2 = imread( samples::findFile( img_path ), 0);
         
        retPointcloud s = createPointClouds(disparity1, disparity2, keypoints1, keypoints2, knn_matches);
        Mat m1 = s.ret[0];
        Mat m2 = s.ret[1];
        vector<Point2f> src = s.src;
        vector<Point2f> dst = s.dst;

        //if(frames[i] == "15") {
        //    cout << descriptors1.size() << endl;
        //    cout << descriptors2.size() << endl;
        //    cout << i << " | " << keypoints1[0].pt << " | " << keypoints2[0].pt << endl;
        //    cout << i << " | " << src[0] << " | " << dst[0] << endl;        
        //}
        

        
        filteredOutput.push_back(s);
        //cout << "+++++++++++++++++++++++++++++++++++++++++" << endl;
        //cout << m1.size().height << endl;
        if(m1.size().height < 100) {
            cout << "Number of matches " << m1.size().height<< " found between frames " << i-1 << " and  "<< i << " is too low. Skipping frame." << i << endl;
            if(tmp_considered.size() > considered_poses.size()){
                considered_poses = tmp_considered;
            }
            tmp_considered.clear();        
            continue;
        }
        tmp_considered.push_back(stoi(frames[i]));
    }
    
    
    if(tmp_considered.size() > considered_poses.size()){
        considered_poses = tmp_considered;
    }
    vector<retPointcloud> finalFilteredOutput;
    for(size_t i=0; i<frames.size(); ++i) {
        if (std::count(considered_poses.begin(), considered_poses.end(), stoi(frames[i]))) { 
            finalFilteredOutput.push_back(filteredOutput[i-1]);
        }
    }
    rf.considered_poses = considered_poses;
    rf.filteredOutput = finalFilteredOutput;
    //rf.descriptors1 = d1;
    //rf.descriptors2 = d2;
    //cout << rf.filteredOutput[0].ret[0] << endl;
    //cout << "==================================================================" << endl;
    //cout << rf.filteredOutput[27].ret[0] << endl;
    return rf;
}


retPointcloud createPointClouds(Mat disparity1, Mat disparity2, std::vector<KeyPoint> keypoints1, std::vector<KeyPoint> keypoints2, std::vector< std::vector<DMatch> > knn_matches)
{
    vector<float> errors;
    vector<float> rawcloud1;
    vector<float> rawcloud2;
    vector<Point2f> src;
    vector<Point2f> dst;    
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
        if (errors[l] < mean/10 ) {
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
    cout << m1.size() << endl;
    memcpy(m1.data, pointcloud1.data(), pointcloud1.size()*sizeof(float)); 
    
    Mat m2 = Mat(pointcloud2.size()/3, 1, CV_32FC3);
    memcpy(m2.data, pointcloud2.data(), pointcloud2.size()*sizeof(float));
    
    vector<Mat> ret;
    ret.push_back(m1);
    ret.push_back(m2);
    
    retPointcloud s;
    s.ret = ret;
    s.src = src;
    s.dst = dst;
        
    return s;
}