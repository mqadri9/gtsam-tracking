#include "pointcloud.h"

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