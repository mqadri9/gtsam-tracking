#include "test_sfm.h"


int NumDigits(int x)  
{  
    x = abs(x);  
    return (x < 10 ? 1 :   
        (x < 100 ? 2 :   
        (x < 1000 ? 3 :   
        (x < 10000 ? 4 :   
        (x < 100000 ? 5 :   
        (x < 1000000 ? 6 :   
        (x < 10000000 ? 7 :  
        (x < 100000000 ? 8 :  
        (x < 1000000000 ? 9 :  
        10)))))))));  
}  

void test_sfm(gtsam::Values result, Cal3_S2::shared_ptr Kgt, vector<int> considered_poses)
{
    int pose_id = 2;
    int target_frame_id = considered_poses[pose_id];
    int initial_frame_id = considered_poses[0];
    string img_path = image_folder + "/frame" + to_string(initial_frame_id) + ".jpg";
    cout << "==================== TESTING =================" << endl;
    cout << "image 0 :" << img_path << endl;
    Mat img = imread( samples::findFile( img_path ));
    
    string idx; 
    if (NumDigits(initial_frame_id) == 1) idx = "00" + to_string(initial_frame_id);
    if (NumDigits(initial_frame_id) == 2) idx = "0" + to_string(initial_frame_id);
    if (NumDigits(initial_frame_id) == 3) idx = to_string(initial_frame_id);
    
    string filename =  "/mnt/c/Users/moham/OneDrive/Desktop/others/151_6_6/bbox/frame000" + idx + "_FRUITLET.csv";
    cout << "CSV FILE " << filename << endl;
    vector<vector<float>> csv = read_csv(filename);
    img_path = data_folder + "/frame"+  to_string(initial_frame_id) + ".jpg";
    cout << "disparity FILE " << img_path << endl;
    Mat disparity = imread( samples::findFile( img_path ), 0);
    
    for(int j = 0; j < csv.size(); j++) {
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
    Pose3 P0 = result.at(Symbol('x', 0).key()).cast<Pose3>();
    Pose3 P = result.at(Symbol('x', pose_id).key()).cast<Pose3>();
    //cout << P0 << endl;
    //landmark3d = P0.rotation()*(landmark3d - P0.translation());
    cout << landmark3d << endl;
    PinholeCamera<Cal3_S2> camera(P, *Kgt);
    Point2 measurement = camera.project(landmark3d);
    Point2f mmm;
    mmm.x = measurement(0);
    mmm.y = measurement(1);
         
    cout << measurement << endl;
    ShowBlackCircle(img, pt3, 5);
    img_path = image_folder + "/frame" + to_string(target_frame_id) + ".jpg";
    img = imread( samples::findFile( img_path ));        
    
    ShowBlackCircle(img, mmm, 5);
}