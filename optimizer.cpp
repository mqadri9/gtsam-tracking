#include "optimizer.h"


gtsam::Values Optimize(map<int, map<int, Point2f>> KeypointMapper, 
                       Cal3_S2::shared_ptr Kgt, 
                       vector<string> frames, 
                       vector<int> considered_poses, 
                       vector<Pose3> poses) {
    // Define the camera observation noise model
    auto noise = noiseModel::Isotropic::Sigma(2, 2.0);  // one pixel in u and v
    
    // Create a Factor Graph and Values to hold the new data
    NonlinearFactorGraph graph;
    auto poseNoise = noiseModel::Diagonal::Sigmas(
        (Vector(6) << Vector3::Constant(0.1), Vector3::Constant(0.3))
            .finished());  // 30cm std on x,y,z 0.1 rad on roll,pitch,yaw
    graph.addPrior(Symbol('x', 0), poses[0], poseNoise);  // add directly to graph        

    int N = poses.size();
    map<int, map<int, Point2f> > FilteredKeypointMapper;
    map<int, map<int, Point3f> > landmarks;
    map<int, map<int, Point2f> >::iterator it;
    map<int, Point2f>::iterator itt;
    vector<int>::iterator iti;
    int j = 0;
    for ( it=KeypointMapper.begin() ; it != KeypointMapper.end(); it++ ) {
        // Filter out keypoints that do not appear in at least N images
        if((it->second).size() < N) {
            continue;
        }
        FilteredKeypointMapper.insert(make_pair(j++, it->second));
    }
    if (FilteredKeypointMapper.size() == 0) {
        throw std::invalid_argument( "Size of FilteredKeypointMapper is 0");
    }
    //cout << "Created FilteredKeypointMapper" << endl;
    //printKeypointMapper(FilteredKeypointMapper);
    vector<int> good_poses;
    for (size_t i = 0; i < poses.size(); ++i) {
        //iti = find (considered_poses.begin(), considered_poses.end(), i);
        //if (iti == considered_poses.end()) continue;        
        PinholeCamera<Cal3_S2> camera(poses[i], *Kgt);
        map<int, map<int, Point2f> >::iterator it;
        int j=0;
        for ( it=FilteredKeypointMapper.begin() ; it != FilteredKeypointMapper.end(); it++ ) {
            //int frame_id = considered_poses[i];
            itt = (it->second).find(i);
            //if (itt == (it->second).end()) continue;
            Point2f measurement_cv2 = itt->second;
            Point2 measurement;
            measurement(0) = measurement_cv2.x;
            measurement(1) = measurement_cv2.y;            
            graph.emplace_shared<GenericProjectionFactor<Pose3, Point3, Cal3_S2> >(
              measurement, noise, Symbol('x', i), Symbol('l', j), Kgt);
            j++;
        }
        good_poses.push_back(i);
    }
    
    //cout << "populated good_poses" << endl;
    //for(int i=0; i<good_poses.size(); ++i)
    //  std::cout << good_poses[i] << ' ';
  
    //if(N != frames.size())
    //    throw invalid_argument( "This only works if N=frames.size(). Update code" );
    
    //graph.print("graph: ");
    // map keys are ordered
    vector<Point3> landmarks3d;
    //printKeypointMapper(FilteredKeypointMapper);
    for ( it=FilteredKeypointMapper.begin(); it != FilteredKeypointMapper.end(); it++ ) {
        Point2f landmark = (it->second).begin()->second; 
        int img_index =  (it->second).begin()->first;
        string img_path = data_folder + "/frame" + to_string(considered_poses[img_index]) + ".jpg";
        Mat disparity = imread( samples::findFile( img_path ), 0);
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
    
         
    // Create the data structure to hold the initial estimate to the solution
    Values initialEstimate;
    for (size_t i = 0; i < poses.size(); ++i) {
        if(std::find(good_poses.begin(), good_poses.end(), i) == good_poses.end()) continue;
        initialEstimate.insert(
            Symbol('x', i), poses[i]);
    }
    for (size_t j = 0; j < landmarks3d.size(); ++j) {
        initialEstimate.insert<Point3>(Symbol('l', j), landmarks3d[j]);
    }
    graph.print("Factor Graph:\n");
    initialEstimate.print("initial");
    //cout << "===================== POSES =========================" << endl;
    //for (size_t i = 0; i < poses.size(); ++i) {
    //        cout << poses[i] << endl;
    //}       

    //cout << "===================== LANDMARKS =========================" << endl;
    //for (size_t i = 0; i < landmarks3d.size(); ++i) {
    //       cout << landmarks3d[i] << endl;
    //}
    gtsam::Values result;
    result = DoglegOptimizer(graph, initialEstimate).optimize();
    //result.print("Final results:\n");
    cout << "initial error = " << graph.error(initialEstimate) << endl;
    cout << "final error = " << graph.error(result) << endl;
   
    //cout << poses.size() << endl;;
    //cout << landmarks3d.size() << endl;
    //cout << landmarks3d.size() << endl;
    return result;
}    
