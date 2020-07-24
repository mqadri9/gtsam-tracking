 #include "pose.h"
 
 
 retPose getPose(Mat m1, Mat m2, const std::string& method) {
    Rot3 R;
    Point3 t;
    if(method == "procrustes") {
        Procrustes proc(false, false);
        proc.procrustes(m1, m2);
        //cout << proc.error << endl;
        Mat Rr = proc.rotation ;
        Mat Tr = proc.translation;
        Rot3 R1(
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
        
        t(0) = Tr.at<float>(0);
        t(1) = Tr.at<float>(1);
        t(2) = Tr.at<float>(2);
        R = R1;
    }
    
    else if(method == "icp") {   
        typedef PointMatcher<float> PM;
        typedef PM::DataPoints DP;    
        PM::ICP icp;
        
        const DP ref = create_datapoints(m1);
        const DP data = create_datapoints(m2);
        icp.setDefault();
        PM::TransformationParameters T = icp(ref, data);
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
        
        t(0) = T(0,3);
        t(1) = T(1,3);
        t(2) = T(2,3);
        R = R1;     
    }
    retPose s;
    s.R = R;
    s.t = t;
    //Mat m2_transformed = matrix_transform<float>(m2, Rr, Tr);

    //save_vtk<float>(m2_transformed, "m2_transformed.vtk");
    //save_vtk<float>(m2, "m2.vtk"); 
    //save_vtk<float>(m1, "m1.vtk"); 
        
    return s;
}