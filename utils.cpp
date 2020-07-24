#include "utils.h"    
    

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

void printKeypointMapper(map<int, map<int, Point2f>> mainMap) {
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