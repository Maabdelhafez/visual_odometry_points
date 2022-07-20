#include "vop.h"
#include <filesystem>
#include <sstream>
//int f= 1 ; int j = 1, lop=0; 


void find_feature_matches(
   Mat &img_1,  Mat &img_2,
  std::vector<KeyPoint> &keypoints_1,
  std::vector<KeyPoint> &keypoints_2,
  std::vector<DMatch> &matches);

// Pixel coordinate to camera normalized coordinate
Point2d pixel2cam(const Point2d &p, const Mat &K);

// BA by g2o
typedef vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>> VecVector2d;
typedef vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> VecVector3d;

//void bundleAdjustmentG2O(const VecVector3d &points_3d, const VecVector2d &points_2d, const Mat &K, Sophus::SE3d &pose);

// BA by gauss-newton
//void bundleAdjustmentGaussNewton(const VecVector3d &points_3d, const VecVector2d &points_2d, const Mat &K, Sophus::SE3d &pose);

Mat generateDepthMap(int j);


//----------------------------------------------------------MAIN-------------------------------------------------//
namespace{
  
  //----
  std::vector<String> fn0;
  std::vector<String> fn1;
  std::vector<String> fn3;
  bool enShow_ = true;
  int stride_= 4;
}
//---------------------
string img_frm_idx(const string& s1)
{
  string s= std::filesystem::path(s1).stem();
  int i=0;
  for(;i<s.length();i++)
    if(s[i]!='0') break;
  s = s.substr(i);
  return s;
}
//--------
string kitti_line(const Mat& Rw, const Mat& tw, 
            const string& sfrm)
{
  stringstream s;
  s.precision(7);
  s << std::fixed;
  //---- save for kitti evaluation
    s << img_frm_idx(sfrm); // current frame index
    cv::Mat_<double> Tw(3,4);
    for(int i=0;i<3;i++)
      Tw.col(i) = Rw.col(i);
    Tw.col(3) = tw;
    for(int i=0; i<Tw.rows; i++)
      for(int j=0; j<Tw.cols; j++)
        s << " " << Tw.at<double>(i, j);
    s << endl;
    string sr = s.str();
    return sr;
}
//------------------------

extern void run_vop()
{

   cv::glob("seq/image_0", fn3);

  //--- world transform
  Mat Rw = (Mat_<double>(3, 3) << 1, 0, 0,  0, 1, 0, 0, 0, 1);
  Mat tw = (Mat_<double>(3, 1) << 0,0,0);

  //---- open log file for world pose
  ofstream ofs;
  ofs.open("pose_log.txt");
  ofstream ofs2;
  ofs2.open("Tw.txt"); 
  
  //ofs2 << kitti_line(Rw, tw);

  //-----
  int ic = stride_; // current frame
  while(ic<fn3.size()) 
  {
    int ip = ic - stride_; // previous frm
    std::vector<KeyPoint> keypoints_1;
    std::vector<KeyPoint> keypoints_2;
    std::vector<DMatch> matches;
    cout << "gen depth map of frm:" << ip << endl;
    Mat depthMap =  generateDepthMap(ip);

  //-- read the image
  
    Mat img_1 = imread(fn3[ip], CV_LOAD_IMAGE_GRAYSCALE);
    Mat img_2 = imread(fn3[ic], CV_LOAD_IMAGE_GRAYSCALE);
    assert(img_1.data != nullptr && img_2.data != nullptr);
    cout <<"Doing Feature match:" <<fn3[ip] << ", " << fn3[ic] << endl;
    cout << "Feature match frm:" << ip <<", " << ic << endl;
    // find features and match them
    find_feature_matches(img_1, img_2, keypoints_1, keypoints_2, matches);
    Mat K = (Mat_<double>(3, 3) << 718.856, 0, 607.1928, 0, 718.856, 185.2157, 0, 0, 1);
    vector<Point3f> pts_3d;
    vector<Point2f> pts_2d;
    vector<double> dds;
    cout << "Found matches:" << matches.size() << endl;  
    cout << "depth:";
    for (DMatch m:matches) {
      auto& kp1 = keypoints_1[m.queryIdx].pt;
      auto& kp2 = keypoints_2[m.trainIdx].pt;
      auto& kpd = kp1; // ori
    //  auto& kpd = kp2;
      ushort d = depthMap.ptr<unsigned short>(int(kpd.y))[int(kpd.x)]; 
      
      if (d == 0)   // bad depth
        continue;
      float dd = d / 5000.0; 
      cout << dd << ", "; 
      dds.push_back(dd);
      Point2d p1 = pixel2cam(kp1, K);
      pts_3d.push_back(Point3f((p1.x) * dd, (p1.y) * dd, dd));
      pts_2d.push_back(kp2);
    }
    cout << endl;
    //-----
    int N = pts_3d.size();
    cout << "Found good depth: " << N << endl;
    if(N<6)
    {
      ic += stride_;
      continue;
    }

    Mat R;
//    Vec3f t ;
    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    cout << "solvePnPRansac()" << endl;
//    solvePnP(pts_3d, pts_2d, K, Mat(), r, t, false); //  Call OpenCV's PnP solution, choose EPNP , DLS and other methods-->ex: ADD ,SOLVEPNP_EPNP after false

  cv::Mat r(3,1,cv::DataType<double>::type);
  cv::Mat t(3,1,cv::DataType<double>::type);
 
  cv::solvePnPRansac(pts_3d, pts_2d, K, Mat(), r, t);
 
    Mat ot; // outliers
    cv::Mat D;

    //solvePnPRansac(pts_3d, pts_2d, K, D, r, t, false, 100, 8.0, 0.99); 
    cout << "Outliers:(" << ot.rows << ","<< ot.cols << ") of " << pts_2d.size()<<endl;
    cout << "Solved motion: r=" << r << ", ";
    cout << "t=" << t << endl; // delt t, delta R
    cv::Rodrigues(r, R); // r is in the form of a rotation vector, converted to a matrix using the Rodrigues formula 
    // T = [ R' | t'] = [ RT | -RT*t]
    // t1 = -R.transpose()*t;
    // R1 = R.transpose();
    //---- inverse transform to relative motion
    Mat dRw; transpose(R, dRw);
    Mat dt = -dRw*t;
    //---- to world transform
    Rw = Rw * dRw;
    tw = tw + dt;
    cv::Mat rw(3,1,cv::DataType<double>::type);
    cv::Rodrigues(Rw, rw);
    Mat ew = rw*180.0/M_PI; // to degree
    cout << "World pose: ew=" << ew << ", tw=" << tw << endl; 
    //---- save to log
    Mat ew1, tw1; 
    transpose(ew, ew1);
    transpose(tw, tw1); 
    ofs << ew1 << ",   " << tw1 << endl; 
    //---- save for kitti evaluation
    ofs2 << kitti_line(Rw, tw, fn3[ic]);

    //--------------
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
  
    //pointsTranslationVector = pointsTranslationVector + t ;

//    cout << "Translation total =" << endl << pointsTranslationVector << endl;
  // lop++;
    //---- draw depth pnts
    Mat imd = img_2;
    for(int i=0;i<dds.size();i++)
    {
      Point2f q = pts_2d[i];
      float d = dds[i];
      stringstream s;
      s << std::setprecision(2)<<d;
      cv::putText(imd, s.str(), q,FONT_HERSHEY_COMPLEX, 1,{255,0,0}, 2);//Putting the text in the matrix//
    }
      //cvtcolor(imd,cv::COLOR_BGR2GRAY);
      if(enShow_) {
        cv::namedWindow("dbg1", cv::WINDOW_KEEPRATIO);
      imshow("dbg1", imd);
    }
    // resizeWindow("dbg1", 1200,400);
    ic += stride_;
  }
  ofs.close();
}
//-------------
void find_feature_matches(
   Mat &img_1,  Mat &img_2,
  std::vector<KeyPoint> &keypoints_1,
  std::vector<KeyPoint> &keypoints_2,
  std::vector<DMatch> &matches) {

  Mat descriptors_1, descriptors_2;

  Ptr<FeatureDetector> detector = ORB::create();
  Ptr<DescriptorExtractor> descriptor = ORB::create();
  
  Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");
  
  //-- Step 1: Detect Oriented FAST corner positions
  detector->detect(img_1, keypoints_1);
  detector->detect(img_2, keypoints_2);

  //-- Step 2: Calculate the BRIEF descriptor based on the corner position
  descriptor->compute(img_1, keypoints_1, descriptors_1);
  descriptor->compute(img_2, keypoints_2, descriptors_2);

  //-- Step 3: Match the Brief descriptors in the two images, using the Hamming distance
  vector<DMatch> match;
  // BFMatcher matcher ( NORM_HAMMING );
  matcher->match(descriptors_1, descriptors_2, match);

  //-- Match point pair filter
  double min_dist = 10000, max_dist = 0;

  //Find the minimum and maximum distances between all matches, that is, 
  //the distances between the most similar and least similar two sets of points
  for (int i = 0; i < descriptors_1.rows; i++) {
    double dist = match[i].distance;
    if (dist < min_dist) min_dist = dist;
    if (dist > max_dist) max_dist = dist;
  }

//When the distance between descriptors is greater than twice the minimum distance, 
//it is considered that the matching is wrong. But sometimes the minimum distance will be very small, 
//and an empirical value of 30 is set as the lower limit.
  for (int i = 0; i < descriptors_1.rows; i++) {
//  if (match[i].distance <= max(min_dist, 20.0)) {
    if (match[i].distance <= max(min_dist*2, 30.0)) {
      matches.push_back(match[i]);
    }
  }
  Mat img_match;
  drawMatches(img_1, keypoints_1, img_2, keypoints_2, matches, img_match);
  if(enShow_) {
        cv::namedWindow("Matched 2D ORB Features", cv::WINDOW_KEEPRATIO);
        imshow("Matched 2D ORB Features", img_match);
      resizeWindow("Matched 2D ORB Features", 1800,400);
      waitKey(35);
  }

  
}
//---------------
Point2d pixel2cam(const Point2d &p, const Mat &K) {
  return Point2d
    (
      (p.x - K.at<double>(0, 2)) / K.at<double>(0, 0),
      (p.y - K.at<double>(1, 2)) / K.at<double>(1, 1)
    );
}

//---------------------------------------------------generate Depth Map---------------//

Mat generateDepthMap(int j) {

  float fx = 718.856, fy = 718.856, cx = 607.1928, cy = 185.2157; // dim in pixels
  float b = 0.573; // in meters

  cv::glob("seq/image_0", fn0);
  cv::glob("seq/image_1", fn1);

  cout <<"Depth map of frm :" << j << ", with" << fn0[j] << ", " << fn1[j] << endl;
   Mat imgDp_1, imgDp_2;
   
    imgDp_1 = imread(fn0[j], CV_LOAD_IMAGE_GRAYSCALE);
    imgDp_2 = imread(fn1[j], CV_LOAD_IMAGE_GRAYSCALE);
    assert(imgDp_1.data != nullptr && imgDp_2.data != nullptr);

    cv::Ptr<cv::StereoSGBM> sgbm = cv::StereoSGBM::create( 0, 96, 9, 8 * 9 * 9, 32 * 9 * 9, 1, 63, 10, 100, 32);    // tested parameters
    
    cv::Mat disparity_sgbm, disparity, disparityMap;
    sgbm->compute(imgDp_1, imgDp_2, disparity_sgbm);
    disparity_sgbm.convertTo(disparity, CV_32F, 1.0 / 16.0f);
    disparityMap = disparity/96.0; // 16.0 ;
 //   disparityMap = disparity/40.0; // 16.0 ;
    if(enShow_) {

      cv::namedWindow("Disparity", cv::WINDOW_KEEPRATIO);
      cv::imshow("Disparity", disparityMap );
      cv::resizeWindow("Disparity", 800,300);
      cv::waitKey(35);
    }


   
   Mat depthMap= disparity;


   
    for (int i = 0; i <depthMap.rows; i++)
        {
        for (int j = 0; j <depthMap.cols; j++)
          {
               if (depthMap.at<float>(i, j) <= 10.0 || depthMap.at<float>(i, j) >= 96.0) continue;
               double depth = ((fx*b)) / (depthMap.at<float>(i,j)) ;
               depthMap.at<float>(i,j) = depth; 
          }
        }

    if(enShow_) {

        cv::namedWindow("Live depth Map", cv::WINDOW_KEEPRATIO);
      cv::imshow("Live depth Map", depthMap );
        cv::resizeWindow("Live depth Map", 800,300);
       cv::waitKey(35);
  }
return  depthMap; 
}

