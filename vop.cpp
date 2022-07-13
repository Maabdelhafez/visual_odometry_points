#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <Eigen/Core>
/*
#include <g2o/core/base_vertex.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/solver.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
*/
//#include <sophus/se3.hpp>
#include <sophus/se3.h>
#include <chrono>
//#include <opencv2/imgcodecs/legacy/constants_c.h>
#include <unistd.h>
#include <vector>
#include <string>
//#include <pangolin/pangolin.h>


using namespace std;
using namespace cv;
using namespace Eigen;


std::vector<String> fn0;
std::vector<String> fn1;
std::vector<String> fn3;

Vec3f  pointsTranslationVector = 0.0  ;



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



int main(int argc, char **argv) {

   cv::glob("seq/image_0", fn3);
  int j=1; // frm num
  while (j<fn3.size()) {
    std::vector<KeyPoint> keypoints_1;
    std::vector<KeyPoint> keypoints_2;
    std::vector<DMatch> matches;

    Mat depthMap =  generateDepthMap(j);

  //-- read the image
  
    Mat img_1 = imread(fn3[j-1], CV_LOAD_IMAGE_GRAYSCALE);
    Mat img_2 = imread(fn3[j], CV_LOAD_IMAGE_GRAYSCALE);
    assert(img_1.data != nullptr && img_2.data != nullptr);
    cout <<"Doing Feature match:" <<fn3[j-1] << ", " << fn3[j] << endl;
    j++;
    // find features and match them
    find_feature_matches(img_1, img_2, keypoints_1, keypoints_2, matches);
    Mat K = (Mat_<double>(3, 3) << 718.856, 0, 607.1928, 0, 718.856, 185.2157, 0, 0, 1);
    vector<Point3f> pts_3d;
    vector<Point2f> pts_2d;
    cout << "Found matches:" << matches.size() << endl;  
    for (DMatch m:matches) {
      auto& kp1 = keypoints_1[m.queryIdx].pt;
      auto& kp2 = keypoints_2[m.trainIdx].pt;
      ushort d = depthMap.ptr<unsigned short>(int(kp1.y))[int(kp1.x)]; 
      
      if (d == 0)   // bad depth
        continue;
      float dd = d / 5000.0; 
      Point2d p1 = pixel2cam(kp1, K);
      pts_3d.push_back(Point3f((p1.x) * dd, (p1.y) * dd, dd));
      pts_2d.push_back(kp2);
    }
    //-----
    int N = pts_3d.size();
    cout << "Found good depth: " << N << endl;
    if(N<6)
      continue;

    Mat r, R;
    Vec3f t ;
    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();

    solvePnP(pts_3d, pts_2d, K, Mat(), r, t, false); //  Call OpenCV's PnP solution, choose EPNP , DLS and other methods-->ex: ADD ,SOLVEPNP_EPNP after false
    
    cv::Rodrigues(r, R); // r is in the form of a rotation vector, converted to a matrix using the Rodrigues formula 
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
  
    pointsTranslationVector = pointsTranslationVector + t ;

    cout << "Translation total =" << endl << pointsTranslationVector << endl;
  // lop++;

  }
  return 0;
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
  cv::namedWindow("Matched 2D ORB Features", cv::WINDOW_KEEPRATIO);
  imshow("Matched 2D ORB Features", img_match);
  resizeWindow("Matched 2D ORB Features", 1800,400);
  waitKey(35);

  
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

   Mat imgDp_1, imgDp_2;
   
    imgDp_1 = imread(fn0[j], CV_LOAD_IMAGE_GRAYSCALE);
    imgDp_2 = imread(fn1[j], CV_LOAD_IMAGE_GRAYSCALE);
    assert(imgDp_1.data != nullptr && imgDp_2.data != nullptr);

    cv::Ptr<cv::StereoSGBM> sgbm = cv::StereoSGBM::create( 0, 96, 9, 8 * 9 * 9, 32 * 9 * 9, 1, 63, 10, 100, 32);    // tested parameters
    
    cv::Mat disparity_sgbm, disparity, disparityMap;
    sgbm->compute(imgDp_1, imgDp_2, disparity_sgbm);
    disparity_sgbm.convertTo(disparity, CV_32F, 1.0 / 16.0f);
    disparityMap = disparity/96.0; // 16.0 ;
  cv::namedWindow("Disparity", cv::WINDOW_KEEPRATIO);
  cv::imshow("Disparity", disparityMap );
  cv::resizeWindow("Disparity", 800,300);


  cv::waitKey(35);

   
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

  cv::namedWindow("Live depth Map", cv::WINDOW_KEEPRATIO);
 cv::imshow("Live depth Map", depthMap );
  cv::resizeWindow("Live depth Map", 800,300);


   cv::waitKey(35);
return  depthMap; 
}

