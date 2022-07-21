#pragma once

#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include<opencv2/opencv.hpp>
#include <Eigen/Core>


#include "vop.h"

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
//#include <sophus/se3.h>
#include <chrono>
//#include <opencv2/imgcodecs/legacy/constants_c.h>
#include <unistd.h>
#include <vector>
#include <string>
#include <iomanip>
#include <fstream>
//#include <pangolin/pangolin.h>


using namespace std;
using namespace cv;
using namespace Eigen;


extern void run_vop();
