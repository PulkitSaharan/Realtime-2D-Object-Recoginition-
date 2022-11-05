//
//  recognition.hpp
//  RealTimeObjectRecognition
//  
//  Created by Pulkit Saharan on 25/10/22.
//

#ifndef recognition_hpp
#define recognition_hpp

#include <stdio.h>
#include<opencv2/opencv.hpp>
#include <iostream>
using namespace cv;
using namespace std;
vector<float> segment_and_get_features(cv::Mat &img_bw, cv::Mat &color_regions, cv::Mat &src);
vector<float> get_largest_region_features(cv::Mat &img_bw, cv::Mat &dst, cv::Mat &src, cv::Mat &labels, cv::Mat &stats, cv::Mat &centroids);
vector<float> get_moments_opencv(cv::Mat binary_img, cv::Mat &dst, cv::Mat &src, cv::Mat stats, int region_label, Point);
int opening(cv::Mat& src, cv::Mat& dst);
int closing(cv::Mat& src, cv::Mat& dst);
int threshold(cv::Mat& src, cv::Mat& dst, int threshold);
int hsv_threshold(cv::Mat& src, cv::Mat& dst, int sat_thresh, int val_thresh);
int erosion(cv::Mat& src, cv::Mat& dst);
int dilation(cv::Mat& src, cv::Mat& dst);
vector<pair<char*, double>> sorted_sum_squared_distances(std::vector<char*> filenames, std::vector<std::vector<float>> data, vector<float> target_feature, int N);
bool ascending(const pair<char*, double>& a, const pair<char*, double>& b);
bool descending(const pair<char*, double>& a, const pair<char*, double>& b);
double sum_squared_difference(vector<float> x1, vector<float> x2);
double scaled_euclidean_dist(vector<float> x1, vector<float> x2, vector<float> std_dev);
vector<pair<char*, double>> sorted_scaled_euclidean_dist(std::vector<char*> filenames, std::vector<std::vector<float>> data, vector<float> target_feature, int N);
string KNN_Classifier(std::vector<char*> filenames, std::vector<std::vector<float>> data, vector<float> target_feature, int K);
vector<float> standard_deviation(std::vector<std::vector<float>> data);
#endif /* recognition_hpp */
