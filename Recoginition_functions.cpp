//
//  recognition.cpp
//  RealTimeObjectRecognition
// This code file consists of all the functions used in the Object detection. 
//  Created by Pulkit Saharan on 25/10/22.
//

#include "recognition.hpp"
using namespace cv;
using namespace std;

//creating function to create a binary image using thresholding technique
int threshold(cv::Mat& src, cv::Mat& dst, int threshold) {
    // declaring variables
    int i, j;
    cv::Mat blur;
    //creating destination image of same size and type as source image
    dst.create(src.size(), CV_8U);

    // We will smoothen our image using gaussian blur filter to make the thresholding better
    GaussianBlur(src, blur, Size(5, 5), 0, 0);

    // creating binary image by assigning 0 pixel to background and 255 to foreground/object
    for (i = 0; i < blur.rows; i++) {
        const cv::Vec3b* rptr = blur.ptr<cv::Vec3b>(i); // getting ith row of blurred image
        uchar* dptr = dst.ptr<uchar>(i); //getting ith row of destination image
        for (j = 0; j < dst.cols; j++) {
            if (rptr[j][0] < threshold || rptr[j][1] < threshold || rptr[j][2] < threshold) {
                dptr[j] = 255;
            }
            else {
                dptr[j] = 0;;
            }
        }
    }
    return(0);
}

//creating function to create a binary image using HSV thresholding technique
int hsv_threshold(cv::Mat& src, cv::Mat& dst, int sat_thresh, int val_thresh) {
    // declaring variables
    int i, j;
    cv::Mat hsv_img, blur;
    //creating destination image of same size and type as source image
    dst.create(src.size(), CV_8U);
    //COnvert image to hsv
    cvtColor(src, hsv_img, COLOR_BGR2HSV),

        // We will smoothen our image using gaussian blur filter to make the thresholding better
        GaussianBlur(hsv_img, blur, Size(5, 5), 0, 0);

    // creating binary image by assigning 0 pixel to background and 255 to foreground/object
    for (i = 0; i < blur.rows; i++) {
        const cv::Vec3b* rptr = blur.ptr<cv::Vec3b>(i); // getting ith row of blurred image
        uchar* dptr = dst.ptr<uchar>(i); //getting ith row of destination image
        for (j = 0; j < dst.cols; j++) {
            if (rptr[j][1] < sat_thresh && rptr[j][2] > val_thresh) {
                dptr[j] = 255;
            }
            else {
                dptr[j] = 0;;
            }
        }
    }
    return(0);
}


//Morphological filtering using erosion and dilation & closing and opening
//Function for erosion with 4-connected neighbor
int erosion(cv::Mat& src, cv::Mat& dst) {

    dst.create(src.size(), src.type());

    int i, j;

    for (i = 0; i < src.rows; i++) {
        for (j = 0; j < src.cols; j++) {

            bool flag = 0;
            // check if background pixel
            if (src.at<uchar>(i, j) == 0) {
                dst.at<uchar>(i, j) = src.at<uchar>(i, j);
            }
            // 4-connected
            //if pixel on the top of current pixel is 0 then flag =1
            if (i > 0) {
                if (src.at<uchar>(i - 1, j) == 0) {
                    flag = 1;
                }
            }
            //if pixel on the bottom of current pixel is 0 then flag =1
            if (i < src.rows - 1) {
                if (src.at<uchar>(i + 1, j) == 0) {
                    flag = 1;
                }
            }
            //if pixels to the left of current pixel is 0 then flag =1
            if (j > 0 && src.at<uchar>(i, j - 1) == 0) {
                flag = 1;
            }
            //if pixel to the right of current pixel is 0 then flag =1
            if (j < src.cols - 1 && src.at<uchar>(i, j + 1) == 0) {
                flag = 1;
            }
            //if flag is 1 then make the current pixel 0 else not
            if (flag == 1) {
                dst.at<uchar>(i, j) = 0;
            }
            else {
                dst.at<uchar>(i, j) = src.at<uchar>(i, j);
            }
        }

    }
    return(0);

}

//Function for dilation with 8-connected neighbor
int dilation(cv::Mat& src, cv::Mat& dst) {

    dst.create(src.size(), src.type());

    int i, j;

    for (i = 0; i < src.rows; i++) {
        for (j = 0; j < src.cols; j++) {
            bool flag = 0;
            // Checking for foreground pixels
            if (src.at<uchar>(i, j) == 255) {
                dst.at<uchar>(i, j) = src.at<uchar>(i, j);
            }
            // 8-connected
            //if pixel on the top of current pixel is 255 then flag =1
            if (i > 0) {
                if (j > 0 && src.at<uchar>(i - 1, j - 1) == 255) {
                    flag = 1;
                }
                if (src.at<uchar>(i - 1, j) == 255) {
                    flag = 1;
                }
                if (j < src.cols - 1 && src.at<uchar>(i - 1, j + 1) == 255) {
                    flag = 1;
                }
            }
            //if pixel on the bottom of current pixel is 255 then flag =1
            if (i < src.rows - 1) {
                if (j > 0 && src.at<uchar>(i + 1, j - 1) == 255) {
                    flag = 1;
                }
                if (src.at<uchar>(i + 1, j) == 255) {
                    flag = 1;
                }
                if (j < src.cols - 1 && src.at<uchar>(i + 1, j + 1) == 255) {
                    flag = 1;
                }
            }
            //if pixel on the left of current pixel is 255 then flag =1
            if (j > 0 && src.at<uchar>(i, j - 1) == 255) {
                flag = 1;
            }
            //if pixel on the right of current pixel is 255 then flag =1
            if (j < src.cols - 1 && src.at<uchar>(i, j + 1) == 255) {
                flag = 1;
            }
            // if flag is true, convert to foreground else keep original
            if (flag == 1) {
                dst.at<uchar>(i, j) = 255;
            }
            else {
                dst.at<uchar>(i, j) = src.at<uchar>(i, j);
            }
        }

    }
    return(0);

}

// Function for opening by applying erosion first and then dilation
int opening(cv::Mat& src, cv::Mat& dst) {

    cv::Mat intermediate(src.size(), CV_8UC1);

    erosion(src, intermediate);
    dilation(intermediate, dst);

    return(0);
}

// Function for closing by applying dilation first and then erosion
int closing(cv::Mat& src, cv::Mat& dst) {

    cv::Mat intermediate(src.size(), CV_8UC1);
    dilation(src, intermediate);
    dilation(src, intermediate);
    dilation(src, intermediate);

    erosion(intermediate, dst);

    return(0);
}

// Function to segment image and get features
vector<float> segment_and_get_features(cv::Mat &img_bw, cv::Mat &color_regions, cv::Mat &src)
{
    Mat labels;
    Mat stats;
    Mat centroids;
    vector<float> moments_features;
    
    // Get connected components
    int num_labels = cv::connectedComponentsWithStats(img_bw, labels, stats, centroids);
    color_regions.create(img_bw.size(), CV_8UC3);
    //Colors equal to number of labels
    std::vector<Vec3b> colors(num_labels);
    colors[0] = Vec3b(0, 0, 0); //background
    // Assign random color to each region
    for(int label = 1; label < num_labels; ++label)
        colors[label] = Vec3b( (rand()&255), (rand()&255), (rand()&255) );

    for(int i = 0; i < color_regions.rows; i++)
    {
        for(int j = 0; j < color_regions.cols; j++)
        {
            int label = labels.at<int>(i, j);
            //get pixel and assign specific color according to region
            Vec3b &pixel = color_regions.at<Vec3b>(i, j);
            pixel = colors[label];
        }
    }
    // Get largest region and features of this region
    moments_features = get_largest_region_features(img_bw,color_regions,src, labels, stats, centroids);
    return moments_features;
}

// Function to get largest region in the center of image and return feature vector
vector<float> get_largest_region_features(cv::Mat &img_bw, cv::Mat &dst, cv::Mat &src, cv::Mat &labels, cv::Mat &stats, cv::Mat &centroids)
{
    vector<float> moments_features;
    int mid_region_label = -1;
    Scalar color(255,0,0);
    int min_pixel_area = 100;
    int max_pixel_area = img_bw.total()*0.7;
    
    // iterating over regions detected
    for(int i = 0; i<stats.rows;i++)
    {
        Scalar color(255,0,0);
        // centroid of region
        double centroid_x = centroids.at<double>(i,0);
        double centroid_y = centroids.at<double>(i,1);
        
        // check if centroid is in center 1/3 image and area is between min area and max area
        if((centroid_x <= img_bw.cols *2/3) && (centroid_x >= img_bw.cols *1/3) &&  (centroid_y <= img_bw.rows *2/3) && (centroid_y >= img_bw.rows *1/3) && stats.at<int>(i, CC_STAT_AREA) >= min_pixel_area
           && stats.at<int>(i, CC_STAT_AREA) <= max_pixel_area )
        {
            mid_region_label = i;
        }
    }
    if(mid_region_label !=-1)
    {
        // calculate moments of selected region
        double centroid_x = centroids.at<double>(mid_region_label,0);
        double centroid_y = centroids.at<double>(mid_region_label,1);
        Point centroid(centroid_x, centroid_y);
        moments_features = get_moments_opencv(img_bw, dst, src, stats, mid_region_label, centroid);
    }
    return moments_features;
}

//Function to get moments of a specified region
vector<float> get_moments_opencv(cv::Mat binary_img, cv::Mat &dst, cv::Mat &src, cv::Mat stats, int region_label, Point centroid)
{
    vector<float> moments_features;

    Scalar color(255,0,0);
    
    // Get bounding box coordinated for region
    int x = stats.at<int>(Point(0, region_label));
    int y = stats.at<int>(Point(1, region_label));
    int w = stats.at<int>(Point(2, region_label));
    int h = stats.at<int>(Point(3, region_label));
    int area =stats.at<int>(Point(4, region_label));
    
    // Create ROI and calculate moments
    Rect roi(x,y,w,h);
    cv::Mat small_img =binary_img(roi);

    cv::Moments moment = cv::moments(small_img);

    // Calculating moment angle
    double angle_alpha = 0.5 * atan2((2 * moment.nu11),(moment.nu20 - moment.nu02));
    float angle = angle_alpha * (180 / (float)CV_PI);
    if (angle < 0) {
            angle = angle + 360;
    }
    if (angle >= 180) {
            angle = angle - 180;
    }
    
    // getting centroid of region
    int x_centroid =(moment.m10/moment.m00);
    int y_centroid =(moment.m01/moment.m00);
    int min_xp = INT_MAX;
    int max_xp = INT_MIN;
    int min_yp = INT_MAX;
    int max_yp = INT_MIN;
    int xpp, ypp;
    // Calculating converted x and y coordinates to axis of least moment
    for(int i = 0; i<binary_img(roi).rows;i++)
    {
        uchar *ptr = binary_img(roi).ptr<uchar>(i);
        for(int j=0;j<binary_img(roi).cols;j++)
        {
            if(ptr[j] == 255)
            {
                xpp =( j - x_centroid) * cos(angle_alpha) + (i - y_centroid) * sin(angle_alpha);
                ypp = (j - x_centroid) * -sin(angle_alpha) + (i - y_centroid) * cos(angle_alpha);
                if (xpp < min_xp)
                    min_xp = xpp;
                if (xpp > max_xp)
                    max_xp = xpp;
                if (ypp < min_yp)
                    min_yp = ypp;
                if (ypp > max_yp)
                    max_yp = ypp;
            }
        }
    }

    Scalar red(0,0,255);
    
    // Unrotate and get bounding box coordinates in XY plane and translate by adding centroid
    int unrotate_right_top_xp = max_xp * cos(angle_alpha) - max_yp * sin(angle_alpha) + centroid.x;
    int unrotate_right_top_yp = max_xp * sin(angle_alpha) + max_yp * cos(angle_alpha)+ centroid.y;
    int unrotate_right_bottom_xp = max_xp * cos(angle_alpha) - min_yp * sin(angle_alpha)+ centroid.x;
    int unrotate_right_bottom_yp = max_xp * sin(angle_alpha) + min_yp * cos(angle_alpha)+ centroid.y;
    int unrotate_left_top_xp = min_xp * cos(angle_alpha) - max_yp * sin(angle_alpha)+ centroid.x;
    int unrotate_left_top_yp = min_xp * sin(angle_alpha) + max_yp * cos(angle_alpha)+ centroid.y;
    int unrotate_left_bottom_xp = min_xp * cos(angle_alpha) - min_yp * sin(angle_alpha)+ centroid.x;
    int unrotate_left_bottom_yp = min_xp * sin(angle_alpha) + min_yp * cos(angle_alpha)+ centroid.y;
    

    Point ver_1 = Point(unrotate_left_top_xp, unrotate_left_top_yp);
    Point ver_2 = Point(unrotate_right_top_xp, unrotate_right_top_yp);
    Point ver_3 = Point(unrotate_right_bottom_xp, unrotate_right_bottom_yp);
    Point ver_4 = Point(unrotate_left_bottom_xp, unrotate_left_bottom_yp);
    
    float length = cv::norm(ver_1 - ver_2);
    float width = cv::norm(ver_2 - ver_3);
    float aspect_ratio =  width/length;
    float percent_filled = moment.m00 / (255*(max_xp - min_xp) * (max_yp - min_yp)) * 100 ;
   // cout << moment.m00 << " " << area << endl;
    
    // oriented bounding box on segmented image

    cv::line(dst, ver_1, ver_2, red, 2, LINE_8);
    cv::line(dst, ver_2, ver_3, red, 2, LINE_8);
    cv::line(dst, ver_3, ver_4, red, 2, LINE_8);
    cv::line(dst, ver_4, ver_1, red, 2, LINE_8);
    
    // oriented bounding box on source image
    cv::line(src, ver_1, ver_2, red, 2, LINE_8);
    cv::line(src, ver_2, ver_3, red, 2, LINE_8);
    cv::line(src, ver_3, ver_4, red, 2, LINE_8);
    cv::line(src, ver_4, ver_1, red, 2, LINE_8);
 
    // Creating feature vector using normalized moments, alpha and aspect ratio
    moments_features.push_back(moment.nu11);
    moments_features.push_back(moment.nu02);
    moments_features.push_back(moment.nu20);
    moments_features.push_back(aspect_ratio);
    moments_features.push_back(percent_filled);
    
    int len = 200;
    int x1 = roi.x + x_centroid ;
    int y1 = roi.y + y_centroid;
    int x2 = x1 + len * cos(angle_alpha);
    int y2 = y1 + len * sin(angle_alpha);
    
    // drawing angle of least moment axis
    line(dst, Point2f(x1, y1), Point2f(x2, y2), Scalar(0, 0, 255), 2, LINE_8);
    line(src, Point2f(x1, y1), Point2f(x2, y2), Scalar(0, 0, 255), 2, LINE_8);

    return moments_features;
}

//Classifiers
//function to compute the sum squared difference between two vectors
double sum_squared_difference(vector<float> x1, vector<float> x2) {
    double ssd = 0.0;

    for (int i = 0; i < x1.size(); i++)
    {
        ssd += (x1[i] - x2[i]) * (x1[i] - x2[i]);
    }
    return ssd;
}

// Function to get ascending values, for sorting
bool ascending(const pair<char*, double>& a, const pair<char*, double>& b)
{
    return a.second < b.second;
}

// Function to get descending values, for sorting
bool descending(const pair<char*, double>& a, const pair<char*, double>& b)
{
    return a.second > b.second;
}

// Function to get vector from m to n position
vector<pair<char*, double>> slice(vector<pair<char*, double>> const& v, int m, int n) {
    auto first = v.begin() + m;
    auto last = v.begin() + n + 1;
    vector<pair<char*, double>> vector(first, last);
    return vector;
}

//Function to get the top N similar images using sum squared distance
vector<pair<char*, double>> sorted_sum_squared_distances(std::vector<char*> filenames, std::vector<std::vector<float>> data, vector<float> target_feature, int N)
{
    vector<pair<char*, double>> distances_vector, top_n_distances_vector;
    string no = "No_label";
    vector<pair<char*, double>> p = {{&no[0],0.0}};
    // Show 'No label' when there is no object in the frame
    if(target_feature.size() == 0)
        return p;
    // iterate over all images in database
    for (int i = 0; i < filenames.size(); i++)
    {
        // calculate intersection
        double intersection = sum_squared_difference(data[i], target_feature);
        distances_vector.push_back(make_pair(filenames[i], intersection));
    }
    //sort and return top n
    sort(distances_vector.begin(), distances_vector.end(), ascending);
    top_n_distances_vector = slice(distances_vector, 1, N);
    return top_n_distances_vector;
}


// Function to find the scaled euclidean distance metrics
double scaled_euclidean_dist(vector<float> x1, vector<float> x2, vector<float> std_dev) {
    float difference=0;
    int i;
    // iterate over all features
    for (i = 0; i < x1.size(); i++) {
        difference += (x1[i] - x2[i]) * (x1[i] - x2[i]) / (std_dev[i] * std_dev[i]);
    }
    return difference;
}

// Function to find standard deviation of all features
vector<float> standard_deviation(std::vector<std::vector<float>> data)
{
    vector<float> std_dev(data[0].size() , 0), mean(data[0].size(), 0), count(data[0].size(), 0);
    // calculate mean of all features
    for(int i=0;i<data.size();i++)
    {
      for(int j=0;j<data[i].size();j++)
      {
          mean[j] += data[i][j];
          count[j] +=1;
      }
    }
    for(int i = 0;i<mean.size();i++)
    {
        mean[i] /= count[i];
    }
    // Calculate standard deviation using mean for all features
    for(int i=0;i<data.size();i++)
    {
      for(int j=0;j<data[i].size();j++)
      {
          std_dev[j] += (data[i][j] - mean[j]) * (data[i][j] - mean[j]);
      }
    }
    // divide sum by count - 1 for all features
    for(int i = 0;i<std_dev.size();i++)
    {
        std_dev[i] /= count[i]-1;
        std_dev[i] = sqrt(std_dev[i]);
    }
    
    return std_dev;
}

//Function to get the top N similar images using scaled euclidean distance
vector<pair<char*, double>> sorted_scaled_euclidean_dist(std::vector<char*> filenames, std::vector<std::vector<float>> data, vector<float> target_feature, int N)
{
    vector<pair<char*, double>> distances_vector, top_n_distances_vector;
    string no = "No_label";
    vector<pair<char*, double>> p = {{&no[0],0.0}};
    
    vector<float> std_dev = standard_deviation(data);
    // Show 'No label' when there is no object in the frame
    if(target_feature.size() == 0)
        return p;
    // iterate over all images in database
    for (int i = 0; i < filenames.size(); i++)
    {
        // calculate intersection
        double intersection = scaled_euclidean_dist(data[i], target_feature, std_dev);
        distances_vector.push_back(make_pair(filenames[i], intersection));
    }
    //sort and return top n
    sort(distances_vector.begin(), distances_vector.end(), ascending);
    top_n_distances_vector = slice(distances_vector, 1, N);
    return top_n_distances_vector;
}

//Function for KNN Classifier
string KNN_Classifier(std::vector<char*> filenames, std::vector<std::vector<float>> data, vector<float> target_feature, int K) {
   
    vector<pair<char*, double>> distances_vector, top_n_distances_vector;

    // Show 'No label' when there is no object in the frame
    if(target_feature.size() == 0)
        return "No_label";
    // iterate over all objects in database
    vector<float> std_dev = standard_deviation(data);
    for (int i = 0; i < filenames.size(); i++)
    {
        // calculate intersection
        double intersection = scaled_euclidean_dist(data[i], target_feature, std_dev);
        distances_vector.push_back(make_pair(filenames[i], intersection));
    }
    //sort distances
    sort(distances_vector.begin(), distances_vector.end(), ascending);
    
    string predicted_label = " ";
    int a = distances_vector.size();
    // label and label_count pair
    map<string, int>  label_count;

    // check K closest examples and keep count of all labels occurring in k examples
    for (int i=0; i < min(a,K); i++) {

        string label = distances_vector[i].first;

        if (label_count.find(label) == label_count.end()) {
            label_count[label]=1;
        }
        else {
            label_count[label]++;
        }
    }
    //find the actual label by extracting the label from the map with maximum value (majority vote)
    std::map<string, int>::iterator itr;
    int max = -1;
    for (itr = label_count.begin(); itr != label_count.end(); itr++) {
        if (itr->second > max) {
            max = itr->second;
            predicted_label = itr->first;
        }
    }

    // Showing 'Unknown' when there is an unknown object in the frame 
    double max_dist = 0.8;
    double min_predicted_dist = 10000;
        for (int i = 0; i < distances_vector.size(); i++) {
            if (distances_vector[i].first  ==  predicted_label)
            {
                if(distances_vector[i].second < min_predicted_dist)
                    min_predicted_dist =distances_vector[i].second;
            }
        }
  

    if(min_predicted_dist > max_dist)
        return "Unknown";
    return predicted_label;
}
