//
//  main.cpp
//  RealTimeObjectRecognition
// This code is to recognize the objects in real time by using the features and classifiers. 
// This takes in a feature_DB csv as an input to get the train data and then classify the object. 
// We can put the system in training mode by pressing 'n' and the features will be appended in the features_DB csv.
//  Created by Pulkit Saharan on 25/10/22.
//

#include <iostream>
#include<opencv2/opencv.hpp>
#include "recognition.hpp"
#include "csv_util.hpp"
using namespace std;
using namespace cv;



int main(int argc, char* argv[]) {
   // Declaring all the variables 
    cv::VideoCapture* capdev;
    vector<float> features;
    string label;
    std::vector<char*> filenames;
    std::vector<std::vector<float>> data;
    vector<pair<char*, double>> top_labels;
    cv::Mat frame, frame_thresh, frame_hsv, frame_clean, color_regions_frame;
    char pressed_key = 'o';
    char buffer[256];
    char* object_label, * csv;
    int key;
    string predicted_label;
    char* csv_filename;

    //Load feature_DB csv to train and classify objects
    string feature_csv = "C:\\Users\\ASUS\\Documents\\CS5330\\Project3\\features_DB.csv";
    csv_filename = &feature_csv[0];

    // open the video device
    capdev = new cv::VideoCapture(1);
    capdev->set(cv::CAP_PROP_FRAME_WIDTH, 1280);//Setting the width of the video 1280
    capdev->set(cv::CAP_PROP_FRAME_HEIGHT, 720);//Setting the height of the video// 720
   //CHeck if the camera is able to open or not
    if (!capdev->isOpened()) {
        printf("Unable to open video device\n");
        return(-1);
    }

    // get some properties of the image
    cv::Size refS((int)capdev->get(cv::CAP_PROP_FRAME_WIDTH),
        (int)capdev->get(cv::CAP_PROP_FRAME_HEIGHT));
    printf("Expected size: %d %d\n", refS.width, refS.height);

    //Reading the data from the feature csv
    read_image_data_csv(csv_filename, filenames, data);
    cout << "Read image database" << endl;

    cv::namedWindow("Segmentation");
    cv::namedWindow("Video", 1); // identifies a window

    for (;;) {
        
        *capdev >> frame; // get a new frame from the camera, treat as a stream
        if (frame.empty()) {
            printf("frame is empty\n");
            break;
        }

        //Thresholding the video frame
        threshold(frame, frame_thresh, 100);
        //Cleaning the object in the frame
        closing(frame_thresh, frame_clean);
        closing(frame_thresh, frame_clean);
        closing(frame_thresh, frame_clean);
        //Strong object features in features vector 
        features = segment_and_get_features(frame_clean, color_regions_frame, frame);
        //write feature values to frame
        if (features.size() != 0)
        {
            //Putting text on the frame which will show the values of features for the object present in frame
            string aspect_feature_string = "Aspect Ratio: " + to_string(features[3]);
            string nu11_feature_string = "NU11:" + to_string(features[0]);
            string nu02_feature_string = "NU02:" + to_string(features[1]);
            string nu20_feature_string = "NU20:" + to_string(features[2]);
            string percent_feature_string = "Percent Filled:" + to_string(features[4]);

            cv::putText(frame, aspect_feature_string, cv::Point(50, 100), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 1, false);
            cv::putText(frame, nu11_feature_string, cv::Point(50, 150), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 1, false);
            cv::putText(frame, nu02_feature_string, cv::Point(50, 200), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 1, false);
            cv::putText(frame, nu20_feature_string, cv::Point(50, 250), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 1, false);
            cv::putText(frame, percent_feature_string, cv::Point(50, 300), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 1, false);

        }

        // see if there is a waiting keystroke
        key = cv::waitKey(10);
        // store the keystroke for future frames
        if (key != -1)
            pressed_key = key;
        // taking actions based on pressed key
        switch (pressed_key)
        {
            // display thresholded video if pressed key is t
        case 't':
            cv::namedWindow("Thresholded Video");
            threshold(frame, frame_thresh, 100);
            cv::imshow("Thresholded Video", frame_thresh);
            cv::imshow("Video", frame);
            break;

            // display cleaned video if pressed key is h
        case 'c':
            cv::namedWindow("Cleaned Video");
            cv::imshow("Cleaned Video", frame_clean);
            cv::imshow("Video", frame);
            break;
            // display Segmented video if pressed key is s
        case 's':
            cv::namedWindow("Segmented Video");
            cv::imshow("Segmented Video", color_regions_frame);
            cv::imshow("Video", frame);
            break;
            // capture training image if key pressed is n
        case 'n':
            std::cout << "Enter the label for the object: ";
            std::cin >> label;
            std::cout << endl;

            object_label = &label[0];
            csv = &feature_csv[0];
            strcpy(buffer, object_label);
            append_image_data_csv(csv, buffer, features, false);
            pressed_key = 'o';
            break;
            // Baseline object detecting classifier
        case 'b':
            top_labels = sorted_scaled_euclidean_dist(filenames, data, features, 1);
            predicted_label = top_labels[0].first;
            cv::putText(frame, predicted_label, cv::Point(400, 100), cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(0, 255, 0), 2, false);
            cv::imshow("Video", frame);
            break;
            // KNN object detection  
        case 'k':
            predicted_label = KNN_Classifier(filenames, data, features, 3);
            cv::putText(frame, predicted_label, cv::Point(400, 100), cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(0, 255, 0), 2, false);
            cv::imshow("Video", frame);
            break;
            // Sum Squared difference classifier 
        case 'e':
            top_labels = sorted_sum_squared_distances(filenames, data, features, 1);
            predicted_label = top_labels[0].first;
            cv::putText(frame, predicted_label, cv::Point(400, 100), cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(0, 255, 0), 2, false);
            cv::imshow("Video", frame);
            break;
        default:
            cv::imshow("Video", frame);
            break;
        }
        cv::imshow("Segmentation", color_regions_frame);

        // quit if key pressed is 'q'
        if (key == 'q')
            break;
    }

    delete capdev;
    return(0);
}
