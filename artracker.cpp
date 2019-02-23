#include <iostream>
#include <vector>
#include <opencv2/features2d.hpp>
#include <opencv/cv.hpp>
#include <opencv2/stitching.hpp>

using namespace std;
using namespace cv;
using namespace detail;

int main()
{
int bit = 0;
Ptr<FeatureDetector> detector = BRISK::create(); // create a detector
VideoCapture cap;
cap.open("your_video_file_goes_here.mp4");

vector<KeyPoint> keypoints_marker,keypoints_frame;
Mat descriptors_marker,descriptors_frame;  // definitions, definitions
Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED); // Nearest neighbor based matching

Mat marker = imread("your_marker_image_goes_here.jpg");
Mat res;

detector->detectAndCompute(marker,noArray(),keypoints_marker,descriptors_marker);   // detect keypoints and descriptor for template image

vector <Point2f> newpoints;
vector<Point2f> xpoints;
if(!cap.isOpened())
{
cout << "\n Error opening video file! Exiting!! \n";
exit(1);
}
Mat frame,f1,f2,f3;
vector<ImageFeatures> common_points;

while(cap.get(CV_CAP_PROP_POS_FRAMES) < cap.get(CV_CAP_PROP_FRAME_COUNT))       // loop through the video frames while (curr pos < total_frames)
    {

    ////////////////////////////      ///----Tracking----\\\     /////////////////////////////////////////
    if(bit)                                 // if the marker is detected then stopping detection just focusing on tracking
    {
    f1 = f2.clone();
    frame = f3.clone();
    cap >> f3;
    cvtColor(f3,f2,COLOR_BGR2GRAY);
    vector<uchar> status;
    vector<float> err;
    
    calcOpticalFlowPyrLK(f1, f2, newpoints, xpoints, status, err, Size(10,10),3, TermCriteria(TermCriteria::COUNT|TermCriteria::EPS,20,0.03), 0, 0.001);    // using the LK tracker to determine where the points are in the next image
    // newpoints are the points in the current frame and xpoints are points in the next frame
    if(count(status.begin(),status.end(),0) == status.size())       // if none of the points found at all then marker is disappered, go back to detection again
    {
    bit = 0;
    continue;
    }
    vector<KeyPoint> newkeys;
    for(int i = 0; i < newpoints.size(); i++ ) 
    {
    newkeys.push_back(KeyPoint(newpoints[i], 1.f));
    }
    
    common_points.push_back(newpoints);     // for later use
    drawKeypoints(frame,newkeys,res,Scalar(0,0,255),DrawMatchesFlags::DEFAULT);
    resize(res,res,res.size()/2);
    imshow("Mat",res);
    waitKey(25);
    
    newpoints = xpoints;        // next keypoints now current keypoints
    continue;           // skip detection as marker found
    }

    ////////////////////////////      ///----Detection----\\\     /////////////////////////////////////////
    cap >> frame;       // read 

    detector->detectAndCompute(frame,noArray(),keypoints_frame,descriptors_frame);      // detect feature keypoint and descriptors in this frame
    vector<vector<DMatch> > matches12;        
   
    if(descriptors_frame.empty() || descriptors_marker.empty())     // if empty handle it
    {
    cout << "\n EMPTY DESCRIPTOR! \n";
    continue;    
    }
    if(descriptors_frame.type()!=CV_32F) 
    {
    descriptors_frame.convertTo(descriptors_frame, CV_32F);
    }
    if(descriptors_marker.type()!=CV_32F)                       // if wrong type handle it
    {
    descriptors_marker.convertTo(descriptors_marker, CV_32F);
    }   
    matcher->knnMatch(descriptors_marker,descriptors_frame,matches12,2); // query,train, match the descriptors from the template to the frame

    vector<Point2f> mark;
    vector<Point2f> fram;
    vector<KeyPoint> newframe;
    float lowe_ratio = 0.75;
    for(int i = 0; i < matches12.size(); i++)
    {
    if(matches12[i][0].distance < lowe_ratio*matches12[i][1].distance)
    {
        
        mark.push_back(keypoints_marker[matches12[i][0].queryIdx].pt);           // extract the marker and the frame matches as points to estimate the homography between the two
        fram.push_back(keypoints_frame[matches12[i][0].trainIdx].pt);                
        newframe.push_back(keypoints_frame[matches12[i][0].trainIdx]);                

    }
    }

    if(mark.size() == 0 || fram.size() == 0)
    {
    continue;    
    }
    Mat inliers;
    Mat H = findHomography(mark,fram,CV_RANSAC,3.0,inliers);                // calculate the homography and obtain the number of inliers from it
    cout << "\n Inliers: " << countNonZero(inliers) << "      ";


    if(countNonZero(inliers) > 20)                                          // if inliers greater than a certain number, successful detection
    {
    bit = 1;
    cap >> f3;
    
    cvtColor(f3,f2,COLOR_BGR2GRAY);
    cvtColor(frame,f1,COLOR_BGR2GRAY);
    vector<uchar> status;
    vector<float> err;
    
    calcOpticalFlowPyrLK(f1, f2, fram, newpoints, status, err, Size(10,10),3, TermCriteria(TermCriteria::COUNT|TermCriteria::EPS,20,0.03), 0, 0.001);       // track the first set once
    
    drawKeypoints(frame,newframe,res,Scalar(0,0,255),DrawMatchesFlags::DEFAULT);        // display the first frame
    resize(res,res,res.size()/2);
    imshow("Mat",res);
    waitKey(10);

    }
    else
    {
    cout << "\nNo match on frameID:  "<<cap.get(CV_CAP_PROP_POS_FRAMES);    
    }   

    

    }

cap.release();                              // release the VideoCapture object

// 3D reconstruction here
return 0;
}
