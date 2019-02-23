# Detection-and-Tracking
Detection and Tracking for an AR marker


Given an image of a marker to track and a video containing it, assuming only one marker instance is present in the video at any time, this will track it within the video. Not exclusively tested, still under development. Can do detection and tracking as described within this paper:  https://stacks.stanford.edu/file/druid:bf950qp8995/Toole_Dolben.pdf

Detection Process using feature matching: 

1. Feature detection + descriptor extraction in marker image (BRISK features as opposed to SIFT in original paper)
2. Feature detection + descriptor extraction in video frame
3. Matching
  A. Nearest neighbors based matching scheme
  B. Lowes ratio test
4. Homography calculation between matches across video frame and marker image
  - Number of inliers calculated 
    - If inliers above a certain number then detection successful
    - Pass of feature locations to the tracking system

Tracking:

Use the KLT Tracker


At this point, you should use your own video and marker image. I'm not releasing the video / marker image I have but feel free to use your own.
