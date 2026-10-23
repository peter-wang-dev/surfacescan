 #include <opencv2/opencv.hpp>
cv::Mat dehaze(cv::Mat image);
std::vector<double> estimate_column_intensity_stat(const cv::Mat &image);
//cv::Mat flatten(cv::Mat image,std::string debugfilename="");
cv::Mat flatten(cv::Mat image,float r_inner=1.0f, float r_outer=1.0f, std::string debugfilename="");
cv::Mat get_merged_image(const std::vector<cv::Mat>& fullmaps);
int ransac_fit_circle(const std::vector<cv::Point2d>& points,double& best_circle_x,double& best_circle_y,double& best_circle_r,int iterations,double threshold);
std::vector<cv::Point2d> get_edge_points(const cv::Mat& image);
bool find_notch(const std::vector<cv::Point2d>& edge_points,cv::Point2d& valley,cv::Point2d& center);
std::vector<float> get_intensity_distribution(const cv::Mat& image,int column_index=-1);//samples the pixels in the image and returns a ascendingly sorted vector of intensity values. The input image could be CV_8UC1, CV_16UC1, or CV_32FC1. The output is a vector of floats in ascending order.
