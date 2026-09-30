 #include <opencv2/opencv.hpp>
cv::Mat dehaze(cv::Mat image);
std::vector<double> estimate_column_intensity_stat(const cv::Mat &image);
//cv::Mat flatten(cv::Mat image,std::string debugfilename="");
cv::Mat flatten(cv::Mat image,float r_inner=1.0f, float r_outer=1.0f, std::string debugfilename="");
