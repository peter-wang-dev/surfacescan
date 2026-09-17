 #include <opencv2/opencv.hpp>
cv::Mat dehaze(cv::Mat image);
std::vector<double> estimate_column_intensity_stat(cv::Mat image);
cv::Mat flatten(cv::Mat image,std::string debugfilename="");
