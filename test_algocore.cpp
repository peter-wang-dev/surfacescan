#include <gtest/gtest.h>
#include "algocore.h"
#if defined(_WIN32)
#include <stdlib.h>
#define SET_ENV(key, val) _putenv_s(key, val)
#else
#include <stdlib.h>
#define SET_ENV(key, val) setenv(key, val, 1)
#endif
import std;
std::string path_output="test_output"; 
//std::string path_output="C:/Users/cyber/test_output"; 
std::string path_input="d:/dev/cns_testinput"; 
std::string path_userinput;
//const std::string path_input="./test_output/"; 
TEST(core,flatten)
{ 
	auto datadir=path_input+"/dehaze";
	//auto datadir="D:/dev/test_output";
	for(int kr=0; kr<11; ++kr)
	{//index of ring 
		auto imgpath=std::format("{}/ch1r{}.png",datadir,kr);
		cv::Mat img8b=cv::imread(imgpath,CV_8U);
		cv::Mat img;
		img8b.convertTo(img,CV_16U,16.0f); // Convert to 16-bit unsigned
		ASSERT_FALSE(img.empty())<<"Failed to load image: "<<imgpath;
		cv::Mat flattened=flatten(img,1.0f,1.0f,std::format("{}/flattenning_{}.txt",path_output,kr));
		ASSERT_FALSE(flattened.empty())<<"Flattening failed.";
		//cv::imwrite(std::format("{}/flatten_result_{}.png",path_output,kr),flattened*4.0f);
	}
}
//TEST(core,dehaze)
//{
//	//auto datadir=path_input+"/dehaze";
//	auto datadir="D:/dev/test_output";
//	int kr=3;//index of ring 
//	auto imgpath=std::format("{}/ch1r{}.png",datadir,kr); 
//	cv::Mat img=cv::imread(imgpath,CV_8U);
//	ASSERT_FALSE(img.empty())<<"Failed to load image: "<<imgpath;
//	cv::Mat dehazed=dehaze(img);
//	ASSERT_FALSE(dehazed.empty())<<"Dehazing failed.";
//	cv::imwrite(path_output+"/dehaze_result.png",dehazed*25.0f);
//}
TEST(core, merge)
{
     SET_ENV("OPENCV_IO_MAX_IMAGE_PIXELS", "1099511627776");
    const std::filesystem::path datadir = path_input + "/merge2deg";
    std::vector<cv::Mat> fullmaps;

    ASSERT_TRUE(std::filesystem::exists(datadir)) << "Input directory does not exist: " << datadir;

    for (const auto& entry : std::filesystem::directory_iterator(datadir))
    {
        if (!entry.is_regular_file())
            continue; 
        std::string extension = entry.path().extension().string();
        std::ranges::transform(extension, extension.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); }); 
        if (extension != ".tif" && extension != ".tiff" &&extension!=".png")
            continue;

        cv::Mat image = cv::imread(entry.path().string(), cv::IMREAD_UNCHANGED);
        ASSERT_FALSE(image.empty()) << "Failed to load image: " << entry.path(); 
        fullmaps.push_back(image);
    }

    ASSERT_FALSE(fullmaps.empty()) << "No TIFF or PNG images found in: " << datadir; 
    cv::Mat merged=fullmaps[0]*0;
    for (const cv::Mat& image : fullmaps)
    {
        ASSERT_EQ(image.size(), fullmaps.front().size())
            << "All images must have the same dimensions.";
        ASSERT_EQ(image.type(), fullmaps.front().type())
            << "All images must have the same type.";
		merged+=image;
    }

    //cv::Mat merged;
    //cv::merge(fullmaps, merged); 
	//cv::Mat merged=fullmaps[0]-fullmaps[1];
    ASSERT_TRUE(cv::imwrite((std::filesystem::path(path_output)/"merge.png").string(),merged))<<"Failed to save merged image: ";
    ASSERT_TRUE(cv::imwrite((std::filesystem::path(path_output)/"merge_4ev.png").string(),merged*16))<<"Failed to save merged image: ";
    //ASSERT_TRUE(cv::imwrite((std::filesystem::path(path_output)/"merge_8ev.png").string(),merged*256))<<"Failed to save merged image: ";
    //ASSERT_TRUE(cv::imwrite((std::filesystem::path(path_output)/"merge_10ev.png").string(),merged*1024))<<"Failed to save merged image: ";
}
int main(int argc, char **argv) 
{
    std::filesystem::create_directories(path_output); // Create test output directory if it doesn't exist
	//std::string resdir="C:/astri2/lasv/lasvrepo/res"; //default resource directory
	//std::println("Using resource directory: {}",resdir);
	//std::filesystem::current_path(resdir); // Set working directory
	//std::filesystem::current_path(path_output);
     //_putenv_s("OPENCV_IO_MAX_IMAGE_PIXELS", "5000000000");
     SET_ENV("OPENCV_IO_MAX_IMAGE_PIXELS", "1099511627776");
    ::testing::InitGoogleTest(&argc, argv);
	if(argc>1) 
		path_userinput=argv[1];
    return RUN_ALL_TESTS();
}
