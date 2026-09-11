#include <gtest/gtest.h>
#include "algocore.h"
import std;
std::string path_output="test_output"; 
//std::string path_output="C:/Users/cyber/test_output"; 
std::string path_input="C:/astri2/cnsvision/testinput"; 
std::string path_userinput;
//const std::string path_input="./test_output/"; 
TEST(core,flatten)
{ 
	auto datadir=path_input+"/dehaze";
	for(int kr=0; kr<11; ++kr)
	{//index of ring 
		auto imgpath=std::format("{}/ch1r{}.png",datadir,kr);
		cv::Mat img=cv::imread(imgpath,CV_8U);
		ASSERT_FALSE(img.empty())<<"Failed to load image: "<<imgpath;
		cv::Mat flattened=flatten(img,std::format("{}/flattenning_{}.txt",path_output,kr));
		ASSERT_FALSE(flattened.empty())<<"Flattening failed.";
		cv::imwrite(std::format("{}/flatten_result_{}.png",path_output,kr),flattened*4.0f);
	}
}
TEST(core,dehaze)
{
	//auto datadir=path_input+"/dehaze";
	auto datadir="D:/dev/test_output";
	int kr=3;//index of ring 
	auto imgpath=std::format("{}/ch1r{}.png",datadir,kr); 
	cv::Mat img=cv::imread(imgpath,CV_8U);
	ASSERT_FALSE(img.empty())<<"Failed to load image: "<<imgpath;
	cv::Mat dehazed=dehaze(img);
	ASSERT_FALSE(dehazed.empty())<<"Dehazing failed.";
	cv::imwrite(path_output+"/dehaze_result.png",dehazed*25.0f);
}
int main(int argc, char **argv) 
{
    std::filesystem::create_directories(path_output); // Create test output directory if it doesn't exist
	//std::string resdir="C:/astri2/lasv/lasvrepo/res"; //default resource directory
	//std::println("Using resource directory: {}",resdir);
	//std::filesystem::current_path(resdir); // Set working directory
	//std::filesystem::current_path(path_output);
    ::testing::InitGoogleTest(&argc, argv);
	if(argc>1) 
		path_userinput=argv[1];
    return RUN_ALL_TESTS();
}
