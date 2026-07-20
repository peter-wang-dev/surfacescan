#include <gtest/gtest.h>
#include "algorithm.h"
#include <opencv2/opencv.hpp>
import std;
std::string path_output="./test_output/"; 
std::string path_input="C:/astri2/cnsvision/testinput/"; 
//const std::string path_input="./test_output/"; 
class AlgoTest: public ::testing::Test
{
protected:
	void SetUp() override
	{
		auto res_init=Initialize();
		ASSERT_EQ(res_init.IsSuccess,true)<<"Initialization failed: "<<res_init.ErrorMessage;
	}
	void TearDown() override
	{
	}
};
TEST_F(AlgoTest,DISABLED_calib)
{ 
	auto res_calbegin=BeginCenterCalculation({300.0, ContourFeature::Notch, 45.0, 1200, 1920, 0.1, SearchDirection::TopToBottom, 128});
	ASSERT_EQ(res_calbegin.IsSuccess,true)<<"Center calculation failed: "<<res_calbegin.ErrorMessage;

	//load images from the directory
	std::vector<cv::Mat> images;
	namespace fs = std::filesystem;
	fs::path dir = path_input+"stage_center_cal"; 
	ASSERT_EQ(fs::exists(dir),true)<<"Directory not found: "<<dir.string();
	ASSERT_EQ(fs::is_directory(dir),true)<<"Not a directory: "<<dir.string();
	std::vector<fs::path> files;
	for (auto const& entry : fs::directory_iterator(dir)) 
	{
		if (!entry.is_regular_file()) continue;
		auto ext = entry.path().extension().string();
		//std::println("Found file: {} with extension: {}",entry.path().string(),ext);
		std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c){ return std::tolower(c); });
		if (ext == ".png" || ext == ".jpg" || ext == ".jpeg" || ext == ".bmp" || ext == ".tif" || ext == ".tiff")
			files.push_back(entry.path());
	} 
	std::sort(files.begin(), files.end());
	for (auto const& p : files) {
		cv::Mat img = cv::imread(p.string(), cv::IMREAD_UNCHANGED);
		if (img.empty()) {
			std::cerr << "Failed to load image: " << p.string() << std::endl;
			continue;
		}
		images.push_back(img);
		std::println("Loaded image: {} ({}x{})",p.string(),img.cols,img.rows);
		auto res_add=AddCenterCalculationImage(img.data,0.0); 
		if(!res_add.IsSuccess)
			std::cerr << "Failed to add center calculation image: " << res_add.ErrorMessage << std::endl;
	}

	ASSERT_GT(images.size(), 0u) << "No images loaded from " << dir.string();
	double dx=0.0,dy=0.0,angle=0.0;
	auto res_calend=EndCenterCalculation(&dx,&dy,&angle);
	std::println("EndCenterCalculation result: dx={}, dy={}, angle={}\n",dx,dy,angle);
	ASSERT_EQ(res_calend.IsSuccess,true)<<"End center calculation failed: "<<res_calend.ErrorMessage;
}
TEST_F(AlgoTest,ChannelProcess_Single)
{
	auto channel=DetectChannel::Narrow; 
	int FrameWidth=1024,FrameHeight=274,TotalRings=10;
	int RingWidth=FrameWidth,RingHeight=27400;
	float PixelSize=10; // um/pixel
	std::string ringsettings = std::format("{{\"FrameWidth\":{}, \"FrameHeight\":{}, \"TotalRings\":{}, \"RingWidth\":{}, \"RingHeight\":{}, \"PixelSize\":{}}}",
                                      FrameWidth, FrameHeight, TotalRings, RingWidth, RingHeight,PixelSize);
	auto res_beginchannel = BeginChannelProcess(channel, ringsettings.c_str(),
                                            "cluster_setting.json","classify_setting.json",
                                            "coord_cali_setting.json","DSizeCurve.json");
	ASSERT_EQ(res_beginchannel.IsSuccess,true)<<"BeginChannelProcess failed: "<<res_beginchannel.ErrorMessage;

	for(int kr=0;kr<TotalRings;kr++)//index of ring
	{
		float theta0=kr*20.0f,theta1=theta0+360.0f; //triggered start and end angles in degrees
		std::string coordCaliJson_ring = std::format("{{\"RingWidth\":{}, \"RingHeight\":{}, \"TriggeredStart\":{{\"T\":{}}}, \"TriggeredEnd\":{{\"T\":{}}}}}", RingWidth, RingHeight, theta0,theta1);
		auto res_beginring = BeginRingProcess(channel, kr, "process_setting.json",
                                      "haze_cali_setting.json", coordCaliJson_ring.c_str());
		ASSERT_EQ(res_beginring.IsSuccess,true)<<"BeginRingProcess failed: "<<res_beginring.ErrorMessage;
		int numFrames=RingHeight/FrameHeight; //number of frames in this ring
		float dI=255.0f/TotalRings; //intensity increment per ring
		float dIframe=dI/numFrames; //intensity increment per frame
		for(int kf=0;kf<numFrames;kf++)//index of frame in ring kr
		{ 
			int I=static_cast<int>(kr*dI+kf*dIframe); //intensity for this frame
			std::vector<uint8_t> frameData(FrameWidth * FrameHeight, static_cast<uint8_t>(I));
			auto res_addblock=AddRingProcessFrame(channel,kr,frameData.data(),FrameWidth,FrameHeight,FrameHeight,0,1);
			ASSERT_EQ(res_addblock.IsSuccess,true)<<"AddRingProcessFrame failed: "<<res_addblock.ErrorMessage;
		}

		auto res_endring=EndRingProcess(channel,kr,nullptr,nullptr);
		std::println("EndRingProcess result for ring {}: IsSuccess={}, ErrorMessage=\"{}\"",kr,res_endring.IsSuccess,res_endring.ErrorMessage);
		ASSERT_EQ(res_endring.IsSuccess,true)<<"EndRingProcess failed: "<<res_endring.ErrorMessage;
	}

	auto res_endchannel=EndChannelProcess(channel,nullptr); 
	std::println("EndChannelProcess result: IsSuccess={}, ErrorMessage=\"{}\"",res_endchannel.IsSuccess,res_endchannel.ErrorMessage);
	ASSERT_EQ(res_endchannel.IsSuccess,true)<<"EndChannelProcess failed: "<<res_endchannel.ErrorMessage;
}
TEST_F(AlgoTest,ChannelProcess_Multiple)
{

}
int main(int argc, char **argv) 
{
    std::filesystem::create_directories(path_output); // Create test output directory if it doesn't exist
	//std::string resdir="C:/astri2/lasv/lasvrepo/res"; //default resource directory
	//std::println("Using resource directory: {}",resdir);
	//std::filesystem::current_path(resdir); // Set working directory
	std::filesystem::current_path(path_output);
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
