#include <gtest/gtest.h>
#include "algorithm.h"
#include <opencv2/opencv.hpp>
#include "io.h"
import std;
std::string path_output="test_output"; 
//std::string path_output="C:/Users/cyber/test_output"; 
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
	//fs::path dir = path_input+"stage_center_cal/main"; 
	fs::path dir = path_input+"stage_center_cal/20260814/181739"; 
	//fs::path dir = path_input+"stage_center_cal/20260814/181856"; 
	ASSERT_EQ(fs::exists(dir),true)<<"Directory not found: "<<dir.string();
	ASSERT_EQ(fs::is_directory(dir),true)<<"Not a directory: "<<dir.string();
	std::vector<fs::path> files;
	for (auto const& entry : fs::directory_iterator(dir)) 
	{
		if (!entry.is_regular_file()) continue;
		auto ext = entry.path().extension().string();
		std::println("Found file: {} with extension: {}",entry.path().string(),ext);
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
TEST_F(AlgoTest,ChannelProcess_Single_Synthetic)
{
	auto channel=DetectChannel::Narrow; 
	int FrameWidth=256,FrameHeight=274,TotalRings=10;
	//float PixelSize=10; // um/pixel
	std::string ringsettings = std::format("{{\"FrameWidth\":{}, \"FrameHeight\":{}, \"TotalRings\":{},  \"ImageSaveDirectory\":\"{}\"}}",
                                      FrameWidth, FrameHeight, TotalRings, std::filesystem::absolute(path_output).generic_string());
	std::println("Ring settings JSON: {}",ringsettings);
	auto res_beginchannel = BeginChannelProcess(channel, ringsettings.c_str(),
                                            "cluster_setting.json","classify_setting.json",
                                            "coord_cali_setting.json","DSizeCurve.json");
	ASSERT_EQ(res_beginchannel.IsSuccess,true)<<"BeginChannelProcess failed: "<<res_beginchannel.ErrorMessage;

	for(int kr=0;kr<TotalRings;kr++)//index of ring
	{
		float theta0=kr*20.0f,theta1=theta0+360.0f; //triggered start and end angles in degrees
		//int RingWidth=FrameWidth,RingHeight=27400;
		int RingWidth=FrameWidth*(std::rand()%3+1),RingHeight=27400;
		std::string coordCaliJson_ring = std::format("{{\"RingWidth\":{}, \"RingHeight\":{}, \"TriggerStart\":{{\"T\":{}}}, \"TriggerEnd\":{{\"T\":{}}}}}", RingWidth, RingHeight, theta0,theta1);
		auto res_beginring = BeginRingProcess(channel, kr, "process_setting.json",
                                      "haze_cali_setting.json", coordCaliJson_ring.c_str());
		ASSERT_EQ(res_beginring.IsSuccess,true)<<"BeginRingProcess failed: "<<res_beginring.ErrorMessage;
		int numFrames=RingHeight/FrameHeight; //number of frames in this ring
		float dI=255.0f/TotalRings; //intensity increment per ring
		float dIframe=dI/numFrames; //intensity increment per frame
		//FrameHeight=float(FrameHeight)*(std::rand()%10+95)/100.0f; //randomize frame height slightly
		for(int kf=0;kf<numFrames;kf++)//index of frame in ring kr
		{ 
			int I=static_cast<int>(kr*dI+kf*dIframe); //intensity for this frame
			std::vector<uint8_t> frameData(RingWidth * FrameHeight, static_cast<uint8_t>(I));
			auto res_addblock=AddRingProcessFrame(channel,kr,frameData.data(),RingWidth,FrameHeight,FrameHeight,0,1);
			ASSERT_EQ(res_addblock.IsSuccess,true)<<"AddRingProcessFrame failed: "<<res_addblock.ErrorMessage;
		}

		auto res_endring=EndRingProcess(channel,kr,nullptr,nullptr);
		std::println("EndRingProcess result for ring {}: IsSuccess={}, ErrorMessage=\"{}\"",kr,res_endring.IsSuccess,res_endring.ErrorMessage);
		ASSERT_EQ(res_endring.IsSuccess,true)<<"EndRingProcess failed: "<<res_endring.ErrorMessage;
	}

	DefectInfoListStruct desc={"desc",0,0};
	auto res_endchannel=EndChannelProcess(channel,&desc); 
	auto numParticles=desc.ParticleCount;
	auto dataSize=desc.DataSize;
	std::println("EndChannelProcess result: IsSuccess={}, ErrorMessage=\"{}\", ParticleCount={}, DataSize={}",res_endchannel.IsSuccess,res_endchannel.ErrorMessage,numParticles,dataSize);
	DefectInfoStruct* pData=nullptr;
	size_t loadedSize=0;
	load_from_mmap(desc.Name,(void**)&pData,&loadedSize);
	ASSERT_NE(pData,nullptr)<<"load_from_mmap failed for name: "<<desc.Name;
	std::println("Loaded {} bytes from shared memory \"{}\". Expected DataSize={}",loadedSize,desc.Name,dataSize);
	std::println("First  DefectInfoStruct :");
	std::println("DefectID={}, ChannelID={}, DefectType={}, BinCode={}",pData->DefectID,static_cast<int>(pData->ChannelID),static_cast<int>(pData->DefectType),pData->BinCode);
	std::println("CoordR={}, CoordT={}, CoordX={}, CoordY={}",pData->CoordR,pData->CoordT,pData->CoordX,pData->CoordY);
	ASSERT_EQ(res_endchannel.IsSuccess,true)<<"EndChannelProcess failed: "<<res_endchannel.ErrorMessage;
}
TEST_F(AlgoTest,ChannelProcess_Single_Offline)
{
	auto channel=DetectChannel::Narrow; 
	namespace fs = std::filesystem;
	
	struct RingInfo {
		int kr;
		std::vector<fs::path> images;
		int ringWidth = 0;
		int ringHeight = 0;
	};
	
	std::vector<RingInfo> rings;
	int FrameWidth = 0, FrameHeight = 0;
	int idx = 1;

	float theta0s[] = { 737.21f, 1817.21f, 2897.21f, 3977.21f, 5057.21f, 6137.21f, 7217.21f, 8297.21f, 9377.21f, 10457.21f, 11537.21f, 12617.21f, 13697.21f, 14777.21f, 15857.21f, 16937.21f };
	float theta1s[] = { 1099.482f, 2184.677f, 3270.662f, 4357.632f, 5419.944f, 6507.232f, 7596.114f, 8687.176f, 9740.922f, 10833.112f, 11930.04f, 13035.134f, 14064.387f, 15180.914f, 16361.184f, 17351.218f };
	int validRows[]={188809, 176343, 163877, 151411, 138945, 126480, 114014, 101548, 89082, 76616, 64150, 51684, 39219, 26753, 14287, 6233}; 
	// Pre-scan directories to determine parameters before calling APIs
	while (true)
	{
		fs::path ring_dir = fs::path(path_input) / "fullmap/0826" / std::format("Ring{}", idx);
		if (!fs::exists(ring_dir) || !fs::is_directory(ring_dir))
			break;

		RingInfo info;
		info.kr = idx - 1;
		for (auto const& entry : fs::directory_iterator(ring_dir)) 
		{
			if (!entry.is_regular_file()) continue;
			auto ext = entry.path().extension().string();
			std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c){ return std::tolower(c); });
			if (ext == ".png" || ext == ".jpg" || ext == ".jpeg" || ext == ".bmp" || ext == ".tif" || ext == ".tiff")
				info.images.push_back(entry.path());
		} 
		std::sort(info.images.begin(), info.images.end());

		if (!info.images.empty())
		{
			cv::Mat first_img = cv::imread(info.images[0].string(), cv::IMREAD_UNCHANGED);
			ASSERT_FALSE(first_img.empty()) << "Failed to load pre-scan image: " << info.images[0].string();

			if (FrameWidth == 0) FrameWidth = first_img.cols;
			if (FrameHeight == 0) FrameHeight = first_img.rows;
			info.ringWidth = first_img.cols;
			//info.ringHeight = first_img.rows * static_cast<int>(info.images.size());
			info.ringHeight = validRows[info.kr];
		}
		rings.push_back(info);
		idx++;
	}

	int TotalRings = static_cast<int>(rings.size());
	std::println("FrameWidth={}, FrameHeight={}, TotalRings={}", FrameWidth, FrameHeight, TotalRings);
	for (const auto& ring : rings)
	{
		std::println("Ring {}: RingWidth={}, RingHeight={}", ring.kr, ring.ringWidth, ring.ringHeight);
	}

	ASSERT_GT(TotalRings, 0) << "No ring directories found.";
	ASSERT_GT(FrameWidth, 0) << "Could not determine FrameWidth.";
	ASSERT_GT(FrameHeight, 0) << "Could not determine FrameHeight.";

	std::string ringsettings = std::format("{{\"FrameWidth\":{}, \"FrameHeight\":{}, \"TotalRings\":{},  \"ImageSaveDirectory\":\"{}\"}}",
                                      FrameWidth, FrameHeight, TotalRings, path_output);
	auto res_beginchannel = BeginChannelProcess(channel, ringsettings.c_str(),
                                            "cluster_setting.json","classify_setting.json",
                                            "coord_cali_setting.json","DSizeCurve.json");
	ASSERT_EQ(res_beginchannel.IsSuccess,true)<<"BeginChannelProcess failed: "<<res_beginchannel.ErrorMessage;

	for(const auto& ring : rings)//index of ring
	{
		float theta0 = theta0s[ring.kr], theta1 = theta1s[ring.kr]; //triggered start and end angles in degrees
		//float theta0 = -68.0f, theta1 = theta0 + 360.0f; //triggered start and end angles in degrees
		std::string coordCaliJson_ring = std::format("{{\"RingWidth\":{}, \"RingHeight\":{}, \"TriggerStart\":{{\"T\":{}}}, \"TriggerEnd\":{{\"T\":{}}}}}", 
                                             ring.ringWidth, ring.ringHeight, theta0, theta1);
		auto res_beginring = BeginRingProcess(channel, ring.kr, "process_setting.json",
                                      "haze_cali_setting.json", coordCaliJson_ring.c_str());
		ASSERT_EQ(res_beginring.IsSuccess,true)<<"BeginRingProcess failed: "<<res_beginring.ErrorMessage; 
		int rows_remaining = validRows[ring.kr];
		std::println("Ring {} started with ringWidth={}, ringHeight={}, validRows={}",ring.kr,ring.ringWidth,ring.ringHeight,rows_remaining);
		for (const auto& img_path : ring.images)
		{
			cv::Mat img = cv::imread(img_path.string(), cv::IMREAD_UNCHANGED);
			ASSERT_FALSE(img.empty()) << "Failed to load image: " << img_path.string(); 
			int frame_height=std::min(img.rows,rows_remaining);
			//std::println("Adding frame from {} ({}x{}) to ring {} with frame_height={} and rows_remaining={}",img_path.string(),img.cols,img.rows,ring.kr,frame_height,rows_remaining);
			auto res_addblock = AddRingProcessFrame(channel, ring.kr, img.data, img.cols, img.rows, frame_height, 0, 1);
			//auto res_addblock = AddRingProcessFrame(channel, ring.kr, img.data, img.cols, img.rows, img.rows, 0, 1);
			rows_remaining-=frame_height;
			//if(rows_remaining<=0) break;
			ASSERT_EQ(res_addblock.IsSuccess,true)<<"AddRingProcessFrame failed: "<<res_addblock.ErrorMessage;
		}

		auto res_endring = EndRingProcess(channel, ring.kr, nullptr, nullptr);
		ASSERT_EQ(res_endring.IsSuccess,true)<<"EndRingProcess failed: "<<res_endring.ErrorMessage;
		std::println("Ring {} ended. {} rows in total; {} rows valid.",ring.kr,ring.ringHeight,validRows[ring.kr]);
	}
    
	auto res_endchannel=EndChannelProcess(channel,nullptr); 
	std::println("EndChannelProcess result: IsSuccess={}, ErrorMessage=\"{}\"",res_endchannel.IsSuccess,res_endchannel.ErrorMessage);
	ASSERT_EQ(res_endchannel.IsSuccess,true)<<"EndChannelProcess failed: "<<res_endchannel.ErrorMessage;
}
TEST_F(AlgoTest,ChannelProcess_Multiple)
{

}
TEST_F(AlgoTest,ErrorLogs)
{
	Initialize();
	SetLogMessageCallBack([](LogType level, const char* source, const char* message){
		std::println("CustomizedLog [{}] {}: {}", static_cast<int>(level), source, message);
	});
	Initialize();
}
int main(int argc, char **argv) 
{
    std::filesystem::create_directories(path_output); // Create test output directory if it doesn't exist
	//std::string resdir="C:/astri2/lasv/lasvrepo/res"; //default resource directory
	//std::println("Using resource directory: {}",resdir);
	//std::filesystem::current_path(resdir); // Set working directory
	//std::filesystem::current_path(path_output);
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
