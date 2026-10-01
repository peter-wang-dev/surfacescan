#include <gtest/gtest.h>
#include "algorithm.h"
#include <opencv2/opencv.hpp>
#include "io.h"
import std;
std::string path_output="test_output"; 
//std::string path_output="C:/Users/cyber/test_output"; 
std::string path_input="C:/astri2/cnsvision/testinput"; 
std::string path_userinput;
double detection_threshold=15.0;
//const std::string path_input="./test_output/"; 
//cv::Mat dehaze(cv::Mat image);
//TEST(core,dehaze)
//{
//	auto datadir=path_input+"cal/200nm/0904 200nm Wide1 800mW/20260904 172742";
//	int kr=3;//index of ring 
//	auto ringdir=std::format("{}/Ring {}",datadir,kr);
//	int ii=0;//index of image
//	auto imgpath=std::format("{}/{}.bmp",ringdir,kr,ii);
//
//	cv::Mat img=cv::imread(path_input+"dehaze_test.png",CV_8UC1);
//	ASSERT_FALSE(img.empty())<<"Failed to load image: "<<path_input+"dehaze_test.png";
//	cv::Mat dehazed=dehaze(img);
//	ASSERT_FALSE(dehazed.empty())<<"Dehazing failed.";
//	cv::imwrite(path_output+"dehaze_result.png",dehazed*255.0f);
//}
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
	std::println("Ring settings JSON string: {}",ringsettings);
	std::string DSizeCurveStr=R"({"CurvePoints": [{"Intensity": 0.0, "DSize": 0.0}, {"Intensity": 100.0, "DSize": 300.0}, {"Intensity": 200.0, "DSize": 1000.0}]})";
	//DSizeCurveStr+='\n';
	std::println("DSizeCurve JSON string: {}",DSizeCurveStr);
	auto res_beginchannel = BeginChannelProcess(channel, ringsettings.c_str(),
                                            "cluster_setting.json","classify_setting.json",
                                            "coord_cali_setting.json",DSizeCurveStr.c_str());
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
			//frameData[0]=255;
			auto res_addblock=AddRingProcessFrame(channel,kr,frameData.data(),RingWidth,FrameHeight,FrameHeight,0,1);
			ASSERT_EQ(res_addblock.IsSuccess,true)<<"AddRingProcessFrame failed: "<<res_addblock.ErrorMessage;
		}

		auto res_endring=EndRingProcess(channel,kr,nullptr,nullptr);
		//std::println("EndRingProcess result for ring {}: IsSuccess={}, ErrorMessage=\"{}\"",kr,res_endring.IsSuccess,res_endring.ErrorMessage);
		ASSERT_EQ(res_endring.IsSuccess,true)<<"EndRingProcess failed: "<<res_endring.ErrorMessage;
	}

	DefectInfoListStruct desc={"desc",0,0};
	auto res_endchannel=EndChannelProcess(channel,&desc); 
	ASSERT_EQ(res_endchannel.IsSuccess,true)<<"EndChannelProcess failed: "<<res_endchannel.ErrorMessage;
	auto numParticles=desc.ParticleCount;
	auto dataSize=desc.DataSize;
	std::println("EndChannelProcess result: IsSuccess={}, ErrorMessage=\"{}\", ParticleCount={}, DataSize={}",res_endchannel.IsSuccess,res_endchannel.ErrorMessage,numParticles,dataSize);
#ifdef _WIN64
	DefectInfoStruct* pData=nullptr;
	size_t loadedSize=0;
	load_from_mmap(desc.Name,(void**)&pData,&loadedSize);
	ASSERT_NE(pData,nullptr)<<"load_from_mmap failed for name: "<<desc.Name;
	std::println("Loaded {} bytes from shared memory \"{}\". Expected DataSize={}",loadedSize,desc.Name,dataSize);
	if(numParticles>0)
	{
		std::println("First  DefectInfoStruct :");
		std::println("DefectID={}, ChannelID={}, DefectType={}, BinCode={}",pData->DefectID,static_cast<int>(pData->ChannelID),static_cast<int>(pData->Type),pData->BinCode);
		std::println("CoordR={}, CoordT={}, CoordX={}, CoordY={}",pData->CoordR,pData->CoordT,pData->CoordX,pData->CoordY); 
	}
#endif
}
std::map<std::string, std::vector<float>> readRingParaCSV(const std::string& filename) {
    std::map<std::string, std::vector<float>> dataMap;
    std::ifstream file(filename);

    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + filename);
    }

    std::string line;
    std::vector<std::string> headers;

    // 1. Parse the header line
    if (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string header;
        
        while (std::getline(ss, header, ',')) {
            // Strip any trailing carriage returns (e.g., from Windows CRLF files)
            if (!header.empty() && header.back() == '\r') {
                header.pop_back();
            }
            headers.push_back(header);
            dataMap[header] = std::vector<float>(); 
        }
    }
	//std::println("Parsed headers from CSV");
	//for(auto const& h:headers)
	//	std::println("Header: {}", h);

    // 2. Parse the data lines
    while (std::getline(file, line)) {
        // Skip completely empty lines
        if (line.empty() || line.find_first_not_of(" \r\n\t") == std::string::npos) {
            continue; 
        }

        std::stringstream ss(line);
        std::string valueStr;
        size_t colIndex = 0;

        while (std::getline(ss, valueStr, ',')) {
            if (colIndex < headers.size()) {
                try {
                    // Convert the string to float and add to the specific column vector
                    float value = std::stof(valueStr);
                    dataMap[headers[colIndex]].push_back(value);
                } catch (const std::invalid_argument&) {
                    // Fallback for empty strings or invalid float data
                    dataMap[headers[colIndex]].push_back(0.0f);
                }
                colIndex++;
            }
        }
    }
	//std::println("Parsed {} rows from CSV",dataMap.begin()->second.size());

    return dataMap;
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
		float radius_mm=1;
	};
	
	std::vector<RingInfo> rings;
	int kr = 0;

	std::string path_channel=path_input+"/cal/200nm/0905/Narrow";
	if(path_userinput!="")
		path_channel=path_userinput;
	auto ringParamMap = readRingParaCSV(path_channel+"/RingsPara.csv");
	int FrameWidth=static_cast<int>(ringParamMap["FrameWidth"][0]+0.001f);
	int FrameHeight=static_cast<int>(ringParamMap["FrameHeight"][0]+0.001f);
	const std::vector<float> theta0s=ringParamMap["StartDegree"]; 
	const std::vector<float> theta1s=ringParamMap["EndDegree"];
	const std::vector<float> radii=ringParamMap["Radius"]; 
	const std::vector<float> validRows_float=ringParamMap["ValidLines"];
	const float pixelsize=ringParamMap["PixelSizeUm"][0];
	std::vector<int> validRows(validRows_float.size());
	std::transform(validRows_float.begin(), validRows_float.end(), validRows.begin(), [](float f){ return static_cast<int>(f+0.001f); });
	std::println("Read {} rings from CSV. ValidRows: {}",validRows.size(),validRows[0]);

	while(true)
	{ 
		fs::path ring_dir=fs::path(path_channel)/std::format("Ring {}",kr);
		if(!fs::exists(ring_dir)||!fs::is_directory(ring_dir))
			break;
		std::println("Found ring directory: {}",ring_dir.string());

		RingInfo info;
		info.kr=kr;
		info.radius_mm=radii[kr];
		auto numImages=std::distance(fs::directory_iterator(ring_dir),fs::directory_iterator{});
		info.images.reserve(static_cast<size_t>(numImages));
		for(int i=0; i<static_cast<int>(numImages); ++i)
			info.images.emplace_back(ring_dir/std::format("{}.tiff",i));

		ASSERT_NE(info.images.size(),0u)<<"No images found in ring directory: "<<ring_dir.string();
		cv::Mat first_img=cv::imread(info.images[0].string(),CV_8UC1);
		ASSERT_FALSE(first_img.empty())<<"Failed to load image: "<<info.images[0].string();
		info.ringWidth=first_img.cols;
		info.ringHeight=validRows[info.kr]; // Use the validRows from CSV for ringHeight
		rings.push_back(info);
		kr++;
	}

	int TotalRings = static_cast<int>(rings.size());
	std::println("FrameWidth={}, FrameHeight={}, TotalRings={}", FrameWidth, FrameHeight, TotalRings);
	for (const auto& ring : rings)
		std::println("Ring {}: RingWidth={}, RingHeight={}, StartDegree={}, EndDegree={}, Radius={}", ring.kr, ring.ringWidth, ring.ringHeight, theta0s[ring.kr], theta1s[ring.kr],ring.radius_mm);

	ASSERT_GT(TotalRings, 0) << "No ring directories found.";
	ASSERT_GT(FrameWidth, 0) << "Could not determine FrameWidth.";
	ASSERT_GT(FrameHeight, 0) << "Could not determine FrameHeight.";

	std::string ringsettings = std::format(R"({{"FrameWidth":{}, "FrameHeight":{}, "TotalRings":{},  "ImageSaveDirectory":"{}", "DebugOutput":true}})",
                                      FrameWidth, FrameHeight, TotalRings, path_output);
	//std::string DSizeCurveStr=std::format(R"({{"CurvePoints": [{"Intensity": 0.0, "DSize": 0.0}, {"Intensity": {}, "DSize": 200.0}, {"Intensity": 255.0, "DSize": 1000.0}]}})", detection_threshold); 
	std::string DSizeCurveStr=std::format(
		R"({{"CurvePoints": [{{"Intensity": 0.0, "DSize": 0.0}}, {{"Intensity": {}, "DSize": 0.5}}, {{"Intensity": 1000000.0, "DSize": 10.0}}]}})",
		//R"({{"CurvePoints": [{{"Intensity": 0.0, "DSize": 0.0}}, {{"Intensity": 140, "DSize": 0.1}},{{"Intensity": {}, "DSize": 0.2}}, {{"Intensity": 420000, "DSize": 0.5}},{{"Intensity": 1000000.0, "DSize": 10.0}}]}})",
		detection_threshold);
	std::string classifySettingStr=std::format(R"({{"PixelSize": {}}})", pixelsize);
	auto res_beginchannel = BeginChannelProcess(channel, ringsettings.c_str(),
                                            "cluster_setting.json",classifySettingStr.c_str(),
                                            "coord_cali_setting.json",DSizeCurveStr.c_str());
	ASSERT_EQ(res_beginchannel.IsSuccess,true)<<"BeginChannelProcess failed: "<<res_beginchannel.ErrorMessage;

	for(const auto& ring : rings)//index of ring
	{
		float theta0 = theta0s[ring.kr], theta1 = theta1s[ring.kr]; //triggered start and end angles in degrees
		//float theta0 = -68.0f, theta1 = theta0 + 360.0f; //triggered start and end angles in degrees
		std::string coordCaliSetting_jsonstr = std::format("{{\"RingWidth\":{}, \"RingHeight\":{}, \"TriggerStart\":{{\"T\":{},\"R\":{}}}, \"TriggerEnd\":{{\"T\":{}}}}}", 
                                             ring.ringWidth, ring.ringHeight, theta0,ring.radius_mm, theta1);
		//std::println("Ring {} coordCaliSetting JSON string: {}",ring.kr,coordCaliSetting_jsonstr);
		std::string processSetting_jsonstr=std::format("{{\"RingWidth\":{}, \"RingHeight\":{}, \"FrameWidth\":{}, \"FrameHeight\":{}}}",
			ring.ringWidth,ring.ringHeight,FrameWidth,FrameHeight);
		auto res_beginring = BeginRingProcess(channel, ring.kr, processSetting_jsonstr.c_str(),
                                      "haze_cali_setting.json", coordCaliSetting_jsonstr.c_str());
		ASSERT_EQ(res_beginring.IsSuccess,true)<<"BeginRingProcess failed: "<<res_beginring.ErrorMessage; 
		int rows_remaining = validRows[ring.kr];
		std::println("Ring {} started with ringWidth={}, ringHeight={}, validRows={}",ring.kr,ring.ringWidth,ring.ringHeight,rows_remaining);
		for (const auto& img_path : ring.images)
		{
			cv::Mat imgin = cv::imread(img_path.string(), cv::IMREAD_UNCHANGED);
			cv::Mat img;
			if(imgin.type()==CV_8UC1)
				imgin.convertTo(img,CV_16UC1,16.0); // we are simulating 12-bit data in a 16-bit container, so the lowest 4 bits and the highest 4 bits are zero, and the middle 8 bits contain the actual data
			else if(imgin.type()==CV_16UC1)
				img=imgin;
			else
				throw std::runtime_error(std::format("Input image {} is neither CV_8U nor CV_16U",img_path.string()));
			//cv::Mat img = cv::imread(img_path.string(), CV_16UC1);
			//cv::Mat img8 = cv::imread(img_path.string(), CV_8UC1);
			//cv::Mat img;//convert to 16-bit unsigned single channel, but the greatest 4 bit are zero, simulating 12-bit data in a 16-bit container
			ASSERT_FALSE(img.empty()) << "Failed to load image: " << img_path.string(); 
			int frame_height=std::min(img.rows,rows_remaining);
			//std::println("Adding frame from {} ({}x{}) to ring {} with frame_height={} and rows_remaining={}",img_path.string(),img.cols,img.rows,ring.kr,frame_height,rows_remaining);
			auto res_addblock = AddRingProcessFrame(channel, ring.kr, img.data, img.cols, img.rows, frame_height, 0, 1);
			rows_remaining-=frame_height;
			ASSERT_EQ(res_addblock.IsSuccess,true)<<"AddRingProcessFrame failed: "<<res_addblock.ErrorMessage;
		}

		auto res_endring = EndRingProcess(channel, ring.kr, nullptr, nullptr);
		ASSERT_EQ(res_endring.IsSuccess,true)<<"EndRingProcess failed: "<<res_endring.ErrorMessage;
		std::println("Ring {} ended. {} rows in total; {} rows valid.",ring.kr,ring.ringHeight,validRows[ring.kr]);
	}
    
	auto res_endchannel=EndChannelProcess(channel,nullptr); 
	std::println("EndChannelProcess result: IsSuccess={}, ErrorMessage=\"{}\"",res_endchannel.IsSuccess,res_endchannel.ErrorMessage);
	ASSERT_EQ(res_endchannel.IsSuccess,true)<<"EndChannelProcess failed: "<<res_endchannel.ErrorMessage;
	std::println("Channel test completed using data from directory {}.",path_channel);
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
	if(argc>1) 
		path_userinput=argv[1];
	if(argc>2) 
		detection_threshold=std::stod(argv[2]);
    return RUN_ALL_TESTS();
}
