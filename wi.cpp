#include <cxxopts.hpp>
#include "algocore.h"
#include "algorithm.h"
#include "io.h"
#include <opencv2/opencv.hpp>
import std;
const std::string pathout_default="test_output"; 
const std::string pathin_default="C:/astri2/cnsvision/testinput"; 
const float thr_snr_default=1.8f;//signal-to-noise ratio threshold for integration calculation
const float thr_intg_default=1000.0f;//integration threshold for particle detection
int main(int argc,char* argv[])
{
	cxxopts::Options options("wi","Command-line interface for wafer inspection");
	options.add_options()
		("i,pathin","Input directory",cxxopts::value<std::string>()->default_value(pathin_default))
		("o,pathout","Output directory",cxxopts::value<std::string>()->default_value(pathout_default))
		("thr_snr","Signal-to-noise ratio threshold",cxxopts::value<float>()->default_value(std::to_string(thr_snr_default)))
		("thr_intg","Integration threshold",cxxopts::value<float>()->default_value(std::to_string(thr_intg_default)))
		("h,help","Print usage");
	auto arguments=options.parse(argc,argv);
	if(arguments.count("help")||!arguments.count("pathin")||!arguments.count("pathout")) {
		std::cout<<options.help()<<std::endl;
		return 0;
	}
	try // wafer inspection
	{ 
		auto channel=DetectChannel::Narrow; 
		namespace fs=std::filesystem;

		struct RingInfo {
			int kr;
			std::vector<fs::path> images;
			int ringWidth=0;
			int ringHeight=0;
			float radius_mm=1.0f;
		};

		std::vector<RingInfo> rings;
		int kr=0;

		std::string pathin=arguments["pathin"].as<std::string>();
		auto ringParamMap=readRingParaCSV(pathin+"/RingsPara.csv");
		int FrameWidth=static_cast<int>(ringParamMap["FrameWidth"][0]+0.001f);
		int FrameHeight=static_cast<int>(ringParamMap["FrameHeight"][0]+0.001f);
		const std::vector<float> theta0s=ringParamMap["StartDegree"];
		const std::vector<float> theta1s=ringParamMap["EndDegree"];
		const std::vector<float> radii=ringParamMap["Radius"];
		const std::vector<float> validRows_float=ringParamMap["ValidLines"];
		const float pixelsize=ringParamMap["PixelSizeUm"][0];
		std::vector<int> validRows(validRows_float.size());
		std::transform(validRows_float.begin(),validRows_float.end(),validRows.begin(),[](float f){ return static_cast<int>(f+0.001f); });
		std::println("Read {} rings from CSV. ValidRows: {}",validRows.size(),validRows[0]);

		while(true)
		{
			fs::path ring_dir=fs::path(pathin)/std::format("Ring {}",kr);
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

			if (info.images.empty())
				throw std::runtime_error(std::format("No images found in ring directory: {}", ring_dir.string()));

			cv::Mat first_img=cv::imread(info.images[0].string(),CV_8UC1);
			if (first_img.empty())
				throw std::runtime_error(std::format("Failed to load image: {}", info.images[0].string()));

			info.ringWidth=first_img.cols;
			info.ringHeight=validRows[info.kr]; // Use the validRows from CSV for ringHeight
			rings.push_back(info);
			kr++;
		}

		int TotalRings=static_cast<int>(rings.size());
		std::println("FrameWidth={}, FrameHeight={}, TotalRings={}",FrameWidth,FrameHeight,TotalRings);
		for(const auto& ring:rings)
			std::println("Ring {}: RingWidth={}, RingHeight={}, StartDegree={}, EndDegree={}, Radius={}",ring.kr,ring.ringWidth,ring.ringHeight,theta0s[ring.kr],theta1s[ring.kr],ring.radius_mm);

		if (TotalRings <= 0)
			throw std::runtime_error("No ring directories found."); 
		if (FrameWidth <= 0)
			throw std::runtime_error("Could not determine FrameWidth."); 
		if (FrameHeight <= 0)
			throw std::runtime_error("Could not determine FrameHeight.");

		std::string ringsettings=std::format(R"({{"FrameWidth":{}, "FrameHeight":{}, "TotalRings":{},  "ImageSaveDirectory":"{}", "DebugOutput":true}})",
			FrameWidth,FrameHeight,TotalRings,pathout_default);
		//std::string DSizeCurveStr=std::format(R"({{"CurvePoints": [{"Intensity": 0.0, "DSize": 0.0}, {"Intensity": {}, "DSize": 200.0}, {"Intensity": 255.0, "DSize": 1000.0}]}})", detection_threshold); 
		float thr_intg=arguments["thr_intg"].as<float>();
		std::string DSizeCurveStr=std::format(
			R"({{"CurvePoints": [{{"Intensity": 0.0, "DSize": 0.0}}, {{"Intensity": {}, "DSize": 0.5}}, {{"Intensity": 1000000.0, "DSize": 10.0}}]}})",
			//R"({{"CurvePoints": [{{"Intensity": 0.0, "DSize": 0.0}}, {{"Intensity": 140, "DSize": 0.1}},{{"Intensity": {}, "DSize": 0.2}}, {{"Intensity": 420000, "DSize": 0.5}},{{"Intensity": 1000000.0, "DSize": 10.0}}]}})",
			thr_intg);
		float thr_snr=arguments["thr_snr"].as<float>();
		std::string classifySettingStr=std::format(R"({{"PixelSize": {},"SNRThreshold":{}}})",pixelsize,thr_snr);
		auto res_beginchannel=BeginChannelProcess(channel,ringsettings.c_str(),
			"cluster_setting.json",classifySettingStr.c_str(),
			"coord_cali_setting.json",DSizeCurveStr.c_str());
		if (!res_beginchannel.IsSuccess)
			throw std::runtime_error(std::format("BeginChannelProcess failed: {}", res_beginchannel.ErrorMessage));

		for(const auto& ring:rings)//index of ring
		{
			float theta0=theta0s[ring.kr],theta1=theta1s[ring.kr]; //triggered start and end angles in degrees
			//float theta0 = -68.0f, theta1 = theta0 + 360.0f; //triggered start and end angles in degrees
			std::string coordCaliSetting_jsonstr=std::format("{{\"RingWidth\":{}, \"RingHeight\":{}, \"TriggerStart\":{{\"T\":{},\"R\":{}}}, \"TriggerEnd\":{{\"T\":{}}}}}",
				ring.ringWidth,ring.ringHeight,theta0,ring.radius_mm,theta1);
			//std::println("Ring {} coordCaliSetting JSON string: {}",ring.kr,coordCaliSetting_jsonstr);
			std::string processSetting_jsonstr=std::format("{{\"RingWidth\":{}, \"RingHeight\":{}, \"FrameWidth\":{}, \"FrameHeight\":{}}}",
				ring.ringWidth,ring.ringHeight,FrameWidth,FrameHeight);
			auto res_beginring=BeginRingProcess(channel,ring.kr,processSetting_jsonstr.c_str(),
				"haze_cali_setting.json",coordCaliSetting_jsonstr.c_str());
			if (!res_beginring.IsSuccess)
				throw std::runtime_error(std::format("BeginRingProcess failed: {}", res_beginring.ErrorMessage));
			int rows_remaining=validRows[ring.kr];
			std::println("Ring {} started with ringWidth={}, ringHeight={}, validRows={}",ring.kr,ring.ringWidth,ring.ringHeight,rows_remaining);
			for(const auto& img_path:ring.images)
			{
				cv::Mat imgin=cv::imread(img_path.string(),cv::IMREAD_UNCHANGED);
				cv::Mat img;
				if(imgin.type()==CV_8UC1)
					imgin.convertTo(img,CV_16UC1,16.0); // we are simulating 12-bit data in a 16-bit container, so the lowest 4 bits and the highest 4 bits are zero, and the middle 8 bits contain the actual data
				else if(imgin.type()==CV_16UC1)
					img=imgin;
				else
					throw std::runtime_error(std::format("Input image {} is neither CV_8U nor CV_16U",img_path.string()));
				//if(ring.kr>=5) // invalidate some rings for testing
				//	img=1;
				//if(ring.kr==0) //
				//	img=1;
				//cv::Mat img = cv::imread(img_path.string(), CV_16UC1);
				//cv::Mat img8 = cv::imread(img_path.string(), CV_8UC1);
				//cv::Mat img;//convert to 16-bit unsigned single channel, but the greatest 4 bit are zero, simulating 12-bit data in a 16-bit container
				if(img.empty())
					throw std::runtime_error(std::format("Failed to load image: {}", img_path.string()));
				int frame_height=std::min(img.rows,rows_remaining);
				//std::println("Adding frame from {} ({}x{}) to ring {} with frame_height={} and rows_remaining={}",img_path.string(),img.cols,img.rows,ring.kr,frame_height,rows_remaining);
				auto res_addblock=AddRingProcessFrame(channel,ring.kr,img.data,img.cols,img.rows,frame_height,0,1);
				rows_remaining-=frame_height;
				if(!res_addblock.IsSuccess)
					throw std::runtime_error(std::format("AddRingProcessFrame failed: {}", res_addblock.ErrorMessage));
			}

			auto res_endring=EndRingProcess(channel,ring.kr,nullptr,nullptr);
			if(!res_endring.IsSuccess)
				throw std::runtime_error(std::format("EndRingProcess failed: {}", res_endring.ErrorMessage));
			std::println("Ring {} ended. {} rows in total; {} rows valid.",ring.kr,ring.ringHeight,validRows[ring.kr]);
		}

		auto res_endchannel=EndChannelProcess(channel,nullptr);
		std::println("EndChannelProcess result: IsSuccess={}, ErrorMessage=\"{}\"",res_endchannel.IsSuccess,res_endchannel.ErrorMessage);
		if(!res_endchannel.IsSuccess)
			throw std::runtime_error(std::format("EndChannelProcess failed: {}", res_endchannel.ErrorMessage));
	}
	catch(const std::exception& e) {
		std::cerr<<"Error during flattening: "<<e.what()<<std::endl;
		return 1;
	}

	//print the parsed options using std::println
	std::println("Parsed options: pathin={}, pathout={}, thr_snr={}, thr_intg={}",
		arguments["pathin"].as<std::string>(),
		arguments["pathout"].as<std::string>(),
		arguments["thr_snr"].as<double>(),
		arguments["thr_intg"].as<double>());
	return 0;
}
