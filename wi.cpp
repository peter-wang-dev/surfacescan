#include <cxxopts.hpp>
#include "algocore.h"
#include "algorithm.h"
#include "io.h"
#include <opencv2/opencv.hpp>
import std;
const std::string pathout_default="test_output"; 
const std::string pathin_default="C:/astri2/cnsvision/testinput"; 
const float thr_snr_default=1.8f;//signal-to-noise ratio threshold for integration calculation

std::string readDSizeCurve(const std::string& path)
{
	std::ifstream file(path);
	if(!file)
		throw std::runtime_error(std::format("Could not open DSizecurve file: {}",path));
	auto hasColumns=[](std::string_view text,std::string_view first,std::string_view second)
	{
		const auto separator=text.find_first_of(" \t");
		const auto secondColumn=text.find_first_not_of(" \t",separator);
		return separator!=std::string_view::npos&&secondColumn!=std::string_view::npos
			&&text.substr(0,separator)==first&&text.substr(secondColumn)==second;
	};
	std::string line;
	size_t lineNumber=0;
	auto readDataLine=[&]()
	{
		while(std::getline(file,line))
		{
			++lineNumber;
			const auto comment=std::min(line.find('#'),line.find("//"));
			if(comment!=std::string::npos)
				line.erase(comment);
			const auto end=line.find_last_not_of(" \t\r");
			if(end==std::string::npos)
				continue;
			line.erase(end+1);
			return true;
		}
		return false;
	};
	if(!readDataLine()||!hasColumns(line,"Intensity","Size(um)"))
		throw std::runtime_error(std::format("Invalid DSizecurve header in '{}': expected Intensity and Size(um) separated by spaces or tabs.",path));
	if(!readDataLine()||!hasColumns(line,"0","0"))
		throw std::runtime_error(std::format("Invalid DSizecurve origin in '{}': the first point must contain 0 and 0 separated by spaces or tabs.",path));
	std::string curve=R"({"CurvePoints": [{"Intensity": 0.0, "DSize": 0.0})";
	double previousIntensity=0.0,previousSize=0.0;
	bool hasPoint=false;
	while(readDataLine())
	{
		std::istringstream row(line);
		double intensity,size;
		char separator;
		if(!(row>>intensity)||!row.get(separator)||(separator!='\t'&&separator!=' ')
			||!(row>>size)||!(row>>std::ws).eof()
			||!std::isfinite(intensity)||!std::isfinite(size))
			throw std::runtime_error(std::format("Invalid DSizecurve point in '{}' at line {}: expected two finite numbers separated by spaces or tabs.",path,lineNumber));
		if(intensity<=previousIntensity)
			throw std::runtime_error(std::format("Invalid DSizecurve point in '{}' at line {}: Intensity values must be strictly increasing.",path,lineNumber));
		if(size<=previousSize)
			throw std::runtime_error(std::format("Invalid DSizecurve point in '{}' at line {}: Size(um) values must be strictly increasing.",path,lineNumber));
		previousIntensity=intensity;
		previousSize=size;
		curve+=std::format(R"(, {{"Intensity": {}, "DSize": {}}})",intensity,size);
		hasPoint=true;
	}
	if(file.bad())
		throw std::runtime_error(std::format("Could not read DSizecurve file: {}",path));
	if(!hasPoint)
		throw std::runtime_error(std::format("DSizecurve file '{}' must contain at least one point after the origin (0,0).",path));
	curve+="]}";
	return curve;
}

int main(int argc,char* argv[])
{
	// Parse command-line arguments
	cxxopts::Options options("wi","Command-line interface for wafer inspection");
	options.add_options()
		("i,pathin","Input directory",cxxopts::value<std::string>()->default_value(pathin_default))
		("o,pathout","Output directory",cxxopts::value<std::string>()->default_value(pathout_default))
		("material","Wafer material: \"silicon\", \"glass\", \"tialsi\"",cxxopts::value<std::string>()->default_value("silicon"))
		("thr_snr","Signal-to-noise ratio threshold",cxxopts::value<float>()->default_value(std::to_string(thr_snr_default)))
		("thr_intg","Integration threshold (mutually exclusive with --dsc)",cxxopts::value<std::string>())
		("dsc","Path to a space- or tab-separated DSizecurve file (# and // comments; mutually exclusive with --thr_intg)",cxxopts::value<std::string>())
		("minpx","Minimal number of pixels for a region to be considered as a defect",cxxopts::value<int>()->default_value(std::to_string(1)))
		("h,help","Print usage");
	auto arguments=options.parse(argc,argv); 
	std::string material=arguments["material"].as<std::string>();
	if(material!="silicon"&&material!="glass"&&material!="tialsi") 
	{
		std::cerr<<"Invalid wafer material: "<<material<<". Must be \"silicon\" or \"glass\" or \"tialsi\"."<<std::endl;
		return 1;
	} 
	if(arguments.count("dsc")&&arguments.count("thr_intg"))
	{
		std::cerr<<"Options --dsc and --thr_intg cannot be used together. Provide exactly one."<<std::endl;
		return 1;
	}
	if(!arguments.count("help")&&!arguments.count("dsc")&&!arguments.count("thr_intg"))
	{
		std::cerr<<"Either --dsc <path> or --thr_intg <value> must be provided."<<std::endl;
		return 1;
	}
	if(arguments.count("help")||!arguments.count("pathin")||!arguments.count("pathout")) {
		std::cout<<options.help()<<std::endl;
		return 0;
	}

	try // wafer inspection
	{
		auto timestamp_start=std::chrono::high_resolution_clock::now();
		std::string DSizeCurveStr;
		if(arguments.count("dsc"))
			DSizeCurveStr=readDSizeCurve(arguments["dsc"].as<std::string>());
		else
		{
			const auto& value=arguments["thr_intg"].as<std::string>();
			std::istringstream input(value);
			float thr_intg;
			if(!(input>>thr_intg)||!(input>>std::ws).eof()||!std::isfinite(thr_intg))
				throw std::runtime_error(std::format("Invalid --thr_intg value: {}",value));
			DSizeCurveStr=std::format(
				R"({{"CurvePoints": [{{"Intensity": 0.0, "DSize": 0.0}}, {{"Intensity": {}, "DSize": 0.5}}, {{"Intensity": 1000000.0, "DSize": 10.0}}]}})",
				thr_intg);
		}
		Initialize();
		auto channel=DetectChannel::Narrow;
		namespace fs=std::filesystem;

		struct RingInfo {
			int kr;
			std::vector<fs::path> images;
			int ringWidth=0;
			int ringHeight=0;
			float radius_mm=1.0f;
			int overlapPixels=0;
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
		const std::vector<float> overlapPixels=ringParamMap["OverlapPixels"];
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

			if(info.images.empty())
				throw std::runtime_error(std::format("No images found in ring directory: {}",ring_dir.string()));

			cv::Mat first_img=cv::imread(info.images[0].string(),CV_8UC1);
			if(first_img.empty())
				throw std::runtime_error(std::format("Failed to load image: {}",info.images[0].string()));

			info.ringWidth=first_img.cols;
			info.ringHeight=validRows[info.kr]; // Use the validRows from CSV for ringHeight
			info.overlapPixels=static_cast<int>(overlapPixels[info.kr]+0.001f);
			rings.push_back(info);
			kr++;
		}

		int TotalRings=static_cast<int>(rings.size());
		std::println("FrameWidth={}, FrameHeight={}, TotalRings={}",FrameWidth,FrameHeight,TotalRings);
		for(const auto& ring:rings)
			std::println("Ring {}: RingWidth={}, RingHeight={}, StartDegree={}, EndDegree={}, Radius={}",ring.kr,ring.ringWidth,ring.ringHeight,theta0s[ring.kr],theta1s[ring.kr],ring.radius_mm);

		if(TotalRings<=0)
			throw std::runtime_error("No ring directories found.");
		if(FrameWidth<=0)
			throw std::runtime_error("Could not determine FrameWidth.");
		if(FrameHeight<=0)
			throw std::runtime_error("Could not determine FrameHeight.");

		std::string pathout=arguments["pathout"].as<std::string>();
		if(pathout!=""&&!std::filesystem::exists(pathout))
			std::filesystem::create_directories(pathout);
		std::string channelSettings=std::format(R"({{"FrameWidth":{}, "FrameHeight":{}, "TotalRings":{},"ROE":{{"Rings":[{{"Rmin":0,"Rmax":0,"Tmin":0.0,"Tmax":360.0}}]}},  "ImageSaveDirectory":"{}", "DebugOutput":true, "Material":"{}"}})",
			FrameWidth,FrameHeight,TotalRings,pathout,material);
		float thr_snr=arguments["thr_snr"].as<float>();
		float minpx=arguments["minpx"].as<int>();
		std::string classifySettingStr=std::format(R"({{"PixelSize": {},"MinDefectPixels":{},"SNRThreshold":{}}})",pixelsize,minpx,thr_snr);
		auto res_beginchannel=BeginChannelProcess(channel,channelSettings.c_str(),
			"cluster_setting.json",classifySettingStr.c_str(),
			"coord_cali_setting.json",DSizeCurveStr.c_str());
		if(!res_beginchannel.IsSuccess)
			throw std::runtime_error(std::format("BeginChannelProcess failed: {}",res_beginchannel.ErrorMessage));

		for(const auto& ring:rings)//index of ring
		{
			float theta0=theta0s[ring.kr],theta1=theta1s[ring.kr]; //triggered start and end angles in degrees
			//float theta0 = -68.0f, theta1 = theta0 + 360.0f; //triggered start and end angles in degrees
			std::string coordCaliSetting_jsonstr=std::format("{{\"RingWidth\":{}, \"RingHeight\":{}, \"TriggerStart\":{{\"T\":{},\"R\":{}}}, \"TriggerEnd\":{{\"T\":{}}}, \"OverlapPixels\":{}}}",
				ring.ringWidth,ring.ringHeight,theta0,ring.radius_mm,theta1,ring.overlapPixels);
			//std::println("Ring {} coordCaliSetting JSON string: {}",ring.kr,coordCaliSetting_jsonstr);
			std::string processSetting_jsonstr=std::format("{{\"RingWidth\":{}, \"RingHeight\":{}, \"FrameWidth\":{}, \"FrameHeight\":{}}}",
				ring.ringWidth,ring.ringHeight,FrameWidth,FrameHeight);
			auto res_beginring=BeginRingProcess(channel,ring.kr,processSetting_jsonstr.c_str(),
				"haze_cali_setting.json",coordCaliSetting_jsonstr.c_str());
			if(!res_beginring.IsSuccess)
				throw std::runtime_error(std::format("BeginRingProcess failed: {}",res_beginring.ErrorMessage));
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
				//cv::Mat img = cv::imread(img_path.string(), CV_16UC1);
				//cv::Mat img8 = cv::imread(img_path.string(), CV_8UC1);
				//cv::Mat img;//convert to 16-bit unsigned single channel, but the greatest 4 bit are zero, simulating 12-bit data in a 16-bit container
				if(img.empty())
					throw std::runtime_error(std::format("Failed to load image: {}",img_path.string()));
				int frame_height=std::min(img.rows,rows_remaining);
				//std::println("Adding frame from {} ({}x{}) to ring {} with frame_height={} and rows_remaining={}",img_path.string(),img.cols,img.rows,ring.kr,frame_height,rows_remaining);
				auto res_addblock=AddRingProcessFrame(channel,ring.kr,img.data,img.cols,img.rows,frame_height,0,1);
				rows_remaining-=frame_height;
				if(!res_addblock.IsSuccess)
					throw std::runtime_error(std::format("AddRingProcessFrame failed: {}",res_addblock.ErrorMessage));
			}

			auto res_endring=EndRingProcess(channel,ring.kr,nullptr,nullptr);
			if(!res_endring.IsSuccess)
				throw std::runtime_error(std::format("EndRingProcess failed: {}",res_endring.ErrorMessage));
			std::println("Ring {} ended. {} rows in total; {} rows valid.",ring.kr,ring.ringHeight,validRows[ring.kr]);
		}
		
		auto timestamp_dataloaded=std::chrono::high_resolution_clock::now();

		auto res_endchannel=EndChannelProcess(channel,nullptr);
		if(!res_endchannel.IsSuccess)
			throw std::runtime_error(std::format("EndChannelProcess failed: {}", res_endchannel.ErrorMessage));
		//print the parsed options using std::println
		auto timestamp_end=std::chrono::high_resolution_clock::now();
		std::println("Parsed options:\n\tpathin={}\n\tpathout={}\n\tmaterial={}\n\tminpx={}\n\tthr_snr={}",
			arguments["pathin"].as<std::string>(),
			arguments["pathout"].as<std::string>(),
			arguments["material"].as<std::string>(),
			arguments["minpx"].as<int>(),
			arguments["thr_snr"].as<float>());
		if(arguments.count("dsc"))
			std::println("\tdsc={}",arguments["dsc"].as<std::string>());
		else
			std::println("\tthr_intg={}",arguments["thr_intg"].as<std::string>());
		std::chrono::duration<double> duration_dataloaded=timestamp_dataloaded-timestamp_start;
		std::println("Data loading time: {}s",duration_dataloaded.count());
		std::chrono::duration<double> duration_total=timestamp_end-timestamp_dataloaded;
		std::println("Inspection & export time: {}s",duration_total.count());	
		std::println("Wafer inspection completed.");
	}
	catch(const std::exception& e) 
	{
		std::cerr<<"Error: "<<e.what()<<std::endl;
		return 1;
	}

	return 0;
}
