#include "algorithm.h"
#include "algocore.h"
#include <tiffio.h> // Requires libtiff-dev
#include <nlohmann/json.hpp>
#include "io.h"
using json = nlohmann::json;

const float twopi=2.0f*static_cast<float>(CV_PI);
static LogMessageCallBack g_logCallback=nullptr; 
ContourCalcParameter param_contour;
std::vector <cv::Mat> centerCalculationImages,contourCalculationImages;
std::vector <float> centerCalculationAngles,contourCalculationAngles;
std::vector <cv::Point2d> CwaferEstimated;
std::mutex mtx_chsettings;
std::map<DetectChannel, std::map<std::string, json>> chsettings;
//const double intensity_normalizer=100.0;
bool DebugOutput=false;
float pixelsize;
bool flag_glass=false;//true: glass wafer, false: silicon wafer
bool flag_tialsi=false;//true: TiAlSi coating, false: no coating
struct Ring
{
	int index;
	cv::Mat img; // accumulated image, CV_8U \in [0,255] or CV_16U \in [0,4096]
	cv::Mat dehazed; // dehazed image, CV_32F \in [0,1]
	std::vector<double> column_intensity;
	int cursor=0; // current row index for adding new frames
	float theta0,theta1; // [in radians] starting angle for this ring
	json processSetting, hazeCaliSetting, coordCaliSetting;
	float radius_mm; // radius of this ring in mm 
	int overlapPixels;// number of pixels overlapping with the next ring
};
std::mutex mtx_rings;
std::map<DetectChannel,std::vector<Ring>> rings; //data structure to hold rings for each channel
void write_log(LogType level,const char* source,const char* message)
{
	if(g_logCallback)
	{
		g_logCallback(level,source,message);
		return;
	}
	//if(LogType::Info==level) return; 
	auto timestamp=std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
	printf("[%d @ %lld] %s: %s\n",static_cast<int>(level),static_cast<long long>(timestamp),source,message);
}
void imgprobe(cv::Mat img, bool continue_flag=false)
{
	double minVal,maxVal;
	cv::minMaxLoc(img,&minVal,&maxVal);
	write_log(LogType::Info,"imgprobe",std::format("Probed image: type={}, height={}, width={}",img.type(),img.rows,img.cols).c_str());
	write_log(LogType::Info,"imgprobe",std::format("Range of elements in probed image : [{},{}]",minVal,maxVal).c_str());
	if(!continue_flag)
		throw std::runtime_error("Intentional exception for image probing");
}

std::vector<DefectInfoStruct> inspect(const cv::Mat& image,const std::vector<double>& Intensities,const std::vector<double>& DSizes,const float pixelsize,float thr_snr,DetectChannel channel,std::string directory="")
{
	//CV_Assert(image.type() == CV_16UC1);
	CV_Assert(image.type() == CV_32F);
	//std::cout<<"\n";
	if(Intensities.size()!=DSizes.size())
	{
		write_log(LogType::Error,"inspect","Intensities and DSizes vectors must have the same size");
		return {};
	}
	if(Intensities.size()<2)
	{
		write_log(LogType::Error,"inspect","Intensities and DSizes vectors must contain at least two elements");
		return {};
	}
	std::stringstream ss; 
	ss<<"DSizeCurve : ";
	for(size_t k=0;k<Intensities.size();++k)
		ss<<std::format("({}|->{}um) ",Intensities[k],DSizes[k]);
	write_log(LogType::Info,"inspect",ss.str().c_str());

	std::vector<DefectInfoStruct> defects;
	const float intensity_threshold=(thr_snr-1.0f)*100;

	// Binary mask
	//cv::Mat binary=(image>=intensity);//&(image<intensity_next); // yields 8-bit mask (0 or 255) in OpenCV expression context on most builds
	cv::Mat binary=(image>=intensity_threshold);//&(image<intensity_next); // yields 8-bit mask (0 or 255) in OpenCV expression context on most builds
	// Ensure 8-bit 0/255
	if(binary.type()!=CV_8U)
		binary.convertTo(binary,CV_8U,255);
	cv::Mat closure;
	cv::morphologyEx(binary,closure,cv::MORPH_CLOSE,cv::getStructuringElement(cv::MORPH_ELLIPSE,cv::Size(11,11)));
	//cv::morphologyEx(binary,closure,cv::MORPH_CLOSE,cv::getStructuringElement(cv::MORPH_ELLIPSE,cv::Size(49,49)));
 
	// Connected components with stats
	cv::Mat labels,stats,centroids;
	int nLabels=cv::connectedComponentsWithStats(closure,labels,stats,centroids,8,CV_32S);
	write_log(LogType::Info,"inspect",std::format("SNR threshold: {},  Found {} potential regions. Analyzing...",thr_snr,nLabels-1).c_str()	);
	if(nLabels<=1) // no foreground components found
		return {};
	auto f_size=[&Intensities,&DSizes](double intensity)->double //linear interpolation
		{
			if (intensity <= Intensities.front())
				return DSizes.front();
			for(size_t k=0;k<Intensities.size()-1;++k)
				if(intensity<Intensities[k+1])
				{
					double x0=Intensities[k],x1=Intensities[k+1];
					double y0=DSizes[k],y1=DSizes[k+1];
					return y0+(intensity-x0)*(y1-y0)/(x1-x0);
				}
			return DSizes.back();
		};

	// Find the largest component (excluding background label 0)
	int maxLabel=1;
	int maxArea=stats.at<int>(1,cv::CC_STAT_AREA);
	std::ofstream ofs_detreg(directory+std::format("/channel{}_detected_regions.txt",static_cast<int>(channel)));//save a file for statistics of detected regions of column id,area,intg 
	ofs_detreg<<"ID\tArea(pixels)\tintg\tX\tY"<<std::endl;
	std::ofstream ofs_defectlog(directory+std::format("/channel{}_defectlog.txt",static_cast<int>(channel)));//save a file for statistics of detected regions of column id,area,intg 
	ofs_defectlog<<"ID\tArea(pixels)\tintg\tX\tY"<<std::endl;

	for(int lbl=1; lbl<nLabels; ++lbl)
	{
		float intg=0;
		int area=stats.at<int>(lbl,cv::CC_STAT_AREA);
		// Extract bounding box and centroid for the selected component
		int left=stats.at<int>(lbl,cv::CC_STAT_LEFT);
		int top=stats.at<int>(lbl,cv::CC_STAT_TOP);
		int width=stats.at<int>(lbl,cv::CC_STAT_WIDTH);
		int height=stats.at<int>(lbl,cv::CC_STAT_HEIGHT);
		cv::Rect bbox(left,top,width,height);
		for(int i=top; i<top+height; ++i)
			for(int j=left; j<left+width; ++j)
				if(labels.at<int>(i,j)==lbl)
				//if (labels.at<int>(i, j) == lbl && binary.at<ushort>(i, j) != 0)
					intg+=image.at<float>(i,j);
		int center_x=static_cast<int>(centroids.at<double>(lbl,0));
		int center_y=static_cast<int>(centroids.at<double>(lbl,1));
		if(flag_glass&&(center_x>10000&&center_x<31000&&center_y>22000))//ignore a region for glass wafer
		//if(flag_glass&&(!(center_x>10000&&center_x<31000&&center_y>22000)))//ignore a region for glass wafer
			continue;
		ofs_detreg<<std::format("{}\t{}\t{}\t{}\t{}\n",lbl,area,intg,left,top);

		//ignore the superficial components that are not part of the binary mask
		cv::Mat componentMask;
		cv::compare(labels(bbox),lbl,componentMask,cv::CMP_EQ); 
		cv::Mat binaryPixelsInComponent;
		cv::bitwise_and(binary(bbox),componentMask,binaryPixelsInComponent); 
		const int binaryPixelCount=cv::countNonZero(binaryPixelsInComponent);
		if(binaryPixelCount==0)
			continue;

		//if(intg<0.1f)//should not happen, but just in case
		//{
		//	std::println("Found region with integrated intensity {} below 0.1",intg);
		//	std::println( "channel={}, label={}: area={}, binary pixels in closure component={}, intg={}",
		//		static_cast<int>(channel), lbl, area, binaryPixelCount, intg);
		//	for(int i=top; i<top+height; ++i)
		//		for(int j=left; j<left+width; ++j)
		//			if(labels.at<int>(i,j)==lbl)
		//				//std::println("Pixel ({},{}) intensity {}",j,i,image.at<uchar>(i,j)); 
		//				std::println(
		//					"inspect channel={}, label={}: integrated intensity {} below 0.1; pixel ({}, {}) intensity {}",
		//					static_cast<int>(channel), lbl, intg, j, i, image.at<float>(i,j));
		//}

		double cx=centroids.at<double>(lbl,0);
		double cy=centroids.at<double>(lbl,1);
		//cv::Point2d centroid(cx,cy);

		DefectInfoStruct defect{0};
		defect.DefectID=static_cast<int>(1);
		defect.ChannelID=channel;
		defect.Type=DefectType::LPD;
		defect.CoordX=static_cast<float>((cx-image.cols/2)*pixelsize);
		defect.CoordY=static_cast<float>((cy-image.rows/2)*pixelsize);
		defect.CoordR=static_cast<float>(std::sqrt(defect.CoordX*defect.CoordX+defect.CoordY*defect.CoordY));
		defect.CoordT=static_cast<float>(std::atan2(defect.CoordY,defect.CoordX)*180.0f/3.14159265f); // in degrees
		defect.XSize=static_cast<float>(width)*pixelsize;
		defect.YSize=static_cast<float>(height)*pixelsize;
		defect.Area=pixelsize*pixelsize*area;
		//if(area>10)
		//	defect.DSize=std::max(width,height)*pixelsize;
		defect.DSize=static_cast<float>(f_size(intg));
		defect.MaxDSize=static_cast<float>(f_size(intg));
		defect.SNR=intg;//todo
		defect.SumSNR=1;//todo
		defect.SpaceResolution=1.0f;//todo
		defect.RawPixelsOffset=0;
		defect.RawPixelsCount=area;
		defect.OutlinePointsOffset=0;
		defect.OutlinePointsCount=0;
		defect.BinCode=0;

		if(area>maxArea)
		{
			maxArea=area;
			maxLabel=lbl;
		}
		const float Rmax=image.cols/2*pixelsize*0.98f;
		if(defect.CoordR>Rmax) continue;
		if(intg<intensity_threshold)
			continue; // skip defects less than threshold the intensity range
		if(intg<Intensities[1])
			continue; // skip defects less than threshold the intensity range 
		//if(defect.CoordX>0||defect.CoordY>0) continue; // skip defects 
			//std::println("Skipping region at ({},{}) with integrated intensity {} below threshold {}",defect.CoordX,defect.CoordY,intg,Intensities[1]);
		//finally, add the defect to the list
		defects.push_back(defect);
		ofs_defectlog<<std::format("{}\t{}\t{}\t{}\t{}\n",lbl,area,intg,defect.CoordX,defect.CoordY);
	}
	std::sort(defects.begin(),defects.end(),[](const DefectInfoStruct& a,const DefectInfoStruct& b) { return a.Area>b.Area; }); // descending order
	return defects;
}
cv::Mat drawmap(cv::Mat image,const std::vector<DefectInfoStruct> &defects,const float pixelsize)
{
	write_log(LogType::Info,"drawmap","Drawing defect annotations");
	cv::Mat defect_annotation;
	cv::cvtColor(image,defect_annotation,cv::COLOR_GRAY2BGR);
	//int numdraw=0;
	write_log(LogType::Info,"drawmap",std::format("Total defects {}",defects.size()).c_str());
	//std::ofstream defect_log("defect_log.txt");
	for(const auto& defect:defects)
	{
		//if(numdraw++>500) break; // limit to the first few defects for drawing
		if(defect.Area<=0) continue; // skip invalid entries
		cv::Point center(static_cast<int>(defect.CoordX/pixelsize+image.cols/2),static_cast<int>(defect.CoordY/pixelsize+image.rows/2));
		cv::Size axes(static_cast<int>(defect.XSize/2/pixelsize)+20,static_cast<int>(defect.YSize/2/pixelsize)+20);
		cv::ellipse(defect_annotation,center,axes,0,0,360,cv::Scalar(0,0,65530),2); // red ellipse
		//defect_log<<std::format("Drawing defect at ({},{})px. R={}um, T={}deg, sizes=({},{})um, area={}px({}um^2), bin={}\n",center.x,center.y,defect.CoordR,defect.CoordT,defect.XSize,defect.YSize,defect.RawPixelsCount,defect.Area,defect.BinCode);

		// place Area and BinCode text at the top of the ellipse, with a small offset to avoid overlap
		int offsetY = std::max(axes.height, 10);
		cv::Point textOrg(center.x , center.y - offsetY);

		// clamp text position inside image
		if(textOrg.x < 2) textOrg.x = 2;
		if(textOrg.y < 12) textOrg.y = 12; // ensure baseline is visible

		// prepare text strings
		//std::string text=std::format("Area: {}, BinCode: {}",static_cast<int>(defect.Area+0.5),defect.BinCode);
		//std::string text=std::format("{}",defect.RawPixelsCount);
		std::string text=std::format("I={}",defect.SNR);

		int fontFace = cv::FONT_HERSHEY_SIMPLEX;
		double fontScale = 4;
		int thickness = 1;

		// draw text with a thin black outline for readability, then white text
		cv::putText(defect_annotation, text, textOrg, fontFace, fontScale, cv::Scalar(0,0,0), thickness+2, cv::LINE_AA);
		cv::putText(defect_annotation, text, textOrg, fontFace, fontScale, cv::Scalar(255,255,255)*256, thickness, cv::LINE_AA); 
	}
	write_log(LogType::Info,"drawmap","Finished drawing defect annotations");
	return defect_annotation;
}
bool saveBigTiff(const std::string& filename, cv::Mat& image) 
{
    // Open file with '8' flag to explicitly force BigTIFF format
    TIFF* tif = TIFFOpen(filename.c_str(), "w8"); 
    if (!tif) 
		throw std::runtime_error(std::format("Failed to open {} for writing BigTIFF.",filename));

    // Set mandatory TIFF tags based on cv::Mat properties
    TIFFSetField(tif, TIFFTAG_IMAGEWIDTH, image.cols);
    TIFFSetField(tif, TIFFTAG_IMAGELENGTH, image.rows);
    TIFFSetField(tif, TIFFTAG_SAMPLESPERPIXEL, image.channels());
    TIFFSetField(tif, TIFFTAG_BITSPERSAMPLE, image.elemSize1() * 8);
    TIFFSetField(tif, TIFFTAG_ORIENTATION, ORIENTATION_TOPLEFT);
    TIFFSetField(tif, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);

    // Set Color Space tag
    if (image.channels() == 1) 
        TIFFSetField(tif, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_MINISBLACK);
    else if (image.channels() == 3 || image.channels() == 4) 
        TIFFSetField(tif, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_RGB);

    // Set Compression (e.g., LZW)
    TIFFSetField(tif, TIFFTAG_COMPRESSION, COMPRESSION_LZW);

    // Write row by row to prevent massive single-allocation spikes
    //size_t row_stride = image.step; 
    for (int y = 0; y < image.rows; ++y)
        if (TIFFWriteScanline(tif, image.ptr<uchar>(y), y, 0) < 0) 
		{
            TIFFClose(tif);
            throw std::runtime_error(std::format("Error writing scanline at row {}", y));
        }

    TIFFClose(tif);
    return true;
}
extern "C"
{
	AlgoResult SetLogMessageCallBack(LogMessageCallBack callBackPointer)
	{
		g_logCallback=callBackPointer;
		return AlgoResult::Success();
	}
	AlgoResult Initialize()
	{
		g_logCallback=nullptr; // Reset log callback to null
		centerCalculationImages.clear(); // Clear any previously stored images
		centerCalculationAngles.clear(); // Clear any previously stored angles
		contourCalculationImages.clear(); // Clear any previously stored images
		contourCalculationAngles.clear(); // Clear any previously stored angles
		CwaferEstimated.clear(); // Clear any previously stored estimated points 
		{
			std::lock_guard<std::mutex> lock(mtx_chsettings);
			chsettings.clear(); // Clear any previously stored channel settings
		}
		release_all_mmap(); // Release all memory-mapped files
		{
			std::lock_guard<std::mutex> lock(mtx_rings);
			rings.clear(); // Clear any previously stored rings
		}
#if defined(_WIN32)
		_putenv_s("OPENCV_IO_ENABLE_OPENEXR","1");
#else
		setenv("OPENCV_IO_ENABLE_OPENEXR","1",1);
#endif
		write_log(LogType::Info,"Initialize","Algorithm module initialized");
		return AlgoResult::Success();
	}

	AlgoResult BeginCenterCalculation(ContourCalcParameter parameters)
	{
		param_contour=parameters;
		return AlgoResult::Success();
	}
	AlgoResult BeginContourCalculation(ContourCalcParameter parameters)
	{
		param_contour=parameters;
		return AlgoResult::Success();
	}
	AlgoResult AddContourCalculationImage(const unsigned char* imageData,double stageAngle)
	{
		write_log(LogType::Info,"AddContourCalculationImage",std::format("Adding image of size {}x{} at stage angle {}",param_contour.ImageWidth,param_contour.ImageHeight,stageAngle).c_str());
		cv::Mat img(param_contour.ImageHeight,param_contour.ImageWidth,CV_8UC1,const_cast<unsigned char*>(imageData));
		if(img.empty()||img.type()!=CV_8UC1)
		{
			std::string hexAddress=std::format("{:p}",static_cast<const void*>(imageData));
			write_log(LogType::Error,"AddContourCalculationImage",std::format("invalid image on address {}",hexAddress).c_str());
			return AlgoResult::Failure("Invalid image");
		}
		contourCalculationAngles.push_back(static_cast<float>(stageAngle));
		contourCalculationImages.push_back(img.clone()); // Store a copy of the image
		return AlgoResult::Success();
	}
	AlgoResult EndContourCalculation(double* transX,double* transY,double* angle)
	{
		for(size_t i=0;i<contourCalculationImages.size();++i)
		{
			//write_log(LogType::Info,"EndContourCalculation",std::format("Processing contour image {} at stage angle {}",i,contourCalculationAngles[i]).c_str());
			//process the image to find the contour and calculate the translation
			cv::Mat img=contourCalculationImages[i];
			if(img.empty())
			{
				write_log(LogType::Error,"EndContourCalculation",std::format("Contour image {} is empty",i).c_str());
				continue;
			}
			auto edge_points=get_edge_points(img);
			cv::Point2d valley,center;
			bool notch_found=find_notch(edge_points,valley,center);// use find_notch() to determine if a notch is found 
			if(notch_found)
			{
				*transX=center.x*param_contour.PixelSize/1000.0; // convert to mm
				*transY=-center.y*param_contour.PixelSize/1000.0; // convert to mm
				*angle=contourCalculationAngles[i]+std::atan2(-valley.y+center.y,valley.x-center.x)*180.0/3.1416;//y axis is inverted in image coordinates
				write_log(LogType::Info,"EndContourCalculation",std::format("Contour image {}: Notch found at ({},{}) using image with stage angle {}",i,valley.x,valley.y,contourCalculationAngles[i]).c_str());
				return  AlgoResult::Success();
			}
			//write_log(LogType::Info,"EndContourCalculation",std::format("Contour image {} size: {}x{}",i,img.cols,img.rows).c_str());
		}
		write_log(LogType::Error,"EndContourCalculation",std::format("Notch not found ").c_str());
		return AlgoResult::Failure("Notch not found");
	}
	AlgoResult AddCenterCalculationImage(const unsigned char* imageData,double stageAngle)
	{
		(void)stageAngle;
		write_log(LogType::Info,"AddCenterCalculationImage",std::format("Adding image of size {}x{} at stage angle {}",param_contour.ImageWidth,param_contour.ImageHeight,stageAngle).c_str());	
		cv::Mat img(param_contour.ImageHeight,param_contour.ImageWidth,CV_8UC1,const_cast<unsigned char*>(imageData));
		if(img.empty()||img.type()!=CV_8UC1) 
		{
			std::string hexAddress=std::format("{:p}",static_cast<const void*>(imageData));
			write_log(LogType::Error,"AddCenterCalculationImage",std::format("invalid image on address {}",hexAddress).c_str());
			return AlgoResult::Failure("Invalid image");
		}

		auto pts=get_edge_points(img);

		//write_log(LogType::Debug, "AddCenterCalculationImage", std::format("collected {} edge points from contour_b", pts.size()).c_str());

		if(pts.size()<100)
		{
			auto err=std::format("AddCenterCalculationImage: insufficient edge points {}\n",pts.size());
			return AlgoResult::Failure(err.c_str());
		}

		//auto id=centerCalculationImages.size();//define a unique id for this image based on the current size of the vector

		/*
		//  Fit circle (Kasa least-squares method)
		// Solve [x y 1] * [A B C]^T = -(x^2 + y^2)
		cv::Mat1d M(static_cast<int>(pts.size()), 3);
		cv::Mat1d b(static_cast<int>(pts.size()), 1);
		for (size_t i = 0; i < pts.size(); ++i) {
			double x = pts[i].x;
			double y = pts[i].y;
			M(static_cast<int>(i), 0) = x;
			M(static_cast<int>(i), 1) = y;
			M(static_cast<int>(i), 2) = 1.0;
			b(static_cast<int>(i), 0) = -(x * x + y * y);
		}
		cv::Mat1d sol;
		bool ok = cv::solve(M, b, sol, cv::DECOMP_SVD);
		if (!ok || sol.rows != 3) {
			//write_log(LogType::Error, "AddCenterCalculationImage", "EndCenterCalculation: circle fit failed");
			return AlgoResult::Success();
		}
		double A = sol(0, 0), B = sol(1, 0), C = sol(2, 0);
		double xc = -A / 2.0, yc = -B / 2.0, r2 = xc * xc + yc * yc - C;
		double r = (r2 > 0.0) ? std::sqrt(r2) : 0.0;
		//write_log(LogType::Info, "AddCenterCalculationImage", std::format("Fitted circle: center=({},{}), radius={}", xc, yc, r).c_str());
		cv::Mat imgWithCircle = img.clone();
		cv::circle(imgWithCircle, cv::Point(static_cast<int>(xc), static_cast<int>(yc)), static_cast<int>(r), cv::Scalar(255), 2);
		cv::imwrite(std::format("{}_fitted_circle.png", id), imgWithCircle);
		*/

		//fit a line to the edge points using cv::fitLine
		cv::Vec4f lineParams;
		cv::fitLine(pts,lineParams,cv::DIST_L2,0,0.01,0.01);
		//double R_wafer_px=param_contour.WaferSize/2*208.4;
		double R_wafer_px=param_contour.WaferSize/param_contour.PixelSize;
		//R_wafer_px=10000;

		{ // draw the fitted line on a new overlay image
			// lineParams: [vx, vy, x0, y0]
			const double vx=lineParams[0];
			const double vy=lineParams[1];
			const double x0=lineParams[2];
			const double y0=lineParams[3];

			// Choose a sufficiently long segment to cover the image
			double L=std::max(img.cols,img.rows)*1.5;
			cv::Point p1(cvRound(x0-vx*L),cvRound(y0-vy*L));
			cv::Point p2(cvRound(x0+vx*L),cvRound(y0+vy*L));

			cv::Mat lineOverlay;
			cv::cvtColor(img,lineOverlay,cv::COLOR_GRAY2BGR);

			// 1. Find M, the mid point of the tangent line
			cv::Point2d M(x0,y0);

			// Determine the normal vector direction (towards the bright wafer region)
			double nx=-vy,ny=vx;
			int testDist=15;
			int tx=cvRound(x0+nx*testDist),ty=cvRound(y0+ny*testDist);

			bool normalPointsInward=false;
			cv::Mat imgb;
			cv::threshold(img,imgb,30,255,cv::THRESH_BINARY);
			if(tx>=0&&tx<imgb.cols&&ty>=0&&ty<imgb.rows)
				if(imgb.at<uchar>(ty,tx)>128)
					normalPointsInward=true;
			if(!normalPointsInward)
			{
				nx=-nx;
				ny=-ny;
			}

			// Find Cwafer, the wafer center, using R_wafer_px
			cv::Point2d Cwafer(x0+nx*R_wafer_px,y0+ny*R_wafer_px);
			write_log(LogType::Info, "AddCenterCalculationImage", std::format("M=({},{}), Cwafer=({},{}), R_wafer_px={}", M.x, M.y, Cwafer.x, Cwafer.y, R_wafer_px).c_str());
			////Alternative code path: use ransac_fit_circle to fit a circle to the edge points and estimate Cwafer
			//int num_inliers=ransac_fit_circle(pts,Cwafer.x,Cwafer.y,R_wafer_px,10000,0.1);
			//write_log(LogType::Info,"AddCenterCalculationImage",std::format("RANSAC fitted circle: inlier={} Cwafer=({},{}), R_wafer_px={}, Inliers={}",num_inliers,Cwafer.x,Cwafer.y,R_wafer_px,num_inliers).c_str());
			//cv::line(lineOverlay,M,Cwafer,cv::Scalar(255,0,0),2,cv::LINE_AA); //Blue for the radius connecting M and Cwafer 
			//cv::line(lineOverlay,p1,p2,cv::Scalar(0,0,255),2,cv::LINE_AA); //Red for fitted line
			//Draw the circle on the overlay image (yellow)
			//cv::circle(lineOverlay, Cwafer, cvRound(R_wafer_px), cv::Scalar(0, 255, 128), 2, cv::LINE_8);//red
			std::vector<cv::Point> arc_pts;
			double start_angle=std::atan2(-ny,-nx); // The angle from the center Cwafer pointing strictly back towards M
			double angle_span=0.2; // Radian span to render (sufficient to cross the image)
			double angle_step=0.001; // Ultra-fine step to defeat chord/polygon approximation 
			for(double a=start_angle-angle_span; a<=start_angle+angle_span; a+=angle_step) {
				int px=cvRound(Cwafer.x+R_wafer_px*std::cos(a));
				int py=cvRound(Cwafer.y+R_wafer_px*std::sin(a));
				arc_pts.push_back(cv::Point(px,py));
			}
			cv::polylines(lineOverlay,arc_pts,false,cv::Scalar(0,255,128),2,cv::LINE_AA);// draw the arc of the circle (green)
			cv::circle(lineOverlay,M,4,cv::Scalar(255,255,255),-1,cv::LINE_AA); // mark the point M on the line used by fitLine (white circle)
			// mark Cwafer (red circle)
			//cv::circle(lineOverlay,Cwafer,6,cv::Scalar(0,0,255),-1,cv::LINE_AA);
			//save overlayimage
			cv::imwrite(std::format("{}_fitted_line_overlay.png",stageAngle),lineOverlay);

			CwaferEstimated.emplace_back(static_cast<double>(Cwafer.x),static_cast<double>(-Cwafer.y));
			centerCalculationAngles.push_back(static_cast<float>(stageAngle));
		}
		//{// Save processed contour points to an overlay image for visualization
		//	std::vector<cv::Point> cpts;
		//	cpts.reserve(pts.size());
		//	for(const auto& p:pts) cpts.emplace_back(static_cast<int>(p.x),static_cast<int>(p.y));

		//	// overlay adjacent-point lines on original image (red)
		//	std::vector<std::vector<cv::Point>> drawVec{cpts};
		//	cv::Mat overlay;
		//	cv::cvtColor(img,overlay,cv::COLOR_GRAY2BGR);
		//	cv::polylines(overlay,drawVec,false,cv::Scalar(0,0,255),2,cv::LINE_AA);
		//	cv::imwrite(std::format("{}_contour_overlay.png",id),overlay); 
		//} 

		centerCalculationImages.push_back(img.clone());
		return AlgoResult::Success();
	}
	AlgoResult EndCenterCalculation(double* dx,double* dy,double* angle)
	{
		if(CwaferEstimated.size()<3)
			return AlgoResult::Failure("Not enough estimated wafer centers");

		std::ofstream ofs_center("CwaferEstimated.txt");
		ofs_center<<"ID\tAngle(deg)\tX(pixels)\tY(pixels)\n";
		for(size_t i = 0; i < CwaferEstimated.size(); ++i) {
			const auto& p = CwaferEstimated[i];
			ofs_center << i << "\t" << centerCalculationAngles[i] << "\t" << p.x << "\t" << p.y << "\n";
		}
		double sumX=0,sumY=0,sumX2=0,sumY2=0,sumXY=0;
		double sumX3=0,sumY3=0,sumX2Y=0,sumXY2=0;
		size_t n=CwaferEstimated.size();

		for(size_t i = 0; i < CwaferEstimated.size(); ++i) {
			const auto& p = CwaferEstimated[i];
			sumX+=p.x;
			sumY+=p.y;
			sumX2+=p.x*p.x;
			sumY2+=p.y*p.y;
			sumXY+=p.x*p.y;
			sumX3+=p.x*p.x*p.x;
			sumY3+=p.y*p.y*p.y;
			sumX2Y+=p.x*p.x*p.y;
			sumXY2+=p.x*p.y*p.y;
		}

		double a=n*sumX2-sumX*sumX;
		double b=n*sumXY-sumX*sumY;
		double c=n*sumY2-sumY*sumY;
		double d=0.5*(n*sumX3+n*sumXY2-sumX*(sumX2+sumY2));
		double e=0.5*(n*sumX2Y+n*sumY3-sumY*(sumX2+sumY2));

		double det=a*c-b*b;
		if(std::abs(det)<1e-12)
			return AlgoResult::Failure("Points are collinear, cannot fit circle");

		if(dx) *dx=(d*c-b*e)/det;
		if(dy) *dy=(a*e-b*d)/det;
		if(angle) *angle=17.0; // Angle (from C_SA to C_CA) usually derived from a notch/flat, defaulting to 17.0

		return AlgoResult::Success();
	}
 
	ALGO_API AlgoResult BeginChannelProcess(DetectChannel channelID,
		const char* ring,const char* cluster,const char* classification,
		const char* coordCaliSetting,const char* DSizeCurve)
	{
		(void)cluster, (void)coordCaliSetting; 
		std::lock_guard<std::mutex> lock(mtx_chsettings);
		write_log(LogType::Info,"BeginChannelProcess",std::format("channelID={}, ring={}",static_cast<int>(channelID),ring).c_str());
		chsettings[channelID]["ring"]=json::parse(ring); 
		write_log(LogType::Info,"BeginChannelProcess",std::format("channelID={}, classifcation={}",static_cast<int>(channelID),classification).c_str());
		chsettings[channelID]["classification"]=json::parse(classification); 
		chsettings[channelID]["DSizeCurve"]=json::parse(DSizeCurve); 
		std::lock_guard<std::mutex> lock_imgs(mtx_rings); 
		rings[channelID]=std::vector<Ring>(chsettings[channelID]["ring"]["TotalRings"]);
		if(chsettings[channelID]["ring"].find("DebugOutput")!=chsettings[channelID]["ring"].end())
			DebugOutput=chsettings[channelID]["ring"]["DebugOutput"];
		if(chsettings[channelID]["ring"].find("Material")!=chsettings[channelID]["ring"].end())
		{
			flag_glass=chsettings[channelID]["ring"]["Material"]=="glass";
			flag_tialsi=chsettings[channelID]["ring"]["Material"]=="tialsi";
		}
		pixelsize=chsettings[channelID]["classification"]["PixelSize"];
		//std::println("setting pixelsize as {}",pixelsize);
		return AlgoResult::Success();
	}
	ALGO_API AlgoResult BeginRingProcess(DetectChannel channelID,int ringIndex,
		const char* processSetting,const char* hazeCaliSetting,const char* coordCaliSetting)
	{
		(void)hazeCaliSetting; 
		write_log(LogType::Info,"BeginRingProcess",std::format("processSetting: {}",processSetting).c_str());
		write_log(LogType::Info,"BeginRingProcess",std::format("coordCaliSetting: {}",coordCaliSetting).c_str());
		json coordCaliJson=json::parse(coordCaliSetting); 
		//std::cout<<coordCaliJson.dump(4)<<std::endl;
		int RingWidth=coordCaliJson["RingWidth"];
		int RingHeight=coordCaliJson["RingHeight"];
		//std::println("BeginRingProcess: RingWidth={}, RingHeight={}",RingWidth,RingHeight);
		//std::println("BeginRingProcess: channelID={}, #rings={}",static_cast<int>(channelID),rings[channelID].size());
		std::lock_guard<std::mutex> lock_settings(mtx_chsettings);
		std::lock_guard<std::mutex> lock_imgs(mtx_rings);
		// allocate the Ring's image
		auto& ring=rings[channelID][ringIndex];
		ring.img = cv::Mat(RingHeight, RingWidth, CV_16UC1);
		ring.cursor = 0;
		float theta0_angle=coordCaliJson["TriggerStart"]["T"];
		float theta1_angle=coordCaliJson["TriggerEnd"]["T"];
		float thetadiff_angle=theta1_angle-theta0_angle;
		ring.radius_mm=coordCaliJson["TriggerStart"]["R"];
		//std::println("BeginRing: radius_mm={}",ring.radius_mm);
		ring.theta0=theta0_angle/180.0f * static_cast<float>(CV_PI); //coverting to radians if needed, but assuming the input is in degrees or radians as required
		while(ring.theta0>twopi) ring.theta0-=twopi;
		ring.theta1=ring.theta0+thetadiff_angle/180.0f * static_cast<float>(CV_PI); 
		ring.overlapPixels=coordCaliJson["OverlapPixels"];

		//ring.theta1=ring.theta0+thetadiff_angle/180.0f * static_cast<float>(CV_PI); 
		//std::println("BeginRingProcess: ringIndex={}, theta0={} rad, theta1={} rad",ringIndex,ring.theta0,ring.theta1);
		write_log(LogType::Info,"BeginRingProcess",std::format(" ringIndex={}, theta0={} rad, theta1={} rad",ringIndex,ring.theta0,ring.theta1).c_str());
		return AlgoResult::Success();
	}

	ALGO_API AlgoResult AddRingProcessFrame(DetectChannel channelID,int ringIndex,
		const unsigned char* imageData,int width,int height,int validRows,
		int timestamp,int softBinning)
	{
		(void)height;(void)timestamp; (void)softBinning;
		std::lock_guard<std::mutex> lock_imgs(mtx_rings); 
		//validRows=std::min(validRows,rings[channelID][ringIndex].img.rows-rings[channelID][ringIndex].cursor);

		size_t dataSize=static_cast<size_t>(width)*validRows*2;// 2 bytes per pixel for CV_16UC1
		cv::Mat &dst = rings[channelID][ringIndex].img;
		size_t offset=static_cast<size_t>(rings[channelID][ringIndex].cursor)*static_cast<size_t>(width)*2; // 2 bytes per pixel for CV_16UC1
		if(offset+dataSize>static_cast<size_t>(dst.rows)*static_cast<size_t>(dst.cols)*2)
		{ 
			write_log(LogType::Warning,"AddRingProcessFrame",std::format("Truncated data exceeds allocated ring image size: offset={} + dataSize={} > totalSize={}",offset,dataSize,dst.rows*dst.cols).c_str());
			//return AlgoResult::Failure("AddRingProcessFrame: data exceeds allocated ring image size");
			dataSize=static_cast<size_t>(dst.rows)*static_cast<size_t>(dst.cols)-offset;
		}
		//cv::Mat temp(height,width,CV_16UC1,const_cast<unsigned char*>(imageData));
		//cv::imwrite("d:/tmp/verification.png", temp);
		std::memcpy(dst.data + offset, imageData, dataSize);
		rings[channelID][ringIndex].cursor += validRows;
		write_log(LogType::Info,"AddRingProcessFrame",std::format("channelID={}, ringIndex={}, added {} rows, cursor now at {}",static_cast<int>(channelID),ringIndex,validRows,rings[channelID][ringIndex].cursor).c_str());
		return AlgoResult::Success();
	}

	ALGO_API AlgoResult EndRingProcess(DetectChannel channelID,int ringIndex, bool* isOverLoad,bool* isHazeOverload)
	{
		(void)isOverLoad; (void)isHazeOverload;
		(void)ringIndex;(void)channelID;
		//if(isOverLoad)*isOverLoad=false; 
		//if(isHazeOverload)*isHazeOverload=false;
		//Remove the left columns of 'overlapPixels' from the ring image
		int overlapPixels=rings[channelID][ringIndex].overlapPixels;
		if(overlapPixels>0)
		{
			cv::Mat &img=rings[channelID][ringIndex].img;
			if(overlapPixels<img.cols)
				img=cv::Mat(img,cv::Rect(overlapPixels,0,img.cols-overlapPixels,img.rows)).clone();
			write_log(LogType::Info,"EndRingProcess",std::format("channelID={}, ringIndex={}, removed {} overlap pixels, new size {}x{}",static_cast<int>(channelID),ringIndex,overlapPixels,img.cols,img.rows).c_str());
		}
		return AlgoResult::Success();
	} 

	ALGO_API AlgoResult EndChannelProcess(DetectChannel channelID,DefectInfoListStruct* descriptor)
	{ 
		//PARAMETER EXTRACTION 
		std::vector<Ring> rings_copy;
		std::vector<int> ringWidths;
		std::string dirout; 
		{
			std::lock_guard<std::mutex> lock_imgs(mtx_rings);
			for(const auto& r:rings[channelID])
			{
				rings_copy.push_back(r); 
				ringWidths.push_back(r.img.cols); // store the width of each ring image 
			}
			rings[channelID].clear(); // clear the original rings to free memory
		}
		std::vector<int> ringStart(ringWidths.size()),ringEnd(ringWidths.size());
		{
			int N=static_cast<int>(rings_copy.size());
			ringStart[N-1]=ringWidths[N-1];
			ringEnd[N-1]=0;
			for(int i=N-2;i>=0;i--)
			{
				ringEnd[i] = ringStart[i+1];
				ringStart[i]=ringEnd[i]+ringWidths[i];
			} 
		} 
		const int W=2*std::accumulate(ringWidths.begin(),ringWidths.end(),0);
		const int H=W;
		int N=rings_copy.size();
		float thr_snr=1.8f;//default value, will be overwritten by settings
		try
		{
			std::lock_guard<std::mutex> lock_settings(mtx_chsettings);
			dirout=chsettings[channelID]["ring"]["ImageSaveDirectory"];
			N=chsettings[channelID]["ring"]["TotalRings"]; 
			thr_snr=chsettings[channelID]["classification"]["SNRThreshold"];
			if(thr_snr>20.0f) thr_snr=20.0f; // clamp to a reasonable maximum
			if(thr_snr<1.2f) thr_snr=1.2f; // clamp to a reasonable minimum
			//pixelsize=chsettings[channelID]["classification"]["PixelSize"]; 
		}
		catch(const std::exception& e)
		{
			write_log(LogType::Warning,"EndChannelProcess",std::format("channelID={} using default settings due to exception during extracting settings: {}",static_cast<int>(channelID),e.what()).c_str());
		}

		//PROCESS OF EACH RING 
		try
		{
			#pragma omp parallel for
			for(int kr=0; kr<N; kr++)
			{
				//write_log(LogType::Info,"EndChannelProcess",std::format("channelID={}, analyzing ring {}",static_cast<int>(channelID),kr).c_str());
				const cv::Mat &rimg=rings_copy[kr].img;
				auto col_intensity=estimate_column_intensity_stat(rimg);
				//write_log(LogType::Info,"EndChannelProcess",std::format("channelID={}, column intensity estimated for ring {}",static_cast<int>(channelID),kr).c_str());
				float r_inner_mm,r_outer_mm;
					float w=static_cast<float>(rimg.cols);
					int ninside=static_cast<int>(rings_copy.size())-kr-1;//number of ring inside of current rings
					r_inner_mm=w*ninside*pixelsize/1000.0f;
					r_outer_mm=w*(ninside+1)*pixelsize/1000.0f;
					//write_log(LogType::Info,"EndChannelProcess",std::format("channelID={}, ring {}: r_inner_mm={}, r_outer_mm={}",static_cast<int>(channelID),kr,r_inner_mm,r_outer_mm).c_str());	
					//r_inner_mm=r-pixelsize/1000.0f*W/2;
					//r_outer_mm=r+pixelsize/1000.0f*W/2;
					//std::println("flatten: r_inner={}, r_outer={}",r_inner_mm,r_outer_mm);
					//std::println("flatten: radius={}",r_inner_mm,r_outer_mm);
				//write_log(LogType::Info,"EndChannelProcess",std::format("channelID={}, dehazing ring {}",static_cast<int>(channelID),kr).c_str());
				cv::Mat flattened=flatten(rimg,r_inner_mm,r_outer_mm,DebugOutput?std::format("{}/radical_intensity{}.txt",dirout,kr):"");
				double minval,maxval;
				cv::minMaxLoc(flattened,&minval,&maxval,nullptr,nullptr);
				//std::println("flattened range: min={}, max={}",minval,maxval); 
				cv::Mat haze(flattened.size(),flattened.type(),cv::Scalar(1.0)); //haze: background comes frome scattering of laser by the roughness of the wafer surface
				auto haze_col_intensity=estimate_column_intensity_stat(flattened);
				for(int x=0; x<haze.cols; ++x)
					haze.col(x)=haze_col_intensity[x]*1.0f;
				cv::Mat dehazed=flattened-haze;//allow negative values to remain
				if(flag_glass&&kr>=5) //ignore the central rings for glass wafer
					dehazed=0;
				if(flag_tialsi&&kr>=10) //ignore the central rings for glass wafer
				{
					write_log(LogType::Info,"EndChannelProcess",std::format("channelID={}, ring {} is ignored for tialsi wafer",static_cast<int>(channelID),kr).c_str());
					dehazed=0; 
				}
				rings_copy[kr].dehazed=dehazed;

				write_log(LogType::Info,"EndChannelProcess",std::format("channelID={}, ring {} ready for inspection",static_cast<int>(channelID),kr).c_str());
			}
		}
		catch(const std::exception& e)
		{
			write_log(LogType::Error,"EndChannelProcess",std::format("channelID={}, exception during processing ring images: {}",static_cast<int>(channelID),e.what()).c_str());
			return AlgoResult::Failure(e.what());
		}


		//FULL MAP CONSTRUCTION
		//Now we find the pixel value for each pixel in the mergedImage by mapping it to the corresponding ring image
		//The first ring image corresponds to the outermost ring. In each ring image, the first column corresponds to the angle 0, and the last column corresponds to the angle 2*pi. The first row corresponds to the outer edge of the ring, and the last row corresponds to the inner edge of the ring.
		//The the first row of each ring image corresponds to theta=0, and the last row corresponds to theta=2*pi. 
		cv::Mat fullimage(H,W,CV_16UC1,cv::Scalar(0));
		//cv::Mat haze(H,W,CV_32F,cv::Scalar(100));
		//cv::Mat haze(H,W,fullimage.type(),cv::Scalar(0));
		cv::Mat dehazed(H,W,CV_32F,cv::Scalar(0));
		//cv::Mat dehazed(H,W,fullimage.type(),cv::Scalar(0));
		write_log(LogType::Info,"EndChannelProcess",std::format("channelID={}, constructing full map of size={}x{}",static_cast<int>(channelID),fullimage.cols,fullimage.rows).c_str());
		#pragma omp parallel for
		for(int i=0;i<H;i++)
		{
			for(int j=0;j<W;j++)
			{
				float x=j-W/2.0f;
				float y=H/2.0f-i;
				float r=std::sqrt(x*x+y*y);
				if(r>ringStart[0])
					continue; // outside the outermost ring
				float theta=std::atan2(y,x);
				//float r_mm=r*pixelSize;
				int idxr=0;
				for(;idxr<N;idxr++)
					if(ringEnd[idxr]<r&&r<ringStart[idxr])
						break;
				if(idxr==N)
					continue;

				const float theta0=rings_copy[idxr].theta0;
				//const float theta1=rings_copy[idxr].theta1;
				const float theta1=theta0+twopi;
				while(theta<theta0)
					theta+=twopi;

				int rows=rings_copy[idxr].img.rows;
				int cols=rings_copy[idxr].img.cols;

				int row=static_cast<int>(((theta-theta0)/(theta1-theta0))*(rows-1));
				float r_in_ring=ringStart[idxr]-r;
				int col=static_cast<int>((r_in_ring/ringWidths[idxr])*(cols-1));
 
				if(col>=cols||row<0||col<0)
				{
					write_log(LogType::Warning,"EndChannelProcess",std::format("Out of bounds: ringIndex={}, row={}, col={}, rows={}, cols={}",idxr,row,col,rows,cols).c_str());
					continue; // out of bounds, skip
				}
				row=std::clamp(row,0,rows-1);
				col=std::clamp(col,0,cols-1);
				col=cols-1-col;// flip the column index to match the orientation of the ring image

				//fullimage.at<uchar>(i,j)=rings_copy[idxr].defectmap.at<uchar>(row,col);
				//defectmap.at<float>(i,j)=rings_copy[idxr].defectmap.at<float>(row,col);
				//dehazed.at<float>(i,j)=rings_copy[idxr].dehazed.at<float>(row,col);
				fullimage.at<ushort>(i,j)=rings_copy[idxr].img.at<ushort>(row,col);
				//haze.at<float>(i,j)=rings_copy[idxr].haze.at<float>(row,col);
				dehazed.at<float>(i,j)=rings_copy[idxr].dehazed.at<float>(row,col);
			}
		}
		write_log(LogType::Info,"EndChannelProcess","Full map constructed");
		rings_copy.clear(); // Clear the local copy of ring images
		write_log(LogType::Info,"EndChannelProcess","Input frames cleared");

		//imgprobe(fullimage,true);
		//imgprobe(haze,true);
		//imgprobe(dehazed);

		//DEFECT INSPECTION
		//std::string DSizeCurve;
		//write_log(LogType::Info,"EndChannelProcess",std::format("Using DSizeCurve for channel {}: {}",static_cast<int>(channelID),chsettings[channelID]["DSizeCurve"].dump(4)).c_str());
		json DSizeCurveJson;
		{
			std::lock_guard<std::mutex> lock_settings(mtx_chsettings);
			DSizeCurveJson=chsettings[channelID]["DSizeCurve"]; 
			//std::println("DSizeCurve={}",DSizeCurveJson.dump(4))	;
		} 
		std::vector<double> Intensities,DSizes;
		// Safety check: ensure "CurvePoints" exists and is an array before iterating
		if (DSizeCurveJson.contains("CurvePoints") && DSizeCurveJson["CurvePoints"].is_array()) 
			for(const auto& point:DSizeCurveJson["CurvePoints"])
			{ 
				if (point.contains("Intensity") && point.contains("DSize")) 
				{
					Intensities.push_back(point["Intensity"].get<double>());
					DSizes.push_back(point["DSize"].get<double>());
				}
				else
				{
					write_log(LogType::Warning,"EndChannelProcess","DSizeCurveJson does not contain 'CurvePoints' or it is not an array. Using default values."); 
					Intensities={3000,5000};
					DSizes={0.2,0.5};
				}
			}

		//OUTPUT

		//std::vector<DefectInfoStruct> defects=inspect(dehazed,Intensities,DSizes,pixelsize,channelID);
		std::vector<DefectInfoStruct> defects=inspect(dehazed,Intensities,DSizes,pixelsize,thr_snr,channelID,dirout);
		write_log(LogType::Info,"EndChannelProcess",std::format("Identified {} defects in channel {}",defects.size(),static_cast<int>(channelID)).c_str());
		//return AlgoResult::Success();

		//std::cout<<"Current path right now: "<<std::filesystem::current_path()<<"\n";
		//std::cout<<"Target absolute path: "<<std::filesystem::absolute(dir+"/img.png")<<"\n";
		write_log(LogType::Info,"EndChannelProcess",std::format("Saving images to directory: {}",dirout).c_str());
		if(dirout!="")
		//if(false)
		{
			std::vector<int> outputparams; 
			outputparams.push_back(cv::IMWRITE_PNG_COMPRESSION);
			outputparams.push_back(1);
			outputparams.push_back(cv::IMWRITE_EXR_COMPRESSION);
			outputparams.push_back(0);
			//outputparams.push_back(cv::IMWRITE_JPG_COMPRESSION);
			//outputparams.push_back(0);
			write_log(LogType::Info,"EndChannelProcess","Saving fullmap");
			//cv::imwrite(dirout+std::format("/ch{}_fullmap.exr",static_cast<int>(channelID)),fullimage,outputparams); 
			//save a vertically flipped version of the full image to match the coordinate system used in the inspection
			//saveBigTiff(dirout+std::format("/ch{}_fullmap.tiff",static_cast<int>(channelID)),fullimage);
			auto imwrite_as16bit=[](const std::string& path,const cv::Mat& img32f)
				{
					cv::Mat img;
					img32f.convertTo(img,CV_16UC1);
					cv::imwrite(path,img);
				};
			auto imwrite_vflipped=[](const std::string& path,const cv::Mat& img)
				{
					cv::Mat img_flipped;
					cv::flip(img, img_flipped, 0);
					cv::imwrite(path,img_flipped);
				};
			imwrite_vflipped(dirout+std::format("/ch{}_fullmap.png",static_cast<int>(channelID)),fullimage);
			if(DebugOutput)
			{
				imwrite_vflipped(dirout+std::format("/ch{}_fullmap_4ev.png",static_cast<int>(channelID)),fullimage*16);
				//write_log(LogType::Info,"EndChannelProcess","Saving haze map");
				//imwrite_as16bit(dirout+std::format("/ch{}_haze.png",static_cast<int>(channelID)),haze);
				//cv::imwrite(dirout+std::format("/ch{}_haze.png",static_cast<int>(channelID)),haze);
				//imwrite_as16bit(dirout+std::format("/ch{}_dehazed.png",static_cast<int>(channelID)),dehazed);
				//imwrite_as16bit(dirout+std::format("/ch{}_dehazed_8ev.png",static_cast<int>(channelID)),dehazed*256);
				cv::Mat dehazed_16UC1; 
				dehazed.convertTo(dehazed_16UC1,CV_16UC1); 
				cv::Mat dehazed_annotation=drawmap(dehazed_16UC1,defects,pixelsize);
				//cv::imwrite(dirout+std::format("/ch{}_dehazed_annotated.png",static_cast<int>(channelID)),dehazed_annotation,outputparams);
				//saveBigTiff(dirout+std::format("/ch{}_dehazed_annotated.tiff",static_cast<int>(channelID)),dehazed_annotation);
				write_log(LogType::Info,"EndChannelProcess","Saving dehazed image");
				imwrite_vflipped(dirout+std::format("/ch{}_dehazed_annotated_enhanced.png",static_cast<int>(channelID)),dehazed_annotation*256);
				imwrite_vflipped(dirout+std::format("/ch{}_dehazed_enhanced.png",static_cast<int>(channelID)),dehazed_16UC1*256);
				//dehazed_annotation*=256;
				//saveBigTiff(dirout+std::format("/ch{}_dehazed_annotated_enhanced.tiff",static_cast<int>(channelID)),dehazed_annotation);
				dehazed.release();
				cv::Mat defect_annotation=drawmap(fullimage,defects,pixelsize);
				fullimage.release();
				write_log(LogType::Info,"EndChannelProcess","saving defect annotation on full image");
				//cv::imwrite(dirout+std::format("/ch{}_annotated.png",static_cast<int>(channelID)),defect_annotation,outputparams);
				imwrite_vflipped(dirout+std::format("/ch{}_annotated_4ev.png",static_cast<int>(channelID)),defect_annotation*16);
				//saveBigTiff(dirout+std::format("/ch{}_annotated.tiff",static_cast<int>(channelID)),defect_annotation);
				//defect_annotation*=16;
				//saveBigTiff(dirout+std::format("/ch{}_annotated_4ev.tiff",static_cast<int>(channelID)),defect_annotation);
				imwrite_vflipped(dirout+std::format("/ch{}_annotated_8ev.png",static_cast<int>(channelID)),defect_annotation*256); 
			}

			write_log(LogType::Info,"EndChannelProcess","Finished saving images");
		}
		if(!descriptor)
		{ 
			write_log(LogType::Error,"EndChannelProcess","Descriptor is null, therefore DefectInfoList will not be filled");
			return AlgoResult::Success();
			//return AlgoResult::Failure("descriptor is null");
		} 
		write_log(LogType::Info,"EndChannelProcess","Preparing output data structures");
		size_t numDefects=defects.size(); // For demonstration, we will fill in some dummy defect info 
		write_log(LogType::Info,"EndChannelProcess",std::format("Number of defects: {}", numDefects).c_str());
		descriptor->ParticleCount=static_cast<int>(numDefects);
		descriptor->DataSize=static_cast<int>(numDefects*sizeof(DefectInfoStruct)); 
		if(descriptor->DataSize==0)
			return AlgoResult::Success(); // No defects found, return success with zero count
		write_log(LogType::Info,"EndChannelProcess","Saving mmap");
		if(!save_to_mmap(descriptor->Name,(void*)defects.data(),numDefects*sizeof(DefectInfoStruct)))
		{
			write_log(LogType::Error,"EndChannelProcess","Failed to save mmap");
			return AlgoResult::Failure("Failed to save mmap");
		}
		write_log(LogType::Info,"EndChannelProcess","Finished saving mmap");
		return AlgoResult::Success();
	}
	ALGO_API AlgoResult SaveImage16bit(char* path,void* data_ptr,int width,int height,float scale)
	{ // Save a 16-bit image to disk, scaling the pixel values by 'scale'
		if(!data_ptr)
			return AlgoResult::Failure("SaveImage16bit: data_ptr is null");
		cv::Mat img(height,width,CV_16UC1,data_ptr);
		if(scale!=1.0f)
			img.convertTo(img,CV_16UC1,scale);
		if(!cv::imwrite(path,img))
			return AlgoResult::Failure(std::format("SaveImage16bit: failed to write image to {}",path).c_str());
		return AlgoResult::Success(); 
	}
	ALGO_API AlgoResult SaveImage8bit(char* path,void* data_ptr,int width,int height)
	{
		if(!data_ptr)
			return AlgoResult::Failure("SaveImage8bit: data_ptr is null");
		cv::Mat img(height,width,CV_8UC1,data_ptr);
		if(!cv::imwrite(path,img))
			return AlgoResult::Failure(std::format("SaveImage8bit: failed to write image to {}",path).c_str());
		return AlgoResult::Success();
	}
}