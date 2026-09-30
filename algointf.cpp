#include "algorithm.h"
#include "algocore.h"
#include <nlohmann/json.hpp>
#include "io.h"
import std;
using json = nlohmann::json;

const float twopi=2.0f*static_cast<float>(CV_PI);
const double ppmm_prealign=208.4; // pixels per mm for prealign images, used to convert wafer size in mm to pixels
static LogMessageCallBack g_logCallback=nullptr; 
ContourCalcParameter param_contour;
std::vector <cv::Mat> centerCalculationImages;
std::vector <cv::Point2d> CwaferEstimated;
std::mutex mtx_chsettings;
std::map<DetectChannel, std::map<std::string, json>> chsettings;
const double intensity_normalizer=100.0;
bool DebugOutput=false;
float pixelsize;
struct Ring
{
	int index;
	cv::Mat img; // accumulated image, CV_8U \in [0,255] or CV_16U \in [0,4096]
	//cv::Mat defectmap; // defect map 
	//cv::Mat haze; // 
	cv::Mat dehazed; // dehazed image, CV_32F \in [0,1]
	std::vector<double> column_intensity;
	int cursor=0; // current row index for adding new frames
	float theta0,theta1; // [in radians] starting angle for this ring
	json processSetting, hazeCaliSetting, coordCaliSetting;
	float radius_mm; // radius of this ring in mm 
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
	printf("[%d] %s: %s\n",static_cast<int>(level),source,message);
}
void imgprobe(cv::Mat img, bool continue_flag=false)
{
	double minVal,maxVal;
	cv::minMaxLoc(img,&minVal,&maxVal);
	std::println("Probed image: type={}, height={}, width={}",img.type(),img.rows,img.cols);
	std::println("Range of elements in probed image : [{},{}]",minVal,maxVal);
	if(!continue_flag)
		throw std::runtime_error("Intentional exception for image probing");
}

std::vector<DefectInfoStruct> inspect(const cv::Mat& image,const std::vector<double>& Intensities,const std::vector<double>& DSizes,const float pixelsize,DetectChannel channel)
{
	//CV_Assert(image.type() == CV_16UC1);
	CV_Assert(image.type() == CV_32F);
	// Output verification
	std::cout<<"Intensities: ";
	for(double val:Intensities) std::cout<<val<<" ";
	std::cout<<"\nDSizes: ";
	for(double val:DSizes) std::cout<<val<<" ";
	std::cout<<"\n";
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

	std::vector<DefectInfoStruct> defects;
	//auto intensity=Intensities[1]/2+1.0f;
	const float intensity_threshold=80;
	//auto intensity_next=4096;
	//if(Intensities.size()>2)
	//	intensity_next=Intensities[2];
	//auto dsize=DSizes[1];

	// Binary mask
	//cv::Mat binary=(image>=intensity);//&(image<intensity_next); // yields 8-bit mask (0 or 255) in OpenCV expression context on most builds
	cv::Mat binary=(image>=intensity_threshold);//&(image<intensity_next); // yields 8-bit mask (0 or 255) in OpenCV expression context on most builds
	// Ensure 8-bit 0/255
	if(binary.type()!=CV_8U)
		binary.convertTo(binary,CV_8U,255);
	cv::Mat closure;
	cv::morphologyEx(binary,closure,cv::MORPH_CLOSE,cv::getStructuringElement(cv::MORPH_ELLIPSE,cv::Size(11,11)));
 
	// Connected components with stats
	cv::Mat labels,stats,centroids;
	int nLabels=cv::connectedComponentsWithStats(closure,labels,stats,centroids,8,CV_32S);
	std::println("Integration threshold: {},  Found {} potential regions. Analyzing...",intensity_threshold,nLabels-1);
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
	std::ofstream ofs_detreg("detected_regions.txt");//save a file for statistics of detected regions of column id,area,intg 
	ofs_detreg<<"ID\tArea(pixels)\tintg"<<std::endl;

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
		ofs_detreg<<std::format("{}\t{}\t{}\n",lbl,area,intg);

		//ignore the superficial components that are not part of the binary mask
		cv::Mat componentMask;
		cv::compare(labels(bbox),lbl,componentMask,cv::CMP_EQ); 
		cv::Mat binaryPixelsInComponent;
		cv::bitwise_and(binary(bbox),componentMask,binaryPixelsInComponent); 
		const int binaryPixelCount=cv::countNonZero(binaryPixelsInComponent);
		if(binaryPixelCount==0)
			continue;

		if(intg<0.1f)//should not happen, but just in case
		{
			std::println("Found region with integrated intensity {} below 0.1",intg);
			std::println( "channel={}, label={}: area={}, binary pixels in closure component={}, intg={}",
				static_cast<int>(channel), lbl, area, binaryPixelCount, intg);
			for(int i=top; i<top+height; ++i)
				for(int j=left; j<left+width; ++j)
					if(labels.at<int>(i,j)==lbl)
						//std::println("Pixel ({},{}) intensity {}",j,i,image.at<uchar>(i,j)); 
						std::println(
							"inspect channel={}, label={}: integrated intensity {} below 0.1; pixel ({}, {}) intensity {}",
							static_cast<int>(channel), lbl, intg, j, i, image.at<float>(i,j));
		}

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
		if(defect.CoordR>100000) continue;
		if(intg<Intensities[1])
		{
			//std::println("Skipping region at ({},{}) with integrated intensity {} below threshold {}",defect.CoordX,defect.CoordY,intg,Intensities[1]);
			continue; // skip defects less than threshold the intensity range 
		}
		else
			defects.push_back(defect);
	}
	std::sort(defects.begin(),defects.end(),[](const DefectInfoStruct& a,const DefectInfoStruct& b) { return a.Area>b.Area; }); // descending order
	return defects;
}
cv::Mat drawmap(cv::Mat image,const std::vector<DefectInfoStruct> &defects,const float pixelsize)
{
	write_log(LogType::Info,"drawmap","Drawing defect annotations on image");
	cv::Mat defect_annotation;
	cv::cvtColor(image,defect_annotation,cv::COLOR_GRAY2BGR);
	//int numdraw=0;
	write_log(LogType::Info,"drawmap",std::format("Total defects {}",defects.size()).c_str());
	std::ofstream defect_log("defect_log.txt");
	for(const auto& defect:defects)
	{
		//if(numdraw++>500) break; // limit to the first few defects for drawing
		if(defect.Area<=0) continue; // skip invalid entries
		cv::Point center(static_cast<int>(defect.CoordX/pixelsize+image.cols/2),static_cast<int>(defect.CoordY/pixelsize+image.rows/2));
		cv::Size axes(static_cast<int>(defect.XSize/2/pixelsize)+20,static_cast<int>(defect.YSize/2/pixelsize)+20);
		cv::ellipse(defect_annotation,center,axes,0,0,360,cv::Scalar(0,0,255*256),2); // red ellipse
		defect_log<<std::format("Drawing defect at ({},{})px. R={}um, T={}deg, sizes=({},{})um, area={}px({}um^2), bin={}\n",center.x,center.y,defect.CoordR,defect.CoordT,defect.XSize,defect.YSize,defect.RawPixelsCount,defect.Area,defect.BinCode);

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
extern "C"
{
	AlgoResult SetLogMessageCallBack(LogMessageCallBack callBackPointer)
	{
		g_logCallback=callBackPointer;
		return AlgoResult::Success();
	}
	AlgoResult Initialize()
	{
		centerCalculationImages.clear(); // Clear any previously stored images
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
		write_log(LogType::Info,"Initialize","Algorithm module initialized");
		return AlgoResult::Success();
	}

	AlgoResult BeginCenterCalculation(ContourCalcParameter parameters)
	{
		param_contour=parameters;
		return AlgoResult::Success();
	}
	AlgoResult AddCenterCalculationImage(const unsigned char* imageData,double stageAngle)
	{
		(void)stageAngle;
		cv::Mat img(param_contour.ImageHeight,param_contour.ImageWidth,CV_8UC1,const_cast<unsigned char*>(imageData));
		if(img.empty()||img.type()!=CV_8UC1) {
			write_log(LogType::Error, "AddCenterCalculationImage", "EndCenterCalculation: invalid image");
			return AlgoResult::Failure("Invalid image");
		}

		// Binary image 'imgb' from 'img' with threshold 30
		cv::Mat imgb;
		cv::threshold(img,imgb,30,255,cv::THRESH_BINARY);

		// Find contour of the largest black blob in imgb and save to contour_b
		std::vector<std::pair<int,int>> contour_b;
		{
			// Invert so black regions become white for findContours
			cv::Mat imgb_inv;
			cv::bitwise_not(imgb,imgb_inv);

			std::vector<std::vector<cv::Point>> contours;
			cv::findContours(imgb_inv,contours,cv::RETR_EXTERNAL,cv::CHAIN_APPROX_NONE);

			if(contours.empty())
				write_log(LogType::Warning, "AddCenterCalculationImage", "no contours found");
			else 
			{ // select largest by area
				double maxArea=0.0;
				int maxIdx=-1;
				for(size_t i=0; i<contours.size(); ++i) {
					double a=std::abs(cv::contourArea(contours[i]));
					if(a>maxArea) { maxArea=a; maxIdx=static_cast<int>(i); }
				}
				if(maxIdx>=0) {
					const auto& c=contours[maxIdx];
					contour_b.reserve(c.size());
					for(const auto& pt:c) contour_b.emplace_back(pt.x,pt.y);
					// optional debug print
					//write_log(LogType::Debug, "AddCenterCalculationImage", std::format("largest contour size={}, area={}", contour_b.size(), maxArea).c_str());
				}
			}
		}

		// Collect edge points from contour_b
		std::vector<cv::Point2d> pts;
		pts.reserve(contour_b.size());
		for(const auto& p:contour_b)
		{
			auto x=static_cast<int>(p.first),y=static_cast<int>(p.second);
			if(x==0||y==0||x>=img.cols-1||y>=img.rows-1)
				continue; // skip border points
			pts.emplace_back(static_cast<double>(x),static_cast<double>(y));
		}
		//write_log(LogType::Debug, "AddCenterCalculationImage", std::format("collected {} edge points from contour_b", pts.size()).c_str());

		if(pts.size()<100)
		{
			auto err=std::format("EndCenterCalculation: insufficient edge points {}\n",pts.size());
			return AlgoResult::Failure(err.c_str());
		}

		auto id=centerCalculationImages.size();//define a unique id for this image based on the current size of the vector

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
		double R_wafer_px=param_contour.WaferSize/2*ppmm_prealign;
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
			cv::line(lineOverlay,M,Cwafer,cv::Scalar(255,0,0),2,cv::LINE_AA); //Blue for the radius connecting M and Cwafer 
			cv::line(lineOverlay,p1,p2,cv::Scalar(0,0,255),2,cv::LINE_AA); //Red for fitted line
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
			cv::polylines(lineOverlay,arc_pts,false,cv::Scalar(0,255,128),2,cv::LINE_AA);
			// mark the point M on the line used by fitLine (white circle)
			cv::circle(lineOverlay,M,4,cv::Scalar(255,255,255),-1,cv::LINE_AA);
			// mark Cwafer (red circle)
			//cv::circle(lineOverlay,Cwafer,6,cv::Scalar(0,0,255),-1,cv::LINE_AA);
			//save overlayimage
			cv::imwrite(std::format("{}_fitted_line_overlay.png",id),lineOverlay);

			//double theta=std::rand()%360; // random angle for demonstration
			//Cwafer.x=-10.0+R_wafer_px*std::cos(theta*3.14159265358979323846/180.0);
			//Cwafer.y=-13.0+R_wafer_px*std::sin(theta*3.14159265358979323846/180.0);
			CwaferEstimated.emplace_back(static_cast<double>(Cwafer.x),static_cast<double>(Cwafer.y));
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

		double sumX=0,sumY=0,sumX2=0,sumY2=0,sumXY=0;
		double sumX3=0,sumY3=0,sumX2Y=0,sumXY2=0;
		size_t n=CwaferEstimated.size();

		for(const auto& p:CwaferEstimated) {
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
		if(angle) *angle=0.0; // Angle usually derived from a notch/flat, defaulting to 0

		return AlgoResult::Success();
	}
 
	ALGO_API AlgoResult BeginChannelProcess(DetectChannel channelID,
		const char* ring,const char* cluster,const char* classification,
		const char* coordCaliSetting,const char* DSizeCurve)
	{
		(void)cluster, (void)coordCaliSetting; 
		std::lock_guard<std::mutex> lock(mtx_chsettings);
		chsettings[channelID]["ring"]=json::parse(ring); 
		chsettings[channelID]["classification"]=json::parse(classification); 
		chsettings[channelID]["DSizeCurve"]=json::parse(DSizeCurve); 
		std::lock_guard<std::mutex> lock_imgs(mtx_rings); 
		rings[channelID]=std::vector<Ring>(chsettings[channelID]["ring"]["TotalRings"]);
		//std::println("BeginChannelProcess: channelID={}, ringSetting={}",static_cast<int>(channelID),ringSetting);
		write_log(LogType::Info,"BeginChannelProcess",std::format("channelID={}, ring={}",static_cast<int>(channelID),ring).c_str());
		write_log(LogType::Info,"BeginChannelProcess",std::format("channelID={}, classifcation={}",static_cast<int>(channelID),classification).c_str());
		if(chsettings[channelID]["ring"].find("DebugOutput")!=chsettings[channelID]["ring"].end())
			DebugOutput=chsettings[channelID]["ring"]["DebugOutput"];
		pixelsize=chsettings[channelID]["classification"]["PixelSize"];
		std::println("setting pixelsize as {}",pixelsize);
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
		std::println("BeginRing: radius_mm={}",ring.radius_mm);
		ring.theta0=theta0_angle/180.0f * static_cast<float>(CV_PI); //coverting to radians if needed, but assuming the input is in degrees or radians as required
		while(ring.theta0>twopi) ring.theta0-=twopi;
		ring.theta1=ring.theta0+thetadiff_angle/180.0f * static_cast<float>(CV_PI); 

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
		//if(isOverLoad)*isOverLoad=false; 
		//if(isHazeOverload)*isHazeOverload=false;
		try
		{ 
			cv::Mat rimg;
			write_log(LogType::Info,"EndRingProcess",std::format("Trying to copy ring data...channelID={}, ringIndex={}, W={}, H={}",static_cast<int>(channelID),ringIndex,rimg.cols,rimg.rows).c_str());
			{// lock scope for ring image processing
				std::lock_guard<std::mutex> lock_imgs(mtx_rings);
				rimg=rings[channelID][ringIndex].img.clone(); // or assign without clone if you prefer shared header
				//imgprobe( rimg=rings[channelID][ringIndex].img.clone()); 
				//rings[channelID][ringIndex].img.convertTo(rimg,CV_32FC1); // convert to float for processing
				//cv::flip(rimg.clone(),rimg,0); //flip the image vertically
			}
			//write_log(LogType::Info,"EndRingProcess",std::format("Trying to copy ring data...channelID={}, ringIndex={}, W={}, H={}",static_cast<int>(channelID),ringIndex,rimg.cols,rimg.rows).c_str());
			//imgprobe(rimg,true);
			write_log(LogType::Info,"EndRingProcess",std::format("Estimating column intensity...channelID={}, ringIndex={}, W={}, H={}",static_cast<int>(channelID),ringIndex,rimg.cols,rimg.rows).c_str());
			auto col_intensity=estimate_column_intensity_stat(rimg);
			write_log(LogType::Info,"EndRingProcess",std::format("Dehazing...channelID={}, ringIndex={}, W={}, H={}",static_cast<int>(channelID),ringIndex,rimg.cols,rimg.rows).c_str());
			float r_inner_mm,r_outer_mm;
			{
				std::lock_guard<std::mutex> lock_settings(mtx_chsettings);
				std::lock_guard<std::mutex> lock_imgs(mtx_rings);
				float W=static_cast<float>(rimg.cols);
				float r=rings[channelID][ringIndex].radius_mm;
				int ninside=rings[channelID].size()-ringIndex-1;//number of ring inside of current rings
				r_inner_mm=W*ninside*pixelsize/1000.0f;
				r_outer_mm=W*(ninside+1)*pixelsize/1000.0f;
				//r_inner_mm=r-pixelsize/1000.0f*W/2;
				//r_outer_mm=r+pixelsize/1000.0f*W/2;
				std::println("flatten: r_inner={}, r_outer={}",r_inner_mm,r_outer_mm);
				std::println("flatten: radius={}",r_inner_mm,r_outer_mm);
			}
			cv::Mat flattened=flatten(rimg,r_inner_mm,r_outer_mm,DebugOutput?std::format("test_output/flattenning_{}.txt",ringIndex):"");
			write_log(LogType::Info,"EndRingProcess",std::format("Column intensity calibrated. ChannelID={}, ringIndex={}, W={}, H={}",static_cast<int>(channelID),ringIndex,rimg.cols,rimg.rows).c_str());
			double minval,maxval;
			cv::minMaxLoc(flattened,&minval,&maxval,nullptr,nullptr);
			//std::println("flattened range: min={}, max={}",minval,maxval);
			write_log(LogType::Info,"EndRingProcess",std::format("Data range after calibration: [{},{}]. ChannelID={}, ringIndex={}, W={}, H={}",minval,maxval,static_cast<int>(channelID),ringIndex,rimg.cols,rimg.rows).c_str());


			cv::Mat haze(flattened.size(),flattened.type(),cv::Scalar(1.0)); //haze: background comes frome scattering of laser by the roughness of the wafer surface
			auto haze_col_intensity=estimate_column_intensity_stat(flattened);
			for(int x=0; x<haze.cols; ++x)
				haze.col(x)=haze_col_intensity[x];

			// Minimum filter: each haze pixel is the minimum in its nnbh × nnbh neighborhood.
			//const int nnbh=31; // neighborhood size for min filter, must be odd and > 1
			//cv::erode( flattened, haze, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(nnbh, nnbh)), cv::Point(-1, -1), 1, cv::BORDER_REPLICATE);
			//{ // Apply a median filter vertically within each column only.
			//	constexpr int verticalMedianKernel=31; // Must be odd and > 1.  
			//	#pragma omp parallel for
			//	for(int x=0; x<flattened.cols; ++x)
			//		cv::medianBlur(flattened.col(x),haze.col(x),verticalMedianKernel);
			//}
			cv::Mat dehazed;
			//haze.convertTo(haze_float,CV_32FC1);	
			cv::max(0,flattened-haze,dehazed);
			//imgprobe(rimg,true);
			//imgprobe(flattened,true);
			//imgprobe(haze,true);
			//imgprobe(dehazed);

			//update the ring structure with the processed images
			{// Process the ring image to detect defects and populate the defect map
				std::lock_guard<std::mutex> lock_imgs(mtx_rings);
				//rings[channelID][ringIndex].defectmap=dehazed.clone(); // For demonstration, copy the ring image to defect map 
				//rings[channelID][ringIndex].haze=haze.clone();
				rings[channelID][ringIndex].dehazed=dehazed.clone();
				rings[channelID][ringIndex].column_intensity=col_intensity;
			}
			write_log(LogType::Info,"EndRingProcess",std::format("Dehazing completed for channelID={}, ringIndex={}",static_cast<int>(channelID),ringIndex).c_str());
			double min_val,max_val,mean_val;
			const auto [min_it,max_it]=std::minmax_element(col_intensity.begin(),col_intensity.end());
			min_val=*min_it;
			max_val=*max_it;
			mean_val=std::accumulate(col_intensity.begin(),col_intensity.end(),0.0)/col_intensity.size();
			write_log(LogType::Info,"EndRingProcess",std::format("Ring intensity before calibration: min={},max={},mean={}",min_val,max_val,mean_val).c_str());

			//save data to disk 
			std::string dir;
			{
				std::lock_guard<std::mutex> gd(mtx_chsettings);
				//std::cout<<chsettings[channelID]["ring"];
				dir=chsettings[channelID]["ring"]["ImageSaveDirectory"];
			}
			if(dir!="")
			{
				//auto fn=dir+std::format("/ch{}r{}.png",static_cast<int>(channelID),ringIndex);
				//cv::imwrite(fn,rimg);
				//write_log(LogType::Info,"EndRingProcess",std::format("Images saved for channelID={}, ringIndex={}, image size={}x{}",static_cast<int>(channelID),ringIndex,rimg.cols,rimg.rows).c_str());
				////auto fn_dehazed=dir+std::format("/ch{}r{}_dehazed.png",static_cast<int>(channelID),ringIndex);
				////cv::imwrite(fn_dehazed,dehazed); 
				//flatten(rimg,std::format("{}/flattenning_{}.txt",dir,ringIndex));
			}
			return AlgoResult::Success();
		}
		catch(const std::exception& e)
		{
			write_log(LogType::Error,"EndRingProcess",std::format("Exception (channel {}, ring {}): {}",static_cast<int>(channelID),ringIndex,e.what()).c_str());
			return AlgoResult::Failure(e.what());
		}
	} 

	ALGO_API AlgoResult EndChannelProcess(DetectChannel channelID,DefectInfoListStruct* descriptor)
	{ 
		//PARAMETER EXTRACTION 
		std::vector<Ring> rings_copy;
		std::vector<int> ringWidths;
		{
			std::lock_guard<std::mutex> lock_imgs(mtx_rings);
			for(const auto& r:rings[channelID])
			{
				rings_copy.push_back(r); 
				//rings_copy.back().defectmap=r.defectmap.clone(); // deep copy of the defect map
				ringWidths.push_back(r.img.cols); // store the width of each ring image 
			}
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
		int N=0;// number of rings
		{
			std::lock_guard<std::mutex> lock_settings(mtx_chsettings);
			N=chsettings[channelID]["ring"]["TotalRings"]; 
			//pixelsize=chsettings[channelID]["classification"]["PixelSize"]; 
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
					std::println("Out of bounds: ringIndex={}, row={}, col={}, rows={}, cols={}",idxr,row,col,rows,cols);
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
		{
			std::lock_guard<std::mutex> lock_imgs(mtx_rings);
			rings[channelID].clear(); // Clear the ring images for this channel to free memory 
		}
		write_log(LogType::Info,"EndChannelProcess","Input frames cleared");

		//imgprobe(fullimage,true);
		//imgprobe(haze,true);
		//imgprobe(dehazed);

		//DEFECT INSPECTION
		//std::string DSizeCurve;
		write_log(LogType::Info,"EndChannelProcess",std::format("DSizeCurve for channel {}: {}",static_cast<int>(channelID),chsettings[channelID]["DSizeCurve"].dump(4)).c_str());
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
		//std::vector<DefectInfoStruct> defects=inspect(dehazed,Intensities,DSizes,pixelsize,channelID);
		std::vector<DefectInfoStruct> defects=inspect(dehazed,Intensities,DSizes,pixelsize,channelID);
		write_log(LogType::Info,"EndChannelProcess",std::format("Identified {} defects in channel {}",defects.size(),static_cast<int>(channelID)).c_str());

		//OUTPUT
		std::string dir; 
		{
			std::lock_guard<std::mutex> gd(mtx_chsettings); 
			//std::cout<<chsettings[channelID]["ring"]<<std::endl;
			dir=chsettings[channelID]["ring"]["ImageSaveDirectory"];
			//std::println("saving to {}",dir);
		} 

		//std::cout<<"Current path right now: "<<std::filesystem::current_path()<<"\n";
		//std::cout<<"Target absolute path: "<<std::filesystem::absolute(dir+"/img.png")<<"\n";
		write_log(LogType::Info,"EndChannelProcess",std::format("Saving images to directory: {}",dir).c_str());
		if(dir!="")
		{
			std::vector<int> outputparams; 
			outputparams.push_back(cv::IMWRITE_PNG_COMPRESSION);
			outputparams.push_back(1);
			write_log(LogType::Info,"EndChannelProcess","Saving fullmap");
			cv::imwrite(dir+std::format("/ch{}_fullmap.png",static_cast<int>(channelID)),fullimage,outputparams); 
			auto imwrite_as16bit=[](const std::string& path,const cv::Mat& img32f)
				{
					cv::Mat img;
					img32f.convertTo(img,CV_16UC1);
					cv::imwrite(path,img);
				};
			if(DebugOutput)
			{
				//write_log(LogType::Info,"EndChannelProcess","Saving haze map");
				//imwrite_as16bit(dir+std::format("/ch{}_haze.png",static_cast<int>(channelID)),haze);
				//cv::imwrite(dir+std::format("/ch{}_haze.png",static_cast<int>(channelID)),haze);
				write_log(LogType::Info,"EndChannelProcess","Saving dehazed image");
				//imwrite_as16bit(dir+std::format("/ch{}_dehazed.png",static_cast<int>(channelID)),dehazed);
				//imwrite_as16bit(dir+std::format("/ch{}_dehazed_8ev.png",static_cast<int>(channelID)),dehazed*256);
				cv::Mat dehazed_16UC1; 
				dehazed.convertTo(dehazed_16UC1,CV_16UC1); 
				cv::Mat dehazed_annotation=drawmap(dehazed_16UC1,defects,pixelsize);
				cv::imwrite(dir+std::format("/ch{}_dehazed_annotated.png",static_cast<int>(channelID)),dehazed_annotation,outputparams);
				cv::imwrite(dir+std::format("/ch{}_dehazed_annotated_8ev.png",static_cast<int>(channelID)),dehazed_annotation*256,outputparams);
				dehazed.release();
				write_log(LogType::Info,"EndChannelProcess","drawing defect annotation");
				//cv::Mat defect_annotation=drawmap(dehazed,defects,pixelsize);
				cv::Mat defect_annotation=drawmap(fullimage,defects,pixelsize);
				fullimage.release();
				write_log(LogType::Info,"EndChannelProcess","saving defect annotation on full image");
				cv::imwrite(dir+std::format("/ch{}_annotated.png",static_cast<int>(channelID)),defect_annotation,outputparams);
				cv::imwrite(dir+std::format("/ch{}_annotated_4ev.png",static_cast<int>(channelID)),defect_annotation*16,outputparams);
				//cv::imwrite(dir+std::format("/ch{}_annotated_8ev.png",static_cast<int>(channelID)),defect_annotation*256,outputparams); 
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
		write_log(LogType::Info,"EndChannelProcess","Saving mmap");
		save_to_mmap(descriptor->Name,(void*)defects.data(),numDefects*sizeof(DefectInfoStruct));
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
