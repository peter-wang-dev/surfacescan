#include "algorithm.h"
#include <opencv2/opencv.hpp>
#include <nlohmann/json.hpp>
import std;
using json = nlohmann::json;

//typedef void(*CallBack)(int level, const char* message);
static LogMessageCallBack g_logCallback = nullptr;
ContourCalcParameter param_contour;
std::vector <cv::Mat> centerCalculationImages;
std::vector <cv::Point2d> CwaferEstimated;
std::mutex mtx_chsettings;
std::map<DetectChannel, std::map<std::string, json>> chsettings;
struct Ring
{
	int index;
	cv::Mat img; // accumulated images 
	cv::Mat defectmap; // defect map 
	int cursor=0; // current row index for adding new frames
	float theta0,theta1; // starting angle for this ring
	json processSetting, hazeCaliSetting, coordCaliSetting;
};
std::mutex mtx_rings;
std::map<DetectChannel,std::vector<Ring>> rings; //data structure to hold rings for each channel
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
			printf("EndCenterCalculation: invalid image\n");
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
				printf("AddCenterCalculationImage: no contours found\n");
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
					//printf("AddCenterCalculationImage: largest contour size=%zu, area=%f\n", contour_b.size(), maxArea);
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
		//std::println("AddCenterCalculationImage: collected {} edge points from contour_b",pts.size());

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
			printf("EndCenterCalculation: circle fit failed\n");
			return AlgoResult::Success();
		}
		double A = sol(0, 0), B = sol(1, 0), C = sol(2, 0);
		double xc = -A / 2.0, yc = -B / 2.0, r2 = xc * xc + yc * yc - C;
		double r = (r2 > 0.0) ? std::sqrt(r2) : 0.0;
		printf("Fitted circle: center=(%f, %f), radius=%f\n", xc, yc, r);
		cv::Mat imgWithCircle = img.clone();
		cv::circle(imgWithCircle, cv::Point(static_cast<int>(xc), static_cast<int>(yc)), static_cast<int>(r), cv::Scalar(255), 2);
		cv::imwrite(std::format("{}_fitted_circle.png", id), imgWithCircle);
		*/

		//fit a line to the edge points using cv::fitLine
		cv::Vec4f lineParams;
		cv::fitLine(pts,lineParams,cv::DIST_L2,0,0.01,0.01);
		double R_wafer_px=param_contour.WaferSize/2*208.4;
		//R_wafer_px=10000;

		// draw the fitted line on a new overlay image
		{
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
			cv::Point M(cvRound(x0),cvRound(y0));

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
				nx=-nx; ny=-ny;
			}

			// Find Cwafer, the wafer center, using R_wafer_px
			cv::Point Cwafer(cvRound(x0+nx*R_wafer_px),cvRound(y0+ny*R_wafer_px));
			//Draw the radius connecting M and Cwafer on the overlay image (blue)
			cv::line(lineOverlay,M,Cwafer,cv::Scalar(255,0,0),2,cv::LINE_AA);
			cv::line(lineOverlay,p1,p2,cv::Scalar(0,0,255),2,cv::LINE_AA); // draw fitted line (red) 
			//Draw the circle on the overlay image (yellow)
			//cv::circle(lineOverlay, Cwafer, cvRound(R_wafer_px), cv::Scalar(0, 255, 128), 2, cv::LINE_8);//red
			std::vector<cv::Point> arc_pts;
			// The angle from the center Cwafer pointing strictly back towards M
			double start_angle=std::atan2(-ny,-nx);
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
			cv::circle(lineOverlay,Cwafer,6,cv::Scalar(0,0,255),-1,cv::LINE_AA);
			//save overlayimage
			cv::imwrite(std::format("{}_fitted_line_overlay.png",id),lineOverlay);

			//double theta=std::rand()%360; // random angle for demonstration
			//Cwafer.x=-10.0+R_wafer_px*std::cos(theta*3.14159265358979323846/180.0);
			//Cwafer.y=-13.0+R_wafer_px*std::sin(theta*3.14159265358979323846/180.0);
			CwaferEstimated.emplace_back(static_cast<double>(Cwafer.x),static_cast<double>(Cwafer.y));
		}

		// Save processed contour points to an overlay image for visualization
		std::vector<cv::Point> cpts;
		cpts.reserve(pts.size());
		for(const auto& p:pts) cpts.emplace_back(static_cast<int>(p.x),static_cast<int>(p.y));

		// overlay adjacent-point lines on original image (red)
		std::vector<std::vector<cv::Point>> drawVec{cpts};
		cv::Mat overlay;
		cv::cvtColor(img,overlay,cv::COLOR_GRAY2BGR);
		cv::polylines(overlay,drawVec,false,cv::Scalar(0,0,255),2,cv::LINE_AA);
		cv::imwrite(std::format("{}_contour_overlay.png",id),overlay);

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
		const char* ringSetting,const char* clusterSetting,const char* classifySetting,
		const char* coordCaliSetting,const char* DSizeCurve)
	{
		(void)clusterSetting,(void)classifySetting; (void)coordCaliSetting; (void)DSizeCurve;
		std::lock_guard<std::mutex> lock(mtx_chsettings);
		chsettings[channelID]["ring"]=json::parse(ringSetting); // Store the coordCaliSetting for this channel
		std::lock_guard<std::mutex> lock_imgs(mtx_rings); 
		rings[channelID]=std::vector<Ring>(chsettings[channelID]["ring"]["TotalRings"]);
		std::println("BeginChannelProcess: channelID={}, ringSetting={}",static_cast<int>(channelID),ringSetting);
		return AlgoResult::Success();
	}
	ALGO_API AlgoResult BeginRingProcess(DetectChannel channelID,int ringIndex,
		const char* processSetting,const char* hazeCaliSetting,const char* coordCaliSetting)
	{
		(void)channelID; (void)ringIndex; (void)processSetting; (void)hazeCaliSetting; 
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
		ring.img = cv::Mat(RingHeight, RingWidth, CV_8UC1);
		ring.cursor = 0;
		float theta0_angle=coordCaliJson["TriggeredStart"]["T"];
		float theta1_angle=coordCaliJson["TriggeredEnd"]["T"];
		float thetadiff_angle=theta1_angle-theta0_angle;
		ring.theta0=theta0_angle/180.0f * static_cast<float>(CV_PI); //coverting to radians if needed, but assuming the input is in degrees or radians as required
		ring.theta1=ring.theta0+thetadiff_angle/180.0f * static_cast<float>(CV_PI); 
		//std::println("BeginRingProcess: ringIndex={}, theta0={} rad, theta1={} rad",ringIndex,ring.theta0,ring.theta1);
		return AlgoResult::Success();
	}

	ALGO_API AlgoResult AddRingProcessFrame(DetectChannel channelID,int ringIndex,
		const unsigned char* imageData,int width,int height,int validRows,
		int timestamp,int softBinning)
	{
		(void)height;(void)timestamp; (void)softBinning;
		std::lock_guard<std::mutex> lock_imgs(mtx_rings); 

		size_t dataSize = static_cast<size_t>(width) * validRows;
		cv::Mat &dst = rings[channelID][ringIndex].img;
		size_t offset = static_cast<size_t>(rings[channelID][ringIndex].cursor) * static_cast<size_t>(width);
		if (offset + dataSize > static_cast<size_t>(dst.rows) * static_cast<size_t>(dst.cols))
			return AlgoResult::Failure("AddRingProcessFrame: data exceeds allocated ring image size");
		std::memcpy(dst.data + offset, imageData, dataSize);
		rings[channelID][ringIndex].cursor += validRows;
		return AlgoResult::Success();
	}

	ALGO_API AlgoResult EndRingProcess(DetectChannel channelID,int ringIndex,
		bool* isOverLoad,bool* isHazeOverload)
	{
		(void)isOverLoad; (void)isHazeOverload;
		cv::Mat rimg;
		{// lock scope for ring image processing
			std::lock_guard<std::mutex> lock_imgs(mtx_rings);
			rimg = rings[channelID][ringIndex].img.clone(); // or assign without clone if you prefer shared header
		}
		cv::Mat defectmap = rimg.clone(); // For demonstration, copy the ring image to defect map
		{// Process the ring image to detect defects and populate the defect map
			std::lock_guard<std::mutex> lock_imgs(mtx_rings);
			rings[channelID][ringIndex].defectmap=defectmap.clone(); // For demonstration, copy the ring image to defect map
		}
		auto fn=std::format("ch{}r{}.png",static_cast<int>(channelID),ringIndex);
		cv::imwrite(fn,rimg);
		auto fn_defect=std::format("ch{}r{}_defect.png",static_cast<int>(channelID),ringIndex);
		cv::imwrite(fn_defect,defectmap);
		return AlgoResult::Success();
	}

	ALGO_API AlgoResult EndChannelProcess(DetectChannel channelID,DefectInfoListStruct* descriptor)
	{
		(void)descriptor;
		std::vector<Ring> rings_copy;
		std::vector<int> ringWidths;
		{
			std::lock_guard<std::mutex> lock_imgs(mtx_rings);
			for(const auto& r:rings[channelID])
			{
				rings_copy.push_back(r); 
				rings_copy.back().defectmap=r.defectmap.clone(); // deep copy of the defect map
				ringWidths.push_back(r.defectmap.cols); // store the width of each ring image 
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

		int W=2*std::accumulate(ringWidths.begin(),ringWidths.end(),0);
		int H=W;
		std::lock_guard <std::mutex> lock_settings(mtx_chsettings);
		//float pixelSize=chsettings[channelID]["PixelSize"];
		int N=chsettings[channelID]["ring"]["TotalRings"];
		//float Wr=chsettings[channelID]["ring"]["RingWidth"];
		cv::Mat mergedImage(H,W,CV_8UC1,cv::Scalar(0));
		//Now we find the pixel value for each pixel in the mergedImage by mapping it to the corresponding ring image
		//The first ring image corresponds to the outermost ring. In each ring image, the first column corresponds to the angle 0, and the last column corresponds to the angle 2*pi. The first row corresponds to the outer edge of the ring, and the last row corresponds to the inner edge of the ring.
		//The the first row of each ring image corresponds to theta=0, and the last row corresponds to theta=2*pi. 
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
				const float theta1=rings_copy[idxr].theta1;
				while(theta<theta0)
					theta+=2.0f*static_cast<float>(CV_PI);

				int rows=rings_copy[idxr].defectmap.rows;
				int cols=rings_copy[idxr].defectmap.cols;

				int row=static_cast<int>(((theta-theta0)/(theta1-theta0))*(rows-1));
				float r_in_ring=ringStart[idxr]-r;
				int col=static_cast<int>((r_in_ring/ringWidths[idxr])*(cols-1));

				row=std::clamp(row,0,rows-1);
				col=std::clamp(col,0,cols-1);

				mergedImage.at<uchar>(i,j)=rings_copy[idxr].defectmap.at<uchar>(row,col);
			}
		}
		cv::imwrite(std::format("ch{}_defect.png",static_cast<int>(channelID)),mergedImage);
		return AlgoResult::Success();
	}
}
