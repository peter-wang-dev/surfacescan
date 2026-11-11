#include "algorithm.h"
#include "algocore.h"
import std;

namespace
{
	ContourCalcParameter param_contour;
	std::vector<cv::Mat> centerCalculationImages,contourCalculationImages;
	std::vector<float> centerCalculationAngles,contourCalculationAngles;
	std::vector<cv::Point2d> CwaferEstimated;
}

void write_log(LogType level,const char* source,const char* message);

void reset_calibration_state()
{
	centerCalculationImages.clear();
	centerCalculationAngles.clear();
	contourCalculationImages.clear();
	contourCalculationAngles.clear();
	CwaferEstimated.clear();
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
