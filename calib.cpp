#include "algorithm.h"
#include "algocore.h"
#include "calibcore.h"
import std;

void write_log(LogType level,const char* source,const char* message);

namespace
{
	ContourCalcParameter param_contour;
	constexpr double kPi=3.14159265358979323846;
	std::vector<cv::Mat> centerCalculationImages;
	std::vector<float> centerCalculationAngles;
	std::vector<cv::Point2d> CwaferEstimated;

	struct ContourState
	{
		ContourCalcParameter parameters{};
		std::vector<cv::Mat> images;
		std::vector<double> angles;
		std::array<double,9> camera_matrix{};
		std::vector<double> distortion;
		cv::Point2d stage_center_px{};
		cv::Point2d wafer_center_px{};
		bool has_camera=false;
		bool ready=false;
	} contourState;

	struct CircleFit
	{
		double cx=0.0,cy=0.0,radius=0.0;
		int inliers=0;
		bool valid=false;
	};

	void reset_contour_state()
	{
		contourState.images.clear();
		contourState.angles.clear();
		contourState.distortion.clear();
		contourState.parameters={};
		contourState.stage_center_px={};
		contourState.wafer_center_px={};
		contourState.has_camera=false;
		contourState.ready=false;
	}

	void bridge_log(const char* message)
	{
		write_log(LogType::Info,"joint_solve_v16",message);
	}
}

void reset_calibration_state()
{
	centerCalculationImages.clear();
	centerCalculationAngles.clear();
	CwaferEstimated.clear();
	reset_contour_state();
}

AlgoResult BeginCenterCalculation(ContourCalcParameter parameters)
{
	param_contour=parameters;
	return AlgoResult::Success();
}
AlgoResult BeginContourCalculation(ContourCalcParameter parameters)
{
	reset_contour_state();
	if(parameters.ImageWidth<=0||parameters.ImageHeight<=0)
		return AlgoResult::Failure("Invalid image dimensions");
	if(!std::isfinite(parameters.PixelSize)||parameters.PixelSize<=0.0)
		return AlgoResult::Failure("PixelSize must be > 0 um/pixel");
	if(!std::isfinite(parameters.WaferSize)||parameters.WaferSize<=0.0)
		return AlgoResult::Failure("WaferSize must be > 0 mm");
	const double nominal_radius=parameters.WaferSize*500.0/parameters.PixelSize;
	if(!std::isfinite(nominal_radius)||nominal_radius<=0.0)
		return AlgoResult::Failure("Nominal radius is outside the supported numeric range");
	if(!std::isfinite(parameters.FeatureAngle))
		return AlgoResult::Failure("FeatureAngle must be finite");
	if(parameters.RoundnessHarmonics<0||parameters.RoundnessHarmonics>3)
		return AlgoResult::Failure("RoundnessHarmonics must be 0..3");
	if(!std::isfinite(parameters.FixedRadiusPx)||parameters.FixedRadiusPx<0.0)
		return AlgoResult::Failure("FixedRadiusPx must be >= 0");
	if(parameters.FrameWeightMode<ContourFrameWeightMode::None||parameters.FrameWeightMode>ContourFrameWeightMode::Density)
		return AlgoResult::Failure("FrameWeightMode must be 0..2");
	if(!std::isfinite(parameters.FrameWeightWindowDeg))
		return AlgoResult::Failure("FrameWeightWindowDeg must be finite");
	if(!std::isfinite(parameters.RadiusTolRatio)||parameters.RadiusTolRatio<0.0)
		return AlgoResult::Failure("RadiusTolRatio must be >= 0");
	if(!std::isfinite(parameters.SubpixelSearchPx)||!std::isfinite(parameters.SubpixelStepPx))
		return AlgoResult::Failure("Subpixel search and step must be finite");
	const double search=parameters.SubpixelSearchPx>0.0?parameters.SubpixelSearchPx:1.5;
	const double step=parameters.SubpixelStepPx>0.0?parameters.SubpixelStepPx:0.25;
	if(parameters.EnableSubpixel&&search/step>static_cast<double>((std::numeric_limits<int>::max()-1)/2))
		return AlgoResult::Failure("Subpixel search has too many samples");
	if(parameters.DistCoeffsCount<0||(!parameters.DistCoeffs&&parameters.DistCoeffsCount>0))
		return AlgoResult::Failure("Invalid distortion coefficient buffer");
	if(parameters.DistCoeffsCount!=0&&parameters.DistCoeffsCount!=4&&parameters.DistCoeffsCount!=5&&
		parameters.DistCoeffsCount!=8&&parameters.DistCoeffsCount!=12&&parameters.DistCoeffsCount!=14)
		return AlgoResult::Failure("DistCoeffsCount must be 0, 4, 5, 8, 12, or 14");
	if(parameters.CameraMatrix)
	{
		for(size_t i=0;i<9;++i)
			if(!std::isfinite(parameters.CameraMatrix[i])) return AlgoResult::Failure("CameraMatrix must be finite");
		if(parameters.CameraMatrix[0]<=0.0||parameters.CameraMatrix[4]<=0.0)
			return AlgoResult::Failure("CameraMatrix must have positive focal lengths");
	}
	for(int i=0;i<parameters.DistCoeffsCount;++i)
		if(!std::isfinite(parameters.DistCoeffs[i])) return AlgoResult::Failure("DistCoeffs must be finite");
	try
	{
		contourState.parameters=parameters;
		contourState.has_camera=parameters.CameraMatrix!=nullptr;
		if(contourState.has_camera) std::copy_n(parameters.CameraMatrix,9,contourState.camera_matrix.begin());
		if(parameters.DistCoeffsCount>0)
			contourState.distortion.assign(parameters.DistCoeffs,parameters.DistCoeffs+parameters.DistCoeffsCount);
		contourState.parameters.CameraMatrix=nullptr;
		contourState.parameters.DistCoeffs=nullptr;
		contourState.ready=true;
		write_log(LogType::Info,"BeginContourCalculation",std::format(
			"Begin(V16) image={}x{} threshold={} pixelSize={} waferSize={} fixedRadiusPx={} harmonics={} subpixel={} frameWeight={}",
			parameters.ImageWidth,parameters.ImageHeight,parameters.Threshold,parameters.PixelSize,parameters.WaferSize,
			parameters.FixedRadiusPx,parameters.RoundnessHarmonics,parameters.EnableSubpixel,
			static_cast<int>(parameters.FrameWeightMode)).c_str());
		return AlgoResult::Success();
	}
	catch(const std::exception& error)
	{
		reset_contour_state();
		return AlgoResult::Failure(error.what());
	}
}
AlgoResult AddContourCalculationImage(const unsigned char* imageData,double stageAngle)
{
	if(!contourState.ready) return AlgoResult::Failure("Call BeginContourCalculation before adding images");
	if(!imageData) return AlgoResult::Failure("Image data is null");
	if(!std::isfinite(stageAngle)) return AlgoResult::Failure("Stage angle is not finite");
	try
	{
		const auto& parameters=contourState.parameters;
		cv::Mat view(parameters.ImageHeight,parameters.ImageWidth,CV_8UC1,const_cast<unsigned char*>(imageData));
		cv::Mat image=view.clone();
		if(contourState.has_camera&&!contourState.distortion.empty())
		{
			cv::Mat camera(3,3,CV_64F,contourState.camera_matrix.data());
			cv::Mat distortion(1,static_cast<int>(contourState.distortion.size()),CV_64F,contourState.distortion.data());
			try
			{
				cv::Mat undistorted;
				cv::undistort(image,undistorted,camera,distortion,camera);
				image=undistorted;
			}
			catch(const cv::Exception& error)
			{
				write_log(LogType::Warn,"AddContourCalculationImage",std::format("undistort skipped: {}",error.what()).c_str());
			}
		}
		contourState.images.push_back(std::move(image));
		try { contourState.angles.push_back(stageAngle); }
		catch(...) { contourState.images.pop_back(); throw; }
		write_log(LogType::Info,"AddContourCalculationImage",std::format("Added frame {}, {}x{}, stageAngle={} deg",
			contourState.images.size(),parameters.ImageWidth,parameters.ImageHeight,stageAngle).c_str());
		return AlgoResult::Success();
	}
	catch(const std::exception& error)
	{
		return AlgoResult::Failure(error.what());
	}
}
AlgoResult EndContourCalculation(double* transX,double* transY,double* angle)
{
	if(!transX||!transY||!angle) return AlgoResult::Failure("Output pointer is null");
	*transX=*transY=*angle=0.0;
	if(!contourState.ready) return AlgoResult::Failure("Call BeginContourCalculation before ending calculation");
	auto log=[](LogType level,const std::string& message) { write_log(level,"EndContourCalculation",message.c_str()); };
	try
	{
		const auto& parameters=contourState.parameters;
		const size_t count=contourState.images.size();
		log(LogType::Info,std::format("End(V16) frames={} image={}x{}",count,parameters.ImageWidth,parameters.ImageHeight));
		if(count<3) return AlgoResult::Failure("At least 3 frames are required");
		const double pixel_size=parameters.PixelSize;
		const double nominal_radius=parameters.WaferSize*500.0/pixel_size;
		const int threshold=parameters.Threshold>0?parameters.Threshold:30;
		std::vector<std::vector<cv::Point2d>> all_edges(count);
		std::vector<CircleFit> fits(count);
		std::vector<unsigned char> notch_found(count,0);
		std::vector<cv::Point2d> valleys(count),notch_centers(count);
		for(size_t i=0;i<count;++i)
		{
			auto edges=calibration::get_edge_points(contourState.images[i],threshold);
			if(edges.size()>=100)
			{
				auto& fit=fits[i];
				fit.inliers=calibration::ransac_fit_circle(edges,fit.cx,fit.cy,fit.radius,1800,3.0);
				fit.valid=fit.inliers>0&&std::isfinite(fit.cx)&&std::isfinite(fit.cy)&&
					std::isfinite(fit.radius)&&fit.radius>0.0;
			}
			const auto& fit=fits[i];
			if(parameters.EnableSubpixel&&fit.valid&&fit.inliers>=50)
				edges=calibration::refine_edges_subpixel(contourState.images[i],edges,fit.cx,fit.cy,
					parameters.SubpixelSearchPx,parameters.SubpixelStepPx);
			all_edges[i]=std::move(edges);
			double smooth_depth=0.0;
			if(calibration::find_notch_with_fit(all_edges[i],contourState.images[i].size(),fit.cx,fit.cy,
				fit.radius,fit.inliers,valleys[i],notch_centers[i],smooth_depth)) notch_found[i]=1;
			log(LogType::Info,std::format("[Frame {}] angle={} edges={} notch={}",
				i,contourState.angles[i],all_edges[i].size(),notch_found[i]));
		}

		double radius=parameters.FixedRadiusPx;
		if(radius>0.0) log(LogType::Info,std::format("radius: use calibrated FixedRadiusPx={}px",radius));
		else
		{
			std::vector<double> radii;
			for(const auto& fit:fits)
				if(fit.valid&&fit.inliers>=50&&fit.radius>0.0&&std::isfinite(fit.radius)) radii.push_back(fit.radius);
			if(radii.empty()) return AlgoResult::Failure("Radius calibration failed: no valid circle");
			const double median=calibration::median_value(radii);
			std::vector<double> kept;
			for(double value:radii)
				if(std::abs(value-median)<=parameters.RadiusTolRatio*median) kept.push_back(value);
			radius=kept.empty()?median:calibration::median_value(kept);
			log(LogType::Info,std::format("radius: auto-calibrated R0={}px (nominal={}, n={}/{})",
				radius,nominal_radius,kept.size(),radii.size()));
		}
		if(!std::isfinite(radius)||radius<=0.0) return AlgoResult::Failure("Invalid fixed radius");

		size_t notch_index=count;
		cv::Point2d notch_valley;
		double best_depth=-1.0;
		for(size_t i=0;i<count;++i)
		{
			if(!notch_found[i]) continue;
			std::vector<double> radii;
			radii.reserve(all_edges[i].size());
			for(const auto& point:all_edges[i])
				radii.push_back(std::hypot(point.x-notch_centers[i].x,point.y-notch_centers[i].y));
			const double depth=calibration::median_value(radii)-
				std::hypot(valleys[i].x-notch_centers[i].x,valleys[i].y-notch_centers[i].y);
			log(LogType::Info,std::format("[Frame {}] NOTCH candidate=({},{}) center=({},{}) depth={}",
				i,valleys[i].x,valleys[i].y,notch_centers[i].x,notch_centers[i].y,depth));
			if(depth>best_depth) { best_depth=depth; notch_index=i; notch_valley=valleys[i]; }
		}
		if(notch_index==count) return AlgoResult::Failure("Notch not found; check diagnostics");
		log(LogType::Info,std::format("selected notch frame={} depth={} point=({},{})",
			notch_index,best_depth,notch_valley.x,notch_valley.y));
		std::vector<double> coverage;
		for(double value:contourState.angles) coverage.push_back(calibration::normalize_angle_360(value));
		std::sort(coverage.begin(),coverage.end());
		log(LogType::Info,std::format("coverage: frames={} angular_span={}deg",count,coverage.back()-coverage.front()));

		std::vector<calibration::FrameData> frames;
		std::vector<size_t> source;
		for(size_t i=0;i<count;++i)
		{
			auto edges=all_edges[i];
			if(edges.size()<50) continue;
			if(i==notch_index) edges=calibration::remove_notch_contour_segment(edges,notch_valley,6.0,20.0);
			if(edges.size()<30)
			{
				log(LogType::Info,std::format("[Frame {}] skipped after notch filtering: edge_count={}",i,edges.size()));
				continue;
			}
			frames.push_back({contourState.angles[i],std::move(edges),i==notch_index});
			source.push_back(i);
		}
		if(frames.size()<3) return AlgoResult::Failure("Not enough valid frames after edge/notch filtering");
		std::vector<size_t> order(frames.size());
		std::iota(order.begin(),order.end(),size_t{0});
		std::stable_sort(order.begin(),order.end(),[&](size_t a,size_t b) {
			const double aa=calibration::normalize_angle_360(frames[a].angle);
			const double bb=calibration::normalize_angle_360(frames[b].angle);
			return aa!=bb?aa<bb:source[a]<source[b];
		});
		std::vector<calibration::FrameData> sorted_frames;
		std::vector<size_t> sorted_source;
		for(size_t i:order) { sorted_frames.push_back(std::move(frames[i])); sorted_source.push_back(source[i]); }
		frames.swap(sorted_frames);
		source.swap(sorted_source);
		const double theta_ref=calibration::normalize_angle_360(frames.front().angle);
		std::vector<cv::Point2d> initial_centers(frames.size());
		std::vector<unsigned char> initial_valid(frames.size(),0);
		for(size_t i=0;i<frames.size();++i)
		{
			const auto& fit=fits[source[i]];
			if(fit.valid&&fit.inliers>=50&&fit.radius>0.0)
			{
				initial_centers[i]=cv::Point2d(fit.cx,fit.cy);
				initial_valid[i]=1;
			}
		}
		double ox=0.0,oy=0.0,dx=0.0,dy=0.0,rmse=0.0;
		std::vector<double> coeffs;
		if(!calibration::joint_solve_v16(frames,theta_ref,radius,parameters.RoundnessHarmonics,initial_centers,initial_valid,
			0.05,parameters.FrameWeightMode,parameters.FrameWeightWindowDeg,ox,oy,dx,dy,coeffs,rmse,bridge_log))
			return AlgoResult::Failure("Joint solve failed");

		const auto orders=calibration::harm_orders(parameters.RoundnessHarmonics);
		size_t total_removed=0;
		for(size_t i=0;i<frames.size();++i)
		{
			const double angle_deg=frames[i].angle-theta_ref;
			double cx=0.0,cy=0.0;
			calibration::predict_center(ox,oy,dx,dy,angle_deg,cx,cy);
			const double frame_radius=radius+calibration::harmonics_value(angle_deg*kPi/180.0,coeffs,orders);
			size_t removed=0;
			auto kept=calibration::remove_isolated_residual_outliers(frames[i].edges,cx,cy,frame_radius,&removed);
			if(removed>0)
			{
				total_removed+=removed;
				frames[i].edges.swap(kept);
				log(LogType::Info,std::format("[RobustFilter] angle={} input_index={} removed={} kept={}",
					frames[i].angle,source[i],removed,frames[i].edges.size()));
			}
		}
		if(total_removed>0)
		{
			double ox2=0.0,oy2=0.0,dx2=0.0,dy2=0.0,rmse2=0.0;
			std::vector<double> coeffs2;
			if(calibration::joint_solve_v16(frames,theta_ref,radius,parameters.RoundnessHarmonics,initial_centers,initial_valid,
				0.05,parameters.FrameWeightMode,parameters.FrameWeightWindowDeg,ox2,oy2,dx2,dy2,coeffs2,rmse2,bridge_log))
			{
				ox=ox2; oy=oy2; dx=dx2; dy=dy2; rmse=rmse2;
				coeffs.swap(coeffs2);
			}
			else log(LogType::Warn,"Robust refit failed; retaining initial fit");
			log(LogType::Info,std::format("[RobustFilter] total_removed={} final_rmse={}",total_removed,rmse));
		}

		double wafer_x=0.0,wafer_y=0.0;
		calibration::predict_center(ox,oy,dx,dy,contourState.angles[notch_index]-theta_ref,wafer_x,wafer_y);
		const double theta_notch=std::atan2(-(notch_valley.y-wafer_y),notch_valley.x-wafer_x)*180.0/kPi;
		const double final_angle=contourState.angles[notch_index]+theta_notch+parameters.FeatureAngle;
		const double radius_mm=radius*pixel_size/1000.0,nominal_mm=parameters.WaferSize/2.0;
		const double output_x=wafer_x*pixel_size/1000.0,output_y=-wafer_y*pixel_size/1000.0;
		if(!std::isfinite(radius_mm)||std::abs(radius_mm-nominal_mm)>0.10*nominal_mm||
			!std::isfinite(rmse)||rmse>10.0||!std::isfinite(output_x)||!std::isfinite(output_y)||!std::isfinite(final_angle))
		{
			log(LogType::Error,std::format("Fit validation failed: radius={} mm nominal={} mm edge_RMSE={} px",radius_mm,nominal_mm,rmse));
			return AlgoResult::Failure("Fit validation failed: radius or edge RMSE outside allowed range");
		}
		log(LogType::Info,std::format("Stage center O: ({},{}) px; eccentricity d: ({},{}) px",ox,oy,dx,dy));
		log(LogType::Info,std::format("Wafer center at notch frame: ({},{}) px; radius={} px; harmonics={} rmse={} px",
			wafer_x,wafer_y,radius,parameters.RoundnessHarmonics,rmse));
		*transX=output_x;
		*transY=output_y;
		*angle=final_angle;
		log(LogType::Info,std::format("SUCCESS X={} Y={} Theta={}",*transX,*transY,*angle));
		contourState.stage_center_px={ox,oy};
		contourState.wafer_center_px={wafer_x,wafer_y};
		return AlgoResult::Success();
	}
	catch(const std::exception& error)
	{
		*transX=*transY=*angle=0.0;
		return AlgoResult::Failure(error.what());
	}
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

void ConvertCoordinates(double X_stage,double Y_stage,double* X_wafer,double* Y_wafer)
{
	const auto offset=(contourState.stage_center_px-contourState.wafer_center_px)*contourState.parameters.PixelSize;
	*X_wafer=X_stage+offset.x;
	*Y_wafer=Y_stage-offset.y;
}
