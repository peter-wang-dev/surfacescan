#include "calibcore.h"
#include <opencv2/imgproc.hpp>
import std;

namespace calibration
{
namespace
{
	constexpr double kPi=3.14159265358979323846;
	constexpr int kHarmStart=2; // The first harmonic is degenerate with center translation.
	constexpr double kHuberDelta=3.0;

	bool finite_number(double value)
	{
		return std::isfinite(value);
	}

	double huber_weight(double residual)
	{
		const double magnitude=std::abs(residual);
		return magnitude<=kHuberDelta||magnitude<1e-12?1.0:kHuberDelta/magnitude;
	}

	double huber_rho(double residual)
	{
		const double magnitude=std::abs(residual);
		return magnitude<=kHuberDelta?residual*residual:2.0*kHuberDelta*magnitude-kHuberDelta*kHuberDelta;
	}

	bool solve_dense(std::vector<double> matrix,std::vector<double> rhs,int n,std::vector<double>& solution)
	{
		solution.assign(static_cast<size_t>(n),0.0);
		for(int col=0;col<n;++col)
		{
			int pivot_row=col;
			for(int row=col+1;row<n;++row)
				if(std::abs(matrix[static_cast<size_t>(row)*n+col])>
					std::abs(matrix[static_cast<size_t>(pivot_row)*n+col])) pivot_row=row;
			const double pivot=matrix[static_cast<size_t>(pivot_row)*n+col];
			if(!finite_number(pivot)||std::abs(pivot)<1e-12) return false;
			if(pivot_row!=col)
			{
				for(int c=col;c<n;++c)
					std::swap(matrix[static_cast<size_t>(col)*n+c],matrix[static_cast<size_t>(pivot_row)*n+c]);
				std::swap(rhs[static_cast<size_t>(col)],rhs[static_cast<size_t>(pivot_row)]);
			}
			for(int row=col+1;row<n;++row)
			{
				const double q=matrix[static_cast<size_t>(row)*n+col]/matrix[static_cast<size_t>(col)*n+col];
				if(q==0.0) continue;
				for(int c=col;c<n;++c)
					matrix[static_cast<size_t>(row)*n+c]-=q*matrix[static_cast<size_t>(col)*n+c];
				rhs[static_cast<size_t>(row)]-=q*rhs[static_cast<size_t>(col)];
			}
		}
		for(int row=n-1;row>=0;--row)
		{
			double value=rhs[static_cast<size_t>(row)];
			for(int col=row+1;col<n;++col)
				value-=matrix[static_cast<size_t>(row)*n+col]*solution[static_cast<size_t>(col)];
			solution[static_cast<size_t>(row)]=value/matrix[static_cast<size_t>(row)*n+row];
			if(!finite_number(solution[static_cast<size_t>(row)])) return false;
		}
		return true;
	}

	double sample_bilinear(const cv::Mat& image,double x,double y)
	{
		if(image.cols<=0||image.rows<=0) return 0.0;
		const double xc=std::clamp(x,0.0,static_cast<double>(image.cols-1));
		const double yc=std::clamp(y,0.0,static_cast<double>(image.rows-1));
		const int x0=static_cast<int>(std::floor(xc)),y0=static_cast<int>(std::floor(yc));
		const int x1=std::min(x0+1,image.cols-1),y1=std::min(y0+1,image.rows-1);
		const double fx=xc-x0,fy=yc-y0;
		const float* row0=image.ptr<float>(y0);
		const float* row1=image.ptr<float>(y1);
		return (1.0-fx)*(1.0-fy)*row0[x0]+fx*(1.0-fy)*row0[x1]+
			(1.0-fx)*fy*row1[x0]+fx*fy*row1[x1];
	}

	double cost_v16(const std::vector<FrameData>& frames,const std::vector<double>& frame_weights,
		const std::vector<int>& orders,double theta_ref,const std::vector<double>& parameters,double fixed_radius)
	{
		std::vector<double> coeffs;
		if(parameters.size()>4) coeffs.assign(parameters.begin()+4,parameters.end());
		double total=0.0;
		for(size_t i=0;i<frames.size();++i)
		{
			const double angle=(frames[i].angle-theta_ref)*kPi/180.0;
			const double radius=fixed_radius+harmonics_value(angle,coeffs,orders);
			double cx=0.0,cy=0.0;
			predict_center(parameters[0],parameters[1],parameters[2],parameters[3],frames[i].angle-theta_ref,cx,cy);
			const double weight=i<frame_weights.size()?frame_weights[i]:1.0;
			for(const auto& point:frames[i].edges)
				total+=weight*huber_rho(std::hypot(point.x-cx,point.y-cy)-radius);
		}
		return total;
	}
}

	double normalize_angle_360(double angle)
	{
		double result=std::fmod(angle,360.0);
		if(result<0.0) result+=360.0;
		return result;
	}

	double median_value(std::vector<double> values)
	{
		if(values.empty()) return 0.0;
		const size_t mid=values.size()/2;
		const auto offset=static_cast<std::ptrdiff_t>(mid);
		std::nth_element(values.begin(),values.begin()+offset,values.end());
		double median=values[mid];
		if(values.size()%2==0)
			median=0.5*(median+*std::max_element(values.begin(),values.begin()+offset));
		return median;
	}

	std::vector<int> harm_orders(int n_harm)
	{
		std::vector<int> orders;
		for(int order=kHarmStart;order<=n_harm;++order) orders.push_back(order);
		return orders;
	}

	double harmonics_value(double alpha_rad,const std::vector<double>& coeffs,const std::vector<int>& orders)
	{
		double value=0.0;
		const size_t count=std::min(orders.size(),coeffs.size()/2);
		for(size_t i=0;i<count;++i)
		{
			const double order=static_cast<double>(orders[i]);
			value+=coeffs[2*i]*std::cos(order*alpha_rad)+coeffs[2*i+1]*std::sin(order*alpha_rad);
		}
		return value;
	}

	void predict_center(double ox,double oy,double dx,double dy,double alpha_deg,double& cx,double& cy)
	{
		const double angle=alpha_deg*kPi/180.0;
		const double cosine=std::cos(angle),sine=std::sin(angle);
		cx=ox+dx*cosine-dy*sine;
		cy=oy+dx*sine+dy*cosine;
	}

	bool ls_fit_circle(const std::vector<cv::Point2d>& points,double& cx,double& cy,double& radius)
	{
		if(points.size()<3) return false;
		double mx=0.0,my=0.0;
		for(const auto& point:points) { mx+=point.x; my+=point.y; }
		mx/=static_cast<double>(points.size());
		my/=static_cast<double>(points.size());
		cv::Mat matrix(static_cast<int>(points.size()),3,CV_64F);
		cv::Mat rhs(static_cast<int>(points.size()),1,CV_64F);
		for(int i=0;i<static_cast<int>(points.size());++i)
		{
			const double x=points[static_cast<size_t>(i)].x-mx,y=points[static_cast<size_t>(i)].y-my;
			matrix.at<double>(i,0)=2.0*x;
			matrix.at<double>(i,1)=2.0*y;
			matrix.at<double>(i,2)=1.0;
			rhs.at<double>(i,0)=x*x+y*y;
		}
		cv::Mat solution;
		if(!cv::solve(matrix,rhs,solution,cv::DECOMP_SVD)||solution.rows!=3) return false;
		const double ux=solution.at<double>(0,0),uy=solution.at<double>(1,0);
		const double squared_radius=solution.at<double>(2,0)+ux*ux+uy*uy;
		if(!finite_number(ux)||!finite_number(uy)||!finite_number(squared_radius)||squared_radius<=0.0) return false;
		cx=mx+ux;
		cy=my+uy;
		radius=std::sqrt(squared_radius);
		return finite_number(cx)&&finite_number(cy)&&finite_number(radius)&&radius>0.0;
	}

	int ransac_fit_circle(const std::vector<cv::Point2d>& points,double& cx,double& cy,double& radius,
		int iterations,double threshold)
	{
		cx=cy=radius=0.0;
		if(points.size()<3||iterations<=0||threshold<=0.0) return 0;
		std::mt19937 generator(12345u+static_cast<unsigned int>(points.size()));
		std::uniform_int_distribution<size_t> pick(0,points.size()-1);
		int best_count=0;
		for(int iteration=0;iteration<iterations;++iteration)
		{
			const size_t i1=pick(generator);
			size_t i2=pick(generator),i3=pick(generator);
			for(int tries=0;i2==i1&&tries<20;++tries) i2=pick(generator);
			for(int tries=0;(i3==i1||i3==i2)&&tries<20;++tries) i3=pick(generator);
			if(i1==i2||i1==i3||i2==i3) continue;
			const auto& p1=points[i1];
			const auto& p2=points[i2];
			const auto& p3=points[i3];
			const double a=p2.x-p1.x,b=p2.y-p1.y,c=p3.x-p1.x,d=p3.y-p1.y;
			const double determinant=2.0*(a*d-b*c);
			if(std::abs(determinant)<1e-9) continue;
			const double e=p2.x*p2.x+p2.y*p2.y-p1.x*p1.x-p1.y*p1.y;
			const double f=p3.x*p3.x+p3.y*p3.y-p1.x*p1.x-p1.y*p1.y;
			const double fit_x=(e*d-b*f)/determinant,fit_y=(a*f-e*c)/determinant;
			const double fit_radius=std::hypot(fit_x-p1.x,fit_y-p1.y);
			if(!finite_number(fit_radius)||fit_radius<=0.0) continue;
			int count=0;
			for(const auto& point:points)
				if(std::abs(std::hypot(point.x-fit_x,point.y-fit_y)-fit_radius)<=threshold) ++count;
			if(count>best_count)
			{
				best_count=count;
				cx=fit_x;
				cy=fit_y;
				radius=fit_radius;
			}
		}
		if(best_count>=3)
		{
			std::vector<cv::Point2d> inliers;
			inliers.reserve(static_cast<size_t>(best_count));
			for(const auto& point:points)
				if(std::abs(std::hypot(point.x-cx,point.y-cy)-radius)<=threshold) inliers.push_back(point);
			double fit_x,fit_y,fit_radius;
			if(ls_fit_circle(inliers,fit_x,fit_y,fit_radius))
			{
				cx=fit_x;
				cy=fit_y;
				radius=fit_radius;
			}
		}
		return best_count;
	}

	std::vector<cv::Point2d> get_edge_points(const cv::Mat& image,int threshold)
	{
		std::vector<cv::Point2d> points;
		if(image.empty()||image.type()!=CV_8UC1) return points;
		cv::Mat blurred,binary,closed;
		cv::GaussianBlur(image,blurred,cv::Size(5,5),1.0,1.0,cv::BORDER_REPLICATE);
		cv::Mat otsu_input=blurred.clone();
		const double otsu_threshold=cv::threshold(otsu_input,binary,0,255,cv::THRESH_BINARY|cv::THRESH_OTSU);
		const int requested_threshold=threshold>0?threshold:30;
		const int used_threshold=std::max(1,std::min(requested_threshold,static_cast<int>(std::lround(otsu_threshold))));
		cv::threshold(blurred,binary,used_threshold,255,cv::THRESH_BINARY);
		cv::morphologyEx(binary,closed,cv::MORPH_CLOSE,cv::getStructuringElement(cv::MORPH_ELLIPSE,cv::Size(3,3)));
		std::vector<std::vector<cv::Point>> contours;
		cv::findContours(closed,contours,cv::RETR_EXTERNAL,cv::CHAIN_APPROX_NONE);
		if(contours.empty()) return points;
		int best_index=-1;
		double best_area=0.0;
		for(size_t i=0;i<contours.size();++i)
		{
			const auto& contour=contours[i];
			if(contour.size()<100) continue;
			const cv::Rect box=cv::boundingRect(contour);
			if(box.x<=0||box.y<=0||box.x+box.width>=image.cols||box.y+box.height>=image.rows) continue;
			const double area=std::abs(cv::contourArea(contour));
			if(area>best_area) { best_area=area; best_index=static_cast<int>(i); }
		}
		if(best_index<0)
			for(size_t i=0;i<contours.size();++i)
			{
				const double area=std::abs(cv::contourArea(contours[i]));
				if(area>best_area) { best_area=area; best_index=static_cast<int>(i); }
			}
		if(best_index<0||contours[static_cast<size_t>(best_index)].size()<100) return points;
		const auto& contour=contours[static_cast<size_t>(best_index)];
		points.reserve(contour.size());
		for(const auto& point:contour)
		{
			if(point.x<=0||point.y<=0||point.x>=image.cols-1||point.y>=image.rows-1) continue;
			points.emplace_back(static_cast<double>(point.x),static_cast<double>(point.y));
		}
		if(points.size()<100) points.clear();
		return points;
	}

	std::vector<cv::Point2d> refine_edges_subpixel(const cv::Mat& image,const std::vector<cv::Point2d>& points,
		double cx,double cy,double search,double step)
	{
		if(image.empty()||points.empty()) return points;
		if(search<=0.0) search=1.5;
		if(step<=0.0) step=0.25;
		cv::Mat gray,gx,gy;
		image.convertTo(gray,CV_32F);
		cv::Sobel(gray,gx,CV_32F,1,0,3);
		cv::Sobel(gray,gy,CV_32F,0,1,3);
		const int half=std::max(1,static_cast<int>(std::floor(search/step+1e-9)));
		std::vector<double> samples;
		samples.reserve(static_cast<size_t>(2*half+1));
		for(int k=-half;k<=half;++k) samples.push_back(k*step);
		if(samples.size()<3) return points;
		std::vector<cv::Point2d> result(points);
		std::vector<double> projection(samples.size(),0.0);
		for(size_t i=0;i<points.size();++i)
		{
			const double vx=points[i].x-cx,vy=points[i].y-cy;
			const double norm=std::hypot(vx,vy);
			if(norm<=1e-6) continue;
			const double nx=vx/norm,ny=vy/norm;
			int best=0;
			for(size_t j=0;j<samples.size();++j)
			{
				const double x=points[i].x+samples[j]*nx,y=points[i].y+samples[j]*ny;
				projection[j]=std::abs(sample_bilinear(gx,x,y)*nx+sample_bilinear(gy,x,y)*ny);
				if(projection[j]>projection[static_cast<size_t>(best)]) best=static_cast<int>(j);
			}
			double offset=0.0;
			if(best>=1&&best<=static_cast<int>(projection.size())-2)
			{
				const double f0=projection[static_cast<size_t>(best-1)];
				const double f1=projection[static_cast<size_t>(best)];
				const double f2=projection[static_cast<size_t>(best+1)];
				const double denominator=f0-2.0*f1+f2;
				if(std::abs(denominator)>1e-12)
					offset=samples[static_cast<size_t>(best)]+0.5*step*(f0-f2)/denominator;
			}
			result[i]=cv::Point2d(points[i].x+offset*nx,points[i].y+offset*ny);
		}
		return result;
	}

	bool find_notch_with_fit(const std::vector<cv::Point2d>& edges,const cv::Size& image_size,
		double fit_cx,double fit_cy,double fit_radius,int fit_inliers,
		cv::Point2d& valley,cv::Point2d& center,double& smooth_depth)
	{
		if(edges.size()<100||fit_inliers<50||!finite_number(fit_radius)||fit_radius<=0.0) return false;
		const size_t count=edges.size();
		std::vector<double> inward(count),smooth(count,0.0);
		for(size_t i=0;i<count;++i)
			inward[i]=fit_radius-std::hypot(edges[i].x-fit_cx,edges[i].y-fit_cy);
		const int half=8;
		for(size_t i=0;i<count;++i)
		{
			double sum=0.0;
			for(int k=-half;k<=half;++k)
			{
				const auto j=static_cast<size_t>((static_cast<long long>(i)+k+static_cast<long long>(count))%
					static_cast<long long>(count));
				sum+=inward[j];
			}
			smooth[i]=sum/static_cast<double>(2*half+1);
		}
		const double width=static_cast<double>(image_size.width),height=static_cast<double>(image_size.height);
		const double margin=std::max(30.0,0.025*std::min(width,height));
		std::vector<double> border_distance(count);
		for(size_t i=0;i<count;++i)
			border_distance[i]=std::min({edges[i].x,width-1.0-edges[i].x,edges[i].y,height-1.0-edges[i].y});
		size_t best=count;
		double depth=-std::numeric_limits<double>::infinity();
		for(size_t i=0;i<count;++i)
		{
			if(border_distance[i]<margin) continue;
			bool neighborhood_clear=true;
			for(int k=-half;k<=half;++k)
			{
				const auto j=static_cast<size_t>((static_cast<long long>(i)+k+static_cast<long long>(count))%
					static_cast<long long>(count));
				if(border_distance[j]<margin) { neighborhood_clear=false; break; }
			}
			if(neighborhood_clear&&smooth[i]>depth) { depth=smooth[i]; best=i; }
		}
		if(best==count||!finite_number(depth)||depth<=6.0) return false;
		double x=0.0,y=0.0;
		int samples=0;
		for(int k=-4;k<=4;++k)
		{
			const auto j=static_cast<size_t>((static_cast<long long>(best)+k+static_cast<long long>(count))%
				static_cast<long long>(count));
			x+=edges[j].x;
			y+=edges[j].y;
			++samples;
		}
		valley=cv::Point2d(x/static_cast<double>(samples),y/static_cast<double>(samples));
		if(std::min({valley.x,width-1.0-valley.x,valley.y,height-1.0-valley.y})<margin) return false;
		center=cv::Point2d(fit_cx,fit_cy);
		smooth_depth=depth;
		return true;
	}

	std::vector<cv::Point2d> remove_notch_contour_segment(const std::vector<cv::Point2d>& edges,
		const cv::Point2d& notch_tip,double residual_threshold_px,double margin_arc_px,
		std::vector<unsigned char>* excluded_mask_out)
	{
		std::vector<cv::Point2d> result;
		if(edges.size()<30) return result;
		double cx=0.0,cy=0.0,radius=0.0;
		const int inliers=ransac_fit_circle(edges,cx,cy,radius,1200,3.0);
		std::vector<unsigned char> excluded(edges.size(),0);
		if(inliers<20||radius<=0.0)
		{
			if(excluded_mask_out) excluded_mask_out->assign(edges.size(),1);
			return edges;
		}
		std::vector<unsigned char> seeds(edges.size(),0);
		size_t tip=0;
		double tip_distance=std::numeric_limits<double>::infinity();
		for(size_t i=0;i<edges.size();++i)
		{
			if(std::abs(std::hypot(edges[i].x-cx,edges[i].y-cy)-radius)>=residual_threshold_px) seeds[i]=1;
			const double distance=std::hypot(edges[i].x-notch_tip.x,edges[i].y-notch_tip.y);
			if(distance<tip_distance) { tip_distance=distance; tip=i; }
		}
		seeds[tip]=1;
		for(size_t i=0;i<edges.size();++i)
		{
			if(!seeds[i]) continue;
			excluded[i]=1;
			for(int direction:{-1,1})
			{
				double arc=0.0;
				long long j=static_cast<long long>(i);
				while(j+direction>=0&&j+direction<static_cast<long long>(edges.size())&&arc<margin_arc_px)
				{
					const auto& a=edges[static_cast<size_t>(j)];
					j+=direction;
					const auto& b=edges[static_cast<size_t>(j)];
					arc+=std::hypot(a.x-b.x,a.y-b.y);
					excluded[static_cast<size_t>(j)]=1;
				}
			}
		}
		result.reserve(edges.size());
		for(size_t i=0;i<edges.size();++i)
			if(!excluded[i]) result.push_back(edges[i]);
		if(excluded_mask_out) *excluded_mask_out=std::move(excluded);
		return result;
	}

	std::vector<cv::Point2d> remove_isolated_residual_outliers(const std::vector<cv::Point2d>& points,
		double cx,double cy,double radius,size_t* removed_count)
	{
		if(removed_count) *removed_count=0;
		if(points.size()<50||!finite_number(cx)||!finite_number(cy)||!finite_number(radius)||radius<=0.0) return points;
		std::vector<double> residuals;
		residuals.reserve(points.size());
		for(const auto& point:points) residuals.push_back(std::hypot(point.x-cx,point.y-cy)-radius);
		const double median=median_value(residuals);
		std::vector<double> deviations;
		deviations.reserve(residuals.size());
		for(double residual:residuals) deviations.push_back(std::abs(residual-median));
		const double cutoff=std::max(8.0,6.0*1.4826*median_value(deviations));
		std::vector<cv::Point2d> kept;
		kept.reserve(points.size());
		for(size_t i=0;i<points.size();++i)
			if(std::abs(residuals[i]-median)<=cutoff) kept.push_back(points[i]);
		if(kept.size()<30||static_cast<double>(kept.size())<static_cast<double>(points.size())*0.50) return points;
		if(removed_count) *removed_count=points.size()-kept.size();
		return kept;
	}

	std::vector<double> frame_weights_voronoi(const std::vector<double>& angles)
	{
		const size_t count=angles.size();
		if(count==0) return {};
		if(count==1) return {1.0};
		std::vector<double> sorted(angles);
		for(double& angle:sorted) angle=normalize_angle_360(angle);
		std::sort(sorted.begin(),sorted.end());
		std::vector<double> weights(count,1.0);
		for(size_t i=0;i<count;++i)
		{
			const double left=std::fmod(sorted[i]-sorted[(i+count-1)%count]+360.0,360.0);
			const double right=std::fmod(sorted[(i+1)%count]-sorted[i]+360.0,360.0);
			weights[i]=0.5*(left+right)*static_cast<double>(count)/360.0;
		}
		return weights;
	}

	std::vector<double> frame_weights_density(const std::vector<double>& angles,double window_deg)
	{
		const size_t count=angles.size();
		if(count==0) return {};
		if(count==1) return {1.0};
		if(window_deg<=0.0) window_deg=360.0/static_cast<double>(count);
		std::vector<double> weights(count,1.0);
		for(size_t i=0;i<count;++i)
		{
			double density=0.0;
			for(size_t j=0;j<count;++j)
			{
				double distance=std::fmod(angles[i]-angles[j]+180.0,360.0);
				if(distance<0.0) distance+=360.0;
				distance=std::abs(distance-180.0);
				if(distance<window_deg) density+=1.0-distance/window_deg;
			}
			weights[i]=density>1e-12?1.0/density:1.0;
		}
		double mean=0.0;
		for(double weight:weights) mean+=weight;
		mean/=static_cast<double>(count);
		if(mean>1e-12) for(double& weight:weights) weight/=mean;
		return weights;
	}

	bool joint_solve_v16(const std::vector<FrameData>& frames,double theta_ref,double fixed_radius,int n_harm,
		const std::vector<cv::Point2d>& init_centers,const std::vector<unsigned char>& init_valid,
		double max_harm_ratio,ContourFrameWeightMode weight_mode,double weight_window_deg,
		double& ox,double& oy,double& dx,double& dy,std::vector<double>& coeffs,double& rmse,
		void (*log_cb)(const char*))
	{
		auto log=[&](const char* message) { if(log_cb) log_cb(message); };
		if(frames.size()<3||fixed_radius<=0.0)
		{
			log("joint_solve_v16: not enough frames or invalid fixed radius");
			return false;
		}
		const std::vector<int> orders=harm_orders(std::max(0,n_harm));
		const size_t n_orders=orders.size();
		std::vector<double> angles;
		angles.reserve(frames.size());
		for(const auto& frame:frames) angles.push_back(frame.angle);
		std::vector<double> weights;
		if(weight_mode==ContourFrameWeightMode::Density) weights=frame_weights_density(angles,weight_window_deg);
		else if(weight_mode==ContourFrameWeightMode::None) weights.assign(frames.size(),1.0);
		else weights=frame_weights_voronoi(angles);

		std::vector<cv::Point2d> centers;
		std::vector<double> center_angles;
		for(size_t i=0;i<frames.size();++i)
		{
			cv::Point2d center;
			bool has_center=false;
			if(i<init_centers.size()&&i<init_valid.size()&&init_valid[i]&&
				finite_number(init_centers[i].x)&&finite_number(init_centers[i].y))
			{
				center=init_centers[i];
				has_center=true;
			}
			if(!has_center)
			{
				double cx=0.0,cy=0.0,radius=0.0;
				if(ls_fit_circle(frames[i].edges,cx,cy,radius)) { center=cv::Point2d(cx,cy); has_center=true; }
			}
			if(has_center) { centers.push_back(center); center_angles.push_back(frames[i].angle); }
		}
		if(centers.size()<3)
		{
			log("joint_solve_v16: could not fit enough per-frame circles");
			return false;
		}
		const int count=static_cast<int>(centers.size());
		cv::Mat matrix(count*2,4,CV_64F,cv::Scalar(0)),rhs(count*2,1,CV_64F,cv::Scalar(0));
		for(int i=0;i<count;++i)
		{
			const double angle=(center_angles[static_cast<size_t>(i)]-theta_ref)*kPi/180.0;
			const double cosine=std::cos(angle),sine=std::sin(angle);
			matrix.at<double>(2*i,0)=1.0;
			matrix.at<double>(2*i,2)=cosine;
			matrix.at<double>(2*i,3)=-sine;
			rhs.at<double>(2*i,0)=centers[static_cast<size_t>(i)].x;
			matrix.at<double>(2*i+1,1)=1.0;
			matrix.at<double>(2*i+1,2)=sine;
			matrix.at<double>(2*i+1,3)=cosine;
			rhs.at<double>(2*i+1,0)=centers[static_cast<size_t>(i)].y;
		}
		cv::Mat initial;
		if(!cv::solve(matrix,rhs,initial,cv::DECOMP_SVD)||initial.rows!=4)
		{
			log("joint_solve_v16: linear trajectory initialization failed");
			return false;
		}
		ox=initial.at<double>(0,0);
		oy=initial.at<double>(1,0);
		dx=initial.at<double>(2,0);
		dy=initial.at<double>(3,0);
		if(!finite_number(ox)||!finite_number(oy)||!finite_number(dx)||!finite_number(dy))
		{
			log("joint_solve_v16: non-finite initialization");
			return false;
		}

		coeffs.assign(2*n_orders,0.0);
		if(n_orders>0)
		{
			std::vector<std::vector<double>> rows;
			std::vector<double> harmonic_rhs;
			for(size_t i=0;i<frames.size();++i)
			{
				if(frames[i].edges.size()<50) continue;
				const double angle_deg=frames[i].angle-theta_ref;
				double cx=0.0,cy=0.0;
				predict_center(ox,oy,dx,dy,angle_deg,cx,cy);
				std::vector<double> radii;
				radii.reserve(frames[i].edges.size());
				for(const auto& point:frames[i].edges) radii.push_back(std::hypot(point.x-cx,point.y-cy));
				const double radius=median_value(radii),angle=angle_deg*kPi/180.0;
				const double scale=std::sqrt(std::max(1e-12,i<weights.size()?weights[i]:1.0));
				std::vector<double> row(2*n_orders,0.0);
				for(size_t t=0;t<n_orders;++t)
				{
					const double order=static_cast<double>(orders[t]);
					row[2*t]=scale*std::cos(order*angle);
					row[2*t+1]=scale*std::sin(order*angle);
				}
				rows.push_back(std::move(row));
				harmonic_rhs.push_back(scale*(radius-fixed_radius));
			}
			if(rows.size()>=2*n_orders)
			{
				const int nr=static_cast<int>(rows.size()),nc=static_cast<int>(2*n_orders);
				cv::Mat harmonic_matrix(nr,nc,CV_64F),harmonic_values(nr,1,CV_64F);
				for(int row=0;row<nr;++row)
				{
					for(int col=0;col<nc;++col)
						harmonic_matrix.at<double>(row,col)=rows[static_cast<size_t>(row)][static_cast<size_t>(col)];
					harmonic_values.at<double>(row,0)=harmonic_rhs[static_cast<size_t>(row)];
				}
				cv::Mat solution;
				if(cv::solve(harmonic_matrix,harmonic_values,solution,cv::DECOMP_SVD)&&solution.rows==nc)
					for(int i=0;i<nc;++i) coeffs[static_cast<size_t>(i)]=solution.at<double>(i,0);
				else coeffs.assign(2*n_orders,0.0);
			}
		}

		const int npar=4+2*static_cast<int>(n_orders);
		std::vector<double> parameters(static_cast<size_t>(npar),0.0);
		parameters[0]=ox; parameters[1]=oy; parameters[2]=dx; parameters[3]=dy;
		for(size_t i=0;i<coeffs.size();++i) parameters[4+i]=coeffs[i];
		const double harmonic_bound=max_harm_ratio*fixed_radius;
		double cost=cost_v16(frames,weights,orders,theta_ref,parameters,fixed_radius);
		double damping=1e-3;
		for(int iteration=0;iteration<80;++iteration)
		{
			std::vector<double> hessian(static_cast<size_t>(npar)*npar,0.0),gradient(static_cast<size_t>(npar),0.0);
			std::vector<double> harmonic_coeffs(parameters.begin()+4,parameters.end());
			double effective_points=0.0;
			for(size_t fi=0;fi<frames.size();++fi)
			{
				const double angle_deg=frames[fi].angle-theta_ref,angle=angle_deg*kPi/180.0;
				const double cosine=std::cos(angle),sine=std::sin(angle);
				double cx=0.0,cy=0.0;
				predict_center(parameters[0],parameters[1],parameters[2],parameters[3],angle_deg,cx,cy);
				const double radius=fixed_radius+harmonics_value(angle,harmonic_coeffs,orders);
				const double frame_weight=fi<weights.size()?weights[fi]:1.0;
				std::vector<double> trig(2*n_orders,0.0);
				for(size_t t=0;t<n_orders;++t)
				{
					const double order=static_cast<double>(orders[t]);
					trig[2*t]=std::cos(order*angle);
					trig[2*t+1]=std::sin(order*angle);
				}
				for(const auto& point:frames[fi].edges)
				{
					const double x=point.x-cx,y=point.y-cy,distance=std::hypot(x,y);
					if(distance<1e-9) continue;
					const double residual=distance-radius,weight=frame_weight*huber_weight(residual);
					const double j0=-x/distance,j1=-y/distance;
					std::vector<double> jacobian(static_cast<size_t>(npar),0.0);
					jacobian[0]=j0;
					jacobian[1]=j1;
					jacobian[2]=j0*cosine+j1*sine;
					jacobian[3]=-j0*sine+j1*cosine;
					for(size_t t=0;t<n_orders;++t)
					{
						jacobian[4+2*t]=-trig[2*t];
						jacobian[5+2*t]=-trig[2*t+1];
					}
					for(int row=0;row<npar;++row)
					{
						gradient[static_cast<size_t>(row)]-=weight*jacobian[static_cast<size_t>(row)]*residual;
						for(int col=0;col<npar;++col)
							hessian[static_cast<size_t>(row)*npar+col]+=
								weight*jacobian[static_cast<size_t>(row)]*jacobian[static_cast<size_t>(col)];
					}
					effective_points+=frame_weight;
				}
			}
			if(effective_points<50.0) break;
			if(n_orders>0)
			{
				const double ridge=1e-4*effective_points;
				for(int k=4;k<npar;++k) hessian[static_cast<size_t>(k)*npar+k]+=ridge;
			}
			std::vector<double> damped=hessian;
			for(int k=0;k<npar;++k)
				damped[static_cast<size_t>(k)*npar+k]+=damping*(hessian[static_cast<size_t>(k)*npar+k]+1.0);
			std::vector<double> delta;
			if(!solve_dense(damped,gradient,npar,delta))
			{
				damping*=10.0;
				if(damping>1e12) break;
				continue;
			}
			std::vector<double> candidate(static_cast<size_t>(npar),0.0);
			bool candidate_finite=true;
			for(int k=0;k<npar;++k)
			{
				candidate[static_cast<size_t>(k)]=parameters[static_cast<size_t>(k)]+delta[static_cast<size_t>(k)];
				if(!finite_number(candidate[static_cast<size_t>(k)])) candidate_finite=false;
			}
			if(!candidate_finite) { damping*=10.0; continue; }
			if(n_orders>0)
			{
				bool out_of_bound=false;
				for(size_t k=4;k<candidate.size();++k)
					if(std::abs(candidate[k])>harmonic_bound) { out_of_bound=true; break; }
				if(out_of_bound)
				{
					damping*=10.0;
					if(damping>1e12) break;
					continue;
				}
			}
			const double new_cost=cost_v16(frames,weights,orders,theta_ref,candidate,fixed_radius);
			if(finite_number(new_cost)&&new_cost<cost)
			{
				const double relative_change=std::abs(cost-new_cost)/(std::abs(cost)+1e-12);
				parameters=candidate;
				cost=new_cost;
				damping=std::max(1e-12,damping*0.3);
				if(relative_change<1e-9) break;
			}
			else
			{
				damping*=10.0;
				if(damping>1e12) break;
			}
		}
		if(!finite_number(cost)) return false;

		// RMSE remains unweighted, matching the Python V16 diagnostic.
		std::vector<double> final_coeffs(parameters.begin()+4,parameters.end());
		double sum_squared=0.0;
		size_t point_count=0;
		for(const auto& frame:frames)
		{
			double cx=0.0,cy=0.0;
			predict_center(parameters[0],parameters[1],parameters[2],parameters[3],frame.angle-theta_ref,cx,cy);
			const double angle=(frame.angle-theta_ref)*kPi/180.0;
			const double radius=fixed_radius+harmonics_value(angle,final_coeffs,orders);
			for(const auto& point:frame.edges)
			{
				const double residual=std::hypot(point.x-cx,point.y-cy)-radius;
				sum_squared+=residual*residual;
				++point_count;
			}
		}
		rmse=point_count?std::sqrt(sum_squared/static_cast<double>(point_count)):std::numeric_limits<double>::infinity();
		if(!finite_number(rmse)) return false;
		if(rmse>10.0) log("joint_solve_v16 WARNING: edge RMSE exceeds 10px");
		ox=parameters[0];
		oy=parameters[1];
		dx=parameters[2];
		dy=parameters[3];
		coeffs.assign(parameters.begin()+4,parameters.end());
		return true;
	}
}
