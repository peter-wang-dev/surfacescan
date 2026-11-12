#pragma once

#include <cstddef>
#include <vector>
#include <opencv2/core.hpp>
#include "algorithm.h"

namespace calibration
{
	struct FrameData
	{
		double angle=0.0;
		std::vector<cv::Point2d> edges;
		bool is_notch_frame=false;
	};

	double normalize_angle_360(double angle);
	double median_value(std::vector<double> values);
	std::vector<int> harm_orders(int n_harm);
	double harmonics_value(double alpha_rad,const std::vector<double>& coeffs,const std::vector<int>& orders);
	void predict_center(double ox,double oy,double dx,double dy,double alpha_deg,double& cx,double& cy);
	bool ls_fit_circle(const std::vector<cv::Point2d>& points,double& cx,double& cy,double& radius);
	int ransac_fit_circle(const std::vector<cv::Point2d>& points,double& cx,double& cy,double& radius,int iterations,double threshold);
	std::vector<cv::Point2d> get_edge_points(const cv::Mat& image,int threshold);
	std::vector<cv::Point2d> refine_edges_subpixel(const cv::Mat& image,const std::vector<cv::Point2d>& points,
		double cx,double cy,double search,double step);
	bool find_notch_with_fit(const std::vector<cv::Point2d>& edges,const cv::Size& image_size,
		double fit_cx,double fit_cy,double fit_radius,int fit_inliers,
		cv::Point2d& valley,cv::Point2d& center,double& smooth_depth);
	std::vector<cv::Point2d> remove_notch_contour_segment(const std::vector<cv::Point2d>& edges,
		const cv::Point2d& notch_tip,double residual_threshold_px,double margin_arc_px,
		std::vector<unsigned char>* excluded_mask_out=nullptr);
	std::vector<cv::Point2d> remove_isolated_residual_outliers(const std::vector<cv::Point2d>& points,
		double cx,double cy,double radius,size_t* removed_count=nullptr);
	// Voronoi weights correspond to normalized, sorted angles, as in the Python reference.
	std::vector<double> frame_weights_voronoi(const std::vector<double>& angles);
	std::vector<double> frame_weights_density(const std::vector<double>& angles,double window_deg);
	bool joint_solve_v16(const std::vector<FrameData>& frames,double theta_ref,double fixed_radius,int n_harm,
		const std::vector<cv::Point2d>& init_centers,const std::vector<unsigned char>& init_valid,
		double max_harm_ratio,ContourFrameWeightMode weight_mode,double weight_window_deg,
		double& ox,double& oy,double& dx,double& dy,std::vector<double>& coeffs,double& rmse,
		void (*log_cb)(const char*)=nullptr);
}
