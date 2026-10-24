#include "algocore.h"
#include <omp.h>
import std;

std::vector<double> estimate_column_intensity(cv::Mat image)
{
	if(image.empty())
		throw std::invalid_argument("Input image is empty");
	if(image.type()!=CV_8UC1&&image.type()!=CV_16UC1&&image.type()!=CV_32FC1)
		throw std::invalid_argument("Input image must be single-channel 8-bit, 16-bit, or 32-bit float");
    if (image.dims != 2 || image.rows <= 0 || image.cols <= 0)
        throw std::invalid_argument("Input image must be a non-empty 2D image");

	int W=image.cols;
	// Calculate the average intensity of each column
	cv::Mat col_avg;
	cv::reduce(image,col_avg,0,cv::REDUCE_AVG,CV_64F);
	double* avg_ptr=col_avg.ptr<double>(0);
	std::vector<double> column_intensities(W);
	for(int x=0; x<W; ++x) 
		column_intensities[x]=avg_ptr[x];
	return column_intensities;
}
std::vector<double> estimate_column_intensity_stat(const cv::Mat &image, float minvalid,float maxvalid)
	//float minvalid=-std::numeric_limits<float>::infinity(), float maxvalid=std::numeric_limits<float>::infinity())
{
    if (image.empty())
        throw std::invalid_argument("Input image is empty");
	if(image.type()!=CV_8UC1&&image.type()!=CV_16UC1&&image.type()!=CV_32FC1)
		throw std::invalid_argument("Input image must be single-channel 8-bit, 16-bit, or 32-bit float");
    if (image.dims != 2 || image.rows <= 0 || image.cols <= 0)
        throw std::invalid_argument("Input image must be a non-empty 2D image");

    std::vector<double> result(image.cols, 0.0);
	cv::Mat imgf; image.convertTo(imgf,CV_32F); 

    int W = image.cols;
    int H = image.rows;
    const int N = 10000;
	//const int start_idx=0000,end_idx=2000;
	const int start_idx=4500,end_idx=5500;
	//const int start_idx=000,end_idx=10000;
    const int valid_samples = end_idx - start_idx;

    // Use a fixed base seed to ensure cross-run determinism
    const unsigned int base_seed = 12345; 

    //#pragma omp parallel
    {
        //#pragma omp for schedule(dynamic)
		//#pragma omp for
		std::mt19937 rng(base_seed); 
		std::uniform_int_distribution<int> dist(0, H - 1);
        for (int x = 0; x < W; ++x) 
        { 
			std::vector<float> samples; 
			samples.reserve(N);
            // 1. Sample N=10000 values randomly
			while(samples.size()<N)
			{
                int y = dist(rng);
                float s=imgf.at<float>(y, x);
				//if(minvalid<=s&&s<=maxvalid)
					samples.push_back(s);
            }

            // 2. Sort the values
            std::sort(samples.begin(), samples.end());

            // 3. Calculate average of the part indicated by start_idx and end_idx
            double sum = 0.0;
            for (int i = start_idx; i < end_idx; ++i) 
                sum += samples[i];
            
            result[x] = sum / valid_samples;
        }
    }

    return result;
}

cv::Mat flatten(cv::Mat image,float r_inner, float r_outer,std::string debugfilename) 
{
	(void)r_inner; (void)r_outer;
    if (image.empty()) 
		throw std::invalid_argument("algocore.cpp: flatten(): Input image is empty");
    
    //if (image.type() != CV_8UC1 && image.type() != CV_16UC1 && image.type() != CV_32FC1) 
        //throw std::invalid_argument("Input image must be single-channel 8-bit, 16-bit, or 32-bit float");
    if (image.type() != CV_16UC1) 
        throw std::invalid_argument("Input image must be single-channel 16-bit");

    int W = image.cols;
    int H = image.rows;
	std::vector<double> colavg;
    std::vector<double> inv_scales(W, 1.0f);
    std::vector<double> W_elements(W, 0.0f); // Store for debugging 

    // Step 1: Calculate the average intensity of each column
	//auto colavg=estimate_column_intensity(image);
    try{
		colavg=estimate_column_intensity_stat(image);
	}
	catch(const std::exception& e)
	{
		std::cerr<<"Error in estimate_column_intensity_stat: "<<e.what()<<std::endl;
		throw std::runtime_error("Error in estimate_column_intensity_stat: "+std::string(e.what()));
    }
	//if(colavg[1]>4096)
	//	throw std::runtime_error("Input image is not 12-bit data in a 16-bit container. The average column intensity is greater than 4096, which is not expected for 12-bit data.");
    //correct colavg because the intensity are distributed on circles with radius from r_inner to r_outer
    std::vector<double> correction_factor;
    const float pixelsize=4.832e-3f;//mm/pixel
    //std::println("flatten: r_inner={}, r_outer={}",r_inner,r_outer);
    for(int x=0; x<W; ++x)
    { 
        //double cf=r_outer/(pixelsize*(x+1)+r_inner);
        //double cf=r_outer/(pixelsize*x+r_inner+0.001f);
        //correction_factor.push_back(cf); 
        //colavg[x]*=cf;
    }
	//auto colavg=estimate_column_intensity_stat(image);

    try
	{
		// Step 2: Scale vector so max element is 1.0f, and precompute inverse factors
		double min_val,max_val,mean_val;
		const auto [min_it,max_it]=std::minmax_element(colavg.begin(),colavg.end());
		min_val=*min_it;
		max_val=*max_it;
		mean_val=std::accumulate(colavg.begin(),colavg.end(),0.0)/colavg.size();
		//std::println("Flatten(): min={}, max={}, mean={}",min_val,max_val,mean_val);

		double desired=100.0; // Desired average intensity for the flattened image
		if(max_val>0.0)
		{
			for(int x=0; x<W; ++x)
			{
				W_elements[x]=colavg[x]/desired;
				if(colavg[x]>1e-6)
					inv_scales[x]=desired/colavg[x];
				else
					inv_scales[x]=1.0;
			}
		}

		// Step 3: Multiply each column by its inverse scale
		cv::Mat result(image.size(),CV_32F);
#pragma omp parallel for schedule(static)
		for(int y=0; y<H; ++y) {
			const ushort* src_row=image.ptr<ushort>(y);
			float* dst_row=result.ptr<float>(y);
			for(int x=0; x<W; ++x)
				dst_row[x]=static_cast<float>(src_row[x]*inv_scales[x]);
			//ushort* dst_row = result.ptr<ushort>(y);
			//for (int x = 0; x < W; ++x) 
			//    dst_row[x] = cv::saturate_cast<ushort>(src_row[x] * inv_scales[x]);
		}
		double minval,maxval;
		cv::minMaxLoc(result,&minval,&maxval,nullptr,nullptr);
		//std::println("flattened output range: min={}, max={}",minval,maxval);
		//std::println("Flattening complete. Image type: {}, Size: {}x{}",image.type(),W,H);

		// --- Debugging output generation ---
		if(debugfilename!="")
		{
			// Calculate the average of each column in the output image
			//cv::Mat out_col_avg;
			//cv::reduce(result,out_col_avg,0,cv::REDUCE_AVG,CV_32F);
			//float* out_avg_ptr=out_col_avg.ptr<float>(0);
			auto colavg_out=estimate_column_intensity_stat(result);

			// Save to debug file
			std::ofstream outfile(debugfilename);
			if(outfile.is_open())
			{
				outfile<<"Index\tInput_Avg\tW_Element\tInput_Avg/W_Element\tColAvg_Out\n";
				for(int x=0; x<W; ++x)
					//outfile<<x<<"\t" <<colavg[x]<<"\t" <<W_elements[x]<<"\t" <<out_avg_ptr[x]<<"\n";
					outfile<<std::format("{}\t{:.10f}\t{:.10f}\t{:.10f}\t{:.10f}\n",x,colavg[x],W_elements[x],colavg[x]/W_elements[x],colavg_out[x]);
				//outfile<<x<<"\t" <<colavg[x]<<"\t" <<W_elements[x]<<"\t" <<colavg_out[x]<<"\n";
				outfile.close();
			}
			//print statistics on out_avg_ptr
			//std::cout<<std::format("Output averages ({}): min={:.4f}, max={:.4f}, mean={:.4f}",debugfilename,min_val,max_val,mean_val)<<std::endl;
		}

		return result;
    }
	catch(const std::exception& e)
	{
		throw std::runtime_error("Error during flattening - column scaling process: "+std::string(e.what()));
	} 
}
std::vector<float> get_intensity_distribution(const cv::Mat& image,int column_index)
{
	if(image.empty() || column_index < -1 || column_index >= image.cols)
		return {};
	if(image.channels()!=1)
		throw std::invalid_argument("Input image must be single-channel");

	cv::Mat float_image;
	image.convertTo(float_image,CV_32F);

	const std::size_t total_samples=10000;

	std::vector<float> intensities;
	intensities.reserve(total_samples);

	std::mt19937 generator(12345);
	std::uniform_int_distribution<int> x_distribution(0,float_image.cols-1);
	std::uniform_int_distribution<int> y_distribution(0,float_image.rows-1);
	if(column_index==-1)
		for(int k=0; k<total_samples; ++k)
		{

			int x=x_distribution(generator);
			int y=y_distribution(generator);
			float pixel=float_image.at<float>(y,x);
			intensities.push_back(pixel);
		}
	else
		for(int k=0; k<total_samples; ++k)
		{
			int y=y_distribution(generator);
			float pixel=float_image.at<float>(y,column_index);
			intensities.push_back(pixel);
		}

	std::sort(intensities.begin(),intensities.end());
	return intensities;
}

int ransac_fit_circle(const std::vector<cv::Point2d>& points,double& best_circle_x,double& best_circle_y,double& best_circle_r,int iterations,double threshold)
{
	if(points.size()<3) {
		return 0; // Not enough points to fit a circle
	}
	int best_inliers=0;
	best_circle_x=0.0;
	best_circle_y=0.0;
	best_circle_r=0.0;
	std::mt19937 rng(12345); // Fixed seed for determinism
	std::uniform_int_distribution<size_t> dist(0,points.size()-1);
	for(int i=0; i<iterations; ++i) {
		// Randomly select 3 unique points
		size_t idx1=dist(rng);
		size_t idx2,idx3;
		do { idx2=dist(rng); } while(idx2==idx1);
		do { idx3=dist(rng); } while(idx3==idx1||idx3==idx2);
		const cv::Point2d& p1=points[idx1];
		const cv::Point2d& p2=points[idx2];
		const cv::Point2d& p3=points[idx3];
		// Fit a circle to the three points
		double A=p2.x-p1.x;
		double B=p2.y-p1.y;
		double C=p3.x-p1.x;
		double D=p3.y-p1.y;
		double E=A*(p1.x+p2.x)+B*(p1.y+p2.y);
		double F=C*(p1.x+p3.x)+D*(p1.y+p3.y);
		double G=2.0*(A*D-B*C);
		if(std::abs(G)<1e-6) {
			continue; // Points are collinear or too close to each other
		}
		double cx=(D*E-B*F)/G;
		double cy=(A*F-C*E)/G;
		double r=std::sqrt((cx-p1.x)*(cx-p1.x)+(cy-p1.y)*(cy-p1.y));
		// Count inliers
		int inliers_count=0;
		for(const auto& pt:points) {
			double dist=std::sqrt((pt.x-cx)*(pt.x-cx)+(pt.y-cy)*(pt.y-cy));
			if(std::abs(dist-r)<threshold) {
				++inliers_count;
			}
		}
		if(inliers_count>best_inliers) {
			best_inliers=inliers_count;
			best_circle_x=cx;
			best_circle_y=cy;
			best_circle_r=r;
		}
	}
	return best_inliers;
}
std::vector<cv::Point2d> get_edge_points(const cv::Mat& image)
{ // Binary image 'imgb' from 'image' with threshold 30
		cv::Mat imgb;
		cv::threshold(image,imgb,30,255,cv::THRESH_BINARY);

		std::vector<cv::Point2d> pts;
		// Find contour of the largest black blob in imgb and save to contour_b
		std::vector<std::pair<int,int>> contour_b;
		{
			// Invert so black regions become white for findContours
			cv::Mat imgb_inv;
			cv::bitwise_not(imgb,imgb_inv);

			std::vector<std::vector<cv::Point>> contours;
			cv::findContours(imgb_inv,contours,cv::RETR_EXTERNAL,cv::CHAIN_APPROX_NONE);

			if(contours.empty())
				//write_log(LogType::Warning, "AddCenterCalculationImage", "no contours found");
				return pts; // return empty vector
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
		pts.reserve(contour_b.size());
		for(const auto& p:contour_b)
		{
			auto x=static_cast<int>(p.first),y=static_cast<int>(p.second);
			if(x==0||y==0||x>=imgb.cols-1||y>=imgb.rows-1)
				continue; // skip border points
			pts.emplace_back(static_cast<double>(x),static_cast<double>(y));
		} 
		return pts;
}
bool find_notch(const std::vector<cv::Point2d>& edge_points,cv::Point2d& valley,cv::Point2d & wafer_center)
{
	if(edge_points.size()<10)
		return false;
	// Placeholder implementation
	double cx,cy,cr;
	int inliers=ransac_fit_circle(edge_points,cx,cy,cr,1000,5.0);
	float ratio=static_cast<float>(inliers)/static_cast<float>(edge_points.size());
	if(ratio<0.5f) //invalid if less than 50% inliers
		return false;
	if(ratio>0.95f) //invalid if more than 95% inliers
		return false;
	wafer_center=cv::Point2d(cx,cy);
	double min_dist=1e10;
	for(auto& pt:edge_points)
	{
		double dist=std::sqrt((pt.x-cx)*(pt.x-cx)+(pt.y-cy)*(pt.y-cy));
		if(dist<min_dist)
		{
			min_dist=dist;
			valley=pt;
		}
	}
	return true;
}
//cv::Mat dehaze(cv::Mat image)
//{
//	cv::Mat dehazed;
//	// Convert to float for processing
//	image.convertTo(dehazed,CV_32F,1.0/255.0);
//	// Apply a simple dehazing algorithm (placeholder)
//	cv::Mat darkChannel;
//	cv::erode(dehazed,darkChannel,cv::Mat(),cv::Point(-1,-1),15);
//	cv::Mat transmission=1.0-darkChannel;
//	cv::normalize(transmission, transmission, 0, 1, cv::NORM_MINMAX);
//
//	return dehazed;
//}
