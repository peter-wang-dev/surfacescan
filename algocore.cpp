#include "algocore.h"
import std;

cv::Mat flatten(cv::Mat image,std::string debugfilename) 
{
    if (image.empty()) 
        return image;
    
    if (image.type() != CV_8UC1 && image.type() != CV_16UC1) 
        throw std::invalid_argument("Input image must be single-channel 8-bit or 16-bit");

    int W = image.cols;
    int H = image.rows;

    // Step 1: Calculate the average intensity of each column
    cv::Mat col_avg;
    cv::reduce(image, col_avg, 0, cv::REDUCE_AVG, CV_64F);
    double* avg_ptr = col_avg.ptr<double>(0);

    // Step 2: Scale vector so max element is 1.0f, and precompute inverse factors
    double min_val, max_val;
    cv::minMaxLoc(col_avg, &min_val, &max_val);

    std::vector<double> inv_scales(W, 1.0f);
    std::vector<double> W_elements(W, 0.0f); // Store for debugging

    if (max_val > 0.0) {
        for (int x = 0; x < W; ++x) {
            W_elements[x] = avg_ptr[x] / static_cast<double>(max_val); // Calculate scaled W
            if (avg_ptr[x] > 1e-6) { 
                inv_scales[x] = static_cast<double>(max_val) / avg_ptr[x];
            } else {
                inv_scales[x] = 1.0;
            }
        }
    }

    // Step 3: Multiply each column by its inverse scale
    cv::Mat result(image.size(), image.type());

    if (image.type() == CV_8UC1) {
        #pragma omp parallel for schedule(static)
        for (int y = 0; y < H; ++y) {
            const uchar* src_row = image.ptr<uchar>(y);
            uchar* dst_row = result.ptr<uchar>(y);
            for (int x = 0; x < W; ++x) {
                dst_row[x] = cv::saturate_cast<uchar>(src_row[x] * inv_scales[x]);
            }
        }
    } else if (image.type() == CV_16UC1) {
        #pragma omp parallel for schedule(static)
        for (int y = 0; y < H; ++y) {
            const ushort* src_row = image.ptr<ushort>(y);
            ushort* dst_row = result.ptr<ushort>(y);
            for (int x = 0; x < W; ++x) {
                dst_row[x] = cv::saturate_cast<ushort>(src_row[x] * inv_scales[x]);
            }
        }
    }

    // --- Debugging output generation ---
    if(debugfilename!="") { 
        // Calculate the average of each column in the output image
        cv::Mat out_col_avg;
        cv::reduce(result,out_col_avg,0,cv::REDUCE_AVG,CV_32F);
        float* out_avg_ptr=out_col_avg.ptr<float>(0);

        // Save to debug file
        std::ofstream outfile(debugfilename);
        if(outfile.is_open()) 
        {
            outfile<<"Index\tInput_Avg\tW_Element\tOutput_Avg\n";
            outfile<<std::fixed<<std::setprecision(4); 
            for(int x=0; x<W; ++x) 
                outfile<<x<<"\t" <<avg_ptr[x]<<"\t" <<W_elements[x]<<"\t" <<out_avg_ptr[x]<<"\n";
            outfile.close();
        }
        else 
            std::cerr<<"Warning: Could not open "<<debugfilename<<" for writing.\n";
		//print statistics on out_avg_ptr
		float sum=0.0f;
		float min_out=std::numeric_limits<float>::max();
		float max_out=std::numeric_limits<float>::lowest();
		for(int x=0; x<W; ++x) {
			float val=out_avg_ptr[x];
			sum+=val;
			if(val<min_out) min_out=val;
			if(val>max_out) max_out=val;
		}
		float mean=sum/W;
		std::println("Output averages ({}): min={:.4f}, max={:.4f}, mean={:.4f}",debugfilename,min_out,max_out,mean);
    }

    // Step 4: Return result
    return result;
}

cv::Mat dehaze(cv::Mat image)
{
	cv::Mat dehazed;
	// Convert to float for processing
	image.convertTo(dehazed,CV_32F,1.0/255.0);
	// Apply a simple dehazing algorithm (placeholder)
	cv::Mat darkChannel;
	cv::erode(dehazed,darkChannel,cv::Mat(),cv::Point(-1,-1),15);
	cv::Mat transmission=1.0-darkChannel;
	cv::normalize(transmission, transmission, 0, 1, cv::NORM_MINMAX);

	return dehazed;
}
