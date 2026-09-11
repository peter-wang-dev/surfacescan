#include "algocore.h"
#include <omp.h>
import std;

std::vector<double> estimate_column_intensity(cv::Mat image)
{
	if(image.empty())
		return {};

	if(image.type()!=CV_8UC1&&image.type()!=CV_16UC1)
		throw std::invalid_argument("Input image must be single-channel 8-bit or 16-bit");
	int W=image.cols;
	// Calculate the average intensity of each column
	cv::Mat col_avg;
	cv::reduce(image,col_avg,0,cv::REDUCE_AVG,CV_64F);
	double* avg_ptr=col_avg.ptr<double>(0);
	std::vector<double> column_intensities(W);
	for(int x=0; x<W; ++x) {
		column_intensities[x]=avg_ptr[x];
	}
	return column_intensities;
}
template <typename T>
void estimate_column_intensity_stat_impl(const cv::Mat& image, std::vector<double>& result) {
    int W = image.cols;
    int H = image.rows;
    const int N = 10000;
    const int trim_count = N / 100; 
    const int start_idx = trim_count;
    const int end_idx = N - trim_count; 
    const int valid_samples = end_idx - start_idx;

    // Use a fixed base seed to ensure cross-run determinism
    const unsigned int base_seed = 12345; 

    #pragma omp parallel
    {
        // Allocate thread-local buffer once per thread to maintain memory efficiency
        std::vector<T> samples(N);

        #pragma omp for schedule(dynamic)
        for (int x = 0; x < W; ++x) {
            // Seed RNG deterministically per column
            // This guarantees column 'x' always gets the same sequence of random numbers
            std::mt19937 rng(base_seed + x);
            std::uniform_int_distribution<int> dist(0, H - 1);
            
            // 1. Sample N=10000 values randomly
            for (int i = 0; i < N; ++i) {
                int y = dist(rng);
                samples[i] = image.ptr<T>(y)[x];
            }

            // 2. Sort the values
            std::sort(samples.begin(), samples.end());

            // 3. Calculate average of the middle 80%
            double sum = 0.0;
            for (int i = start_idx; i < end_idx; ++i) 
                sum += samples[i];
            
            result[x] = sum / valid_samples;
        }
    }
}

// Main interface function
std::vector<double> estimate_column_intensity_stat(cv::Mat image) {
    if (image.empty()) {
        return {};
    }

    if (image.type() != CV_8UC1 && image.type() != CV_16UC1) {
        throw std::invalid_argument("Input image must be single-channel 8-bit or 16-bit");
    }

    std::vector<double> result(image.cols, 0.0);

    // Dispatch to the correct template instantiation based on OpenCV type
    if (image.type() == CV_8UC1) {
        estimate_column_intensity_stat_impl<uchar>(image, result);
    } else if (image.type() == CV_16UC1) {
        estimate_column_intensity_stat_impl<ushort>(image, result);
    }

    return result;
}

cv::Mat flatten(cv::Mat image,std::string debugfilename) 
{
    if (image.empty()) 
        return image;
    
    if (image.type() != CV_8UC1 && image.type() != CV_16UC1) 
        throw std::invalid_argument("Input image must be single-channel 8-bit or 16-bit");

    int W = image.cols;
    int H = image.rows;

    // Step 1: Calculate the average intensity of each column
	//auto colavg=estimate_column_intensity(image);
	auto colavg=estimate_column_intensity_stat(image);

    // Step 2: Scale vector so max element is 1.0f, and precompute inverse factors
    double min_val, max_val;
	const auto [min_it, max_it] = std::minmax_element(colavg.begin(), colavg.end());
	min_val = *min_it;
	max_val = *max_it;

    std::vector<double> inv_scales(W, 1.0f);
    std::vector<double> W_elements(W, 0.0f); // Store for debugging

    if (max_val > 0.0) {
        for (int x = 0; x < W; ++x) {
            W_elements[x] = colavg[x] / static_cast<double>(max_val); // Calculate scaled W
            if (colavg[x] > 1e-6) { 
                inv_scales[x] = static_cast<double>(max_val) / colavg[x];
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
                outfile<<x<<"\t" <<colavg[x]<<"\t" <<W_elements[x]<<"\t" <<out_avg_ptr[x]<<"\n";
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
