#include <gtest/gtest.h>
#include "algorithm.h"
#include "algocore.h"
import std;

void write_log(LogType,const char*,const char*)
{
}

namespace
{
	ContourCalcParameter contour_parameters(double pixel_size=4.0)
	{
		return {240.0*pixel_size/1000.0,ContourFeature::Notch,17.0,420,420,pixel_size,SearchDirection::TopToBottom,128};
	}

	cv::Mat contour_image(double stage_angle,double notch_angle)
	{
		const double radians=stage_angle*std::numbers::pi/180.0;
		const cv::Point center(cvRound(200.0+8.0*std::cos(radians)+5.0*std::sin(radians)),
			cvRound(200.0+8.0*std::sin(radians)-5.0*std::cos(radians)));
		cv::Mat image(420,420,CV_8UC1,cv::Scalar(0));
		cv::circle(image,center,120,cv::Scalar(255),cv::FILLED);
		if(stage_angle==notch_angle) cv::rectangle(image,cv::Point(center.x-15,center.y-130),
			cv::Point(center.x+15,center.y-80),cv::Scalar(0),cv::FILLED);
		return image;
	}

	AlgoResult calibrate(ContourCalcParameter parameters,double notch_angle,double& angle)
	{
		auto result=BeginContourCalculation(parameters);
		if(!result.IsSuccess) return result;
		for(double stage_angle:{180.0,0.0,270.0,90.0})
		{
			auto image=contour_image(stage_angle,notch_angle);
			result=AddContourCalculationImage(image.data,stage_angle);
			if(!result.IsSuccess) return result;
		}
		double x=0.0,y=0.0;
		return EndContourCalculation(&x,&y,&angle);
	}

	void expect_conversion(double stage_x,double stage_y,double expected_x,double expected_y,double tolerance=0.0)
	{
		double wafer_x=0.0,wafer_y=0.0;
		ConvertCoordinates(stage_x,stage_y,&wafer_x,&wafer_y);
		EXPECT_NEAR(wafer_x,expected_x,tolerance);
		EXPECT_NEAR(wafer_y,expected_y,tolerance);
	}
}

class CoordinateConversionTest: public ::testing::Test
{
protected:
	void SetUp() override
	{
		reset_calibration_state();
	}
	void TearDown() override
	{
		reset_calibration_state();
	}
};

TEST_F(CoordinateConversionTest,IdentityBeforeCalibration)
{
	expect_conversion(123.25,-456.5,123.25,-456.5);
}

TEST_F(CoordinateConversionTest,UsesCentersInMicrometersWithoutRotation)
{
	for(double pixel_size:{4.0,7.0})
	{
		SCOPED_TRACE(pixel_size);
		double angle=0.0;
		auto result=calibrate(contour_parameters(pixel_size),90.0,angle);
		ASSERT_TRUE(result.IsSuccess)<<result.ErrorMessage;
		EXPECT_NEAR(angle,197.0,5.0);
		expect_conversion(0.0,0.0,-5.0*pixel_size,8.0*pixel_size,pixel_size);
		expect_conversion(5.0*pixel_size,-8.0*pixel_size,0.0,0.0,pixel_size);
		expect_conversion(123.25,-456.5,123.25-5.0*pixel_size,-456.5+8.0*pixel_size,pixel_size);

		double origin_x=0.0,origin_y=0.0,point_x=0.0,point_y=0.0;
		ConvertCoordinates(0.0,0.0,&origin_x,&origin_y);
		ConvertCoordinates(123.25,-456.5,&point_x,&point_y);
		EXPECT_NEAR(point_x-origin_x,123.25,1e-10);
		EXPECT_NEAR(point_y-origin_y,-456.5,1e-10);
	}
}

TEST_F(CoordinateConversionTest,NextSessionClearsAndReplacesCenters)
{
	const auto parameters=contour_parameters();
	double angle=0.0;
	auto result=calibrate(parameters,90.0,angle);
	ASSERT_TRUE(result.IsSuccess)<<result.ErrorMessage;
	expect_conversion(0.0,0.0,-20.0,32.0,parameters.PixelSize);

	ASSERT_TRUE(BeginContourCalculation(parameters).IsSuccess);
	expect_conversion(123.25,-456.5,123.25,-456.5);
	result=calibrate(parameters,180.0,angle);
	ASSERT_TRUE(result.IsSuccess)<<result.ErrorMessage;
	expect_conversion(0.0,0.0,32.0,20.0,parameters.PixelSize);
}

TEST_F(CoordinateConversionTest,ResetAndFailedBeginClearCenters)
{
	for(bool invalid_begin:{false,true})
	{
		SCOPED_TRACE(invalid_begin);
		auto parameters=contour_parameters();
		double angle=0.0;
		auto result=calibrate(parameters,90.0,angle);
		ASSERT_TRUE(result.IsSuccess)<<result.ErrorMessage;
		expect_conversion(0.0,0.0,-20.0,32.0,parameters.PixelSize);
		if(invalid_begin)
		{
			parameters.PixelSize=0.0;
			EXPECT_FALSE(BeginContourCalculation(parameters).IsSuccess);
		}
		else reset_calibration_state();
		expect_conversion(123.25,-456.5,123.25,-456.5);
	}
}

TEST_F(CoordinateConversionTest,FailedFitDoesNotPublishCenters)
{
	auto parameters=contour_parameters();
	parameters.FixedRadiusPx=240.0;
	double angle=1.0;
	auto result=calibrate(parameters,90.0,angle);
	ASSERT_FALSE(result.IsSuccess);
	EXPECT_STREQ(result.ErrorMessage,"Fit validation failed: radius or edge RMSE outside allowed range");
	expect_conversion(123.25,-456.5,123.25,-456.5);

	result=calibrate(contour_parameters(),90.0,angle);
	ASSERT_TRUE(result.IsSuccess)<<result.ErrorMessage;
	double origin_x=0.0,origin_y=0.0;
	ConvertCoordinates(0.0,0.0,&origin_x,&origin_y);
	EXPECT_FALSE(EndContourCalculation(nullptr,&origin_y,&angle).IsSuccess);
	expect_conversion(0.0,0.0,origin_x,origin_y);
}
