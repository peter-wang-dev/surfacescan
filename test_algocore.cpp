#include <gtest/gtest.h>
#include "algocore.h"
#include "calibcore.h"
#if defined(_WIN32)
#include <stdlib.h>
#define SET_ENV(key, val) _putenv_s(key, val)
#else
#include <stdlib.h>
#define SET_ENV(key, val) setenv(key, val, 1)
#endif
import std;
std::string path_output="test_output"; 
//std::string path_output="C:/Users/cyber/test_output"; 
std::string path_input="d:/dev/cns_testinput"; 
std::string path_userinput;
//const std::string path_input="./test_output/"; 
namespace
{
    std::vector<cv::Point2d> calibration_circle(cv::Point2d center,double radius,int count=360)
    {
        std::vector<cv::Point2d> points;
        for(int i=0;i<count;++i)
        {
            const double angle=2.0*std::numbers::pi*i/count;
            points.emplace_back(center.x+radius*std::cos(angle),center.y+radius*std::sin(angle));
        }
        return points;
    }
}
TEST(calibration,FrameWeightsMatchPython)
{
    const std::vector<double> uniform(4,1.0);
    EXPECT_EQ(calibration::frame_weights_voronoi({0.0,90.0,180.0,270.0}),uniform);
    EXPECT_EQ(calibration::frame_weights_voronoi({-90.0,0.0,90.0,180.0}),uniform);
    EXPECT_EQ(calibration::frame_weights_density({0.0,90.0,180.0,270.0},0.0),uniform);
    const std::vector<double> duplicate={1.25,0.625,0.625,1.25,1.25};
    EXPECT_EQ(calibration::frame_weights_voronoi({0.0,90.0,90.0,180.0,270.0}),duplicate);
    const std::vector<double> angles={0.0,5.0,90.0,180.0,270.0};
    const std::vector<double> expected_voronoi={0.6597222222222222,0.625,1.2152777777777777,1.25,1.25};
    const std::vector<double> expected_density={0.6490795217308788,0.631049535016132,1.1956728031884607,
        1.262099070032264,1.262099070032264};
    const auto voronoi=calibration::frame_weights_voronoi(angles);
    const auto density=calibration::frame_weights_density(angles,90.0);
    ASSERT_EQ(voronoi.size(),expected_voronoi.size());
    ASSERT_EQ(density.size(),expected_density.size());
    for(size_t i=0;i<angles.size();++i)
    {
        EXPECT_NEAR(voronoi[i],expected_voronoi[i],1e-12);
        EXPECT_NEAR(density[i],expected_density[i],1e-12);
    }
    EXPECT_NEAR(std::accumulate(voronoi.begin(),voronoi.end(),0.0),5.0,1e-12);
    EXPECT_NEAR(std::accumulate(density.begin(),density.end(),0.0),5.0,1e-12);
    EXPECT_TRUE(calibration::frame_weights_voronoi({}).empty());
    EXPECT_TRUE(calibration::frame_weights_density({},0.0).empty());
    EXPECT_DOUBLE_EQ(calibration::frame_weights_voronoi({12.0}).front(),1.0);
}
TEST(calibration,CircleFitAndOutlierFiltering)
{
    EXPECT_DOUBLE_EQ(calibration::median_value({9.0,1.0,7.0,3.0}),5.0);
    EXPECT_DOUBLE_EQ(calibration::normalize_angle_360(-90.0),270.0);
    EXPECT_DOUBLE_EQ(calibration::normalize_angle_360(720.5),0.5);
    const cv::Point2d center(10000.0,-20000.0);
    auto points=calibration_circle(center,5000.0);
    double x=0.0,y=0.0,radius=0.0;
    ASSERT_TRUE(calibration::ls_fit_circle(points,x,y,radius));
    EXPECT_NEAR(x,center.x,1e-6);
    EXPECT_NEAR(y,center.y,1e-6);
    EXPECT_NEAR(radius,5000.0,1e-6);
    points.emplace_back(30000.0,10000.0);
    points.emplace_back(-20000.0,20000.0);
    EXPECT_EQ(calibration::ransac_fit_circle(points,x,y,radius,400,0.01),360);
    EXPECT_NEAR(x,center.x,1e-6);
    EXPECT_NEAR(y,center.y,1e-6);
    EXPECT_NEAR(radius,5000.0,1e-6);
    size_t removed=0;
    auto kept=calibration::remove_isolated_residual_outliers(points,x,y,radius,&removed);
    EXPECT_EQ(removed,2u);
    EXPECT_EQ(kept.size(),360u);
    const auto coherent=calibration_circle(center,5012.0);
    kept=calibration::remove_isolated_residual_outliers(coherent,center.x,center.y,5000.0,&removed);
    EXPECT_EQ(kept.size(),coherent.size());
    EXPECT_EQ(removed,0u);
    EXPECT_FALSE(calibration::ls_fit_circle({},x,y,radius));
}
TEST(calibration,NotchSegmentFiltering)
{
    const cv::Point2d center(300.0,300.0);
    auto edges=calibration_circle(center,120.0,720);
    cv::Point2d valley,fitted_center;
    double depth=0.0;
    EXPECT_FALSE(calibration::find_notch_with_fit(edges,cv::Size(600,600),center.x,center.y,120.0,720,
        valley,fitted_center,depth));
    for(size_t i=165;i<=195;++i)
    {
        const double angle=2.0*std::numbers::pi*static_cast<double>(i)/720.0;
        edges[i]=cv::Point2d(center.x+80.0*std::cos(angle),center.y+80.0*std::sin(angle));
    }
    ASSERT_TRUE(calibration::find_notch_with_fit(edges,cv::Size(600,600),center.x,center.y,120.0,689,
        valley,fitted_center,depth));
    EXPECT_GT(depth,30.0);
    EXPECT_LT(cv::norm(valley-center),90.0);
    std::vector<unsigned char> excluded;
    const auto kept=calibration::remove_notch_contour_segment(edges,valley,6.0,20.0,&excluded);
    EXPECT_GT(kept.size(),500u);
    EXPECT_LT(kept.size(),edges.size());
    ASSERT_EQ(excluded.size(),edges.size());
    EXPECT_EQ(static_cast<size_t>(std::count(excluded.begin(),excluded.end(),1)),edges.size()-kept.size());
    for(const auto& point:kept) EXPECT_NEAR(cv::norm(point-center),120.0,1e-6);
}
TEST(calibration,EdgeExtractionAndSubpixel)
{
    cv::Mat image(300,300,CV_8UC1,cv::Scalar(0));
    EXPECT_TRUE(calibration::get_edge_points(image,128).empty());
    cv::circle(image,cv::Point(150,150),80,cv::Scalar(80),cv::FILLED);
    const auto edges=calibration::get_edge_points(image,200);
    ASSERT_GT(edges.size(),100u);
    double x=0.0,y=0.0,radius=0.0;
    ASSERT_TRUE(calibration::ls_fit_circle(edges,x,y,radius));
    EXPECT_NEAR(x,150.0,1.0);
    EXPECT_NEAR(y,150.0,1.0);
    EXPECT_NEAR(radius,80.0,2.0);
    const auto refined=calibration::refine_edges_subpixel(image,edges,x,y,1.5,0.25);
    ASSERT_EQ(refined.size(),edges.size());
    for(size_t i=0;i<edges.size();++i)
    {
        EXPECT_TRUE(std::isfinite(refined[i].x)&&std::isfinite(refined[i].y));
        EXPECT_LE(cv::norm(refined[i]-edges[i]),1.75);
    }
    image.setTo(cv::Scalar(0));
    cv::rectangle(image,cv::Rect(0,0,40,300),cv::Scalar(255),cv::FILLED);
    cv::circle(image,cv::Point(200,150),40,cv::Scalar(255),cv::FILLED);
    const auto interior=calibration::get_edge_points(image,128);
    ASSERT_TRUE(calibration::ls_fit_circle(interior,x,y,radius));
    EXPECT_NEAR(x,200.0,1.0);
    EXPECT_NEAR(y,150.0,1.0);
    EXPECT_NEAR(radius,40.0,2.0);
}
TEST(calibration,JointSolverRecoversHarmonics)
{
    const std::vector<double> expected_coeffs={2.0,-1.0,0.5,0.75};
    const auto orders=calibration::harm_orders(3);
    std::vector<calibration::FrameData> frames;
    std::vector<cv::Point2d> initial_centers;
    for(double angle:{0.0,30.0,35.0,90.0,150.0,210.0,270.0,330.0})
    {
        double x=0.0,y=0.0;
        calibration::predict_center(1234.0,5678.0,8.0,-5.0,angle,x,y);
        const double radius=500.0+calibration::harmonics_value(angle*std::numbers::pi/180.0,expected_coeffs,orders);
        frames.push_back({angle,calibration_circle(cv::Point2d(x,y),radius,180),false});
        initial_centers.emplace_back(x+3.0,y-2.0);
    }
    const std::vector<unsigned char> initial_valid(frames.size(),1);
    for(auto mode:{ContourFrameWeightMode::None,ContourFrameWeightMode::Voronoi,ContourFrameWeightMode::Density})
    {
        SCOPED_TRACE(static_cast<int>(mode));
        double ox=0.0,oy=0.0,dx=0.0,dy=0.0,rmse=0.0;
        std::vector<double> coeffs;
        ASSERT_TRUE(calibration::joint_solve_v16(frames,0.0,500.0,3,initial_centers,initial_valid,0.05,mode,45.0,
            ox,oy,dx,dy,coeffs,rmse));
        EXPECT_NEAR(ox,1234.0,1e-4);
        EXPECT_NEAR(oy,5678.0,1e-4);
        EXPECT_NEAR(dx,8.0,1e-4);
        EXPECT_NEAR(dy,-5.0,1e-4);
        ASSERT_EQ(coeffs.size(),expected_coeffs.size());
        for(size_t i=0;i<coeffs.size();++i) EXPECT_NEAR(coeffs[i],expected_coeffs[i],1e-4);
        EXPECT_LT(rmse,1e-4);
    }
}
TEST(calibration,JointSolverFallbackAndFailures)
{
    std::vector<calibration::FrameData> frames;
    for(double angle:{0.0,120.0,240.0})
    {
        double x=0.0,y=0.0;
        calibration::predict_center(10.0,20.0,3.0,4.0,angle,x,y);
        frames.push_back({angle,calibration_circle(cv::Point2d(x,y),80.0),false});
    }
    double ox=0.0,oy=0.0,dx=0.0,dy=0.0,rmse=0.0;
    std::vector<double> coeffs;
    ASSERT_TRUE(calibration::joint_solve_v16(frames,0.0,80.0,0,{}, {},0.05,ContourFrameWeightMode::Voronoi,0.0,
        ox,oy,dx,dy,coeffs,rmse));
    EXPECT_NEAR(ox,10.0,1e-6);
    EXPECT_NEAR(oy,20.0,1e-6);
    EXPECT_NEAR(dx,3.0,1e-6);
    EXPECT_NEAR(dy,4.0,1e-6);
    EXPECT_TRUE(coeffs.empty());
    EXPECT_LT(rmse,1e-6);
    frames.pop_back();
    EXPECT_FALSE(calibration::joint_solve_v16(frames,0.0,80.0,0,{}, {},0.05,ContourFrameWeightMode::Voronoi,0.0,
        ox,oy,dx,dy,coeffs,rmse));
    frames.resize(3);
    for(auto& frame:frames) frame.edges.clear();
    EXPECT_FALSE(calibration::joint_solve_v16(frames,0.0,80.0,0,{}, {},0.05,ContourFrameWeightMode::Voronoi,0.0,
        ox,oy,dx,dy,coeffs,rmse));
}
TEST(core,flatten)
{ 
	auto datadir=path_input+"/dehaze";
	//auto datadir="D:/dev/test_output";
	for(int kr=0; kr<11; ++kr)
	{//index of ring 
		auto imgpath=std::format("{}/ch1r{}.png",datadir,kr);
		cv::Mat img8b=cv::imread(imgpath,CV_8U);
		cv::Mat img;
		img8b.convertTo(img,CV_16U,16.0f); // Convert to 16-bit unsigned
		ASSERT_FALSE(img.empty())<<"Failed to load image: "<<imgpath;
		cv::Mat flattened=flatten(img,1.0f,1.0f,std::format("{}/flattenning_{}.txt",path_output,kr));
		ASSERT_FALSE(flattened.empty())<<"Flattening failed.";
		//cv::imwrite(std::format("{}/flatten_result_{}.png",path_output,kr),flattened*4.0f);
	}
}
//TEST(core,dehaze)
//{
//	//auto datadir=path_input+"/dehaze";
//	auto datadir="D:/dev/test_output";
//	int kr=3;//index of ring 
//	auto imgpath=std::format("{}/ch1r{}.png",datadir,kr); 
//	cv::Mat img=cv::imread(imgpath,CV_8U);
//	ASSERT_FALSE(img.empty())<<"Failed to load image: "<<imgpath;
//	cv::Mat dehazed=dehaze(img);
//	ASSERT_FALSE(dehazed.empty())<<"Dehazing failed.";
//	cv::imwrite(path_output+"/dehaze_result.png",dehazed*25.0f);
//}
TEST(core, merge)
{
    SET_ENV("OPENCV_IO_MAX_IMAGE_PIXELS", "1099511627776");
    const std::filesystem::path datadir = path_input + "/merge1deg";
	std::println("Merging images from directory: {}",datadir.string());
    std::vector<cv::Mat> fullmaps;

    ASSERT_TRUE(std::filesystem::exists(datadir)) << "Input directory does not exist: " << datadir;

    for (const auto& entry : std::filesystem::directory_iterator(datadir))
    {
        if (!entry.is_regular_file())
            continue; 
        std::string extension = entry.path().extension().string(); 
        std::ranges::transform(extension, extension.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); }); 
        if (extension != ".tif" && extension != ".tiff" &&extension!=".png")
            continue;
		std::println("Found image file: {}",entry.path().string());

        cv::Mat image = cv::imread(entry.path().string(), cv::IMREAD_UNCHANGED);
        ASSERT_FALSE(image.empty()) << "Failed to load image: " << entry.path(); 
        fullmaps.push_back(image);
    }

    ASSERT_FALSE(fullmaps.empty()) << "No TIFF or PNG images found in: " << datadir; 
	ASSERT_TRUE(fullmaps.size()>1)<<"At least two images are required for merging.";   
    cv::Mat merged=fullmaps[0]*0;
    for (const cv::Mat& image : fullmaps)
    {
        ASSERT_EQ(image.size(), fullmaps.front().size())
            << "All images must have the same dimensions.";
        ASSERT_EQ(image.type(), fullmaps.front().type())
            << "All images must have the same type.";
		merged+=image;
    }

    //cv::Mat merged;
    //cv::merge(fullmaps, merged); 
	//cv::Mat merged=fullmaps[0]-fullmaps[1];
    ASSERT_TRUE(cv::imwrite((std::filesystem::path(path_output)/"merge.png").string(),merged))<<"Failed to save merged image: ";
    ASSERT_TRUE(cv::imwrite((std::filesystem::path(path_output)/"merge_4ev.png").string(),merged*16))<<"Failed to save merged image: ";
    //ASSERT_TRUE(cv::imwrite((std::filesystem::path(path_output)/"merge_8ev.png").string(),merged*256))<<"Failed to save merged image: ";
    //ASSERT_TRUE(cv::imwrite((std::filesystem::path(path_output)/"merge_10ev.png").string(),merged*1024))<<"Failed to save merged image: ";
}
int main(int argc, char **argv) 
{
    std::filesystem::create_directories(path_output); // Create test output directory if it doesn't exist
    SET_ENV("OPENCV_IO_MAX_IMAGE_PIXELS", "1099511627776");
    ::testing::InitGoogleTest(&argc, argv);
	if(argc>1) 
		path_userinput=argv[1];
    return RUN_ALL_TESTS();
}
