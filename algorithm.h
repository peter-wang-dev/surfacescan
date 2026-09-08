// Algorithm.h
// C++ 端算法接口结构与调用协议，与 C# IAlgorithmInterface2 对应
//
// 本文件应被 LibParticleSystem.dll 工程引用，确保 C++ 内部返回的 AlgoResult
// 与 C# 侧 [StructLayout(LayoutKind.Sequential, CharSet=CharSet.Ansi)]
// 布局完全一致。（Struct 大小: 260 bytes）

#ifndef ALGORITHM_H
#define ALGORITHM_H

#ifdef _WIN32
#  define ALGO_API __declspec(dllexport) 
#else
#  define ALGO_API __attribute__((visibility("default")))
#include <cstddef>
#endif

// ─────────────────────────────────────────────────────────────
// 统一返回值结构
// C# 对应: AlgoResult (IAlgorithmInterface2.cs)
//          [StructLayout(Sequential, CharSet=CharSet.Ansi)]
//          → CharSet.Ansi = 单字节字符 = char
// ─────────────────────────────────────────────────────────────

#define ALGO_ERROR_MSG_MAX 256   // 与 C# [MarshalAs(SizeConst=256)] 保持一致

#pragma pack(push, 4)
struct AlgoResult
{
    /// <summary>结果标志。true = 成功，false = 失败</summary>
    bool IsSuccess;

    /// <summary>错误描述信息（纯英文，单字节编码）。成功时为空串（首字符 '\0'）。</summary>
    char ErrorMessage[ALGO_ERROR_MSG_MAX];

    // ── 便利工厂方法 ──

    /// <summary>快速构造成功结果</summary>
    static AlgoResult Success()
    {
        AlgoResult r = {};
        r.IsSuccess = true;
        return r;               // ErrorMessage 已零初始化
    }

    /// <summary>快速构造失败结果。msg 超过 255 字符时自动截断。</summary>
    static AlgoResult Failure(const char* msg)
    {
        AlgoResult r = {};
        r.IsSuccess = false;
        if (msg)
        {
            size_t i = 0;
            while (i < ALGO_ERROR_MSG_MAX - 1 && msg[i] != '\0')
            {
                r.ErrorMessage[i] = msg[i];
                ++i;
            }
            r.ErrorMessage[i] = '\0';
        }
        return r;
    }
};
#pragma pack(pop)

// ─────────────────────────────────────────────────────────────
// 轮廓特征计算 / Stage中心计算 共用参数
// C# 对应: ContourCalcParameter (IAlgorithmInterface2.cs)
// ─────────────────────────────────────────────────────────────

enum class ContourFeature : int
{
    Notch = 0,
    Flat  = 1
};

enum class SearchDirection : int
{
    LeftToRight  = 0,
    RightToLeft  = 1,
    TopToBottom  = 2,
    BottomToTop  = 3
};

struct ContourCalcParameter
{
    double WaferSize;           // 晶圆直径（mm）
    ContourFeature Feature;     // 轮廓特征类型
    double FeatureAngle;        // 预估角度（度）
    int    ImageHeight;         // 图像高度（像素）
    int    ImageWidth;          // 图像宽度（像素）
    double PixelSize;           // 像素尺寸（mm/pixel）
    SearchDirection Direction;  // 寻边方向
    int    Threshold;           // 二值化阈值（0-255）
};

// ─────────────────────────────────────────────────────────────
// 点坐标结构
// C# 对应: PointStruct (IAlgorithmInterface2.cs)
// ─────────────────────────────────────────────────────────────

#pragma pack(push, 4)
struct PointStruct
{
    float X;
    float Y;
};
#pragma pack(pop)

// ─────────────────────────────────────────────────────────────
// 像素点结构（mmap中单条滤波/邻域像素记录）
// C# 对应: PixelPointStruct (IAlgorithmInterface2.cs)
// sizeof = 24 bytes (Pack=4)
// ─────────────────────────────────────────────────────────────

#pragma pack(push, 4)
struct PixelPointStruct
{
    int           ID;          // 数据点ID
    PointStruct   Point;       // 晶圆坐标（mm）
    unsigned short RingIndex;  // 所属圈号
    unsigned short RowIndex;   // 行号（LineIndex）
    unsigned short ColIndex;   // 列号（PixelIndex）
    float         Value;       // 强度值（校准后）
    unsigned char ResultFlag;  // 结果标记：0=缺陷周围背景，1=缺陷本身
};
#pragma pack(pop)

// ─────────────────────────────────────────────────────────────
// 缺陷列表信息描述符
// C# 对应: DefectInfoListStruct (IAlgorithmInterface2.cs)
// ─────────────────────────────────────────────────────────────

#define MMAP_NAME_MAX 128

#pragma pack(push, 4)
struct DefectInfoListStruct
{
    char Name[MMAP_NAME_MAX];   // 共享内存名称
    int  DataSize;              // 数据总大小（字节）
    int  ParticleCount;         // 粒子/缺陷总数
};
#pragma pack(pop)

// ─────────────────────────────────────────────────────────────
// 缺陷类型
// C# 对应: DefectType (IAlgorithmInterface2.cs)
// ─────────────────────────────────────────────────────────────

enum class DefectType : int
{
    NONE    = 0,
    LPD     = 1,
    LPDN    = 2,
    AREA    = 3,
    SCRATCH = 4,
    SLIPLINE = 5
};

// ─────────────────────────────────────────────────────────────
// 探测通道类型
// C# 对应: DetectChannel (IAlgorithmInterface2.cs)
// ─────────────────────────────────────────────────────────────

enum class DetectChannel : int
{
    None   = 0,
    Narrow = 1,
    Wide1  = 2,
    Wide2  = 3
};

// ─────────────────────────────────────────────────────────────
// 缺陷信息结构（内存映射文件中的单条记录）
// C# 对应: DefectInfoStruct (IAlgorithmInterface2.cs)
// ─────────────────────────────────────────────────────────────

#pragma pack(push, 4)
struct DefectInfoStruct
{
    // ── 标识 ──
    int           DefectID;            // 缺陷唯一ID
    DetectChannel ChannelID;           // 所属探测通道
    DefectType    Type;          // 缺陷类型
    int           BinCode;             // 分级编码

    // ── 位置 ──
    float CoordR;                      // 中心点R坐标（um）
    float CoordT;                      // 中心点Theta坐标（degree）
    float CoordX;                      // 中心点X坐标（um）
    float CoordY;                      // 中心点Y坐标（um）

    // ── 尺寸 ──
    float XSize;                       // 缺陷X方向尺寸（um）
    float YSize;                       // 缺陷Y方向尺寸（um）
    float Area;                        // 缺陷面积（um²）

    // ── 质量 ──
    float DSize;                       // 缺陷DSize值
    float MaxDSize;                    // 缺陷DSize最大值
    float SNR;                         // 信噪比
    float SumSNR;                      // SNR和值

    // ── 标定 ──
    float PixelSize;                   // 像素尺寸（um/pixel）
    float SpaceResolution;             // 空间分辨率

    // ── 数据指针（mmap内偏移/计数）──
    int RawPixelsOffset;               // 原始像素数据偏移（字节），存储 PixelPointStruct[]
    int RawPixelsCount;                // 原始像素数据数量（字节）
    int OutlinePointsOffset;           // 轮廓点数据偏移（字节），存储 PointStruct[]
    int OutlinePointsCount;            // 轮廓点数据数量（字节）
};
#pragma pack(pop)

// ─────────────────────────────────────────────────────────────
// 日志回调
// C# 对应: LogMessageCallBack (IAlgorithmInterface2.cs)
// ─────────────────────────────────────────────────────────────

enum class LogType : int
{
    None  = 0,
    Fatal = 1,
    Error = 2,
    Warning  = 3,
    Debug = 4,
    Info  = 5,
    All   = 6
};

typedef void(*LogMessageCallBack)(LogType logType, const char* source, const char* message);

// ─────────────────────────────────────────────────────────────
// 导出函数声明
// ─────────────────────────────────────────────────────────────

#ifdef __cplusplus
extern "C" {
#endif

// === 基础设施 ===

/// <summary>初始化算法模块，清空内部缓存和中间状态</summary>
ALGO_API AlgoResult Initialize();

/// <summary>注册C#端日志回调，C++内部日志通过此回调输出到C#日志系统</summary>
ALGO_API AlgoResult SetLogMessageCallBack(LogMessageCallBack callBackPointer);

/// <summary>中止当前正在执行的算法操作，释放已分配的资源。线程安全，可在任意线程调用。</summary>
ALGO_API AlgoResult Abort();

// === 直方图峰值计算 ===

/// <summary>计算直方图峰值和标准差（高斯拟合），用于BaseIntensity校准等独立场景</summary>
ALGO_API AlgoResult FitGaussianPeak(const float* values, int count, float* peakValue, float* sigma);

// === Stage中心计算 ===

/// <summary>开始计算相机与Stage旋转中心的相对位置</summary>
ALGO_API AlgoResult BeginCenterCalculation(ContourCalcParameter parameters);

/// <summary>添加一幅边缘图像，C++内部累积计算</summary>
ALGO_API AlgoResult AddCenterCalculationImage(const unsigned char* imageData, double stageAngle);

/// <summary>结束计算，获取结果。dx/dy单位为mm，angle单位为度</summary>
ALGO_API AlgoResult EndCenterCalculation(double* dx, double* dy, double* angle);

// === 轮廓特征计算 ===

/// <summary>开始计算晶圆轮廓特征（Notch/Flat）</summary>
ALGO_API AlgoResult BeginContourCalculation(ContourCalcParameter parameters);

/// <summary>添加一幅轮廓图像，C++内部累积计算</summary>
ALGO_API AlgoResult AddContourCalculationImage(const unsigned char* imageData, double stageAngle);

/// <summary>结束计算，获取圆心偏移和旋转角</summary>
ALGO_API AlgoResult EndContourCalculation(double* transX, double* transY, double* angle);

// === 通道处理流水线 ===

/// <summary>
/// 开始处理一个检测通道。
/// ring: {
///     "FrameWidth": 1024,              // int: 单帧宽度（像素）
///     "FrameHeight": 256,              // int: 单帧高度（像素）
///     "TotalRings": 10,                // int: 总圈数
///     "IntensityCalibration": "Haze",   // string: 光强校准策略 None|Haze|SelfHaze|RefHaze
///     "ImageSaveDirectory": ""
/// }
/// cluster: {
///     "WaferRadius": float,            // 晶圆半径mm
///     "ClusterSearchRadius": float,    // 聚类搜索半径um（在此半径内的点被归为同一聚类）
///     "ClusterMinPixelCount": int      // 聚类最小像素点数（少于此数的聚类被丢弃）
/// }
/// classification: {
///     "PixelSize": float,              // 像素尺寸um
///     "SpaceResolution": float,        // 空间分辨率（um/pixel，影响缺陷尺寸计算）
///     "MinDefectSize": float,          // 最小缺陷尺寸（um）
///     "MaxDefectSize": float,          // 最大缺陷尺寸（um）
///     "AreaMinPixelCount": int,        // AREA类型粒子的最小像素数（低于此数的聚类不归为AREA）
///     "SatToAreaPixelCount": int,      // 饱和像素转AREA的阈值（单聚类中饱和像素超过此数则归为AREA）
///     "AREADisplayPixelSize": float,   // AREA缺陷显示的像素尺寸mm
///     "NeighborPixelCount": int,       // 领域像素采样数量（用于计算单像素周边背景）
///     "SNRThreshold": float,           // SNR阈值（二次分类用）
///     "MaxDSizeKey": float,            // DSize分级曲线的最大Key值（um）
///     "MaxDSizeValue": float           // DSize分级曲线的最大Value值（分类编号上限）
/// }
/// coordCaliSetting: {
///     "XOffset": float,                // Wafer中心X方向偏移量mm（用于坐标补偿）
///     "YOffset": float,                // Wafer中心Y方向偏移量mm
///     "ThetaOffset": float             // Wafer旋转角偏移度
/// }
/// DSizeCurve: {
///     "CurvePoints": [                 // array: 曲线节点
///         {"Intensity": 0.0, "DSize": 0.0},
///         {"Intensity": 100.0, "DSize": 25.0},
///         ...
///     ]
/// }
/// </summary>
ALGO_API AlgoResult BeginChannelProcess(DetectChannel channelID,
    const char* ring, const char* cluster, const char* classification,
    const char* coordCaliSetting, const char* DSizeCurve);

/// <summary>
/// 开始处理当前通道的一圈。
/// processSetting: {
///     "EnableOverLoadDectect": bool,       // 是否启用OverLoad（过曝）检测
///     "PixelFilterAutoThreshold": bool,    // 启用像素筛选自动阈值(一般只能计算低阈值)
///     "PixelFilterSigma1": float,          // 自动阈值第一轮Sigma剔除系数（通常2.0~4.0）
///     "PixelFilterSigma2": float,          // 自动阈值第二轮Sigma剔除系数（通常1.5~2.5）
///     "PixelFilterLowThreshold": float,    // 像素筛选低阈值（手动模式，高于此值才保留）
///     "PixelFilterHighThreshold": float,   // 像素筛选高阈值（高于此值直接判定为缺陷）
///     "FilteredPixelMaxCount": int,        // 最大筛选像素数（性能保护，超过则截断）
///     "WaferValidRadius": float,           // 晶圆有效区域半径mm（排除边缘区域）
///     "WaferContourFeature": int,          // 晶圆轮廓类型：0=Notch缺口, 1=Flat平边
///     "WaferFlatAngleRange": float,        // 平边晶圆的角度范围（度，仅WaferNotch=1时有效）
///     "AutoThreshold": bool,               // 是否启用自动阈值：false=固定阈值, true=自适应阈值
///     "SingleSNRThreshold": float,         // 单像素SNR阈值（低于此值的像素被过滤）
///     "AccumulatedSNRThreshold": float,    // 区域累积SNR阈值（SNR总和低于此值被过滤）
///     "HighThreshold": float,              // 高阈值（用于区分强缺陷和弱缺陷）
///     "LowThresholdCoeff": float,          // 低阈值系数（低阈值 = HighThreshold * LowThresholdCoeff）
///     "MedianFilterKernelSize": int,       // 中值滤波核大小（1D滤波的窗口宽度）
///     "GlobalMedianRatio": float           // 全局中值比率（用于背景估计）
/// }
/// hazeCaliSetting: {
///     "CameraID": "",                      // 相机ID字符串
///     "ImageIndex": int,                   // 图像索引（圈编号，从0开始）
///     "SupposedHaze": float,               // 预期Haze值（用于矫正基准，通常为第二圈的Haze均值）
///     "MedianFilterWindowSize": int,       // Haze中值滤波窗口大小（列方向）
///     "MedianFilter2D": bool,              // false=一维中值滤波, true=二维中值滤波(GPU)
///     "HazeVarianceCoeff": float,          // Haze方差系数（用于判定Haze异常区域）
///     "SaveUnCalibratedHaze": bool,        // 是否保存Haze矫正前的数据（调试用）
///     "UnCalibratedHazePath": "",          // Haze矫正前数据保存路径
///     "SaveCalibratingHaze": bool,         // 是否输出Haze中间数据（调试用）
///     "CalibratingHazePath": "",           // Haze中间数据保存路径
///     "SaveCalibratedHaze": bool,          // 是否输出Haze矫正后数据（调试用）
///     "CalibratedHazePath": "",            // Haze矫正后数据保存路径
///     "ShowSmoothedHazeMap": bool          // false=显示原始HazeMap, true=平滑后显示
/// }
/// coordCaliSetting: {
///     "PixelSize": float,                  // 像素物理尺寸（um/pixel）
///     "TriggerStart": {
///         "R": float,                      // 径向轴位置mm
///         "T": float,                      // Theta旋转角度（度）
///         "Z": float
///     },
///     "TriggerEnd": {
///         "R": float,                      // 径向轴位置mm
///         "T": float,                      // Theta旋转角度（度）
///         "Z": float
///     },
///     "RingWidth": float,                  // 当前圈的宽度（pixel）
///     "RingHeight": float,                 // 当前圈的总高度（pixel）
///     "CurrentRing": int,                  // 当前圈索引
///     "TotalRings": int                    // 总圈数
/// }
/// </summary>
ALGO_API AlgoResult BeginRingProcess(DetectChannel channelID, int ringIndex,
    const char* processSetting, const char* hazeCaliSetting, const char* coordCaliSetting);

/// <summary>添加当前圈的一帧图像，C++内部累积。validRows为有效行数（最后一帧可能不满）</summary>
ALGO_API AlgoResult AddRingProcessFrame(DetectChannel channelID, int ringIndex,
    const unsigned char* imageData, int width, int height, int validRows,
    int timestamp, int softBinning);

/// <summary>结束当前圈，C++内部开始拼接+校准+筛选。返回过载标志</summary>
ALGO_API AlgoResult EndRingProcess(DetectChannel channelID, int ringIndex,
    bool* isOverLoad, bool* isHazeOverload);

/// <summary>结束当前通道的处理，并返回该通道的缺陷检测结果（内存映射文件）</summary>
ALGO_API AlgoResult EndChannelProcess(DetectChannel channelID, DefectInfoListStruct* descriptor);

// === 晶圆图像拼接（离线） ===

/// <summary>将各圈扫描图像拼接为完整晶圆全景图（依赖已保存到磁盘的图像文件）</summary>
ALGO_API AlgoResult StitchWaferImage(float scale, float minRatio, float maxRatio,
    const char* ringImagePath, const char* mergeImageSavePath);

// === 结果获取（内存映射文件） ===

/// <summary>
/// 设置DMO融合参数。
/// jsonDMOSetting: {
///     "WaferRadius": float,                // 晶圆半径mm
///     "TotalRings": int,                   // 扫描总圈数
///     "MergeStrategy": string,             // 融合策略 PixelConnectivity|PhysicalDistance
///     "MaxDefectSize": float,              // 缺陷尺寸范围上限um
///     "DMOSearchR": float,                 // DMO融合时的聚类搜索半径mm（跨通道粒子匹配的距离容差）
///     "OvertopRatio": float,               // 颗粒超出背景的比率阈值（通常1.25），用于区分真假缺陷
///     "LPDMergeMode": string,              // LPD融合模式：Maximum最大值, WidePriority宽场优先
///     "SaveLPDDetails": bool,              // 是否保存LPD（Light Point Defect，亮点缺陷）详细信息
///     "ValueSavePath": "",                 // LPD Centroid值保存路径（CSV文件）
///     "DSizeSavePath": "",                 // LPD DSize值保存路径（CSV文件）
///     "LPDNRatioSource": string            // LPDN比率计算类型：Intensity光强比, Size尺寸比
/// }
/// </summary>
ALGO_API AlgoResult SetDMOParameters(const char* jsonDMOSetting);

/// <summary>获取DMO融合后的缺陷检测结果，C++将结果写入共享内存，返回描述符</summary>
ALGO_API AlgoResult GetDMOResults(DefectInfoListStruct* descriptor);

#ifdef __cplusplus
}
#endif

#endif // ALGORITHM_H
