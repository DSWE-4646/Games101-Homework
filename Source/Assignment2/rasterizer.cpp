// clang-format off
//
// Created by goksu on 4/6/19.
//
/**
 * 光栅化器实现
 * 核心功能: 将三角形投影到屏幕并填充像素
 */

#include <opencv2/opencv.hpp>
#include <algorithm>
#include <vector>
#include <tuple>
#include "rasterizer.hpp"
#include <math.h>

float MultiSamplePointOffset[4][2] = {
    {0.25f, 0.25f}, //左下点
    {0.75f, 0.25f}, //右下点
    {0.25f, 0.75f}, //左上点
    {0.75f, 0.75f} //右上点
};

// 加载顶点位置数据到缓冲区，返回缓冲区ID
rst::pos_buf_id rst::rasterizer::load_positions(const std::vector<Eigen::Vector3f> &positions)
{
    auto id = get_next_id();
    pos_buf.emplace(id, positions);

    return {id};
}

// 加载顶点索引数据到缓冲区，返回缓冲区ID
rst::ind_buf_id rst::rasterizer::load_indices(const std::vector<Eigen::Vector3i> &indices)
{
    auto id = get_next_id();
    ind_buf.emplace(id, indices);

    return {id};
}

// 加载顶点颜色数据到缓冲区，返回缓冲区ID
rst::col_buf_id rst::rasterizer::load_colors(const std::vector<Eigen::Vector3f> &cols)
{
    auto id = get_next_id();
    col_buf.emplace(id, cols);

    return {id};
}

// 将Vector3f转换为Vector4f(齐次坐标)
auto to_vec4(const Eigen::Vector3f& v3, float w = 1.0f)
{
    return Vector4f(v3.x(), v3.y(), v3.z(), w);
}

/******************************************作业填写部分******************************************/
/**
 * 判断点(x,y)是否在三角形内部
 * 使用叉积法: 点在三角形三条边的同一侧(左侧)则判定为内部
 * @param x, y 像素坐标
 * @param _v   三角形三个顶点的指针数组
 * @return     true=在内部, false=在外部
 */
static bool insideTriangle(float x, float y, const Vector3f* _v)
{
    /* 计算三条边向量*/
    Eigen::Vector2f AB; //3个边向量
    Eigen::Vector2f BC;
    Eigen::Vector2f CA;
    
    AB << _v[1].x() - _v[0].x(), _v[1].y() - _v[0].y();
    BC << _v[2].x() - _v[1].x(), _v[2].y() - _v[1].y();
    CA << _v[0].x() - _v[2].x(), _v[0].y() - _v[2].y();

    /* 计算点向量*/
    Eigen::Vector2f AP; //3个点向量
    Eigen::Vector2f BP;
    Eigen::Vector2f CP;

    float px = x + .5f;
    float py = y + .5f;
    
    AP << px - _v[0].x(), py - _v[0].y();
    BP << px - _v[1].x(), py - _v[1].y();
    CP << px - _v[2].x(), py - _v[2].y();
    
    /* 计算二维向量叉乘*/
    float a = AB.cross(AP);
    float b = BC.cross(BP);
    float c = CA.cross(CP);
    
    return (a >= 0 && b >= 0 && c >= 0) || (a <= 0 && b <= 0 && c <= 0); //在边上也属于三角形内
}

/**
 * 计算点(x,y)相对于三角形的重心坐标
 * 重心坐标用于插值: P = α*A + β*B + γ*C, 其中 α+β+γ=1
 * @return (α, β, γ) 三个顶点的权重
 */
static std::tuple<float, float, float> computeBarycentric2D(float x, float y, const Vector3f* v)
{
    float c1 = (x*(v[1].y() - v[2].y()) + (v[2].x() - v[1].x())*y + v[1].x()*v[2].y() - v[2].x()*v[1].y()) / (v[0].x()*(v[1].y() - v[2].y()) + (v[2].x() - v[1].x())*v[0].y() + v[1].x()*v[2].y() - v[2].x()*v[1].y());
    float c2 = (x*(v[2].y() - v[0].y()) + (v[0].x() - v[2].x())*y + v[2].x()*v[0].y() - v[0].x()*v[2].y()) / (v[1].x()*(v[2].y() - v[0].y()) + (v[0].x() - v[2].x())*v[1].y() + v[2].x()*v[0].y() - v[0].x()*v[2].y());
    float c3 = (x*(v[0].y() - v[1].y()) + (v[1].x() - v[0].x())*y + v[0].x()*v[1].y() - v[1].x()*v[0].y()) / (v[2].x()*(v[0].y() - v[1].y()) + (v[1].x() - v[0].x())*v[2].y() + v[0].x()*v[1].y() - v[1].x()*v[0].y());
    return std::make_tuple(c1, c2, c3);
    //return {c1, c2, c3}
}

/**
 * 绘制函数 - 渲染管线入口
 * 流程: MVP变换 -> 齐次除法 -> 视口变换 -> 光栅化
 */
void rst::rasterizer::draw(pos_buf_id pos_buffer, ind_buf_id ind_buffer, col_buf_id col_buffer, Primitive type)
{
    auto& buf = pos_buf[pos_buffer.pos_id];
    auto& ind = ind_buf[ind_buffer.ind_id];
    auto& col = col_buf[col_buffer.col_id];

    // 深度映射参数: 将z从[0.1,50]映射到[0,1]
    float f1 = (50 - 0.1) / 2.0;
    float f2 = (50 + 0.1) / 2.0;

    // MVP组合变换矩阵
    Eigen::Matrix4f mvp = projection * view * model;
    for (auto& i : ind)
    {
        Triangle t;
        // 对每个顶点应用MVP变换
        Eigen::Vector4f v[] = {
                mvp * to_vec4(buf[i[0]], 1.0f),
                mvp * to_vec4(buf[i[1]], 1.0f),
                mvp * to_vec4(buf[i[2]], 1.0f)
        };
        // 齐次除法: 将裁剪空间转换到NDC空间[-1,1]^3
        for (auto& vec : v) {
            vec /= vec.w();
        }
        // 视口变换: 将NDC[-1,1]映射到屏幕空间[0,width]×[0,height]
        for (auto & vert : v)
        {
            vert.x() = 0.5*width*(vert.x()+1.0);
            vert.y() = 0.5*height*(vert.y()+1.0);
            vert.z() = vert.z() * f1 + f2;  // 深度值映射
        }

        for (int i = 0; i < 3; ++i)
        {
            t.setVertex(i, v[i].head<3>());
            t.setVertex(i, v[i].head<3>());
            t.setVertex(i, v[i].head<3>());
        }

        auto col_x = col[i[0]];
        auto col_y = col[i[1]];
        auto col_z = col[i[2]];

        t.setColor(0, col_x[0], col_x[1], col_x[2]);
        t.setColor(1, col_y[0], col_y[1], col_y[2]);
        t.setColor(2, col_z[0], col_z[1], col_z[2]);

        rasterize_triangle(t);
    }
    //MSAA最终绘制
}

/******************************************作业填写部分******************************************/
/*MSAA*/
void rst::rasterizer::rasterize_triangle(const Triangle& t) {
    auto v = t.toVector4();

    // 包围盒
    int min_x = std::floor(std::min({t.v[0].x(), t.v[1].x(), t.v[2].x()}));
    int max_x = std::ceil(std::max({t.v[0].x(), t.v[1].x(), t.v[2].x()}));
    int min_y = std::floor(std::min({t.v[0].y(), t.v[1].y(), t.v[2].y()}));
    int max_y = std::ceil(std::max({t.v[0].y(), t.v[1].y(), t.v[2].y()}));
    
        for(int x = min_x; x <= max_x; x++)
        {
            for(int y = min_y; y <= max_y; y++)
            {
                int SamplesCount = 0; // 包含的采样点数量Depracted
                int idx = get_index(x, y);
                float z_interpolated = std::numeric_limits<float>::infinity();
                float w_reciprocal = 0.0f;
                bool bIsUpdate = false;
                
                for (int i = 0; i < 4; i++) //多重采样
                {
                    float MSAAPixelX = x + MultiSamplePointOffset[i][0];
                    float MSAAPixelY = y + MultiSamplePointOffset[i][1];
                    
                    if(insideTriangle(MSAAPixelX, MSAAPixelY, t.v))
                    {
                        SamplesCount++;
                        auto[alpha, beta, gamma] = computeBarycentric2D(MSAAPixelX, MSAAPixelY, t.v);
                
                        // 透视校正插值
                        w_reciprocal = 1.0/(alpha / v[0].w() + beta / v[1].w() + gamma / v[2].w());
                        z_interpolated = alpha * v[0].z() / v[0].w() + beta * v[1].z() / v[1].w() + gamma * v[2].z() / v[2].w();
                        z_interpolated *= w_reciprocal;
                        
                        if (z_interpolated < MSAADepthBuffer[idx][i])
                        {
                            MSAADepthBuffer[idx][i] = z_interpolated;
                            MSAAColorBuffer[idx][i] = t.getColor();
                            bIsUpdate = true;
                        }
                    }
                }
                if (bIsUpdate)
                {
                    depth_buf[idx] = std::min({MSAADepthBuffer[idx][0],MSAADepthBuffer[idx][1],MSAADepthBuffer[idx][2],MSAADepthBuffer[idx][3]});
                    Vector3f RevolveColor = (MSAAColorBuffer[idx][0] + MSAAColorBuffer[idx][1] + MSAAColorBuffer[idx][2] + MSAAColorBuffer[idx][3])/4;
                    set_pixel(Eigen::Vector3f(x, y, depth_buf[idx]), Vector3f(RevolveColor.x(), RevolveColor.y(), RevolveColor.z()));
                }
                
            }
        }
}



/******************************************作业填写部分******************************************/
/**
 * 三角形光栅化 - 作业核心函数
 * 将屏幕空间的三角形离散化为像素
 * 
 * 实现步骤:
 * 1. 计算三角形包围盒(Bounding Box)
 * 2. 遍历包围盒内的每个像素
 * 3. 判断像素中心是否在三角形内
 * 4. 使用重心坐标插值深度值
 * 5. 深度测试(Z-Buffer)后写入颜色
void rst::rasterizer::rasterize_triangle(const Triangle& t) {
    auto v = t.toVector4();

    // 包围盒
    float min_x = std::min({t.v[0].x(), t.v[1].x(), t.v[2].x()});
    float max_x = std::max({t.v[0].x(), t.v[1].x(), t.v[2].x()});
    float min_y = std::min({t.v[0].y(), t.v[1].y(), t.v[2].y()});
    float max_y = std::max({t.v[0].y(), t.v[1].y(), t.v[2].y()});
    
    for(int x = min_x; x <= max_x; x++)
        {
        for(int y = min_y; y <= max_y; y++)
            {  
            if(insideTriangle(x, y, t.v))
                {
                auto[alpha, beta, gamma] = computeBarycentric2D(x+0.5f, y+0.5f, t.v);
                
                // 透视校正插值
                float w_reciprocal = 1.0/(alpha / v[0].w() + beta / v[1].w() + gamma / v[2].w());
                float z_interpolated = alpha * v[0].z() / v[0].w() + beta * v[1].z() / v[1].w() + gamma * v[2].z() / v[2].w();
                z_interpolated *= w_reciprocal;

                int idx = get_index(x, y);
                if (z_interpolated < depth_buf[idx])
                    {
                    // 用框架官方 t.getColor()，不用自己插值
                    set_pixel(Eigen::Vector3f(x, y, z_interpolated), t.getColor());
                    depth_buf[idx] = z_interpolated;
                }
            }
        }
    }
}
*/

/******************************************作业填写部分******************************************/
/**
MSAA ——————错误，仅适用于黑色背景下，存在多个三角形重叠时会导致边缘出现黑色点


const float MsaaSamples[4][2] = {
    {0.25f, 0.25f}, // 左下角点
    {0.75f, 0.25f}, // 右下角点
    {0.25f, 0.75f}, // 左上角点
    {0.75f, 0.75f}, // 右上角点
};

void rst::rasterizer::rasterize_triangle(const Triangle& t) {
    auto v = t.toVector4();

    // 包围盒
    int min_x = std::min({t.v[0].x(), t.v[1].x(), t.v[2].x()});
    int max_x = std::max({t.v[0].x(), t.v[1].x(), t.v[2].x()});
    int min_y = std::min({t.v[0].y(), t.v[1].y(), t.v[2].y()});
    int max_y = std::max({t.v[0].y(), t.v[1].y(), t.v[2].y()});
    
        for(int x = min_x; x <= max_x; x++)
        {
            for(int y = min_y; y <= max_y; y++)
            {
                int SamplesCount = 0; // 包含的采样点数量
                int idx = get_index(x, y);
                float z_interpolated = 0.0f;
                float w_reciprocal = 0.0f;
                float MinZ = std::numeric_limits<float>::infinity();
                
                for (int i = 0; i < 4; i++) //多重采样
                {
                    float MSAAPixelX = x + MsaaSamples[i][0];
                    float MSAAPixelY = y + MsaaSamples[i][1];
                    
                    if(insideTriangle(MSAAPixelX, MSAAPixelY, t.v))
                    {
                        SamplesCount++;
                        auto[alpha, beta, gamma] = computeBarycentric2D(MSAAPixelX, MSAAPixelY, t.v);
                
                        // 透视校正插值
                        w_reciprocal = 1.0/(alpha / v[0].w() + beta / v[1].w() + gamma / v[2].w());
                        z_interpolated = alpha * v[0].z() / v[0].w() + beta * v[1].z() / v[1].w() + gamma * v[2].z() / v[2].w();
                        z_interpolated *= w_reciprocal;
                        
                        if (z_interpolated < MinZ)
                        {
                            MinZ = z_interpolated;
                        }
                    }
                }
                
                if (SamplesCount > 0 && MinZ < depth_buf[idx])
                {
                    depth_buf[idx] = MinZ;
                    set_pixel(Eigen::Vector3f(x, y, depth_buf[idx]), t.getColor() * (SamplesCount / 4.0f));
                }
            }
        }
}
*/

// 设置模型变换矩阵
void rst::rasterizer::set_model(const Eigen::Matrix4f& m)
{
    model = m;
}

// 设置视图变换矩阵
void rst::rasterizer::set_view(const Eigen::Matrix4f& v)
{
    view = v;
}

// 设置投影变换矩阵
void rst::rasterizer::set_projection(const Eigen::Matrix4f& p)
{
    projection = p;
}

/**
 * 清空缓冲区
 * @param buff 指定清空颜色缓冲区、深度缓冲区或两者
 */
void rst::rasterizer::clear(rst::Buffers buff)
{
    if ((buff & rst::Buffers::Color) == rst::Buffers::Color)
    {
        std::fill(frame_buf.begin(), frame_buf.end(), Eigen::Vector3f{0, 0, 0});
        std::fill(MSAAColorBuffer.begin(), MSAAColorBuffer.end(), 
                    std::array<Eigen::Vector3f, 4>{{{0,0,0}, {0,0,0}, {0,0,0}, {0,0,0}}});
    }
    if ((buff & rst::Buffers::Depth) == rst::Buffers::Depth)
    {
        std::fill(depth_buf.begin(), depth_buf.end(), std::numeric_limits<float>::infinity());
        float inf = std::numeric_limits<float>::infinity();
        std::fill(MSAADepthBuffer.begin(), MSAADepthBuffer.end(), std::array<float,4>{inf, inf, inf, inf});
    }
}

// 构造函数: 初始化帧缓冲区和深度缓冲区
rst::rasterizer::rasterizer(int w, int h) : width(w), height(h)
{
    frame_buf.resize(w * h);
    depth_buf.resize(w * h);
    //MSAA Buffer:
    MSAAColorBuffer.resize(w * h);
    MSAADepthBuffer.resize(w * h);
    
}

// 计算像素(x,y)在缓冲区中的一维索引(原点在左下角)
int rst::rasterizer::get_index(int x, int y)
{
    return (height-1-y)*width + x;
}

// 设置像素颜色(写入帧缓冲区)
void rst::rasterizer::set_pixel(const Eigen::Vector3f& point, const Eigen::Vector3f& color)
{
    //old index: auto ind = point.y() + point.x() * width;
    auto ind = (height-1-point.y())*width + point.x();
    frame_buf[ind] = color;

}

// clang-format on