//
// Created by goksu on 4/6/19.
//
/**
 * 光栅化器头文件
 * 定义了缓冲区类型、图元类型和光栅化器类
 */

#pragma once

#include <Eigen/Eigen>
#include <algorithm>
#include "global.hpp"
#include "Triangle.hpp"
using namespace Eigen;

namespace rst
{
    // 缓冲区类型枚举
    enum class Buffers
    {
        Color = 1,  // 颜色缓冲区
        Depth = 2   // 深度缓冲区(Z-Buffer)
    };

    // 缓冲区类型的位运算(支持 Color | Depth)
    inline Buffers operator|(Buffers a, Buffers b)
    {
        return Buffers((int)a | (int)b);
    }

    inline Buffers operator&(Buffers a, Buffers b)
    {
        return Buffers((int)a & (int)b);
    }

    // 图元类型枚举
    enum class Primitive
    {
        Line,
        Triangle
    };

    /*
     * 类型安全的缓冲区ID结构体
     * 防止混淆位置缓冲区、索引缓冲区和颜色缓冲区的ID
     */
    struct pos_buf_id  // 顶点位置缓冲区ID
    {
        int pos_id = 0;
    };

    struct ind_buf_id  // 顶点索引缓冲区ID
    {
        int ind_id = 0;
    };

    struct col_buf_id  // 顶点颜色缓冲区ID
    {
        int col_id = 0;
    };

    /**
     * 光栅化器类
     * 负责将3D图元转换到2D像素
     * 渲染管线: Vertex Shader -> MVP -> Clipping -> 齐次除法 -> Viewport -> 光栅化 -> Fragment Shader
     */
    class rasterizer
    {
    public:
        rasterizer(int w, int h);
        pos_buf_id load_positions(const std::vector<Eigen::Vector3f>& positions);
        ind_buf_id load_indices(const std::vector<Eigen::Vector3i>& indices);
        col_buf_id load_colors(const std::vector<Eigen::Vector3f>& colors);

        void set_model(const Eigen::Matrix4f& m);
        void set_view(const Eigen::Matrix4f& v);
        void set_projection(const Eigen::Matrix4f& p);

        void set_pixel(const Eigen::Vector3f& point, const Eigen::Vector3f& color);

        void clear(Buffers buff);

        void draw(pos_buf_id pos_buffer, ind_buf_id ind_buffer, col_buf_id col_buffer, Primitive type);

        std::vector<Eigen::Vector3f>& frame_buffer() { return frame_buf; }

    private:
        void draw_line(Eigen::Vector3f begin, Eigen::Vector3f end);

        void rasterize_triangle(const Triangle& t);


        // 渲染管线流程: VERTEX SHADER -> MVP -> Clipping -> /.W -> VIEWPORT -> DRAWLINE/DRAWTRI -> FRAGSHADER

    private:
        // MVP变换矩阵
        Eigen::Matrix4f model;      // 模型变换
        Eigen::Matrix4f view;       // 视图变换
        Eigen::Matrix4f projection; // 投影变换

        // 数据缓冲区(使用ID索引)
        std::map<int, std::vector<Eigen::Vector3f>> pos_buf;  // 顶点位置
        std::map<int, std::vector<Eigen::Vector3i>> ind_buf;  // 顶点索引
        std::map<int, std::vector<Eigen::Vector3f>> col_buf;  // 顶点颜色

        std::vector<Eigen::Vector3f> frame_buf;  // 帧缓冲区(存储像素颜色)
        std::vector<float> depth_buf;            // 深度缓冲区(存储像素深度)
        
        //MSAA缓冲区：
        std::vector<std::array<Eigen::Vector3f, 4>> MSAAColorBuffer;  // MSAA颜色缓冲区(存储像素颜色)
        std::vector<std::array<float, 4>> MSAADepthBuffer;  

        int get_index(int x, int y);  // 像素索引计算

        int width, height;  // 屏幕尺寸

        int next_id = 0;
        int get_next_id() { return next_id++; }  // ID生成器
    };
}
