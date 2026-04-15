//
// Created by LEI XU on 4/11/19.
//
/**
 * 三角形类
 * 存储三角形的顶点属性: 位置、颜色、纹理坐标、法线
 */

#ifndef RASTERIZER_TRIANGLE_H
#define RASTERIZER_TRIANGLE_H

#include <Eigen/Eigen>


using namespace Eigen;
class Triangle{

public:
    Vector3f v[3];         // 顶点位置(逆时针顺序: v0, v1, v2)
    /*Per vertex values*/
    Vector3f color[3];     // 每个顶点的颜色(RGB)
    Vector2f tex_coords[3];// 每个顶点的纹理坐标(u,v)
    Vector3f normal[3];    // 每个顶点的法线向量

    //Texture *tex;
    Triangle();

    void setVertex(int ind, Vector3f ver);  // 设置第ind个顶点的位置
    void setNormal(int ind, Vector3f n);    // 设置第ind个顶点的法线
    void setColor(int ind, float r, float g, float b); // 设置第ind个顶点的颜色(0-255)
    Vector3f getColor() const { return color[0]*255; } // 获取三角形颜色(仅使用第一个顶点)
    void setTexCoord(int ind, float s, float t);       // 设置第ind个顶点的纹理坐标
    std::array<Vector4f, 3> toVector4() const;  // 转换为齐次坐标(Vector4f数组)
};






#endif //RASTERIZER_TRIANGLE_H
