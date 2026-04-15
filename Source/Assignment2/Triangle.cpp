//
// Created by LEI XU on 4/11/19.
//
/**
 * 三角形类实现
 */

#include "Triangle.hpp"
#include <algorithm>
#include <array>


// 构造函数: 初始化所有顶点属性为0
Triangle::Triangle() {
    v[0] << 0,0,0;
    v[1] << 0,0,0;
    v[2] << 0,0,0;

    color[0] << 0.0, 0.0, 0.0;
    color[1] << 0.0, 0.0, 0.0;
    color[2] << 0.0, 0.0, 0.0;

    tex_coords[0] << 0.0, 0.0;
    tex_coords[1] << 0.0, 0.0;
    tex_coords[2] << 0.0, 0.0;
}

void Triangle::setVertex(int ind, Vector3f ver){
    v[ind] = ver;
}
void Triangle::setNormal(int ind, Vector3f n){
    normal[ind] = n;
}

// 设置顶点颜色(输入范围0-255，内部存储为0-1)
void Triangle::setColor(int ind, float r, float g, float b) {
    if((r<0.0) || (r>255.) ||
       (g<0.0) || (g>255.) ||
       (b<0.0) || (b>255.)) {
        fprintf(stderr, "ERROR! Invalid color values");
        fflush(stderr);
        exit(-1);
    }

    color[ind] = Vector3f((float)r/255.,(float)g/255.,(float)b/255.);
    return;
}
void Triangle::setTexCoord(int ind, float s, float t) {
    tex_coords[ind] = Vector2f(s,t);
}

// 将三个顶点转换为齐次坐标(Vector4f)
std::array<Vector4f, 3> Triangle::toVector4() const
{
    std::array<Eigen::Vector4f, 3> res;
    std::transform(std::begin(v), std::end(v), res.begin(), [](auto& vec) { return Eigen::Vector4f(vec.x(), vec.y(), vec.z(), 1.f); });
    return res;
}
