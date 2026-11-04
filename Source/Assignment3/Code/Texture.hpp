//
// Created by LEI XU on 4/27/19.
//

#ifndef RASTERIZER_TEXTURE_H
#define RASTERIZER_TEXTURE_H
#include "global.hpp"
#include <Eigen/Eigen>
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <cmath>
class Texture{
private:
    cv::Mat image_data;

public:
    Texture(const std::string& name)
    {
        image_data = cv::imread(name);
        cv::cvtColor(image_data, image_data, cv::COLOR_RGB2BGR);
        width = image_data.cols;
        height = image_data.rows;
    }

    int width, height;

    Eigen::Vector3f getColor(float u, float v)
    {
        auto u_img = u * width;
        auto v_img = (1 - v) * height;
        auto color = image_data.at<cv::Vec3b>(v_img, u_img);
        return Eigen::Vector3f(color[0], color[1], color[2]);
    }

    Eigen::Vector3f getColorBilinear(float u, float v)
{
    auto u_img = u * width;
    auto v_img = (1 - v) * height;
    
    //首先取uv的整数部分所在像素点
    int x0 = static_cast<int>(std::floor(u_img));
    int y0 = static_cast<int>(std::floor(v_img));
    
    //取uv所在像素的右上角
    int x1 = std::min(x0 + 1, width - 1);
    int y1 = std::min(y0 + 1, height - 1);

    Eigen::Vector3f ColorQ00 = Eigen::Vector3f(image_data.at<cv::Vec3b>(y0, x0)[0], image_data.at<cv::Vec3b>(y0, x0)[1], image_data.at<cv::Vec3b>(y0, x0)[2]);
    Eigen::Vector3f ColorQ01 = Eigen::Vector3f(image_data.at<cv::Vec3b>(y1, x0)[0], image_data.at<cv::Vec3b>(y1, x0)[1], image_data.at<cv::Vec3b>(y1, x0)[2]);
    Eigen::Vector3f ColorQ10 = Eigen::Vector3f(image_data.at<cv::Vec3b>(y0, x1)[0], image_data.at<cv::Vec3b>(y0, x1)[1], image_data.at<cv::Vec3b>(y0, x1)[2]);
    Eigen::Vector3f ColorQ11 = Eigen::Vector3f(image_data.at<cv::Vec3b>(y1, x1)[0], image_data.at<cv::Vec3b>(y1, x1)[1], image_data.at<cv::Vec3b>(y1, x1)[2]);

    float Denominator = 1.f / ((x1 - x0) * (y1 - y0));
    float Q11 = Denominator * (x1 - u_img) * (y1 - v_img);
    float Q21 = Denominator * (u_img - x0) * (y1 - v_img);
    float Q12 = Denominator * (x1 - u_img) * (v_img - y0);
    float Q22 = Denominator * (u_img - x0) * (v_img - y0);
    Eigen::Vector3f ReturnColor = Q11 * ColorQ00 + Q21 * ColorQ01 + Q12 * ColorQ10 + Q22 * ColorQ11;

    return ReturnColor;

}

};
#endif //RASTERIZER_TEXTURE_H