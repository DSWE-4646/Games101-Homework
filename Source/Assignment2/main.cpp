// clang-format off
/**
 * GAMES101 作业2 - 三角形光栅化渲染器
 * 核心流程: 模型变换(M) -> 视图变换(V) -> 投影变换(P) -> 视口变换 -> 光栅化
 */
#include <iostream>
#include <opencv2/opencv.hpp>
#include "rasterizer.hpp"
#include "global.hpp"
#include "Triangle.hpp"

constexpr double MY_PI = 3.1415926;

/**
 * 视图变换矩阵 (View Matrix)
 * 将世界坐标系转换到相机坐标系，本质是将相机移动到原点
 * @param eye_pos 相机在世界坐标系中的位置
 */
Eigen::Matrix4f get_view_matrix(Eigen::Vector3f eye_pos)
{
    Eigen::Matrix4f view = Eigen::Matrix4f::Identity();

    // 将相机平移到原点的逆变换
    Eigen::Matrix4f translate;
    translate << 1,0,0,-eye_pos[0],
                 0,1,0,-eye_pos[1],
                 0,0,1,-eye_pos[2],
                 0,0,0,1;

    view = translate*view;

    return view;
}

/**
 * 模型变换矩阵 (Model Matrix)
 * 将模型从局部坐标系变换到世界坐标系
 * 作业中未实现旋转，返回单位矩阵
 */
Eigen::Matrix4f get_model_matrix(float rotation_angle)
{
    Eigen::Matrix4f model = Eigen::Matrix4f::Identity();
    return model;
}

/**
 * 透视投影矩阵 (Projection Matrix)
 * 将视锥体(Frustum)内的物体投影到[-1,1]^3的规范化立方体中
 * 
 * 投影过程: 透视投影 -> 正交投影
 * 透视投影将视锥体压缩成长方体，再进行正交投影
 * 
 * @param eye_fov      垂直视场角(度)
 * @param aspect_ratio 宽高比
 * @param zNear        近平面距离(正值)
 * @param zFar         远平面距离(正值)
 */
Eigen::Matrix4f get_projection_matrix(float eye_fov, float aspect_ratio, float zNear, float zFar)
{
    Eigen::Matrix4f projection = Eigen::Matrix4f::Identity();

    // 计算视锥体边界
    float FovAngel = (eye_fov * MY_PI / 180.f) * 0.5f;
    float Near = -zNear; // 转为负值(相机看向-Z方向)
    float Far = -zFar;
    float Top = zNear * tan(FovAngel);
    float Bottom = -Top;
    float Right = Top * aspect_ratio;
    float Left = -Right;

    // 透视投影矩阵 = 正交投影矩阵 * 透视->正交压缩矩阵
    Eigen::Matrix4f ProJectionMatrix;
    ProJectionMatrix << 2 * Near / (Right - Left), 0, -(Right + Left) / (Right - Left), 0,
                        0, 2 * Near / (Top - Bottom), -(Top + Bottom) / (Top - Bottom), 0,
                        0, 0, (Far + Near) / (Near - Far), -2 * Far * Near / (Near - Far),
                        0, 0, 1, 0;
    
    projection = ProJectionMatrix * projection;

    return projection;
}

int main(int argc, const char** argv)
{
    float angle = 0;
    bool command_line = false;
    std::string filename = "output.png";

    if (argc == 2)
    {
        command_line = true;
        filename = std::string(argv[1]);
    }

    // 创建光栅化器(700x700像素)
    rst::rasterizer r(700, 700);

    // 相机位置: 在世界坐标系中(0,0,5)，看向-Z方向
    Eigen::Vector3f eye_pos = {0,0,5};

    // 定义两个三角形的顶点位置(世界坐标系)
    std::vector<Eigen::Vector3f> pos
            {
                    {2, 0, -2},    // 三角形1顶点
                    {0, 2, -2},
                    {-2, 0, -2},
                    {3.5, -1, -5}, // 三角形2顶点
                    {2.5, 1.5, -5},
                    {-1, 0.5, -5}
            };

    // 顶点索引: 每个Vector3i定义一个三角形(三个顶点的索引)
    std::vector<Eigen::Vector3i> ind
            {
                    {0, 1, 2}, // 三角形1使用顶点0,1,2
                    {3, 4, 5}  // 三角形2使用顶点3,4,5
            };

    // 每个顶点的颜色(RGB)
    std::vector<Eigen::Vector3f> cols
            {
                    {217.0, 238.0, 185.0}, // 三角形1 - 绿色系
                    {217.0, 238.0, 185.0},
                    {217.0, 238.0, 185.0},
                    {185.0, 217.0, 238.0}, // 三角形2 - 蓝色系
                    {185.0, 217.0, 238.0},
                    {185.0, 217.0, 238.0}
            };

    // 将数据加载到光栅化器中
    auto pos_id = r.load_positions(pos);
    auto ind_id = r.load_indices(ind);
    auto col_id = r.load_colors(cols);

    int key = 0;
    int frame_count = 0;

    // 命令行模式: 直接渲染并保存
    if (command_line)
    {
        r.clear(rst::Buffers::Color | rst::Buffers::Depth);

        r.set_model(get_model_matrix(angle));
        r.set_view(get_view_matrix(eye_pos));
        r.set_projection(get_projection_matrix(45, 1, 0.1, 50));

        r.draw(pos_id, ind_id, col_id, rst::Primitive::Triangle);
        cv::Mat image(700, 700, CV_32FC3, r.frame_buffer().data());
        image.convertTo(image, CV_8UC3, 1.0f);
        cv::cvtColor(image, image, cv::COLOR_RGB2BGR);

        cv::imwrite(filename, image);

        return 0;
    }

    // 交互模式: 按ESC退出
    while(key != 27)
    {
        r.clear(rst::Buffers::Color | rst::Buffers::Depth);

        r.set_model(get_model_matrix(angle));
        r.set_view(get_view_matrix(eye_pos));
        r.set_projection(get_projection_matrix(45, 1, 0.1, 50));

        r.draw(pos_id, ind_id, col_id, rst::Primitive::Triangle);

        cv::Mat image(700, 700, CV_32FC3, r.frame_buffer().data());
        image.convertTo(image, CV_8UC3, 1.0f);
        cv::cvtColor(image, image, cv::COLOR_RGB2BGR);
        cv::imshow("image", image);
        key = cv::waitKey(10);

        std::cout << "frame count: " << frame_count++ << '\n';

        if (key == 's')
        {
            cv::imwrite("output.png", image);
            std::cout << "Saved to output.png\n";
        }
    }

    return 0;
}
// clang-format on