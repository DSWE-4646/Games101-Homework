#include "Triangle.hpp"
#include "rasterizer.hpp"
#include <Eigen/Eigen>
#include <iostream>
#include <opencv2/opencv.hpp>

constexpr double MY_PI = 3.1415926;

Eigen::Matrix4f get_view_matrix(Eigen::Vector3f eye_pos)
{
    Eigen::Matrix4f view = Eigen::Matrix4f::Identity();

    Eigen::Matrix4f translate;
    translate << 1, 0, 0, -eye_pos[0], 0, 1, 0, -eye_pos[1], 0, 0, 1,
        -eye_pos[2], 0, 0, 0, 1;

    view = translate * view;

    return view;
}

Eigen::Matrix4f get_model_matrix(float rotation_angle) //返回角度相应的旋转矩阵
{
    Eigen::Matrix4f model = Eigen::Matrix4f::Identity(); //初始化为单位矩阵，为了拓展性才这么写？或者防止空值？

    float Angel = rotation_angle * MY_PI /180.f;
    Eigen::Matrix4f Rotation;
    Rotation << cos(Angel), sin(Angel), 0, 0,
                -sin(Angel), cos(Angel), 0, 0,
                0, 0, 1, 0,
                0, 0, 0, 1;
    model = Rotation * model;
    
    return model;
}

Eigen::Matrix4f get_projection_matrix(float eye_fov, float aspect_ratio,
                                      float zNear, float zFar)
{
    // Students will implement this function

    Eigen::Matrix4f projection = Eigen::Matrix4f::Identity();

    float FovAngel = (eye_fov * MY_PI / 180.f) * 0.5f;
    float Near = -zNear; //处理zNear和zFar，使其变为负值
    float Far = -zFar;  //处理zNear和zFar，使其变为负值
    float Top = zNear * tan(FovAngel);
    float Bottom = -Top;
    float Right = Top * aspect_ratio;
    float Left = -Right;

    Eigen::Matrix4f ProJectionMatrix;
    ProJectionMatrix << 2 * Near / (Right - Left), 0, -(Right + Left) / (Right - Left), 0,
                        0, 2 * Near / (Top - Bottom), -(Top + Bottom) / (Top - Bottom), 0,
                        0, 0, (Far + Near) / (Near - Far), -2 * Far * Near / (Near - Far),
                        0, 0, 1, 0; //投影变换矩阵
    
    projection = ProJectionMatrix * projection;

    return projection;
}

Eigen::Matrix4f get_rotation(Vector3f axis, float angle)
{
    // 角度转弧度
    float rad = angle * MY_PI / 180.0f;
    
    // 轴向量归一化
    axis = axis.normalized();
    
    float x = axis.x();
    float y = axis.y();
    float z = axis.z();
    
    // 反对称矩阵 K
    Eigen::Matrix3f N;
    N << 0, -z,  y,
         z,  0, -x,
        -y,  x,  0;
    
    // Rodrigues 公式: R = cos(θ)·I + (1-cos(θ))·nnT + sin(θ)·N
    Eigen::Matrix3f I = Eigen::Matrix3f::Identity();
    Eigen::Matrix3f nnt = axis * axis.transpose();
    Eigen::Matrix3f R = cos(rad) * I + (1 - cos(rad)) * nnt + sin(rad) * N;
    
    // 扩展为 4x4 齐次坐标矩阵
    Eigen::Matrix4f rotation = Eigen::Matrix4f::Identity();
    rotation.block<3, 3>(0, 0) = R;
    
    return rotation;
}


int main(int argc, const char** argv)
{
    float angle = 0;
    float XAngle = 0;
    float YAngle = 0;
    bool command_line = false;
    std::string filename = "output.png";

    if (argc >= 3) {
        command_line = true;
        angle = std::stof(argv[2]); // -r by default
        if (argc == 4) {
            filename = std::string(argv[3]);
        }
        else
            return 0;
    }

    rst::rasterizer r(700, 700);

    Eigen::Vector3f eye_pos = {0, 0, 5};

    std::vector<Eigen::Vector3f> pos{{2, 0, -2}, {0, 2, -2}, {-2, 0, -2}};

    std::vector<Eigen::Vector3i> ind{{0, 1, 2}};

    auto pos_id = r.load_positions(pos);
    auto ind_id = r.load_indices(ind);

    int key = 0;
    int frame_count = 0;

    if (command_line) {
        r.clear(rst::Buffers::Color | rst::Buffers::Depth);

        r.set_model(get_model_matrix(angle));
        r.set_view(get_view_matrix(eye_pos));
        r.set_projection(get_projection_matrix(45, 1, 0.1, 50));

        r.draw(pos_id, ind_id, rst::Primitive::Triangle);
        cv::Mat image(700, 700, CV_32FC3, r.frame_buffer().data());
        image.convertTo(image, CV_8UC3, 1.0f);

        cv::imwrite(filename, image);

        return 0;
    }

    while (key != 27) {
        r.clear(rst::Buffers::Color | rst::Buffers::Depth);
        
        // 组合三个轴的旋转
        auto R_z = get_rotation(Vector3f(0, 0, 1), angle);
        auto R_y = get_rotation(Vector3f(0, 1, 0), YAngle);
        auto R_x = get_rotation(Vector3f(1, 0, 0), XAngle);
    
        r.set_model(R_z * R_y * R_x);  // 组合旋转
        //r.set_model(get_model_matrix(angle));
        r.set_view(get_view_matrix(eye_pos));
        r.set_projection(get_projection_matrix(45, 1, 0.1, 50));

        r.draw(pos_id, ind_id, rst::Primitive::Triangle);

        cv::Mat image(700, 700, CV_32FC3, r.frame_buffer().data());
        image.convertTo(image, CV_8UC3, 1.0f);
        cv::imshow("image", image);
        key = cv::waitKey(10);

        std::cout << "frame count: " << frame_count++ << '\n';

        if (key == 'a')
        {
            angle += 1;
            get_rotation(Vector3f(0, 0, 1), angle);
        }
        else if (key == 'd')
        {
            angle -= 1;
            get_rotation(Vector3f(0, 0, 1), angle);
        }
        else if (key == 'w')
        {
            YAngle += 1;
            get_rotation(Vector3f(0, 1, 0), YAngle);
        }
        else if (key == 's')
        {
            YAngle -= 1;
            get_rotation(Vector3f(0, 1, 0), YAngle);
        }
        else if (key == 'q')
        {
            XAngle += 1;
            get_rotation(Vector3f(1, 0, 0), XAngle);
        }
        else if (key == 'e')
        {
            XAngle -= 1;
            get_rotation(Vector3f(1, 0, 0), XAngle);
        }
    }

    return 0;
}
