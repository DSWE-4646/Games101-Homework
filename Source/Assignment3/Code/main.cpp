#include <iostream>
#include <opencv2/opencv.hpp>

#include "global.hpp"
#include "rasterizer.hpp"
#include "Triangle.hpp"
#include "Shader.hpp"
#include "Texture.hpp"
#include "OBJ_Loader.h"

// Shared normal-perturbation strength for bump and displacement.
constexpr float kNormalGain = 2.0f;

Eigen::Matrix4f get_view_matrix(Eigen::Vector3f eye_pos)
{
    Eigen::Matrix4f view = Eigen::Matrix4f::Identity();

    Eigen::Matrix4f translate;
    translate << 1,0,0,-eye_pos[0],
                 0,1,0,-eye_pos[1],
                 0,0,1,-eye_pos[2],
                 0,0,0,1;

    view = translate*view;

    return view;
}

Eigen::Matrix4f get_model_matrix(float angle)
{
    Eigen::Matrix4f rotation;
    angle = angle * MY_PI / 180.f;
    rotation << cos(angle), 0, sin(angle), 0,
                0, 1, 0, 0,
                -sin(angle), 0, cos(angle), 0,
                0, 0, 0, 1;

    Eigen::Matrix4f scale;
    scale << 2.5, 0, 0, 0,
              0, 2.5, 0, 0,
              0, 0, 2.5, 0,
              0, 0, 0, 1;

    Eigen::Matrix4f translate;
    translate << 1, 0, 0, 0,
            0, 1, 0, 0,
            0, 0, 1, 0,
            0, 0, 0, 1;

    return translate * rotation * scale;
}

Eigen::Matrix4f get_projection_matrix(float eye_fov, float aspect_ratio, float zNear, float zFar)
{
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

Eigen::Vector3f vertex_shader(const vertex_shader_payload& payload)
{
    return payload.position;
}

Eigen::Vector3f normal_fragment_shader(const fragment_shader_payload& payload)
{
    Eigen::Vector3f return_color = (payload.normal.head<3>().normalized() + Eigen::Vector3f(1.0f, 1.0f, 1.0f)) / 2.f;
    Eigen::Vector3f result;
    result << return_color.x() * 255, return_color.y() * 255, return_color.z() * 255;
    return result;
}

static Eigen::Vector3f reflect(const Eigen::Vector3f& vec, const Eigen::Vector3f& axis)
{
    auto costheta = vec.dot(axis);
    return (2 * costheta * axis - vec).normalized();
}

struct light
{
    Eigen::Vector3f position;
    Eigen::Vector3f intensity;
};

Eigen::Vector3f texture_fragment_shader(const fragment_shader_payload& payload)
{
    Eigen::Vector3f return_color = {0, 0, 0};
    if (payload.texture)
    {
        //对纹理坐标进行clamp，确保在0-1之间，浮点计算误差会导致超出纹理范围
        float u = std::clamp(payload.tex_coords.x(), 0.0f, 1.0f);
        float v = std::clamp(payload.tex_coords.y(), 0.0f, 1.0f);
        return_color = payload.texture -> getColor(u, v);
    }
    Eigen::Vector3f texture_color;
    texture_color << return_color.x(), return_color.y(), return_color.z();

    Eigen::Vector3f ka = Eigen::Vector3f(0.005, 0.005, 0.005);
    Eigen::Vector3f kd = texture_color / 255.f;
    Eigen::Vector3f ks = Eigen::Vector3f(0.7937, 0.7937, 0.7937);

    auto l1 = light{{20, 20, 20}, {500, 500, 500}};
    auto l2 = light{{-20, 20, 0}, {500, 500, 500}};

    std::vector<light> lights = {l1, l2};
    Eigen::Vector3f amb_light_intensity{10, 10, 10};
    Eigen::Vector3f eye_pos{0, 0, 10};

    float p = 150;

    Eigen::Vector3f color = texture_color;
    Eigen::Vector3f point = payload.view_pos;
    Eigen::Vector3f normal = payload.normal;

    Eigen::Vector3f result_color = {0, 0, 0};

    for (auto& light : lights)
    {
        light.intensity = light.intensity * 2.5;  //提高光照强度，否则高光仅在p = 1时才可见
        float Distance = (light.position - point).norm(); //Eigen使用norm()计算距离
        float DistanceSquare = Distance * Distance;
        Eigen::Vector3f LightDirection = (light.position - point).normalized();
        Eigen::Vector3f ViewDirection = (eye_pos - point).normalized();

        //diffuse,漫反射项 = 物体颜色 + 光照强度RGB(分量相乘) + 标量贡献值(float)，std::max是模板函数不支持左右值隐式转换
        Eigen::Vector3f Ld = kd.cwiseProduct(light.intensity / DistanceSquare) * std::max(0.f, normal.dot(LightDirection));
        //Specular
        Eigen::Vector3f HalfVector = (LightDirection + ViewDirection).normalized();
        Eigen::Vector3f Ls = ks.cwiseProduct(light.intensity / DistanceSquare) * std::pow(std::max(0.f, normal.dot(HalfVector)),p);
        
        result_color  += Ls + Ld; //多条光线应该累加
    }
    
    //Ambient，环境光仅计算一次
    Eigen::Vector3f La = ka.cwiseProduct(amb_light_intensity);
    result_color  += La;

    return result_color * 255.f;
}


Eigen::Vector3f phong_fragment_shader(const fragment_shader_payload& payload)
{
    Eigen::Vector3f ka = Eigen::Vector3f(0.005, 0.005, 0.005);
    Eigen::Vector3f kd = payload.color;
    Eigen::Vector3f ks = Eigen::Vector3f(0.7937, 0.7937, 0.7937);

    auto l1 = light{{20, 20, 20}, {500, 500, 500}};
    auto l2 = light{{-20, 20, 0}, {500, 500, 500}};

    std::vector<light> lights = {l1, l2};
    Eigen::Vector3f amb_light_intensity{10, 10, 10};
    Eigen::Vector3f eye_pos{0, 0, 10};

    float p = 150;

    Eigen::Vector3f color = payload.color;
    Eigen::Vector3f point = payload.view_pos;
    Eigen::Vector3f normal = payload.normal.normalized(); //法线单位化
    
    Eigen::Vector3f result_color = {0, 0, 0};
    for (auto& light : lights)
    {
        light.intensity = light.intensity * 5;  //提高光照强度，否则高光仅在p = 1时才可见
        float Distance = (light.position - point).norm(); //Eigen使用norm()计算距离
        float DistanceSquare = Distance * Distance;
        Eigen::Vector3f LightDirection = (light.position - point).normalized();
        Eigen::Vector3f ViewDirection = (eye_pos - point).normalized();

        //diffuse,漫反射项 = 物体颜色 + 光照强度RGB(分量相乘) + 标量贡献值(float)，std::max是模板函数不支持左右值隐式转换
        Eigen::Vector3f Ld = kd.cwiseProduct(light.intensity / DistanceSquare) * std::max(0.f, normal.dot(LightDirection));
        //Specular
        Eigen::Vector3f HalfVector = (LightDirection + ViewDirection).normalized();
        Eigen::Vector3f Ls = ks.cwiseProduct(light.intensity / DistanceSquare) * std::pow(std::max(0.f, normal.dot(HalfVector)),p);
        
        result_color  += Ls + Ld; //多条光线应该累加
    }
    
    //Ambient，环境光仅计算一次
    Eigen::Vector3f La = ka.cwiseProduct(amb_light_intensity);
    result_color  += La;
    
    return result_color * 255.f;
}



Eigen::Vector3f displacement_fragment_shader(const fragment_shader_payload& payload)
{
    Eigen::Vector3f ka = Eigen::Vector3f(0.005, 0.005, 0.005);
    Eigen::Vector3f kd = payload.color;
    Eigen::Vector3f ks = Eigen::Vector3f(0.7937, 0.7937, 0.7937);

    auto l1 = light{{20, 20, 20}, {500, 500, 500}};
    auto l2 = light{{-20, 20, 0}, {500, 500, 500}};

    std::vector<light> lights = {l1, l2};
    Eigen::Vector3f amb_light_intensity{10, 10, 10};
    Eigen::Vector3f eye_pos{0, 0, 10};

    float p = 150;

    Eigen::Vector3f color = payload.color; 
    Eigen::Vector3f point = payload.view_pos;
    Eigen::Vector3f normal = payload.normal;

    float kn = 0.1f;  // 位移高度：最大 0.1 个 view 单位

    Eigen::Vector3f n = normal.normalized();

    // 极坐标切线：以 y 轴为极轴取经线方向；s≈0 时退化，需要兜底
    float x = n.x();
    float y = n.y();
    float z = n.z();
    float s = std::sqrt(x * x + z * z);
    Eigen::Vector3f t;
    if (s > 1e-6f)
        t = Eigen::Vector3f(x * y / s, -s, z * y / s);
    else
        t = Eigen::Vector3f(1, 0, 0);
    t = (t - t.dot(n) * n).normalized();

    Eigen::Vector3f b = n.cross(t);
    Eigen::Matrix3f TBN;
    TBN << t, b, n;

    float maxU = 1.f - 1.f / payload.texture->width;
    float minV = 1.f / payload.texture->height;
    float u = std::clamp(payload.tex_coords.x(), 0.f, maxU);
    float v = std::clamp(payload.tex_coords.y(), minV, 1.f);

    float UShift = 1.f / payload.texture->width;
    float VShift = 1.f / payload.texture->height;
    float h  = payload.texture->getColor(u, v).x() / 255.f;
    float hu = payload.texture->getColor(std::min(u + UShift, maxU), v).x() / 255.f;
    float hv = payload.texture->getColor(u, std::min(v + VShift, 1.f)).x() / 255.f;

    float dU = kNormalGain * (hu - h);
    float dV = kNormalGain * (hv - h);
    Eigen::Vector3f ln(-dU, -dV, 1);

    Eigen::Vector3f ShiftedPoint = point + kn * n * h;
    Eigen::Vector3f NewNormal = (TBN * ln).normalized();


    // Vector t = (x*y/sqrt(x*x+z*z),sqrt(x*x+z*z),z*y/sqrt(x*x+z*z))
    // Vector b = n cross product t
    // Matrix TBN = [t b n]
    // dU = kh * kn * (h(u+1/w,v)-h(u,v))
    // dV = kh * kn * (h(u,v+1/h)-h(u,v))
    // Vector ln = (-dU, -dV, 1)
    // Position p = p + kn * n * h(u,v)
    // Normal n = normalize(TBN * ln)


    Eigen::Vector3f result_color = {0, 0, 0};

    for (auto& light : lights)
    {
        light.intensity = light.intensity * 5;  //提高光照强度，否则高光仅在p = 1时才可见

        float Distance = (light.position - ShiftedPoint).norm(); //Eigen使用norm()计算距离
        float DistanceSquare = Distance * Distance;
        Eigen::Vector3f LightDirection = (light.position - ShiftedPoint).normalized();
        Eigen::Vector3f ViewDirection = (eye_pos - ShiftedPoint).normalized();

        //diffuse,漫反射项 = 物体颜色 + 光照强度RGB(分量相乘) + 标量贡献值(float)，std::max是模板函数不支持左右值隐式转换
        Eigen::Vector3f Ld = kd.cwiseProduct(light.intensity / DistanceSquare) * std::max(0.f, NewNormal.dot(LightDirection));
        //Specular
        Eigen::Vector3f HalfVector = (LightDirection + ViewDirection).normalized();
        Eigen::Vector3f Ls = ks.cwiseProduct(light.intensity / DistanceSquare) * std::pow(std::max(0.f, NewNormal.dot(HalfVector)),p);
        
        result_color  += Ls + Ld; //多条光线应该累加


        // TODO: For each light source in the code, calculate what the *ambient*, *diffuse*, and *specular* 
        // components are. Then, accumulate that result on the *result_color* object.
    }

    //Ambient，环境光仅计算一次
    Eigen::Vector3f La = ka.cwiseProduct(amb_light_intensity);
    result_color  += La;

    return result_color * 255.f;
}


Eigen::Vector3f bump_fragment_shader(const fragment_shader_payload& payload)
{
    
    Eigen::Vector3f ka = Eigen::Vector3f(0.005, 0.005, 0.005); //环境光系数,还没被使用
    Eigen::Vector3f kd = payload.color;
    Eigen::Vector3f ks = Eigen::Vector3f(0.7937, 0.7937, 0.7937);

    auto l1 = light{{20, 20, 20}, {500, 500, 500}};
    auto l2 = light{{-20, 20, 0}, {500, 500, 500}};

    std::vector<light> lights = {l1, l2};
    Eigen::Vector3f amb_light_intensity{10, 10, 10}; //环境光系数,还没被使用
    Eigen::Vector3f eye_pos{0, 0, 10};

    float p = 150;

    Eigen::Vector3f color = payload.color; 
    Eigen::Vector3f point = payload.view_pos;
    Eigen::Vector3f normal = payload.normal;


    float u_shift = 1.0f / payload.texture->width; //纹素变化的步长
    float v_shift = 1.0f / payload.texture->height;

    float maxU = 1.0f - 1.0f / payload.texture->width;
    float minV = 1.0f / payload.texture->height;
    float u = std::clamp(payload.tex_coords.x(), 0.0f, maxU);
    float v = std::clamp(payload.tex_coords.y(), minV, 1.0f);
    //getColor 不做边界检查，而 Release 版 OpenCV 把断言编译掉了，越界会静默读到相邻行的数据
    Eigen::Vector3f CenterHeight = payload.texture -> getColor(u, v);
    Eigen::Vector3f UHeightShift = payload.texture -> getColor(std::min(u + u_shift, maxU), v);
    Eigen::Vector3f VHeightShift = payload.texture -> getColor(u, std::min(v + v_shift, 1.0f));

    //对于灰度图，RGB通道存储的都是相同的高度值，所以只需要取一个通道即可
    //getColor 返回的是 0~255 的原始值，而 hint 的公式假设 h ∈ [0,1]，所以要除以 255
    float h_center = CenterHeight.x() / 255.f;
    float h_u = UHeightShift.x() / 255.f;
    float h_v = VHeightShift.x() / 255.f;

    // 和 displacement 共用同一个法线强度；hint 的 kh*kn=0.02 在 h 归一化后太弱。
    float dU = kNormalGain * (h_u - h_center);
    float dV = kNormalGain * (h_v - h_center);
    Eigen::Vector3f  n = normal.normalized();
    Eigen::Vector3f ln = Eigen::Vector3f(-dU, -dV, 1);
    Eigen::Vector3f t = (std::abs(n.x()) < 0.9f)
                    ? Eigen::Vector3f(1, 0, 0)
                    : Eigen::Vector3f(0, 1, 0);
    Eigen::Vector3f T = (t - t.dot(n) * n).normalized();
    Eigen::Vector3f B = n.cross(T);
    Eigen::Matrix3f TBN;
    TBN << T, B, n;
    Eigen::Vector3f NewNormal = TBN * ln;
    NewNormal = NewNormal.normalized();
    
    
    
    //所以作业这里的注释实际就是 t = uv方向的切线， b = uv方向的副切线，n = 法线 然后dU dV是纹素变化在世界坐标系的变化，ln是叉乘得到的法线，TBN将其转换到世界坐标系
    // Vector t = (x*y/sqrt(x*x+z*z),sqrt(x*x+z*z),z*y/sqrt(x*x+z*z))
    // Vector b = n cross product t
    // Matrix TBN = [t b n]
    // dU = kh * kn * (h(u+1/w,v)-h(u,v))
    // dV = kh * kn * (h(u,v+1/h)-h(u,v))
    // Vector ln = (-dU, -dV, 1)
    // Normal n = normalize(TBN * ln)


    Eigen::Vector3f result_color = {0, 0, 0};

    for (auto& light : lights)
    {
        light.intensity = light.intensity * 5;  //提高光照强度，否则高光仅在p = 1时才可见
        float Distance = (light.position - point).norm(); //Eigen使用norm()计算距离
        float DistanceSquare = Distance * Distance;
        Eigen::Vector3f LightDirection = (light.position - point).normalized();
        Eigen::Vector3f ViewDirection = (eye_pos - point).normalized();

        //diffuse,漫反射项 = 物体颜色 + 光照强度RGB(分量相乘) + 标量贡献值(float)，std::max是模板函数不支持左右值隐式转换
        Eigen::Vector3f Ld = kd.cwiseProduct(light.intensity / DistanceSquare) * std::max(0.f, NewNormal.dot(LightDirection));
        //Specular
        Eigen::Vector3f HalfVector = (LightDirection + ViewDirection).normalized();
        Eigen::Vector3f Ls = ks.cwiseProduct(light.intensity / DistanceSquare) * std::pow(std::max(0.f, NewNormal.dot(HalfVector)),p);
        
        result_color  += Ls + Ld; //多条光线应该累加
    }

    //Ambient，环境光仅计算一次
    Eigen::Vector3f La = ka.cwiseProduct(amb_light_intensity);
    result_color  += La;

    return result_color * 255.f;
}

int main(int argc, const char** argv)
{
    std::vector<Triangle*> TriangleList;

    float angle = 40.f;
    bool command_line = false;

    std::string filename = "TestBump.png";
    objl::Loader Loader;
    std::string obj_path = "models/spot/";

    // Load .obj File
    bool loadout = Loader.LoadFile("models/spot/spot_triangulated_good.obj");
    for(auto mesh:Loader.LoadedMeshes)
    {
        for(int i=0;i<mesh.Vertices.size();i+=3)
        {
            Triangle* t = new Triangle();
            for(int j=0;j<3;j++)
            {
                t->setVertex(j,Vector4f(mesh.Vertices[i+j].Position.X,mesh.Vertices[i+j].Position.Y,mesh.Vertices[i+j].Position.Z,1.0));
                t->setNormal(j,Vector3f(mesh.Vertices[i+j].Normal.X,mesh.Vertices[i+j].Normal.Y,mesh.Vertices[i+j].Normal.Z));
                t->setTexCoord(j,Vector2f(mesh.Vertices[i+j].TextureCoordinate.X, mesh.Vertices[i+j].TextureCoordinate.Y));
            }
            TriangleList.push_back(t);
        }
    }

    rst::rasterizer r(700, 700);

    auto texture_path = "hmap.jpg";
    r.set_texture(Texture(obj_path + texture_path));

    std::function<Eigen::Vector3f(fragment_shader_payload)> active_shader = phong_fragment_shader;

    if (argc >= 2)
    {
        command_line = true;
        filename = std::string(argv[1]);

        if (argc == 3 && std::string(argv[2]) == "texture")
        {
            std::cout << "Rasterizing using the texture shader\n";
            active_shader = texture_fragment_shader;
            texture_path = "spot_texture.png";
            r.set_texture(Texture(obj_path + texture_path));
        }
        else if (argc == 3 && std::string(argv[2]) == "normal")
        {
            std::cout << "Rasterizing using the normal shader\n";
            active_shader = normal_fragment_shader;
        }
        else if (argc == 3 && std::string(argv[2]) == "phong")
        {
            std::cout << "Rasterizing using the phong shader\n";
            active_shader = phong_fragment_shader;
        }
        else if (argc == 3 && std::string(argv[2]) == "bump")
        {
            std::cout << "Rasterizing using the bump shader\n";
            active_shader = bump_fragment_shader;
        }
        else if (argc == 3 && std::string(argv[2]) == "displacement")
        {
            std::cout << "Rasterizing using the bump shader\n";
            active_shader = displacement_fragment_shader;
        }
    }

    Eigen::Vector3f eye_pos = {0,0,10};

    r.set_vertex_shader(vertex_shader);
    r.set_fragment_shader(active_shader);

    int key = 0;
    int frame_count = 0;

    if (command_line)
    {
        r.clear(rst::Buffers::Color | rst::Buffers::Depth);
        r.set_model(get_model_matrix(angle));
        r.set_view(get_view_matrix(eye_pos));
        r.set_projection(get_projection_matrix(45.0, 1, 0.1, 50));

        r.draw(TriangleList);
        cv::Mat image(700, 700, CV_32FC3, r.frame_buffer().data());
        image.convertTo(image, CV_8UC3, 1.0f);
        cv::cvtColor(image, image, cv::COLOR_RGB2BGR);

        cv::imwrite(filename, image);

        return 0;
    }

    while(key != 27)
    {
        r.clear(rst::Buffers::Color | rst::Buffers::Depth);

        r.set_model(get_model_matrix(angle));
        r.set_view(get_view_matrix(eye_pos));
        r.set_projection(get_projection_matrix(45.0, 1, 0.1, 50));

        //r.draw(pos_id, ind_id, col_id, rst::Primitive::Triangle);
        r.draw(TriangleList);
        cv::Mat image(700, 700, CV_32FC3, r.frame_buffer().data());
        image.convertTo(image, CV_8UC3, 1.0f);
        cv::cvtColor(image, image, cv::COLOR_RGB2BGR);

        cv::imshow("image", image);
        cv::imwrite(filename, image);
        key = cv::waitKey(10);

        if (key == 'a' )
        {
            angle -= 0.1;
        }
        else if (key == 'd')
        {
            angle += 0.1;
        }

    }
    return 0;
}
