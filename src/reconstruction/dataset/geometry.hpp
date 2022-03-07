/**
 * ©2020-2022 ALPhANOV (https://www.alphanov.com/) All Rights Reserved
 * Author: Barjou Emile
 */

#pragma once

/**
 * @brief Class representing the geometry of the reconstruction
 * 
 */
class Geometry {
protected:
    Parameters *_parameters;

    glm::vec3 source_pos, source_at, object_center;
    std::vector<glm::mat4x4> projection_matrices_mat4;
    //std::vector<std::vector<float>> prm_g.projection_matrices; //contains the 4x4 projection matrix and the vec4 viewport vector for each angles 

public:
    Geometry(Parameters *parameters) : 
        _parameters(parameters)
    {
        std::cout << "Creating geometry..." << std::flush;

        //Angles
        if(prm_g.angle_list.empty()) {
            for(int i = 0; i < prm_g.projections; ++i){
                prm_g.angle_list.push_back(i*(prm_g.angle/prm_g.projections));
            }
        } else if (prm_g.angle_list.size() != prm_g.projections) {
            throw std::runtime_error("The size of the angle list is different from the number of projections");
        }

        //Volume dimension
        prm_g.vwidth = prm_g.dwidth;
        prm_g.vheight = prm_g.dheight;
        if(prm_g.heli_step >= 1) {
            prm_g.vheight += int32_t(std::ceil(((prm_g.projections-1.0f)/prm_g.heli_step)*(prm_g.heli_offset/prm_g.vx)));
        }

        //Source-space
        source_pos = {0.0f, 0.0f, 0.0f};
        source_at = source_pos+glm::vec3{0.0f, 0.0f, prm_g.sd};
        glm::vec3 axis_proj = source_at+(glm::vec3{-prm_d.sx+prm_d.rx, -prm_d.sy, 0.0f});
        object_center = source_pos+glm::normalize(axis_proj-source_pos)*prm_g.so;

        //To object-space
        source_pos -= object_center;
        source_at -= object_center;
        object_center = glm::vec3{0.0f, 0.0f, 0.0f};
        prm_g.orig = object_center+glm::vec3{-0.5f*prm_g.vwidth*prm_g.vx, -0.5f*prm_g.vheight*prm_g.vx, -0.5f*prm_g.vwidth*prm_g.vx};
        
        computeMVPs();

        std::cout << "Ok." << std::endl;

        std::cout << "Volume dimension : " << prm_g.vwidth << "x" << prm_g.vheight << "x" << prm_g.vwidth << std::endl;
    }

    /**
     * @brief Rotate a vector around a point along x,y and z axis
     * 
     * @param vec 
     * @param origin 
     * @param x (rad)
     * @param y (rad)
     * @param z (rad)
     * @return glm::vec3 the rotated vector
     */
    glm::vec3 rotate(glm::vec3 vec, glm::vec3 origin, float x, float y, float z){
        glm::vec3 v = vec - origin;
        v = glm::rotateX(v, glm::radians(x));
        v = glm::rotateY(v, glm::radians(y));
        v = glm::rotateZ(v, glm::radians(z));
        v += origin;
        return v;
    }

    /**
     * @brief Rotate a vector around a point along arbitrary axis
     * 
     * @param vec 
     * @param origin 
     * @param x (rad)
     * @param y (rad) 
     * @param z (rad) 
     * @param X X axis
     * @param Y Y axis
     * @param Z Z axis
     * @return glm::vec3 the rotated vector
     */
    glm::vec3 rotate(glm::vec3 vec, glm::vec3 origin, float x, float y, float z, glm::vec3 X, glm::vec3 Y, glm::vec3 Z){
        glm::vec3 v = vec - origin;
        v = glm::rotate(v, glm::radians(x), X);
        v = glm::rotate(v, glm::radians(y), Y);
        v = glm::rotate(v, glm::radians(z), Z);
        v += origin;
        return v;
    }

    /**
     * @brief Get the Projection Matrices
     * 
     * @return std::vector<glm::mat4x4> 
     */
    std::vector<glm::mat4x4> getMatrices() {
        return projection_matrices_mat4;
    }

private:
    /**
     * @brief Compute the 3-points corners coordinates of the detector for a given configuration
     * 
     * @param pos source position
     * @param at point the source is facing on the detector
     * @param module index of the module of the detector
     * @return std::tuple<glm::vec3, glm::vec3, glm::vec3> (upper-left, upper-right, bottom-left)
     */
    std::tuple<glm::vec3, glm::vec3, glm::vec3> detectorFromSource(glm::vec3 pos, glm::vec3 at, int64_t module){
        glm::vec3 v = glm::vec3{0.0f, 1.0f, 0.0f};
        glm::vec3 u = glm::normalize(glm::cross(v,glm::vec3(at-pos)));
        glm::vec3 n = glm::normalize(glm::vec3(at-pos));

        glm::vec3 detector = at-u*prm_d.sx-v*prm_d.sy + (prm_md[module].offset_x*u*prm_d.px + prm_md[module].offset_y*v*prm_d.px + prm_md[module].offset_z*n*prm_d.px);

        uint64_t width = prm_md[module].end_x() - prm_md[module].start_x();
        uint64_t height = prm_md[module].end_y() - prm_md[module].start_y();

        glm::vec3 detector_ul = detector - u*(0.5f*width*prm_d.px) - v*(0.5f*height*prm_d.px);
        glm::vec3 detector_ur = detector_ul + u*(width*prm_d.px);
        glm::vec3 detector_bl = detector_ul + v*(height*prm_d.px);

        detector_ul = rotate(detector_ul, detector, prm_md[module].yaw, prm_md[module].pitch, prm_md[module].roll, u, v, glm::normalize(at-pos));
        detector_ur = rotate(detector_ur, detector, prm_md[module].yaw, prm_md[module].pitch, prm_md[module].roll, u, v, glm::normalize(at-pos));
        detector_bl = rotate(detector_bl, detector, prm_md[module].yaw, prm_md[module].pitch, prm_md[module].roll, u, v, glm::normalize(at-pos));

        return std::make_tuple(detector_ul, detector_ur, detector_bl);
    }

    /**
     * @brief Compute the projection matrix for a given source and detector position
     * 
     * @param pos source position
     * @param at detector position
     * @param module 
     * @return glm::mat4 
     */
    glm::mat4 computeGeneralizedProjection(glm::vec3 pos, glm::vec3 at, int64_t module){
        glm::vec3 pe = pos;

        auto [pa, pb, pc] = detectorFromSource(pos, at, module);

        //Orientation of the detector
        auto vr = glm::normalize(pb-pa);
        auto vu = glm::normalize(pc-pa);
        auto vn = glm::normalize(glm::cross(vu, vr));

        //Source->corner vector
        auto va = pa-pe;
        auto vb = pb-pe;
        auto vc = pc-pe;

        float d = -glm::dot(va,vn);
        float n = 1.0f;
        float f = prm_g.sd*2.0f;

        //Detector offsets
        auto l = glm::dot(vr,va)*(n/d);
        auto r = glm::dot(vr,vb)*(n/d);
        auto b = glm::dot(vu,va)*(n/d);
        auto t = glm::dot(vu,vc)*(n/d);

        auto P = (glm::mat4(
            (2*n)/(r-l),    0.0f,        (r+l)/(r-l),  0.0f,
            0.0f,           (2*n)/(t-b), (t+b)/(t-b),  0.0f,
            0.0f,           0.0f,        -(f+n)/(f-n), -(2*f*n)/(f-n),
            0.0f,           0.0f,        -1.0f,        0.0f
        ));

        auto Mt = (glm::mat4(
            vr.x, vr.y, vr.z, 0.0f, 
            vu.x, vu.y, vu.z, 0.0f,
            vn.x, vn.y, vn.z, 0.0f,
            0.0f, 0.0f, 0.0f, 1.0f
        ));

        auto T = (glm::mat4(
            1.0f,    0.0f,    0.0f,    -pe.x,
            0.0f,    1.0f,    0.0f,    -pe.y,
            0.0f,    0.0f,    1.0f,    -pe.z,
            0.0f,    0.0f,    0.0f,    1.0f
        ));

        //Viewport parameters
        float vpl = float(prm_md[module].start_x());
        float vpr = float(prm_md[module].end_x());
        float vpb = float(prm_md[module].start_y());
        float vpt = float(prm_md[module].end_y());

        auto S = (glm::mat4(
            (vpr-vpl)*0.5f, 0.0f,         0.0f, (vpr+vpl)*0.5f,
            0.0f,         (vpt-vpb)*0.5f, 0.0f, (vpt+vpb)*0.5f,
            0.0f,         0.0f,         0.5f, 0.5f,
            0.0f,         0.0f,         0.0f, 1.0f
        ));

        //return S*P*Mt*T;
        return T*Mt*P*S;
    }

    /**
     * @brief Compute the projection matrix of each detector and each modules
     * 
     */
    void computeMVPs(){
        projection_matrices_mat4.clear();
        prm_g.projection_matrices.clear();

        prm_g.projection_matrices.resize(prm_r.sit);
        
        for(int64_t sit = 0; sit < prm_r.sit; ++sit) { //For each sub-iterations
            for(int64_t i = sit; i < prm_g.projections; i += prm_r.sit) { //For each angles of the current sub-iteration
                for(int64_t module_index = 0; module_index < int64_t(prm_md.size()); ++module_index) { //For each modules
                    float heli_z_offset = prm_g.heli_offset * std::floor((i-prm_g.projections/2.0f)/prm_g.heli_step);
                    float angle_deg = prm_g.angle_list[i];//float(i)*(prm_g.angle/prm_g.projections);
                    auto rotated_source_pos = rotate(source_pos, object_center, 0.0f, angle_deg, 0.0f);
                    auto MVP = computeGeneralizedProjection(
                        rotated_source_pos+glm::vec3{0,0,heli_z_offset}, 
                        rotate(source_at, object_center, 0.0f, angle_deg, 0.0f)+glm::vec3{0,0,heli_z_offset},
                        module_index
                    );
                    std::vector<float> vp = {float(prm_md[module_index].start_x()), float(prm_md[module_index].end_x()), 
                                                float(prm_md[module_index].start_y()), float(prm_md[module_index].end_y())};

                    projection_matrices_mat4.push_back(MVP);
                    prm_g.projection_matrices[sit].insert(std::end(prm_g.projection_matrices[sit]), glm::begin(MVP), glm::end(MVP));
                    prm_g.projection_matrices[sit].insert(std::end(prm_g.projection_matrices[sit]), std::begin(vp), std::end(vp));
                }
            }
        }
    }
};