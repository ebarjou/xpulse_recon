/**
 * ©2020-2022 ALPhANOV (https://www.alphanov.com/) All Rights Reserved
 * Author: Barjou Emile
 */

#pragma once

struct MvpPerLayer {
    uint32_t maxAngles;
    std::vector<std::vector<std::vector<uint16_t>>> mvp_indexes; //For each SIT, for each layer, the list of valid mvp index
    std::vector<std::vector<std::vector<uint16_t>>> image_indexes; //For each SIT, for each layer, the list of image index
    std::vector<float> mvp;
};

struct projData {
    glm::mat4x4 m;
    glm::vec4 v;
};

/**
 * @brief Class representing the geometry of the reconstruction
 * 
 */
class Geometry {
protected:
    Parameters *_parameters;

    glm::vec3 source_pos, source_at, object_center;
    std::vector<glm::mat4x4> projection_matrices_mat4;
    std::vector<glm::vec4> viewports_vec4;
    std::vector<projData> projection_data;
    std::vector<glm::vec2> projection_matrices_layer_minmax;

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

        //Compute voxel size in the center of the volume
        prm_g.vx = prm_d.px*(prm_g.so/prm_g.sd);
        //Compute voxel size in the closest part of the volume to the detector
        //prm_g.vx = prm_d.px*((prm_g.so+prm_g.vx*std::sqrt(prm_g.dwidth))/prm_g.sd);

        //Volume dimension
        prm_g.vwidth = prm_g.dwidth;
        prm_g.vheight = prm_g.dheight + int64_t(std::abs(prm_g.heli_offset/prm_g.vx));

        //Source-space
        source_pos = {0.0f, 0.0f, 0.0f};
        source_at = source_pos+glm::vec3{0.0f, 0.0f, prm_g.sd};
        glm::vec3 axis_proj = source_at+glm::vec3{-prm_d.sx+prm_d.rx, -prm_d.sy, 0.0f};
        object_center = source_pos+glm::normalize(axis_proj-source_pos)*prm_g.so;

        //To object-space
        source_pos -= object_center;
        source_at -= object_center;
        object_center = glm::vec3{0.0f, 0.0f, 0.0f};
        prm_g.orig = object_center+glm::vec3{-0.5f*prm_g.vwidth*prm_g.vx, -0.5f*prm_g.vheight*prm_g.vx, -0.5f*prm_g.vwidth*prm_g.vx};
        
        computeMVPs();

        std::cout << "Ok." << std::endl;

        std::cout << "Volume dimension : " << prm_g.vwidth << "x" << prm_g.vheight << "x" << prm_g.vwidth << std::endl;
        std::cout << "Pixel size : " << prm_d.px << ", Voxel size : " << prm_g.vx << std::endl;
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

    /**
     * @brief Project a point to the detector space for a given projection angle
     * 
     * @param point to project
     * @param angle index of the angle
     * @return glm::vec2 
     */
    inline glm::vec2 project(glm::vec4 point, glm::mat4x4 mvp) {
        glm::vec4 proj = point*mvp;
        glm::vec4 coord = proj/proj.w;
        return glm::vec2{coord.x, coord.y};
    }

    MvpPerLayer getMvpPerLayer() {
        MvpPerLayer output;
        output.maxAngles = 0;
        float* mvp_data = (float*)projection_data.data();
        output.mvp = std::vector(mvp_data, mvp_data+projection_data.size()*20);
        for(int64_t sit = 0; sit < prm_r.sit; ++sit) { //For each sub-iterations
            output.mvp_indexes.push_back(std::vector<std::vector<uint16_t>>(prm_g.vheight));
            output.image_indexes.push_back(std::vector<std::vector<uint16_t>>(prm_g.vheight));
            for(int64_t l = 0; l < prm_g.vheight; ++l) {
                uint16_t sit_index = 0;
                for(int64_t i = sit; i < prm_g.projections; i += prm_r.sit) { //For each angles of the current sub-iteration
                    for(int64_t module_index = 0; module_index < prm_d.module_number; ++module_index) { //For each modules
                        glm::vec2 minmax = minmaxLayerProj(projection_matrices_mat4[i], l);
                        float min = minmax[0];
                        float max = minmax[1];
                        if( !(min >= viewports_vec4[i][3] && max >= viewports_vec4[i][3]) && !(min < viewports_vec4[i][2] && max < viewports_vec4[i][2]) ) {
                            output.mvp_indexes[sit][l].push_back(uint16_t(i)); 
                            output.image_indexes[sit][l].push_back(sit_index); 
                        }
                    }
                    ++sit_index;
                }
                output.maxAngles = std::max(output.maxAngles, uint32_t(output.mvp_indexes[sit][l].size()));
            }
        }
        return output;
    }

private:
    /**
     * @brief Compute the projection matrix for a given source and detector position
     * 
     * @param pos source position
     * @param at detector position
     * @param module 
     * @return glm::mat4 
     */
    glm::mat4 computeGeneralizedProjection(glm::vec3 pe, glm::vec3 pa, glm::vec3 pb, glm::vec3 pc,
                                           float vpl, float vpr, float vpb, float vpt){
        //Orientation of the plane
        auto vr = glm::normalize(pb-pa);
        auto vu = glm::normalize(pc-pa);
        auto vn = glm::normalize(glm::cross(vu, vr));

        //Origin->corner vector
        auto va = pa-pe;
        auto vb = pb-pe;
        auto vc = pc-pe;

        float d = -glm::dot(va,vn);
        float n = 1.0f;
        float f = prm_g.sd*2.0f;

        //Plane offsets
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

        //Viewport
        auto S = (glm::mat4(
            (vpr-vpl)*0.5f, 0.0f,         0.0f, (vpr+vpl)*0.5f,
            0.0f,         (vpt-vpb)*0.5f, 0.0f, (vpt+vpb)*0.5f,
            0.0f,         0.0f,         0.5f, 0.5f,
            0.0f,         0.0f,         0.0f, 1.0f
        ));

        return T*Mt*P*S;
    }

    /**
     * @brief Compute the 3-points corners coordinates of the detector for a given configuration
     * 
     * @param pos source position
     * @param at point the source is facing on the detector
     * @param module index of the module of the detector
     * @return std::tuple<glm::vec3, glm::vec3, glm::vec3> (upper-left, upper-right, bottom-left)
     */
    std::tuple<glm::vec3, glm::vec3, glm::vec3, glm::vec3> detectorFromSource(glm::vec3 pos, glm::vec3 at){
        glm::vec3 v = glm::vec3{0.0f, 1.0f, 0.0f};
        glm::vec3 u = glm::normalize(glm::cross(v,glm::vec3(at-pos)));
        glm::vec3 n = glm::normalize(glm::vec3(at-pos));

        uint64_t width = prm_g.dwidth/prm_d.module_number;
        uint64_t height = prm_g.dheight;

        glm::vec3 detector = at-u*prm_d.sx-v*prm_d.sy;
        // Offset for each modules
        // + (prm_md[module].offset_x*u*prm_d.px + prm_md[module].offset_y*v*prm_d.px + prm_md[module].offset_z*n*prm_d.px);

        glm::vec3 detector_ul = detector - u*(0.5f*width*prm_d.px) - v*(0.5f*height*prm_d.px);
        glm::vec3 detector_ur = detector_ul + u*(width*prm_d.px);
        glm::vec3 detector_bl = detector_ul + v*(height*prm_d.px);

        /*detector_ul = rotate(detector_ul, detector, prm_md[module].yaw, prm_md[module].pitch, prm_md[module].roll, u, v, glm::normalize(at-pos));
        detector_ur = rotate(detector_ur, detector, prm_md[module].yaw, prm_md[module].pitch, prm_md[module].roll, u, v, glm::normalize(at-pos));
        detector_bl = rotate(detector_bl, detector, prm_md[module].yaw, prm_md[module].pitch, prm_md[module].roll, u, v, glm::normalize(at-pos));*/
        
        return std::make_tuple(detector_ul, detector_ur, detector_bl, detector);
    }

    glm::mat4 computeDetectorProjection(int64_t angle_index, int64_t module) {
        float heli_current_offset = (prm_g.heli_offset/prm_g.projections) * angle_index - prm_g.heli_offset*0.5f;
        
        float angle_deg = prm_g.angle_list[angle_index];
        auto rotated_source_pos = rotate(source_pos, object_center, 0.0f, angle_deg, 0.0f)+glm::vec3{0,heli_current_offset,0};
        auto rotated_source_at = rotate(source_at, object_center, 0.0f, angle_deg, 0.0f)+glm::vec3{0,heli_current_offset,0};
        
        auto [pa, pb, pc, center] = detectorFromSource(rotated_source_pos, rotated_source_at);
        auto pe = rotated_source_pos;

        glm::vec3 module_center = center+glm::normalize(rotated_source_pos-rotated_source_at)*(prm_g.sd-prm_d.module_center_offset_z);
        uint64_t width = prm_g.dwidth/prm_d.module_number;

        float module_angle = prm_d.module_angle_offset.assigned ? prm_d.module_angle_offset() : glm::degrees(2.0f*std::atan2(width*0.5f*prm_d.px, glm::distance(module_center, center)));
        float module_current_angle = module_angle*(module-std::floor(prm_d.module_number/2.0f));
        pa = rotate(pa, module_center, 0, module_current_angle, 0);
        pb = rotate(pb, module_center, 0, module_current_angle, 0);
        pc = rotate(pc, module_center, 0, module_current_angle, 0);
        
        float vpl = float(width*module);
        float vpr = float(width*(module+1));
        float vpb = float(0);
        float vpt = float(prm_g.dheight);

        return computeGeneralizedProjection(pe, pa, pb, pc, vpl, vpr, vpb, vpt);
    }

    /**
     * @brief Compute the projection matrix of each detector and each modules
     * 
     */
    void computeMVPs(){
        projection_matrices_mat4.resize(prm_g.projections*prm_d.module_number);
        viewports_vec4.resize(prm_g.projections*prm_d.module_number);
        projection_data.resize(prm_g.projections*prm_d.module_number);

        prm_g.projection_matrices.clear();
        prm_g.projection_matrices.resize(prm_r.sit);

        for(int64_t sit = 0; sit < prm_r.sit; ++sit) { //For each sub-iterations
            for(int64_t i = sit; i < prm_g.projections; i += prm_r.sit) { //For each angles of the current sub-iteration
                glm::vec2 min_max{prm_g.vheight, 0};
                for(int64_t module_index = 0; module_index < prm_d.module_number; ++module_index) { //For each modules
                    auto MVP = computeDetectorProjection(i, module_index);
                    
                    uint64_t width = prm_g.dwidth/prm_d.module_number;
                    std::vector<float> vp = {float(width*module_index), float(width*(module_index+1)), 
                                             0, float(prm_g.dheight)};

                    projection_matrices_mat4[i*prm_d.module_number+module_index] = MVP;
                    viewports_vec4[i*prm_d.module_number+module_index] = {vp[0],vp[1],vp[2],vp[3]};

                    projection_data[i*prm_d.module_number+module_index].m = MVP;
                    projection_data[i*prm_d.module_number+module_index].v = {vp[0],vp[1],vp[2],vp[3]};

                    prm_g.projection_matrices[sit].insert(std::end(prm_g.projection_matrices[sit]), glm::begin(MVP), glm::end(MVP));
                    prm_g.projection_matrices[sit].insert(std::end(prm_g.projection_matrices[sit]), std::begin(vp), std::end(vp));
                    
                }
            }
        }
    }

    /**
     * @brief 
     * 
     */
    glm::ivec2 minmaxLayerProj(glm::mat4x4 mvp, uint32_t layer) {
        std::vector<glm::vec4> bounds{
            { prm_g.orig.x, prm_g.orig.y-1, -prm_g.orig.z, 1.0f},
            {-prm_g.orig.x, prm_g.orig.y-1, -prm_g.orig.z, 1.0f},
            { prm_g.orig.x, prm_g.orig.y-1,  prm_g.orig.z, 1.0f},
            {-prm_g.orig.x, prm_g.orig.y-1,  prm_g.orig.z, 1.0f},
            { prm_g.orig.x, prm_g.orig.y+1, -prm_g.orig.z, 1.0f},
            {-prm_g.orig.x, prm_g.orig.y+1, -prm_g.orig.z, 1.0f},
            { prm_g.orig.x, prm_g.orig.y+1,  prm_g.orig.z, 1.0f},
            {-prm_g.orig.x, prm_g.orig.y+1,  prm_g.orig.z, 1.0f}
        };
        int64_t min = prm_g.vheight*2;
        int64_t max = -prm_g.vheight;
        int64_t end = bounds.size();
        for(int64_t j = 0; j < end; ++j) {
            glm::vec4 pos = bounds[j] + glm::vec4{0, layer*prm_g.vx, 0, 0};
            int64_t line = int64_t(project(pos, mvp).y);
            min = std::min(min, line);
            max = std::max(max, line);
        }
        return {(int)std::floor(min), (int)std::ceil(max)};
    }
};