/**
 * ©2020-2022 ALPhANOV (https://www.alphanov.com/) All Rights Reserved
 * Author: Barjou Emile
 */

#pragma once

/**
 * @brief Specialized geometry class that will divide the reconstruction in autonomous chunks
 * 
 */
class GeometryChunk {
    Parameters *_parameters;
    Geometry *_geometry;
    uint64_t _availableMemoryByte, _maxMemoryAllocByte;
    std::vector<float> _layerMinYProj, _layerMaxYProj;
    std::vector<glm::mat4x4>  _projection_matrices_mat4;
public:
    GeometryChunk(Parameters *parameters, Geometry *geometry, uint64_t availableMemoryByte, uint64_t maxMemoryAllocByte) : 
        _parameters(parameters),
        _geometry(geometry),
        _availableMemoryByte(availableMemoryByte),
        _maxMemoryAllocByte(maxMemoryAllocByte),
        _projection_matrices_mat4(_geometry->getMatrices())
    {
        std::cout << "Creating chunk division..." << std::flush;
        computeLayerYBounds();
        computeChunks();
        std::cout << "Ok" << std::endl;
    }

    /**
     * @brief Compute the maximum size of the buffer necessary for the padding of the given chunk, in bytes
     * 
     * @param chunk 
     * @return uint64_t size in bytes
     */
    uint64_t paddingSizeInMemory(Chunk &chunk) {
        int64_t layerSize = prm_g.vwidth*prm_g.vwidth;
        int64_t paddingSize = std::max(chunk.vPadBot,chunk.vPadTop)*layerSize;
        return paddingSize*sizeof(float);
    }

    /**
     * @brief Compute the size of the buffer necessary for the images of the given chunk, in bytes
     * 
     * @param chunk 
     * @return uint64_t size in bytes
     */
    uint64_t imagesSizeInMemory(Chunk &chunk) {
        int64_t lineSize = prm_g.dwidth;
        int64_t imagesSize = chunk.iSize*lineSize*prm_g.concurrent_projections;
        return imagesSize*sizeof(float);
    }
    
    /**
     * @brief Compute the total memory size necessary for the given chunk, in bytes
     * 
     * @param chunk 
     * @return uint64_t size in bytes
     */
    uint64_t sizeInMemory(Chunk &chunk) {
        int64_t layerSize = prm_g.vwidth*prm_g.vwidth;
        int64_t lineSize = prm_g.dwidth;

        int64_t volumeSize = (chunk.vSize+chunk.vPadTop+chunk.vPadBot)*layerSize;
        int64_t imagesSize = chunk.iSize*lineSize*prm_g.concurrent_projections;
        int64_t sumImagesSize = chunk.iSize*lineSize*prm_g.concurrent_projections;

        int64_t totalSize = (volumeSize+imagesSize+sumImagesSize)*sizeof(float);
        return totalSize;
    }

private:
    /**
     * @brief Project a point to the detector space for a given projection angle
     * 
     * @param point to project
     * @param angle index of the angle
     * @return glm::vec2 
     */
    inline glm::vec2 project(glm::vec4 point, uint32_t angle) {
        glm::vec4 proj = point*_projection_matrices_mat4[angle];
        glm::vec4 coord = proj/proj.w;
        return glm::vec2{coord.x, coord.y};
    }

    /**
     * @brief Generate the list of chunks that have to be reconstructed
     * 
     */
    void computeChunks() {
        prm_m2.chunks.clear();
        int64_t firstCompleteLayer = std::distance(_layerMinYProj.begin(), std::upper_bound( _layerMinYProj.begin(), _layerMinYProj.end(), 0));
        int64_t lastCompleteLayer = std::distance(_layerMaxYProj.begin(), std::upper_bound( _layerMaxYProj.begin(), _layerMaxYProj.end(), prm_g.vheight-1));
        std::cout << "[" << firstCompleteLayer << "-" << lastCompleteLayer << "]...";
        prm_m2.chunks.push_back({firstCompleteLayer, 0, 0, 0, 0, 0});
        for(int i = 0; i < lastCompleteLayer-firstCompleteLayer; ++i) {
            Chunk chunk = prm_m2.chunks.back();
            ++chunk.vSize;
            updateImageViewport(chunk);
            updateVolumePadding(chunk);
            if(fitInMemory(chunk)) {
                prm_m2.chunks.back() = chunk;
            } else {
                if(chunk.vSize <= 1) {
                    throw std::runtime_error(std::string("Not enough GPU memory for the chunk division (")
                                                        + std::to_string(chunk.vPadTop)+"-"+std::to_string(chunk.vSize)+"-"+std::to_string(chunk.vPadBot)+", "
                                                        + std::to_string(sizeInMemory(chunk)/(1024*1024))+"Mo)");
                }
                prm_m2.chunks.push_back({prm_m2.chunks.back().vOffset+prm_m2.chunks.back().vSize, 0, 0, 0, 0, 0});
            }
        }
    }

    /**
     * @brief Pre-compute the vertical bounds of each layers of the volume
     * 
     */
    void computeLayerYBounds() {
        std::vector<glm::vec4> bounds{
            { prm_g.orig.x, prm_g.orig.y, -prm_g.orig.z, 1.0f},
            {-prm_g.orig.x, prm_g.orig.y, -prm_g.orig.z, 1.0f},
            { prm_g.orig.x, prm_g.orig.y,  prm_g.orig.z, 1.0f},
            {-prm_g.orig.x, prm_g.orig.y,  prm_g.orig.z, 1.0f}
        };
        _layerMinYProj.resize(prm_g.vheight);
        _layerMaxYProj.resize(prm_g.vheight);
        #pragma omp parallel for schedule(static)
        for(int l = 0; l < prm_g.vheight; ++l) {
            std::vector<float> values(_projection_matrices_mat4.size()*bounds.size()*2);
            uint64_t index = 0;
            for(int i = 0; i < _projection_matrices_mat4.size(); ++i) {
                for(int j = 0; j < bounds.size(); ++j) {
                    glm::vec4 top = bounds[j] + glm::vec4{0, l*prm_g.vx, 0, 0};
                    glm::vec4 bot = bounds[j] + glm::vec4{0, (l+1)*prm_g.vx, 0, 0};
                    values[index++] = project(top, i).y;
                    values[index++] = project(bot, i).y;
                }
            }
            const auto [min, max] = std::minmax_element(values.begin(), values.end());
            _layerMinYProj[l] = *min;
            _layerMaxYProj[l] = *max;
        }
    }

    /**
     * @brief Return the min and max Y coordinates on the detector of the given volume slab
     * 
     * @return min, max
     */
    glm::vec2 getYProjection(int64_t volumeOffset, int64_t volumeHeight) {
        return glm::vec2{
            std::min(_layerMinYProj[volumeOffset], _layerMinYProj[volumeOffset+volumeHeight-1]), 
            std::max(_layerMaxYProj[volumeOffset], _layerMaxYProj[volumeOffset+volumeHeight-1])
        };
    }

    /**
     * @brief For a given volume chunk size, compute the image chunk that is needed
     * 
     * @param chunk 
     */
    void updateImageViewport(Chunk &chunk) {
        glm::vec2 min_max = getYProjection(chunk.vOffset, chunk.vSize);
        chunk.iOffset = int64_t(std::max(std::floor(min_max.x)-1, 0.0f));
        chunk.iSize = int64_t(std::max(std::min(std::ceil(min_max.y-chunk.iOffset)+2, float(prm_g.dheight-chunk.iOffset)), 0.0f));
    }

    /**
     * @brief For a given volume and image chunk, compute the padding necessary
     * 
     * @param chunk 
     */
    void updateVolumePadding(Chunk &chunk) {
        chunk.vPadTop = 0;
        chunk.vPadBot = 0;
        while(  chunk.vOffset-chunk.vPadTop > 0 &&
                getYProjection(chunk.vOffset-chunk.vPadTop, 1).y > chunk.iOffset - 1) {
            ++chunk.vPadTop;
        }
        while(  chunk.vOffset+chunk.vSize+chunk.vPadBot < prm_g.vheight-1 &&
                getYProjection(chunk.vOffset+chunk.vSize+chunk.vPadBot, 1).x < chunk.iOffset + chunk.iSize + 1) {
            ++chunk.vPadBot;
        }
        chunk.vPadTop = int64_t(std::ceil(chunk.vPadTop*prm_m2.overlap_fct));
        chunk.vPadBot = int64_t(std::ceil(chunk.vPadBot*prm_m2.overlap_fct));
    }

    /**
     * @brief Check if the given chunk fit in the GPU memory
     * 
     * @param chunk 
     */
    bool fitInMemory(Chunk &chunk) {
        return _availableMemoryByte > sizeInMemory(chunk) && _maxMemoryAllocByte-1 > imagesSizeInMemory(chunk) && _maxMemoryAllocByte > paddingSizeInMemory(chunk);
    }
};