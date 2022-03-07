file(GLOB MAIN_SOURCE_FILES
	"*.cpp"
	"*.hpp" 
    "reconstruction/*.hpp" 
    "reconstruction/dataset/*.hpp" 
    "reconstruction/dataset/parameters/*.hpp" 
    "reconstruction/gpu/*.hpp" 
    "reconstruction/gpu/kernels/*.cl" 
)

list(APPEND CPP_LIST ${CPP_LIST} ${MAIN_SOURCE_FILES})
set(CPP_LIST ${CPP_LIST} PARENT_SCOPE)