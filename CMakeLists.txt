cmake_minimum_required (VERSION 2.8)  

project (write_model)  
      
add_executable(write_model wm.cpp)  
      
include_directories (/home/yinyuecheng/caffe/include  
        #/usr/local/include  
        #/usr/local/cuda/include  
        #/usr/include
 )  
      
target_link_libraries(write_model  
        /home/yinyuecheng/caffe/build/lib/libcaffe.so  
        /usr/lib/x86_64-linux-gnu/libglog.so  
        /usr/lib/x86_64-linux-gnu/libboost_system.so  
        )  
