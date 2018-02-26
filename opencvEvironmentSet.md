# Set environment on mac with python 3.6.1 and conda 4.3.30

I've been taking a whole afternoon on setting the enviroment

Either

  ~~conda install -c menpo opencv3~~
  ----
  
or

  ~~conda install -c conda-forge opencv~~
  ----
  ~~conda install -c conda-forge/label/broken opencv~~
  ----
  
is suitable in my situation.


Eventually find a way [传送门](http://blog.csdn.net/k7arm/article/details/78178088) to set it up in the end of afternoon.This blog 
worked out my puzzles effectively, and sort out some problem which is helpful to comprehend conda, python working environment and 
so on...

And take some notes for reviewing

load other additional channels to load resources
    
    conda config --add channels

-c point to the source repository

    conda install -c 
  
construct diverse environment to satisfy different requirement
  
    conda create -n zopencv python=version
  
check out the environment
  
    source activate zopencv
    source deactivate
    conda info --envs

--spec to check the key

    conda search -c conda-forge --spec 'opencv=3*'
    
    
留意一下多环境操作，搭配合适的环境比较方便后续工作
  
    
