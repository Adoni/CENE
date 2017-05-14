# CENE
Content Enhanced Network Embedding

CENE is tool to learn embedding of nodes in a network.

#### Building

First you need to clone DyNet from GitHub and compile it as well as its dependencies.
You can learn how to do that [here](https://github.com/clab/dynet). We use version [v1.0-rc1](https://github.com/clab/dynet/releases/tag/v1.0-rc1)

Typically you should have Eigen installed already after compile DyNet.
If not, you can get it using the following command:
    
    hg clone https://bitbucket.org/eigen/eigen/ -r 346ecdb

Then you should clone this project.

After that, you need to use [`cmake`](http://www.cmake.org/) to generate the makefiles

    mkdir build
    cd build
    cmake .. -DEIGEN3_INCLUDE_DIR=/path/to/eigen -DDYNET_INCLUDE_DIR==/path/to/dynet -DDYNET_LINK_DIR=/path/to/dynet/build/dynet

Then you need to 

    make -j 2

