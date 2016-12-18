//
// Created by Adoni1203 on 16/7/21.
//

#pragma once

#include "dynet/dynet.h"
#include "dynet/training.h"
#include "dynet/expr.h"
#include "dynet/dict.h"
#include "dynet/lstm.h"
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/interprocess/ipc/message_queue.hpp>
#include <boost/interprocess/shared_memory_object.hpp>
#include <boost/interprocess/mapped_region.hpp>
#include <boost/interprocess/sync/interprocess_semaphore.hpp>
#include <boost/interprocess/anonymous_shared_memory.hpp>

#include <sys/types.h>
#include <sys/wait.h>
#include <sys/shm.h>
#include <iostream>
#include <limits>
#include <fstream>
#include <vector>
#include <utility>
#include <sstream>
#include <random>
#include <algorithm>
#include <chrono>

#include "graph_data.h"
#include "network_embedding.h"

using namespace dynet;

namespace mp_train {
    extern std::string queue_name;
    extern std::string shared_memory_name;

    struct WorkloadHeader {
        unsigned vv_or_vc; // to determain which kind of link is used
    };

    // A simple struct to hold information about a child process
    struct Workload {
        pid_t pid;
        int c2p[2]; // Child to parent pipe
        int p2c[2]; // Parent to child pipe
    };

    // This interface is used by the child processes and called
    // once per datum.


    struct SharedObject {
        SharedObject() : update_mutex(1), counter_mutex(1), counter(0) { }

        boost::interprocess::interprocess_semaphore update_mutex;
        boost::interprocess::interprocess_semaphore counter_mutex;
        unsigned counter;
    };


    extern SharedObject *shared_object;

    /// XXX: We never delete these objects
    template<class T>
    T *GetSharedMemory() {
        auto region = new boost::interprocess::mapped_region(boost::interprocess::anonymous_shared_memory(sizeof(T)));
        void *addr = region->get_address();
        T *obj = new(addr) SharedObject();
        return obj;
    }

    // Some simple functions that do IO to/from pipes.
    // These are used to send data from child processes
    // to the parent process or vice/versa.
    template<class T>
    T Read(int pipe) {
        T v;
        int err = read(pipe, &v, sizeof(T));
        assert (err != -1);
        return v;
    }

    template<class T>
    void Write(int pipe, const T &v) {
        int err = write(pipe, &v, sizeof(T));
        assert (err != -1);
    }

    std::string GenerateQueueName();

    std::string GenerateSharedMemoryName();

    dynet::real SumValues(const std::vector<dynet::real> &values);

    dynet::real Mean(const std::vector<dynet::real> &values);

    std::string ElapsedTimeString(const std::chrono::time_point<std::chrono::high_resolution_clock> start,
                                  double fractional_iter);

    unsigned SpawnChildren(std::vector<Workload> &workloads);

    std::vector<Workload> CreateWorkloads(unsigned num_children);

    // Called by the parent to process a chunk of data
    dynet::real RunDataSet(std::vector<unsigned>::iterator begin, std::vector<unsigned>::iterator end,
                    const std::vector<Workload> &workloads, boost::interprocess::message_queue &mq,
                    const WorkloadHeader &header);


    void RunParent(GraphData &graph_data, DLNEModel *learner, Trainer *params_trainer,
                   std::vector<Workload> &workloads, unsigned num_iterations,
                   float alpha, unsigned save_every_i, unsigned update_epoch_every_i, unsigned report_every_i, unsigned batch_size);

    int RunChild(unsigned cid, DLNEModel *learner, Trainer *params_trainer,
                 std::vector<Workload> &workloads, GraphData &graph_data);

    void RunMultiProcess(unsigned num_children, DLNEModel *learner, Trainer *params_trainer,
                         GraphData &graph_data, unsigned num_iterations,
                         float alpha, unsigned save_every_i, unsigned updata_epoch_every_i, unsigned report_every_idate_every_i, unsigned batch_size);
}
