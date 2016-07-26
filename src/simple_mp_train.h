#pragma once
#include "cnn/cnn.h"
#include "cnn/training.h"
#include "cnn/expr.h"
#include "cnn/dict.h"
#include "cnn/lstm.h"
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

#include "graph_data.h"
#include "network_embedding.h"
#include "embedding_methods.h"
#include "seperate_trainer.h"


namespace simple_mp {
    // TODO: Pass these around instead of having them be global
    extern std::string queue_name;
    extern std::string shared_memory_name;
    extern timespec start_time;
    extern bool stop_requested;

    struct WorkloadHeader {
        bool is_dev_set;
        bool end_of_epoch;
        unsigned report_frequency;
    };

    // A simple struct to hold information about a child process
    // TODO: Rename me!
    struct Workload {
        pid_t pid;
        int c2p[2]; // Child to parent pipe
        int p2c[2]; // Parent to child pipe
    };

    // This interface is used by the child processes and called
    // once per datum.
    template<class D, class S>
    class ILearner {
    public:
        virtual ~ILearner() {}
        virtual S LearnFromDatum(const D& datum, bool learn) = 0;
        virtual void SaveModel() = 0;
    };

    struct SharedObject {
        SharedObject() : update_mutex(1), counter_mutex(1), counter(0) {}
        boost::interprocess::interprocess_semaphore update_mutex;
        boost::interprocess::interprocess_semaphore counter_mutex;
        unsigned counter;
    };
    extern SharedObject* shared_object;

    /// XXX: We never delete these objects
    template <class T>
    T* GetSharedMemory() {
        /*std::cerr << "Creating shared memory named " << shared_memory_name << std::endl;
        auto shm = new boost::interprocess::shared_memory_object(boost::interprocess::create_only, shared_memory_name.c_str(), boost::interprocess::read_write);
        shm->truncate(sizeof(T));
        auto region = new boost::interprocess::mapped_region (*shm, boost::interprocess::read_write);*/
        auto region = new boost::interprocess::mapped_region(boost::interprocess::anonymous_shared_memory(sizeof(T)));
        void* addr = region->get_address();
        T* obj = new (addr) SharedObject();
        return obj;
    }

    // Some simple functions that do IO to/from pipes.
    // These are used to send data from child processes
    // to the parent process or vice/versa.
    template <class T>
    T Read(int pipe) {
        T v;
        int err = read(pipe, &v, sizeof(T));
        assert (err != -1);
        return v;
    }

    template <class T>
    void Write(int pipe, const T& v) {
        int err = write(pipe, &v, sizeof(T));
        assert (err != -1);
    }

    std::string GenerateQueueName();
    std::string GenerateSharedMemoryName();

    cnn::real SumValues(const std::vector<cnn::real>& values);
    cnn::real Mean(const std::vector<cnn::real>& values);

    std::string ElapsedTimeString(const timespec& start, const timespec& end);

    unsigned SpawnChildren(std::vector<Workload>& workloads);
    std::vector<Workload> CreateWorkloads(unsigned num_children);

    // Called by the parent to process a chunk of data
    cnn::real RunDataSet(std::vector<unsigned>::iterator begin, std::vector<unsigned>::iterator end, const std::vector<Workload>& workloads,
                 boost::interprocess::message_queue& mq, const WorkloadHeader& header);
    void RunParent(DLNEModel *learner, SeperateSimpleSGDTrainer* trainer, std::vector<Workload>& workloads, GraphData &graph_data);

    int RunChild(unsigned cid, DLNEModel *learner, SeperateSimpleSGDTrainer* trainer,
                 std::vector<Workload>& workloads, GraphData &graph_data);

    void RunMultiProcess(unsigned num_children, DLNEModel *learner, SeperateSimpleSGDTrainer* trainer, GraphData &graph_data);
}

