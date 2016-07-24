//
// Created by Adoni1203 on 16/7/21.
//

#include "mp_train.h"
#include <chrono>

using namespace std;
using namespace boost::interprocess;

namespace mp_train {

    // TODO: Pass these around instead of having them be global
    std::string queue_name = "cnn_mp_work_queue";
    std::string shared_memory_name = "cnn_mp_shared_memory";
    timespec start_time;
    bool stop_requested = false;
    SharedObject *shared_object = nullptr;

    std::string GenerateQueueName() {
        std::ostringstream ss;
        ss << "DBNE_QUEUE";
//        ss << rand();
        return ss.str();
    }

    cnn::real RunDataSet(std::vector<unsigned>::iterator begin, std::vector<unsigned>::iterator end,
                    const std::vector<Workload> &workloads,
                    boost::interprocess::message_queue &mq, const WorkloadHeader &header) {

        const unsigned num_children = workloads.size();

        // Tell all the children to start up
        for (unsigned cid = 0; cid < num_children; ++cid) {
            bool cont = true;
            Write(workloads[cid].p2c[1], cont);
            Write(workloads[cid].p2c[1], header);
        }

        // Write all the indices to the queue for the children to process
        for (auto curr = begin; curr != end; ++curr) {
            unsigned i = *curr;
            mq.send(&i, sizeof(i), 0);
        }

        // Send a bunch of stop messages to the children
        for (unsigned cid = 0; cid < num_children; ++cid) {
            unsigned stop = -1U;
            mq.send(&stop, sizeof(stop), 0);
        }

        // Wait for each child to finish training its load
        cnn::real loss=0.0;
        for (unsigned cid = 0; cid < num_children; ++cid) {
            loss  += Read<cnn::real>(workloads[cid].c2p[0]);
        };
        return loss;
    }

    std::string GenerateSharedMemoryName() {
        std::ostringstream ss;
        ss << "DLNE_shared_memory";
        ss << rand();
        return ss.str();
    }

    cnn::real SumValues(const std::vector<cnn::real> &values) {
        return accumulate(values.begin(), values.end(), 0.0);
    }

    cnn::real Mean(const std::vector<cnn::real> &values) {
        return SumValues(values) / values.size();
    }

    std::string ElapsedTimeString(const std::chrono::time_point<std::chrono::high_resolution_clock> start, double fractional_iter) {
        std::ostringstream ss;
        auto now_time = std::chrono::high_resolution_clock::now();

        ss << std::chrono::duration<double, std::milli>(now_time - start).count() / 3600000 << " hours for " <<fractional_iter<<" epoch"
        << std::endl;
        ss << std::chrono::duration<double, std::milli>(now_time - start).count() / 3600000/ fractional_iter << " hours for each epoch"
        << std::endl;

        return ss.str();
    }

    unsigned SpawnChildren(std::vector<Workload> &workloads) {
        const unsigned num_children = workloads.size();
        assert (workloads.size() == num_children);
        pid_t pid;
        unsigned cid;
        for (cid = 0; cid < num_children; ++cid) {
            pid = fork();
            if (pid == -1) {
                std::cerr << "Fork failed. Exiting ..." << std::endl;
                return 1;
            }
            else if (pid == 0) {
                // children shouldn't continue looping
                break;
            }
            workloads[cid].pid = pid;
        }
        return cid;
    }

    std::vector<Workload> CreateWorkloads(unsigned num_children) {
        int err;
        std::vector<Workload> workloads(num_children);
        for (unsigned cid = 0; cid < num_children; cid++) {
            err = pipe(workloads[cid].p2c);
            assert (err == 0);
            err = pipe(workloads[cid].c2p);
            assert (err == 0);
        }
        return workloads;
    }
}
