#include "simple_mp_train.h"

using namespace std;
using namespace boost::interprocess;

namespace simple_mp {
    // TODO: Pass these around instead of having them be global
    std::string queue_name = "cnn_mp_work_queue";
    std::string shared_memory_name = "cnn_mp_shared_memory";
    timespec start_time;
    bool stop_requested = false;
    SharedObject* shared_object = nullptr;

    std::string GenerateQueueName() {
        std::ostringstream ss;
        ss << "cnn_mp_work_queue";
        ss << rand();
        return ss.str();
    }

    std::string GenerateSharedMemoryName() {
        std::ostringstream ss;
        ss << "cnn_mp_shared_memory";
        ss << rand();
        return ss.str();
    }

    cnn::real SumValues(const std::vector<cnn::real>& values) {
        return accumulate(values.begin(), values.end(), 0.0);
    }

    cnn::real Mean(const std::vector<cnn::real>& values) {
        return SumValues(values) / values.size();
    }

    std::string ElapsedTimeString(const timespec& start, const timespec& end) {
        std::ostringstream ss;
        time_t secs = end.tv_sec - start.tv_sec;
        long nsec = end.tv_nsec - start.tv_nsec;
        ss << secs << " seconds and " << nsec << "nseconds";
        return ss.str();
    }

    unsigned SpawnChildren(std::vector<Workload>& workloads) {
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

    cnn::real RunDataSet(std::vector<unsigned>::iterator begin, std::vector<unsigned>::iterator end, const std::vector<Workload>& workloads,
                         boost::interprocess::message_queue& mq, const WorkloadHeader& header) {
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
            if (stop_requested) {
                break;
            }
        }

        // Send a bunch of stop messages to the children
        for (unsigned cid = 0; cid < num_children; ++cid) {
            unsigned stop = -1U;
            mq.send(&stop, sizeof(stop), (stop_requested ? 1 : 0));
        }

        // Wait for each child to finish training its load
        std::vector<cnn::real> losses(num_children);
        for(unsigned cid = 0; cid < num_children; ++cid) {
            losses[cid] = Read<cnn::real>(workloads[cid].c2p[0]);
        }

        cnn::real total_loss = 0;
        for (auto datum_loss : losses) {
            total_loss += datum_loss;
        }
        return total_loss;
    }

    void RunParent(DLNEModel<WordAvg> *learner, SeperateSimpleSGDTrainer* trainer, std::vector<Workload>& workloads, GraphData &graph_data) {
        const unsigned num_children = workloads.size();
        boost::interprocess::message_queue mq(boost::interprocess::open_or_create, queue_name.c_str(), 10000, sizeof(unsigned));
        std::vector<unsigned> train_indices(graph_data.vv_edgelist.size());
        std::iota(train_indices.begin(), train_indices.end(), 0);

        int num_iterations=100000000;
        unsigned report_frequency=1000;

        for (unsigned iter = 0; iter < num_iterations; ++iter) {
            std::vector<unsigned>::iterator begin = train_indices.begin();
            while (begin != train_indices.end()) {
                std::vector<unsigned>::iterator end = begin + 10000;
                if (end > train_indices.end()) {
                    end = train_indices.end();
                }
                cnn::real batch_loss = RunDataSet(begin, end, workloads, mq, {false, end == train_indices.end()});
                std::cerr << "loss = " << batch_loss << std::endl;
                std::cout<<"Parent Before"<<learner->test_tmp(0, graph_data)<<std::endl;
//                for (auto& p : trainer->model->lookup_parameters_list()) {
//                    std::cout << "# none zero: " << p->non_zero_grads.size() << std::endl;
//                }
//                std::cout<<"No Zero: "<<trainer->model->lookup_parameters_list()<<std::endl;
//                trainer->update();
                std::cout<<"==========="<<std::endl;
            }
        }

        // Kill all children one by one and wait for them to exit
        for (unsigned cid = 0; cid < num_children; ++cid) {
            bool cont = false;
            Write(workloads[cid].p2c[1], cont);
            wait(NULL);
        }
    }

    int RunChild(unsigned cid, DLNEModel<WordAvg> *learner, SeperateSimpleSGDTrainer* trainer,
                 std::vector<Workload>& workloads, GraphData &graph_data) {
        const unsigned num_children = workloads.size();
        assert (cid >= 0 && cid < num_children);
        unsigned i;
        unsigned priority;
        boost::interprocess::message_queue::size_type recvd_size;
        boost::interprocess::message_queue mq(boost::interprocess::open_or_create, queue_name.c_str(), 10000, sizeof(unsigned));
        while (true) {
            // Check if the parent wants us to exit
            bool cont = Read<bool>(workloads[cid].p2c[0]);
            if (cont == 0) {
                break;
            }

            // Check if we're running on the training data or the dev data
            WorkloadHeader header = Read<WorkloadHeader>(workloads[cid].p2c[0]);

            cnn::real total_loss=0.0;
            while (true) {
                mq.receive(&i, sizeof(unsigned), recvd_size, priority);
                if (i == -1U) {
                    break;
                }
                learner->TrainVVEdge(graph_data.vv_edgelist[i],graph_data);
            }
//            trainer->update();
            trainer->update_lookup_params();

            if (header.end_of_epoch) {
                //trainer->update_epoch();
            }

            // Let the parent know that we're done and return the loss value
            Write(workloads[cid].c2p[1], total_loss);
        }
        return 0;
    }

    void RunMultiProcess(unsigned num_children, DLNEModel<WordAvg> *learner, SeperateSimpleSGDTrainer* trainer, GraphData &graph_data) {
        assert (cnn::ps->is_shared());
        queue_name = GenerateQueueName();
        boost::interprocess::message_queue::remove(queue_name.c_str());
        boost::interprocess::message_queue::remove(queue_name.c_str());
        shared_memory_name = GenerateSharedMemoryName();
        shared_object = GetSharedMemory<SharedObject>();
        std::vector<Workload> workloads = CreateWorkloads(num_children);
        unsigned cid = SpawnChildren(workloads);
        if (cid < num_children) {
            RunChild(cid, learner, trainer, workloads, graph_data);
        }
        else {
            RunParent(learner, trainer, workloads, graph_data);
        }
    }
}
