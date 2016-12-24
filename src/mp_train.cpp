//
// Created by Adoni1203 on 16/7/21.
//

#include "mp_train.h"

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
        ss << rand();
        return ss.str();
    }

    dynet::real RunDataSet(std::vector<unsigned>::iterator begin, std::vector<unsigned>::iterator end,
                           const std::vector<Workload> &workloads,
                           boost::interprocess::message_queue &mq) {

        const unsigned num_children = workloads.size();

        // Tell all the children to start up
        for (unsigned cid = 0; cid < num_children; ++cid) {
            bool cont = true;
            Write(workloads[cid].p2c[1], cont);
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
        dynet::real loss = 0.0;
        for (unsigned cid = 0; cid < num_children; ++cid) {
            loss += Read<dynet::real>(workloads[cid].c2p[0]);
        };
        return loss;
    }

    std::string GenerateSharedMemoryName() {
        std::ostringstream ss;
        ss << "DLNE_shared_memory";
        ss << rand();
        return ss.str();
    }

    dynet::real SumValues(const std::vector<dynet::real> &values) {
        return accumulate(values.begin(), values.end(), 0.0);
    }

    dynet::real Mean(const std::vector<dynet::real> &values) {
        return SumValues(values) / values.size();
    }

    std::string ElapsedTimeString(const std::chrono::time_point<std::chrono::high_resolution_clock> start,
                                  double fractional_iter) {
        std::ostringstream ss;
        auto now_time = std::chrono::high_resolution_clock::now();

        ss << std::chrono::duration<double, std::milli>(now_time - start).count() / 3600000 << " hours for " <<
           fractional_iter << " epoch"
           << std::endl;
        ss << std::chrono::duration<double, std::milli>(now_time - start).count() / 3600000 / fractional_iter <<
           " hours for each epoch"
           << std::endl;

        return ss.str();
    }

    int SpawnChildren(std::vector<Workload> &workloads) {
        const unsigned num_children = workloads.size();
        assert (workloads.size() == num_children);
        pid_t pid;
        unsigned cid;
        for (cid = 0; cid < num_children; ++cid) {
            pid = fork();
            if (pid == -1) {
                std::cerr << "Fork failed. Exiting ..." << std::endl;
                return 1;
            } else if (pid == 0) {
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

    void RunParent(NetworkData &network_data, DLNEModel *learner, Trainer *params_trainer,
                   std::vector<Workload> &workloads, unsigned num_iterations,
                   unsigned save_every_i, unsigned report_every_i,
                   unsigned batch_size, unsigned update_epoch_every_i) {

        std::cout << "Iterations: " << batch_size << std::endl;
        std::cout << "Batch size: " << batch_size << std::endl;
        std::cout << "Save every " << save_every_i << "iterations" << std::endl;
        report_every_i = report_every_i / batch_size;
        std::cout << "Report every " << report_every_i << "batches" << std::endl;

        const unsigned num_children = unsigned(workloads.size());
        boost::interprocess::message_queue mq(boost::interprocess::open_or_create, queue_name.c_str(), 10000,
                                              sizeof(unsigned));

        std::vector<unsigned> train_indices(network_data.edge_list.size());
        std::iota(train_indices.begin(), train_indices.end(), 0);
        std::vector<unsigned>::iterator begin = train_indices.begin();

        for (unsigned iter = 1; iter < num_iterations; ++iter) {
            unsigned batch_count = 0;
            float loss;
            while (begin != train_indices.end()) {
                std::vector<unsigned>::iterator end = begin + batch_size;
                if (end > train_indices.end()) {
                    end = train_indices.end();
                }
                loss += RunDataSet(begin, end, workloads, mq);
                begin = end;
                batch_count++;
                if (batch_count % report_every_i == 0) {
                    std::cout << "Eta = " << params_trainer->eta << "\tloss = " << loss << std::endl;
                    loss = 0.0;
                }
                if (batch_count % update_epoch_every_i == 0){
                    params_trainer->update_epoch();
                }
            }
            std::shuffle(train_indices.begin(), train_indices.end(), (*dynet::rndeng));
            begin = train_indices.begin();

            if (iter % save_every_i == 0) {
                std::ostringstream ss;
                ss << learner->get_learner_name() << "_embedding_pid" << getpid() << "_alpha_";
                for (auto alpha:learner->alpha) {
                    ss << std::setprecision(2) << alpha << "_";
                }
                ss << unsigned(iter / save_every_i) << ".data";
                learner->SaveEmbedding(ss.str(), network_data);
            }
//            if (iter % save_model_every_i == 0) {
//                std::ostringstream ss;
//                ss << learner->get_learner_name() << "_mp_model" << getpid() << ".data";
//                std::string fname = ss.str();
//                std::ofstream out(fname);
//                boost::archive::text_oarchive oa(out);
//                std::cerr << "Saving model to " << fname << std::endl;
//                oa << *(params_trainer->model);
//            }
        }

        // Kill all children one by one and wait for them to exit
        for (unsigned cid = 0; cid < num_children; ++cid) {
            bool cont = false;
            Write(workloads[cid].p2c[1], cont);
            wait(NULL);
        }
    }

    int RunChild(unsigned cid, DLNEModel *learner, Trainer *params_trainer,
                 std::vector<Workload> &workloads, NetworkData &network_data) {
        const unsigned num_children = workloads.size();
        assert (cid >= 0 && cid < num_children);
        unsigned i;
        unsigned priority;
        boost::interprocess::message_queue::size_type recvd_size;
        boost::interprocess::message_queue mq(boost::interprocess::open_or_create, queue_name.c_str(), 10000,
                                              sizeof(unsigned));
        while (true) {
            // Check if the parent wants us to exit
            bool cont = Read<bool>(workloads[cid].p2c[0]);
            if (cont == 0) {
                break;
            }

            // Check if we're running on the training data or the dev data
            // Run the actual training loop
            dynet::real loss = 0.0;
            while (true) {
                mq.receive(&i, sizeof(unsigned), recvd_size, priority);
                if (i == -1U) {
                    break;
                }

                assert (i < network_data.edge_list.size());
                const Edge edge = network_data.edge_list[i];
                loss += learner->Train(edge, network_data);
            }
            params_trainer->update();

            Write(workloads[cid].c2p[1], loss);
        }
        return 0;
    }

    void RunMultiProcess(unsigned num_children, DLNEModel *learner, Trainer *params_trainer,
                         NetworkData &network_data, unsigned num_iterations,
                         unsigned save_every_i,
                         unsigned report_every_idate_every_i, unsigned batch_size, unsigned update_epoch_every_i) {
        std::cout << "========" << std::endl << "START TRAINING" << std::endl << "========" << std::endl;
        queue_name = GenerateQueueName();
        boost::interprocess::message_queue::remove(queue_name.c_str());
        boost::interprocess::message_queue::remove(queue_name.c_str());
        shared_memory_name = GenerateSharedMemoryName();
        boost::interprocess::shared_memory_object::remove(shared_memory_name.c_str());
        shared_object = GetSharedMemory<SharedObject>();
        std::vector<Workload> workloads = CreateWorkloads(num_children);
        int cid = SpawnChildren(workloads);
        if (cid < num_children) {
            RunChild(cid, learner, params_trainer, workloads, network_data);
        } else {
            RunParent(network_data, learner, params_trainer, workloads, num_iterations, save_every_i,
                      report_every_idate_every_i, batch_size, update_epoch_every_i);
        }
    }
}
