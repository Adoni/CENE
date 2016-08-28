//
// Created by Adoni1203 on 16/7/21.
//

#include "mp_train.h"
#include <chrono>
#include <iomanip>

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
        cnn::real loss = 0.0;
        for (unsigned cid = 0; cid < num_children; ++cid) {
            loss += Read<cnn::real>(workloads[cid].c2p[0]);
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

    void RunParent(GraphData &graph_data, DLNEModel *learner, Trainer *params_trainer, Trainer *lookup_params_trainer,
                   std::vector<Workload> &workloads, unsigned num_iterations,
                   float alpha, unsigned save_every_i, unsigned update_epoch_every_i, unsigned report_every_i,
                   unsigned batch_size) {
        const unsigned num_children = unsigned(workloads.size());
        boost::interprocess::message_queue mq(boost::interprocess::open_or_create, queue_name.c_str(), 10000,
                                              sizeof(unsigned));
        std::vector<unsigned> vv_train_indices(graph_data.vv_edgelist.size());
        std::iota(vv_train_indices.begin(), vv_train_indices.end(), 0);
        std::vector<unsigned> vc_train_indices(graph_data.vc_edgelist.size());
        std::iota(vc_train_indices.begin(), vc_train_indices.end(), 0);

        auto total_start_time = std::chrono::high_resolution_clock::now();

        std::ostringstream ss_for_loss_err;
        ss_for_loss_err << learner->get_learner_name() << "_mp_vv_losses_" << getpid() << ".data";
        std::ofstream out_for_vv_losses(ss_for_loss_err.str());
        ss_for_loss_err.str("");
        ss_for_loss_err << learner->get_learner_name() << "_mp_vc_losses_" << getpid() << ".data";
        std::ofstream out_for_vc_losses(ss_for_loss_err.str());


        std::vector<unsigned>::iterator vv_begin = vv_train_indices.begin();
        std::vector<unsigned>::iterator vc_begin = vc_train_indices.begin();

        std::cout << "Batch size: " << batch_size << std::endl;
        std::cout << "Update epoch every " << update_epoch_every_i << std::endl;
        std::cout << "Save every " << save_every_i << std::endl;
        std::cout << "Report every " << report_every_i << std::endl;

        std::uniform_real_distribution<> dis(0.0, 1.0);
        save_every_i = save_every_i / batch_size;
        report_every_i = report_every_i / batch_size;
        update_epoch_every_i = update_epoch_every_i / batch_size;
        num_iterations = num_iterations / batch_size;
        int save_model_every_i = 1000000;
        cnn::real loss = 0.0;

        std::cout << "Batch size: " << batch_size << std::endl;
        std::cout << "Iteration number: " << num_iterations << " batch" << std::endl;
        std::cout << "Update epoch every " << update_epoch_every_i << " batch" << std::endl;
        std::cout << "Save every " << save_every_i << " batch" << std::endl;
        std::cout << "Report every " << report_every_i << " batch" << std::endl;


        for (unsigned iter = 1; iter < num_iterations; ++iter) {
            unsigned vv_or_vc;

            if (dis(*cnn::rndeng) < alpha) {
                vv_or_vc = 0;
                if (vv_begin == vv_train_indices.end()) {
                    std::shuffle(vv_train_indices.begin(), vv_train_indices.end(), (*cnn::rndeng));
                    vv_begin = vv_train_indices.begin();
                }
                std::vector<unsigned>::iterator end = vv_begin + batch_size;
                if (end > vv_train_indices.end()) {
                    end = vv_train_indices.end();
                }
                loss += RunDataSet(vv_begin, end, workloads, mq, {vv_or_vc});
                vv_begin = end;
            }
            else {
                vv_or_vc = 1;
                if (vc_begin == vc_train_indices.end()) {
                    vc_begin = vc_train_indices.begin();
                    std::shuffle(vc_train_indices.begin(), vc_train_indices.end(), (*cnn::rndeng));
                }
                std::vector<unsigned>::iterator end = vc_begin + batch_size;
                if (end > vc_train_indices.end()) {
                    end = vc_train_indices.end();
                }
                loss += RunDataSet(vc_begin, end, workloads, mq, {vv_or_vc});
                vc_begin = end;
            }

            params_trainer->update();

            if (iter % update_epoch_every_i == 0) {
                params_trainer->update_epoch();
                lookup_params_trainer->update_epoch();
            }
            if (iter % report_every_i == 0) {
                std::ostringstream ss;
                if (vv_or_vc == 0) {
                    ss << "Eta = " << params_trainer->eta << "\tVV" << " loss = " << loss << std::endl;
                    std::string loss_info = ss.str();
                    std::cout << loss_info;
                    out_for_vv_losses << loss_info;
                } else {
                    ss << "Eta = " << params_trainer->eta << "\tVC" << " loss = " << loss << std::endl;
                    std::string loss_info = ss.str();
                    std::cout << loss_info;
                    out_for_vc_losses << loss_info;
                }
                loss = 0.0;
//                std::string info = ElapsedTimeString(total_start_time, iter*update_every_i / 100000);
//                std::cerr << info;
            }

            if (iter % save_every_i == 0) {
                std::ostringstream ss;
                ss << learner->get_learner_name() << "_embedding_pid" << getpid() << "_alpha_" <<
                std::setprecision(2) << alpha << "_" << int(iter / save_every_i) << ".data";
                learner->SaveEmbedding(ss.str(), graph_data);
            }

            if (iter % save_model_every_i == 0) {
                std::ostringstream ss;
                ss << learner->get_learner_name() << "_mp_model" << getpid() << ".data";
                std::string fname = ss.str();
                std::ofstream out(fname);
                boost::archive::text_oarchive oa(out);
                std::cerr << "Saving model to " << fname << std::endl;
                oa << *(params_trainer->model);
            }
        }

        // Kill all children one by one and wait for them to exit
        for (unsigned cid = 0; cid < num_children; ++cid) {
            bool cont = false;
            Write(workloads[cid].p2c[1], cont);
            wait(NULL);
        }
    }

    int RunChild(unsigned cid, DLNEModel *learner, Trainer *params_trainer, Trainer *lookup_params_trainer,
                 std::vector<Workload> &workloads, GraphData &graph_data) {
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
            WorkloadHeader header = Read<WorkloadHeader>(workloads[cid].p2c[0]);

            // Run the actual training loop
            cnn::real loss = 0.0;
            int child_counter = 0;
            while (true) {
                mq.receive(&i, sizeof(unsigned), recvd_size, priority);
                if (i == -1U) {
                    break;
                }

                if (header.vv_or_vc == 0) {
                    assert (i < graph_data.vv_edgelist.size());
                    const Edge edge = graph_data.vv_edgelist[i];
                    loss += learner->TrainVVEdge(edge, graph_data);
                }
                else if (header.vv_or_vc == 1) {
                    assert (i < graph_data.vc_edgelist.size());
                    const Edge edge = graph_data.vc_edgelist[i];
                    loss += learner->TrainVCEdge(edge, graph_data);
                } else {
                    std::cout << "Error" << std::endl;
                }
                lookup_params_trainer->update();
                child_counter += 1;
            }

            // Let the parent know that we're done and return the loss value
            Write(workloads[cid].c2p[1], loss);
        }
        return 0;
    }

    void RunMultiProcess(unsigned num_children, DLNEModel *learner, Trainer *params_trainer, Trainer *lookup_params_trainer,
                         GraphData &graph_data, unsigned num_iterations,
                         float alpha, unsigned save_every_i, unsigned updata_epoch_every_i,
                         unsigned report_every_idate_every_i, unsigned batch_size) {
        std::cout << "==================" << std::endl << "START TRAINING" << std::endl << "==================" <<
        std::endl;
        queue_name = GenerateQueueName();
        boost::interprocess::message_queue::remove(queue_name.c_str());
        boost::interprocess::message_queue::remove(queue_name.c_str());
        shared_memory_name = GenerateSharedMemoryName();
        boost::interprocess::shared_memory_object::remove(shared_memory_name.c_str());
        shared_object = GetSharedMemory<SharedObject>();
        std::vector<Workload> workloads = CreateWorkloads(num_children);
        unsigned cid = SpawnChildren(workloads);
        if (cid < num_children) {
            RunChild(cid, learner, params_trainer, lookup_params_trainer, workloads, graph_data);
        }
        else {
            RunParent(graph_data, learner, params_trainer, lookup_params_trainer, workloads, num_iterations, alpha, save_every_i,
                      updata_epoch_every_i, report_every_idate_every_i, batch_size);
        }
    }
}
