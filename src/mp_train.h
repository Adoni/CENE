//
// Created by Adoni1203 on 16/7/21.
//

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
#include <chrono>

#include "graph_data.h"
#include "network_embedding.h"
#include "seperate_trainer.h"

using namespace cnn;

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

    cnn::real SumValues(const std::vector<cnn::real> &values);

    cnn::real Mean(const std::vector<cnn::real> &values);

    std::string ElapsedTimeString(const std::chrono::time_point<std::chrono::high_resolution_clock> start,
                                  double fractional_iter);

    unsigned SpawnChildren(std::vector<Workload> &workloads);

    std::vector<Workload> CreateWorkloads(unsigned num_children);

    // Called by the parent to process a chunk of data
    cnn::real RunDataSet(std::vector<unsigned>::iterator begin, std::vector<unsigned>::iterator end,
                    const std::vector<Workload> &workloads, boost::interprocess::message_queue &mq,
                    const WorkloadHeader &header);


    template<class CONTENT_EMBEDDING_METHOD>
    void RunParent(GraphData &graph_data, DLNEModel<CONTENT_EMBEDDING_METHOD> *learner, SeperateSimpleSGDTrainer *trainer,
                   std::vector<Workload> &workloads, unsigned num_iterations,
                   float alpha, unsigned save_every_i, unsigned update_every_i, unsigned report_every_i) {
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

        std::cout<<"Update every "<<update_every_i<<std::endl;
        std::cout<<"Save every "<<save_every_i<<std::endl;
        std::cout<<"Report every "<<report_every_i<<std::endl;

        std::uniform_real_distribution<> dis(0.0, 1.0);
        save_every_i=save_every_i/update_every_i;
        report_every_i=report_every_i/update_every_i;
        int save_model_every_i=1000000;
        cnn::real loss=0.0;

        std::cout<<"Update every "<<update_every_i<<std::endl;
        std::cout<<"Save every "<<save_every_i<<std::endl;
        std::cout<<"Report every "<<report_every_i<<std::endl;

        for (unsigned iter = 1; iter < num_iterations; ++iter) {
            unsigned vv_or_vc;

            if (dis(*cnn::rndeng) < alpha) {
                vv_or_vc=0;
                if (vv_begin == vv_train_indices.end()) {
                    vv_begin = vv_train_indices.begin();
                    std::shuffle(vv_train_indices.begin(), vv_train_indices.end(), (*cnn::rndeng));
                }
                std::vector<unsigned>::iterator end = vv_begin + update_every_i;
                if (end > vv_train_indices.end()) {
                    end = vv_train_indices.end();
                }
                loss+=RunDataSet(vv_begin, end, workloads, mq, {vv_or_vc});

                trainer->update_params();
                vv_begin = end;
            }
            else {
                vv_or_vc=1;
                if (vc_begin == vc_train_indices.end()) {
                    vc_begin = vc_train_indices.begin();
                    std::shuffle(vc_train_indices.begin(), vc_train_indices.end(), (*cnn::rndeng));
                }
                std::vector<unsigned>::iterator end = vc_begin + update_every_i;
                if (end > vc_train_indices.end()) {
                    end = vc_train_indices.end();
                }
                loss+=RunDataSet(vc_begin, end, workloads, mq, {vv_or_vc});
//                trainer->update_params();
                vc_begin = end;
            }

//            trainer->update_epoch();
            if (iter % report_every_i == 0) {
                std::ostringstream ss;
                if (vv_or_vc==0){
                    ss << "VV"<< " loss = " << loss << std::endl;
                    std::string loss_info = ss.str();
                    std::cout << loss_info;
                    out_for_vv_losses << loss_info;
                }else{
                    ss << "VC"<< " loss = " << loss << std::endl;
                    std::string loss_info = ss.str();
                    std::cout << loss_info <<std::endl;
                    out_for_vc_losses << loss_info;
                }
                loss=0.0;
//                std::string info = ElapsedTimeString(total_start_time, iter*update_every_i / 100000);
//                std::cerr << info;
            }

            if (iter % save_every_i == 0) {
                std::ostringstream ss;
                ss << learner->get_learner_name() << "_embedding_pid" << getpid() << "_" << int(iter/save_every_i);
                learner->SaveEmbedding( ss.str(), graph_data);
            }

            if (iter % save_model_every_i ==0) {
                std::ostringstream ss;
                ss << learner->get_learner_name() << "_mp_model" << getpid() << ".data";
                std::string fname = ss.str();
                std::ofstream out(fname);
                boost::archive::text_oarchive oa(out);
                std::cerr << "Saving model to " << fname << std::endl;
                oa << *(trainer->model);
            }
        }

        // Kill all children one by one and wait for them to exit
        for (unsigned cid = 0; cid < num_children; ++cid) {
            bool cont = false;
            Write(workloads[cid].p2c[1], cont);
            wait(NULL);
        }
    }

    template<class CONTENT_EMBEDDING_METHOD>
    int RunChild(unsigned cid, DLNEModel<CONTENT_EMBEDDING_METHOD> *learner, SeperateSimpleSGDTrainer *trainer,
                 std::vector<Workload> &workloads, GraphData &graph_data) {

        std::cout<<"Child trainer: "<<trainer->model->lookup_parameters_list().size()<<std::endl;

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
            while (true) {
                mq.receive(&i, sizeof(unsigned), recvd_size, priority);
                if (i == -1U) {
                    break;
                }

                if (header.vv_or_vc==0){
                    assert (i < graph_data.vv_edgelist.size());
                    const Edge edge = graph_data.vv_edgelist[i];
                    loss += learner->TrainVVEdge(edge, graph_data);
                }
                else if (header.vv_or_vc==1){
                    assert (i < graph_data.vc_edgelist.size());
                    const Edge edge = graph_data.vc_edgelist[i];
                    loss += learner->TrainVCEdge(edge, graph_data);
                } else{
                    std::cout<<"Error"<<std::endl;
                }
            }
            trainer->update_lookup_params(1.0/5000.0);
//            trainer->update_lookup_params();


            // Let the parent know that we're done and return the loss value
            Write(workloads[cid].c2p[1], loss);
        }
        return 0;
    }
    template<class CONTENT_EMBEDDING_METHOD>
    void RunMultiProcess(unsigned num_children, DLNEModel<CONTENT_EMBEDDING_METHOD> *learner, SeperateSimpleSGDTrainer *trainer,
                         GraphData &graph_data, unsigned num_iterations,
                         float alpha, unsigned save_every_i, unsigned updata_every_i, unsigned report_every_idate_every_i) {
        std::cout<< "==================" << std::endl << "START TRAINING" << std::endl << "==================" <<std::endl;
        queue_name = GenerateQueueName();
        boost::interprocess::message_queue::remove(queue_name.c_str());
        boost::interprocess::message_queue::remove(queue_name.c_str());
        shared_memory_name = GenerateSharedMemoryName();
        boost::interprocess::shared_memory_object::remove(shared_memory_name.c_str());
        shared_object = GetSharedMemory<SharedObject>();
        std::vector<Workload> workloads = CreateWorkloads(num_children);
        unsigned cid = SpawnChildren(workloads);
        if (cid < num_children) {
            RunChild<CONTENT_EMBEDDING_METHOD>(cid, learner, trainer, workloads, graph_data);
        }
        else {
            RunParent<CONTENT_EMBEDDING_METHOD>(graph_data, learner, trainer, workloads, num_iterations, alpha, save_every_i,
                                                updata_every_i, report_every_idate_every_i);
        }
    }
}
