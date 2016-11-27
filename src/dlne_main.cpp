//
// Created by Adoni1203 on 16/7/19.
// Diversity Link based Network Embedding (DLNE)
//

#include <boost/program_options.hpp>

#include "graph_data.h"
#include "embedding_methods.h"
#include "network_embedding.h"
#include "mp_train.h"
#include "sp_train.h"

using namespace std;
using namespace dynet;
namespace po = boost::program_options;

void InitCommandLine(int argc, char **argv, po::variables_map *conf) {
    po::options_description opts("Configuration options");
    opts.add_options()
            // Data option
            ("graph_file", po::value<string>(), "Graph file, adjacency list")
            ("content_file", po::value<string>(), "Content file")
            ("word_embedding_file", po::value<string>(), "Emebdding file of word2vec-format")
            ("vertex_embedding_file", po::value<string>(), "Pre-trained vertex embedding")
            ("to_be_saved_index_file_name", po::value<string>(), "Indexes whose embeddings would be saved")
            ("eta0", po::value<float>(), "eta0 for sgd")
            ("eta_decay", po::value<float>(), "eta_decay for sgd")
            ("workers", po::value<unsigned>(), "workers count")
            ("iterations", po::value<unsigned>(), "iterations number")
            ("batch_size", po::value<unsigned>(), "Update frequency")
            ("save_every_i", po::value<unsigned>(), "Save frequency as well as the update_epoch frequency")
            ("update_epoch_every_i", po::value<unsigned>(), "Update frequency")
            ("report_every_i", po::value<unsigned>(), "Report frequency")
            ("vertex_negative", po::value<unsigned>(), "Negative sampling count for vertex")
            ("content_negative", po::value<unsigned>(), "Negative sampling count for content")
            ("alpha", po::value<float>(), "alpha to control the proportion of VV and VC")
            ("embedding_method", po::value<std::string>(), "method to learn embedding from content: WordAvg or GRU")
            ("strictly_content_required", po::value<bool>(), "if the content for each vertex is strictly required")
            ("use_const_lookup", po::value<bool>(), "use const_lookup or lookup")
            ("cnn_filter_count", po::value<unsigned>()->default_value(1), "the count of each filter")
            ("word_embedding_size", po::value<unsigned>(), "word embedding size ")


            ("help", "Help");
    po::options_description dcmdline_options;
    dcmdline_options.add(opts);
    po::store(parse_command_line(argc, argv, dcmdline_options), *conf);
    if (conf->empty()) {
        cerr << "Error: empty conf" << endl;
        cerr << dcmdline_options << endl;
        exit(1);
    }
    if (conf->count("help")) {
        cerr << dcmdline_options << endl;
        exit(1);
    }
    vector<string> required_options{"graph_file", "content_file", "eta0", "eta_decay", "workers",
                                    "iterations", "batch_size", "save_every_i", "update_epoch_every_i",
                                    "report_every_i",
                                    "vertex_negative", "content_negative", "alpha", "strictly_content_required","use_const_lookup"};

    for (auto opt_str:required_options) {
        if (conf->count(opt_str) == 0) {
            cerr << "Error: missed option" << endl;
            cerr << "Please specify --" << opt_str << endl;
            exit(1);
        }
    }
}

void output_all_information(int argc, char **argv) {
    std::ostringstream ss;
    ss << getpid() << "_info.data";
    std::ofstream output_file(ss.str());
    for (int i = 0; i < argc; i++) {
        output_file << argv[i] << std::endl;
    }
}


int main(int argc, char **argv) {
    dynet::initialize(argc, argv, true);
    po::variables_map conf;
    InitCommandLine(argc, argv, &conf);
    dynet::Dict d;
    cout << "Pid: " << getpid() << endl;
    output_all_information(argc, argv);
    GraphData graph_data(conf["graph_file"].as<string>(), conf["content_file"].as<string>(),
                         conf["strictly_content_required"].as<bool>(), d);

    cout << "Vocabulary count: " << d.size() << endl;
    cout << "Node count: " << graph_data.node_count << endl;
    cout << "VV link count: " << graph_data.vv_edgelist.size() << endl;
    cout << "VC link count: " << graph_data.vc_edgelist.size() << endl;
    Model params_model;
    Model lookup_params_model;



    unsigned V_NEG = conf["vertex_negative"].as<unsigned>();
    unsigned C_NEG = conf["content_negative"].as<unsigned>();

    unsigned W_EM_DIM = conf["word_embedding_size"].as<unsigned>();
    unsigned V_EM_DIM = W_EM_DIM/2;
    unsigned C_EM_DIM = W_EM_DIM;

    ContentEmbeddingMethod *content_embedding_method;
    if (conf["embedding_method"].as<std::string>() == "WordAvg") {
        content_embedding_method = new WordAvg_CE(params_model, lookup_params_model, W_EM_DIM, C_EM_DIM, conf["use_const_lookup"].as<bool>(), d);
    } else if (conf["embedding_method"].as<std::string>() == "GRU") {
        content_embedding_method = new GRU_CE(params_model, lookup_params_model, W_EM_DIM, C_EM_DIM, conf["use_const_lookup"].as<bool>(), d);
    } else if (conf["embedding_method"].as<std::string>() == "BiGRU") {
        content_embedding_method = new BiGRU_CE(params_model, lookup_params_model, W_EM_DIM, C_EM_DIM, conf["use_const_lookup"].as<bool>(), d);
    } else if (conf["embedding_method"].as<std::string>() == "CNN") {
        unsigned f_count=conf["cnn_filter_count"].as<unsigned>();
        content_embedding_method = new CNN_CE(params_model, lookup_params_model, W_EM_DIM, C_EM_DIM, {{2, f_count},
                                                                          {3, f_count},
                                                                          {4, f_count},{5,3}}, d);
    } else {
        std::cerr << "Unsupported embedding method" << std::endl;
        return 1;
    }

    if (conf.count("word_embedding_file")) {
        content_embedding_method->initial_look_up_table_from_file(conf["word_embedding_file"].as<string>(), d);
    }

    DLNEModel dlne(params_model, lookup_params_model, graph_data.node_count, V_NEG, C_NEG, V_EM_DIM, content_embedding_method);
    if (conf.count("vertex_embedding_file")) {
        dlne.initialize_from_pretrained_vertex_embedding(conf["vertex_embedding_file"].as<string>(), graph_data);
    }
    if (conf.count("to_be_saved_index_file_name")) {
        dlne.set_to_be_saved_index(conf["to_be_saved_index_file_name"].as<string>(), graph_data);
    }
    Trainer *params_trainer = nullptr;
    params_trainer = new SimpleSGDTrainer(&params_model, 1e-6, conf["eta0"].as<float>());
    params_trainer->eta_decay = conf["eta_decay"].as<float>();
    Trainer *lookup_params_trainer = nullptr;
    lookup_params_trainer = new SimpleSGDTrainer(&lookup_params_model, 1e-6, conf["eta0"].as<float>());
    lookup_params_trainer->eta_decay = conf["eta_decay"].as<float>();
    mp_train::RunMultiProcess(conf["workers"].as<unsigned>(), &dlne, params_trainer, lookup_params_trainer, graph_data, conf["iterations"].as<unsigned>(),
                              conf["alpha"].as<float>(),
                              conf["save_every_i"].as<unsigned>(), conf["update_epoch_every_i"].as<unsigned>(),
                              conf["report_every_i"].as<unsigned>(), conf["batch_size"].as<unsigned>());
    return 0;
}
