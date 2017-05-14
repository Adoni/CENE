//
// Created by Adoni1203 on 16/7/19.
// Diversity Link based Network Embedding (DLNE)
//

#include <boost/program_options.hpp>
#include <iostream>

#include "embedding_methods.h"
#include "mp_train.h"
#include "L2SGD.h"

using namespace std;
using namespace dynet;
namespace po = boost::program_options;

void InitCommandLine(int argc, char **argv, po::variables_map *conf) {
    po::options_description opts("Configuration options");
    opts.add_options()
        // Data option
        ("node_list_file", po::value<vector<string>>()->multitoken(), "Multiple files; the first line is node_count")
        ("edge_list_file",
         po::value<vector<string>>()->multitoken(),
         "Multiple files, the first line is edge_count and edge_type_count")
        ("content_node_file", po::value<string>(), "Content file")
        ("word_embedding_file", po::value<string>(), "Emebdding file of word2vec-format")
        ("to_be_saved_index_file_name", po::value<string>(), "Indexes whose embeddings would be saved")
        ("params_eta0", po::value<float>(), "eta0 for sgd")
        ("params_eta_decay", po::value<float>(), "eta_decay for sgd")
        ("workers", po::value<unsigned>(), "workers count")
        ("iterations", po::value<unsigned>(), "iterations number")
        ("batch_size", po::value<unsigned>(), "Update frequency")
        ("save_every_i", po::value<unsigned>(), "Save frequency as well as the update_epoch ""frequency")
        ("report_every_i", po::value<unsigned>(), "Report frequency")
        ("update_epoch_every_i", po::value<unsigned>(), "Update epoch frequency")
        ("negative", po::value<vector<unsigned>>()->multitoken(), "Negative sampling counts")
        ("alpha", po::value<vector<float>>()->multitoken(), "alpha to control the proportion of VV and VC")
        ("embedding_method", po::value<std::string>(), "method to learn embedding from content: WordAvg or GRU")
        ("score_function", po::value<unsigned>(), "0: simple score; 1: bilinear; 2: matrix score")
        ("lambda", po::value<float>(), "lambda for L2 regularization")
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
    vector<string> required_options{"node_list_file",
                                    "edge_list_file",
                                    "content_node_file",
                                    "word_embedding_file",
                                    "params_eta0",
                                    "params_eta_decay",
                                    "workers",
                                    "iterations",
                                    "batch_size",
                                    "save_every_i",
                                    "report_every_i",
                                    "negative",
                                    "alpha",
                                    "embedding_method",
                                    "lambda"};

    for (auto opt_str : required_options) {
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
    for (unsigned i = 0; i < argc; i++) {
        output_file << argv[i] << std::endl;
    }
}

unsigned get_word_embedding_size(string word_embedding_file_name) {
    ifstream file_in(word_embedding_file_name);
    assert(file_in);
    unsigned word_embedding_size, word_count;
    file_in >> word_count >> word_embedding_size;
    file_in.close();
    return word_embedding_size;
}

void initialize_word_dict(string word_embedding_file_name, dynet::Dict &d) {
    ifstream file_in(word_embedding_file_name);
    assert(file_in);
    unsigned word_embedding_size, word_count;
    file_in >> word_count >> word_embedding_size;
    for (int i = 0; i < word_count; i++) {
        string word;
        float embedding;
        file_in >> word;
        d.convert(word);
        for (int j = 0; j < word_embedding_size; j++) {
            file_in >> embedding;
        }
    }
    d.freeze();
    d.set_unk("<unk>");
    file_in.close();
    cout << "Initialized " << d.size() << " words" << endl;
}

int main(int argc, char **argv) {
    dynet::initialize(argc, argv, true);
    po::variables_map conf;
    InitCommandLine(argc, argv, &conf);

    dynet::Dict d;
    cout << "Pid: " << getpid() << endl;
    output_all_information(argc, argv);
//    initialize_word_dict(conf["word_embedding_file"].as<string>(), d);
    NetworkData network_data(conf["node_list_file"].as<vector<string>>(),
                             conf["edge_list_file"].as<vector<string>>(),
                             conf["content_node_file"].as<string>(), d);
    cout << "Vocabulary count: " << d.size() << endl;
    cout << "NetNode count: " << network_data.node_count << endl;
    cout << "Embedding Size: " << network_data.normal_node_count << endl;
    cout << "Link count: " << network_data.edge_list.size() << endl;
    Model lookup_params_model;
    Model params_model;

    // word embedding dim
    unsigned W_EM_DIM =
        get_word_embedding_size(conf["word_embedding_file"].as<string>());
    // content embedding dim
    unsigned C_EM_DIM = W_EM_DIM;
    // node embedding dim
    unsigned N_EM_DIM = W_EM_DIM;

    ContentEmbeddingMethod *content_embedding_method;
    if (conf["embedding_method"].as<std::string>() == "WordAvg") {
        content_embedding_method = new WordAvg_CE(lookup_params_model, params_model,
                                                  W_EM_DIM, C_EM_DIM, d);
    } else if (conf["embedding_method"].as<std::string>() == "GRU") {
        content_embedding_method =
            new GRU_CE(lookup_params_model, params_model, W_EM_DIM, C_EM_DIM, d);
    } else if (conf["embedding_method"].as<std::string>() == "BiGRU") {
        content_embedding_method =
            new BiGRU_CE(lookup_params_model, params_model, W_EM_DIM, C_EM_DIM, d);
    } else {
        std::cerr << "Unsupported embedding method" << std::endl;
        return 1;
    }

    cout << "Content Embedding Method Done." << endl;
    content_embedding_method->initial_look_up_table_from_file(
        conf["word_embedding_file"].as<string>(), d);

    DLNEModel dlne(lookup_params_model,
                   params_model,
                   network_data.normal_node_count,
                   network_data.content_node_count,
                   N_EM_DIM,
                   network_data.edge_type_count,
                   conf["negative"].as<vector<unsigned>>(),
                   content_embedding_method,
                   conf["alpha"].as<vector<float>>(),
                   network_data);
    if (conf.count("to_be_saved_index_file_name")) {
        dlne.set_to_be_saved_index(conf["to_be_saved_index_file_name"].as<string>(), network_data);
    }
    Trainer *lookup_params_trainer = nullptr;
    Trainer *params_trainer = nullptr;
    lookup_params_trainer = new L2SimpleSGDTrainer(
        lookup_params_model, conf["lookup_params_eta0"].as<float>(),
        conf["lookup_params_eta_decay"].as<float>(), conf["lambda"].as<float>());
    params_trainer =
        new L2SimpleSGDTrainer(params_model, conf["params_eta0"].as<float>(),
                               conf["params_eta_decay"].as<float>(), conf["lambda"].as<float>());

    mp_train::RunMultiProcess(
        conf["workers"].as<unsigned>(), &dlne, lookup_params_trainer,
        params_trainer, network_data, conf["iterations"].as<unsigned>(),
        conf["save_every_i"].as<unsigned>(),
        conf["report_every_i"].as<unsigned>(), conf["batch_size"].as<unsigned>(),
        conf["update_epoch_every_i"].as<unsigned>());

    return 0;
}
