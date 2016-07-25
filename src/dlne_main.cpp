//
// Created by Adoni1203 on 16/7/19.
// Diversity Link based Network Embedding (DLNE)
//

#include <boost/program_options.hpp>

#include "graph_data.h"
#include "embedding_methods.h"
#include "network_embedding.h"
#include "simple_mp_train.h"
#include "mp_train.h"
#include "seperate_trainer.h"

using namespace std;
using namespace cnn;
namespace po = boost::program_options;

void InitCommandLine(int argc, char **argv, po::variables_map *conf) {
    po::options_description opts("Configuration options");
    opts.add_options()
            // Data option
            ("graph_file", po::value<string>(), "Graph file, adjacency list")
            ("content_file", po::value<string>(), "content file")
            ("embedding_file", po::value<string>(), "emebdding file of word2vec-format")
            ("eta0", po::value<float>(), "eta0 for sgd")
            ("eta_decay", po::value<float>(), "eta_decay for sgd")
            ("workers", po::value<unsigned>(), "workers count")
            ("iterations", po::value<unsigned>(), "iterations number")
            ("save_every_i", po::value<unsigned>(), "Save frequency as well as the update_epoch frequency")
            ("update_every_i", po::value<unsigned>(), "Update frequency")
            ("report_every_i", po::value<unsigned>(), "Report frequency")
            ("alpha", po::value<float>(), "alpha to control the proportion of VV and VC")


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
    vector<string> required_options{"graph_file", "content_file", "embedding_file", "eta0", "eta_decay", "workers",
                                    "iterations", "save_every_i", "update_every_i", "report_every_i", "alpha"};

    for (auto opt_str:required_options) {
        if (conf->count(opt_str) == 0) {
            cerr << "Error: missed option" << endl;
            cerr << "Please specify --" << opt_str << endl;
            exit(1);
        }
    }
}

int main(int argc, char **argv) {
    cnn::Initialize(argc, argv, 1, true);
    po::variables_map conf;
    InitCommandLine(argc, argv, &conf);
    cnn::Dict d;
    GraphData graph_data(conf["graph_file"].as<string>(), conf["content_file"].as<string>(), d);
    cout << "Vocabulary size: " << d.size() << endl;
    cout << "Node size: " << graph_data.node_count << endl;
    Model model;
    std::cout<<"DBLP trainer: "<<model.parameters_list().size()<<std::endl;

    DLNEModel<WordAvg> dlne(model, graph_data.node_count, 15, 15, 100, 100, 100, conf["embedding_file"].as<string>(),
                            d);
    SeperateSimpleSGDTrainer *sgd = new SeperateSimpleSGDTrainer(&model, 1e-6, conf["eta0"].as<float>());
    sgd->eta_decay = conf["eta_decay"].as<float>();
    mp_train::RunMultiProcess<WordAvg>(conf["workers"].as<unsigned>(), &dlne, sgd, graph_data, conf["iterations"].as<unsigned>(), conf["alpha"].as<float>(),
                    conf["save_every_i"].as<unsigned>(), conf["update_every_i"].as<unsigned>(), conf["report_every_i"].as<unsigned>());


    return 0;
}