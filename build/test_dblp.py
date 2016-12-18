import subprocess


def main():
    command = [
            './dlne_main',
            '--dynet-mem 4500',
            '--dynet-seed 1',
            '--graph_file ~/DLNE_data/dblp/outputacm_graph.data',
            '--content_file ~/DLNE_data/dblp/outputacm_tokenized_splited_document.data',
            #'--content_file ~/DLNE_data/dblp/outputacm_tokenized_document.data',
            '--word_embedding_file ~/DLNE_data/dblp/outputacm_pre_word_embedding.data',
            '--to_be_saved_index_file_name ~/DLNE_data/dblp/outputacm_to_be_saved_index.data',
            '--eta0 0.03',
            '--eta_decay 0.003',
            '--workers 10',
            '--iterations 600000000',
            '--save_every_i  600000',
            '--batch_size 200',
            '--update_epoch_every_i 10000',
            '--report_every_i 10000',
            '--vertex_negative 15',
            '--content_negative 35',
            '--alpha 0.7',
            '--embedding_method CNN',
            '--strictly_content_required false',
            '--use_const_lookup false',
            '--cnn_filter_count 10',
            '--word_embedding_size 200'
        ]
    command = ' '.join(command)
    print(command)
    subprocess.call(command, shell=True)


if __name__ == '__main__':
    main()
