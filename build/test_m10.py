import subprocess


def main():
    command = [
            './dlne_main',
            '--cnn_mem 800',
            '--graph_file ~/DLNE_data/M10/adjedges.txt',
            '--content_file ~/DLNE_data/M10/docs_tokenized.data',
            #'--content_file ~/DLNE_data/dblp/outputacm_tokenized_document.data',
            '--word_embedding_file ~/DLNE_data/M10/embedding.data',
            '--eta0 0.03',
            '--eta_decay 0.003',
            '--workers 30',
            '--iterations 200000000',
            '--save_every_i 2000000',
            '--batch_size 200',
            '--update_epoch_every_i 10000',
            '--report_every_i 10000',
            '--vertex_negative 15',
            '--content_negative 60',
            '--alpha 0.5',
            '--embedding_method GRU',
            '--strictly_content_required false',
            '--use_const_lookup false',
            '--cnn_filter_count 3',
            '--word_embedding_size 300'
        ]
    command = ' '.join(command)
    print(command)
    subprocess.call(command, shell=True)


if __name__ == '__main__':
    main()
