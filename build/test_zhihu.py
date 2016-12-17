import subprocess


def main():
    command = [
            './dlne_main',
            '--cnn_mem 800',
            '--graph_file ~/DLNE_data/zhihu/zhihu_graph.data',
            '--content_file ~/DLNE_data/zhihu/zhihu_document_40.data',
            #'--content_file ~/DLNE_data/zhihu/zhihu_splited_document.data',
            '--word_embedding_file ~/DLNE_data/zhihu/zhihu_pre_word_embedding.data',
            #'--to_be_saved_index_file_name ~/DLNE_data/zhihu/zhihu_to_be_saved_index.data',
            '--eta0 0.03',
            '--eta_decay 0.003',
            '--workers 20',
            '--iterations  200000000',
            '--save_every_i 100000000',
            '--batch_size 200',
            '--update_epoch_every_i 10000',
            '--report_every_i 1000000',
            '--vertex_negative 15',
            '--content_negative 35',
            '--alpha 0.5',
            '--embedding_method WordAvg',
            '--strictly_content_required false',
            '--use_const_lookup false',
            '--cnn_filter_count 5',
            '--word_embedding_size 200',
        ]
    command = ' '.join(command)
    print(command)
    subprocess.call(command, shell=True)


if __name__ == '__main__':
    main()
