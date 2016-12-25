import subprocess


def main():
    command = [
            './dlne_main',
            '--dynet-mem 500',
            '--dynet-seed 12345',
            '--node_list_file ~/toy_data/nodes_0.data',
            '--edge_list_file ~/toy_data/edge_list_0.data',
            '--content_node_file ~/toy_data/contents.data',
            '--word_embedding_file ~/toy_data/word_embedding.data',
            '--to_be_saved_index_file_name ~/toy_data/labels.data',
            '--eta0 0.3',
            '--eta_decay 0.03',
            '--workers 1',
            '--iterations  100',
            '--batch_size 1000',
            '--save_every_i 10',
            '--report_every_i 100000',
            '--update_epoch_every_i 10000',
            '--negative 1 1 1 1 1',
            '--alpha 1 1 1 1 1',
            '--embedding_method WordAvg',
        ]
    command = ' '.join(command)
    print(command)
    subprocess.call(command, shell=True)


if __name__ == '__main__':
    main()
