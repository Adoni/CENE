import subprocess


def main():
    command = [
            './dlne_main',
            '--dynet-mem 4500',
            '--dynet-seed 1',
            '--node_list_file ~/zhihu_data/nodes.data',
            '--edge_list_file ~/zhihu_data/edge_list.data',
            '--content_node_file ~/zhihu_data/contents.data',
            '--word_embedding_file ~/DLNE_data/zhihu/zhihu_pre_word_embedding.data',
            '--to_be_saved_index_file_name ~/DLNE_data/zhihu/zhihu_to_be_saved_index.data',
            '--eta0 0.03',
            '--eta_decay 0.003',
            '--workers 20',
            '--iterations  100',
            '--batch_size 1000',
            '--save_every_i 2',
            '--report_every_i 10000',
            '--negative 15 15 15 15 15',
            '--alpha 1 1 1 1 1',
            '--embedding_method WordAvg',
        ]
    command = ' '.join(command)
    print(command)
    subprocess.call(command, shell=True)


if __name__ == '__main__':
    main()
