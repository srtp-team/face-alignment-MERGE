import os, argparse
import tensorflow as tf
from tensorflow.python.framework import graph_util

dir = os.path.dirname(os.path.realpath(__file__))
#获得完整路径中的目录部分

def freeze_graph(model_folder):
    # We retrieve our checkpoint fullpath
    checkpoint = tf.train.get_checkpoint_state(model_folder)    #根据checkpoint文件(model_folder)找模型文件名
    input_checkpoint = checkpoint.model_checkpoint_path  #input_checkp将保存最新的tensorflow模型文件的文件路径

    # We precise the file fullname of our freezed graph
    absolute_model_folder = '/'.join(input_checkpoint.split('/')[:-1])  #input_ckp绝对路径的文件夹部分
    output_graph = absolute_model_folder + '/frozen_model.pb'    #输出的图

    # Before exporting our graph, we need to precise what is our output node
    # This is how TF decides what part of the Graph he has to keep and what part it can dump
    # NOTE: this variable is plural, because you can have multiple output nodes
    output_node_names = 'generate_output/output'   #输出模式的名字

    # We clear devices to allow TensorFlow to control on which device it will load operations
    clear_devices = True    #不知道干啥的

    # We import the meta graph and retrieve a Saver
    saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=clear_devices)    #读取元数据

    # We retrieve the protobuf graph definition
    graph = tf.get_default_graph()  #读取图（具体读哪张图啊）
    input_graph_def = graph.as_graph_def()  #（graph的内部结构）

    # We start a session and restore the graph weights
    with tf.Session() as sess:
        saver.restore(sess, input_checkpoint)   #加载参数

        # We use a built-in TF helper to export variables to constants
        output_graph_def = graph_util.convert_variables_to_constants(
            sess,  # The session is used to retrieve the weights
            input_graph_def,  # The graph_def is used to retrieve the nodes
            output_node_names.split(",")  # The output node names are used to select the usefull nodes
        )

        # Finally we serialize and dump the output graph to the filesystem
        with tf.gfile.GFile(output_graph, 'wb') as f:
            f.write(output_graph_def.SerializeToString())
        print('%d ops in the final graph.' % len(output_graph_def.node))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-folder', type=str, help='Model folder to export')
    args = parser.parse_args()

    freeze_graph(args.model_folder)
