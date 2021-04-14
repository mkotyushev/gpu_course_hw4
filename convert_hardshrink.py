import argparse
import onnx
import textwrap

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-m', '--model-path', 
        required=True, type=str, 
        help='path to onnx model file'
    )
    parser.add_argument(
        '-s', '--save-path', 
        required=True, type=str, 
        help='path to save modified onnx model'
    )
    parser.add_argument(
        '-v', '--verbose', 
        type=int, default=0,
        help='verbosity level'
    )
    return parser.parse_args()

def print_graph(graph, verbose):
    for i, node in enumerate(graph.node):
        print('\t', i, node.name)
        if verbose > 1: print(textwrap.indent(str(node), '\t\t'))

def main(args):
    """
        Replace custom layer atomic operations with single 
        Hardshrink operation for hardcoded ONNX model
        acquired from TODO
    """
    # Load model
    onnx_model = onnx.load(args.model_path)
    
    if args.verbose > 0:
        print('1. Before removal: ')
        print_graph(onnx_model.graph, args.verbose)

    # Remove atomic operations
    node_indices_to_remove = [
        *list(range(1, 11)),
        *list(range(12, 22)),
        *list(range(25, 35)),
    ]
    for index in node_indices_to_remove[::-1]:
        node = onnx_model.graph.node[index]
        onnx_model.graph.node.remove(node)

    if args.verbose > 0:
        print('2. After removal: ')
        print_graph(onnx_model.graph, args.verbose)

    # Insert Hardshrink nodes
    for i in [5, 2, 1]:
        node_hs = onnx.NodeProto()
        node_hs.op_type = 'Hardshrink'
        node_hs.name = f'hs_{i}'
        node_hs.output.insert(0, f'hs_output_{i}')
        node_hs.input.insert(0, onnx_model.graph.node[i - 1].output[0])
        onnx_model.graph.node[i].input[0] = f'hs_output_{i}'
        onnx_model.graph.node.insert(i, node_hs)

    if args.verbose > 0:
        print('3. After insertion: ')
        print_graph(onnx_model.graph, args.verbose)

    # Save model
    onnx.save(onnx_model, args.save_path)

if __name__ == '__main__':
    args = parse_args()
    main(args)
