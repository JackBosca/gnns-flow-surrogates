import tensorflow as tf
import functools
import json
import os
import h5py

def _parse(proto, meta):
    '''
    Parse a single tf.Example into image and label tensors.
    
    Args:
        proto: a serialized tf.Example
        meta: metadata dictionary
    Returns:
        out: a dictionary of parsed tensors   
    '''
    # Extract dictionary of sparse features
    feature_lists = {k: tf.io.VarLenFeature(tf.string)
                    for k in meta['field_names']}
    features = tf.io.parse_single_example(proto, feature_lists)

    out = {}
    for key, field in meta['features'].items():
        # Decode from a string back to the dense values
        data = tf.io.decode_raw(features[key].values, getattr(tf, field['dtype']))

        # Reshape according to the original shape
        data = tf.reshape(data, field['shape']) 

        if field['type'] == 'static':
            # Repeat static data along the trajectory dimension
            data = tf.tile(data, [meta['trajectory_length'], 1, 1])
        elif field['type'] == 'dynamic_varlen':
            # Convert to a ragged tensor using the length information
            length = tf.io.decode_raw(features['length_' + key].values, tf.int32)
            length = tf.reshape(length, [-1])
            data = tf.RaggedTensor.from_row_lengths(data, row_lengths=length)
        elif field['type'] != 'dynamic':
            raise ValueError('Invalid data format!')

        # Store in the output dictionary
        out[key] = data

    return out

def load_dataset(path, split):
    '''
    Load a dataset from TFRecord files.

    Args:
        path: the path to the dataset directory
        split: the dataset split to load (train, test, valid)
    Returns:
        ds: a tf.data.Dataset object.
    '''
    # Load metadata
    with open(os.path.join(path, 'meta.json'), 'r') as fp:
        meta = json.loads(fp.read())

    # Create a TFRecordDataset and parse the examples
    ds = tf.data.TFRecordDataset(os.path.join(path, split + '.tfrecord'))
    ds = ds.map(functools.partial(_parse, meta=meta), num_parallel_calls=8)
    ds = ds.prefetch(1)

    return ds


if __name__ == '__main__':
    # Change here to your dataset and output paths
    tf_datasetPath='/work/scitas-share/boscario/cylinder_flow'
    out_dir = '/work/scitas-share/boscario/cylinder_flow_h5'
    os.makedirs(out_dir, exist_ok=True)

    for split in ['train', 'test', 'valid']:
        # Load current dataset split
        ds = load_dataset(tf_datasetPath, split)
        save_path = os.path.join(out_dir, split + '.h5')
        print(save_path)

        # Save to HDF5 file
        with h5py.File(save_path, 'w') as f:
            for index, d in enumerate(ds):
                # Convert tensors to numpy arrays
                pos = d['mesh_pos'].numpy()
                node_type = d['node_type'].numpy()
                velocity = d['velocity'].numpy()
                cells = d['cells'].numpy()
                pressure = d['pressure'].numpy()
                data = ("pos", "node_type", "velocity", "cells", "pressure")

                # Create a group for each trajectory and save the data
                g = f.create_group(str(index))
                for k in data:
                    g[k] = eval(k)

                print(f'Saved trajectory {index} with {pos.shape[0]} steps.')