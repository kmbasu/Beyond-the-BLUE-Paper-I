"""
io_h5.py — HDF5 dataset I/O with latents, parameters, and seeds
===============================================================

Extends the legacy save_dataset_hdf5 with everything the eta-ceiling
pipeline needs while remaining READ-COMPATIBLE with the existing consumers
(the MF notebooks and the HPC ResNet loaders read 'noisy'/'clean'/'I0',
which keep their exact legacy layout, chunking and compression).

File layout
-----------
/noisy       (N, H, W) float32   — observed maps (signal + noise + artifacts)
/clean       (N, H, W) float32   — pre-beam noiseless signal maps
/I0          (N,)      float32   — true source amplitudes
/has_source  (N,)      int8      — 1 = source injected, 0 = noise-only
/latents/*                       — per-image latent variables (see below)
attrs: n_samples, model_name, creation_date, noise_lib_version,
       master_seed, params_json (full config as JSON) + each scalar/string
       parameter as an individual attribute.

Latent storage
--------------
Fixed-size per-image latents (slope, amplitude, angle, n_events, patch
parameter grids, ...) are stored as plain arrays of leading dimension N.
Ragged event lists (glitch positions/amplitudes, crossing rows/amplitudes,
leaked-mode amplitudes) are flattened with ``pack_event_latents`` into

    /latents/<name>_count   (N,) int32      events per image
    /latents/<name>_<field> (sum(count),)   concatenated field values

and can be re-split with ``unpack_event_latents``.
"""

import json
from datetime import datetime

import numpy as np
import h5py


def pack_event_latents(info_lists, fields, name):
    """Flatten a per-image list of event dicts into count + field arrays.

    Parameters
    ----------
    info_lists : list (length N) of lists of dicts, e.g. the ``info``
                 output of the artifact injectors collected per image
    fields     : the dict keys to store, e.g. ('x0', 'y0', 'amp')
    name       : latent name prefix, e.g. 'glitch'

    Returns
    -------
    dict of arrays: {'<name>_count': (N,) int32,
                     '<name>_<field>': (total_events,) float64, ...}
    """
    counts = np.array([len(lst) for lst in info_lists], dtype=np.int32)
    out = {f'{name}_count': counts}
    for f in fields:
        out[f'{name}_{f}'] = np.array(
            [ev[f] for lst in info_lists for ev in lst], dtype=np.float64)
    return out


def unpack_event_latents(latents, fields, name):
    """Inverse of pack_event_latents: return a list (length N) of dicts of
    per-image field arrays."""
    counts = latents[f'{name}_count']
    edges = np.concatenate([[0], np.cumsum(counts)])
    out = []
    for i in range(len(counts)):
        sl = slice(edges[i], edges[i + 1])
        out.append({f: latents[f'{name}_{f}'][sl] for f in fields})
    return out


def save_dataset(outpath, noisy, clean, I0, has_source,
                 params, latents=None, master_seed=None,
                 model_name='', compression='lzf'):
    """Write one simulated image dataset to HDF5.

    Parameters
    ----------
    outpath    : output file path
    noisy      : (N, H, W) array or list of (H, W) arrays — observed maps
    clean      : same layout — noiseless signal maps
    I0         : (N,) true amplitudes
    has_source : (N,) 0/1 flags
    params     : dict of ALL generation parameters (the driver's CONFIG);
                 values must be JSON-serializable scalars/strings/lists
    latents    : dict of per-image latent arrays (see module docstring);
                 optional but strongly recommended
    master_seed: the master RNG seed used by the driver
    model_name : short identifier, e.g. 'T1_PSRAND'
    """
    noisy = np.stack(noisy).astype(np.float32)
    clean = np.stack(clean).astype(np.float32)
    I0_arr = np.asarray(I0, dtype=np.float32)
    hs_arr = np.asarray(has_source, dtype=np.int8)
    n_samples = noisy.shape[0]
    img_chunks = (1,) + noisy.shape[1:]

    with h5py.File(outpath, 'w') as f:
        kw = dict(compression=compression, shuffle=True, dtype=np.float32)
        f.create_dataset('noisy', data=noisy, chunks=img_chunks, **kw)
        f.create_dataset('clean', data=clean, chunks=img_chunks, **kw)
        f.create_dataset('I0', data=I0_arr,
                         chunks=(min(1000, n_samples),),
                         compression='gzip', compression_opts=1,
                         dtype=np.float32)
        f.create_dataset('has_source', data=hs_arr)

        if latents:
            g = f.create_group('latents')
            for key, arr in latents.items():
                g.create_dataset(key, data=np.asarray(arr))

        f.attrs['n_samples'] = n_samples
        f.attrs['model_name'] = model_name
        f.attrs['creation_date'] = datetime.now().isoformat()
        from . import __version__
        f.attrs['noise_lib_version'] = __version__
        if master_seed is not None:
            f.attrs['master_seed'] = int(master_seed)
        f.attrs['params_json'] = json.dumps(params, default=str)
        for key, val in params.items():
            if isinstance(val, (int, float, str, bool, np.integer, np.floating)):
                f.attrs[key] = val

    import os
    mb = os.path.getsize(outpath) / 1024**2
    print(f"Written {n_samples} samples -> {outpath}  ({mb:.1f} MB)")


def load_dataset(path, load_maps=True):
    """Read a noise_lib HDF5 dataset (also reads legacy files gracefully).

    Returns a dict with keys 'noisy', 'clean', 'I0', 'has_source' (None if
    absent, as in legacy files), 'latents' (dict, empty if absent), and
    'attrs' (dict of file attributes, with 'params' parsed from
    params_json when present).
    """
    out = {}
    with h5py.File(path, 'r') as f:
        if load_maps:
            out['noisy'] = f['noisy'][:]
            out['clean'] = f['clean'][:]
        out['I0'] = f['I0'][:]
        out['has_source'] = f['has_source'][:] if 'has_source' in f else None
        out['latents'] = ({k: f['latents'][k][:] for k in f['latents']}
                          if 'latents' in f else {})
        attrs = dict(f.attrs)
        if 'params_json' in attrs:
            attrs['params'] = json.loads(attrs['params_json'])
        out['attrs'] = attrs
    return out
