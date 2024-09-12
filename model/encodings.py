import torch
import numpy as np
#hash_grid_encoding = __import__("Co-SLAM.hash-grid-encoding")
from hash_grid_encoding.encoding import _HashGrid, Frequency, MultiResHashGrid, OneBlob


def get_encoder(encoding, input_dim=3,
                degree=4, n_bins=16, n_frequencies=12,
                n_levels=16, level_dim=2, 
                base_resolution=16, log2_hashmap_size=19, 
                desired_resolution=512):
    
    # # Dense grid encoding
    # if 'dense' in encoding.lower():
    #     n_levels = 4
    #     per_level_scale = np.exp2(np.log2(desired_resolution  / base_resolution) / (n_levels - 1))
    #     embed = tcnn.Encoding(
    #         n_input_dims=input_dim,
    #         encoding_config={
    #                 "otype": "Grid",
    #                 "type": "Dense",
    #                 "n_levels": n_levels,
    #                 "n_features_per_level": level_dim,
    #                 "base_resolution": base_resolution,
    #                 "per_level_scale": per_level_scale,
    #                 "interpolation": "Linear"},
    #             dtype=torch.float
    #     )
    #     out_dim = embed.n_output_dims
    
    # Sparse grid encoding
    if 'hash' in encoding.lower() or 'tiled' in encoding.lower():
        print('Hash size', log2_hashmap_size)
        embed = MultiResHashGrid(
            dim = input_dim,
            n_levels = n_levels,
            n_features_per_level = level_dim,
            log2_hashmap_size = log2_hashmap_size,
            base_resolution = base_resolution,
        )
        out_dim = embed.output_dim

    # # Spherical harmonics encoding
    # elif 'spherical' in encoding.lower():
    #     embed = tcnn.Encoding(
    #             n_input_dims=input_dim,
    #             encoding_config={
    #             "otype": "SphericalHarmonics",
    #             "degree": degree,
    #             },
    #             dtype=torch.float
    #         )
    #     out_dim = embed.n_output_dims
    
    # OneBlob encoding
    elif 'blob' in encoding.lower():
        print('Use blob')
        embed = OneBlob(
                n_input_dims = input_dim,
                n_bins = n_bins,
                n_levels = 1,
            )
        out_dim = embed.output_dim
    
    # Frequency encoding
    elif 'freq' in encoding.lower():
        print('Use frequency')
        embed = Frequency(
                dim=input_dim,
                n_levels=n_frequencies,
            )
        out_dim = embed.output_dim
    
    # # Identity encoding
    # elif 'identity' in encoding.lower():
    #     embed = tcnn.Encoding(
    #             n_input_dims=input_dim,
    #             encoding_config={
    #             "otype": "Identity"
    #             },
    #             dtype=torch.float
    #         )
    #     out_dim = embed.n_output_dims

    return embed, out_dim