import numpy as np
import nibabel as nib
from argparse import ArgumentParser, RawDescriptionHelpFormatter

__description__ = '''
This script combines labels into one binary mask. All labels must exist in the input image with unique label values.
Useful if wanting to combine multiple labels into one signature region.

Author: Tom Veale
Email: tom.veale@ucl.ac.uk
'''

# collect inputs
parser = ArgumentParser(formatter_class=RawDescriptionHelpFormatter,
                        description=__description__)

parser.add_argument('-i', '--input_file',
                    help='Input nifti file with separately labelled regions.',
                    required=True)
parser.add_argument('-r', '--regions',
                    help='List of values in --input_file that correspond to regions to be combined (space separated).'
                    'E.g. 1001 1002 1003',
                    required=True,
                    type=int,
                    nargs='+')
parser.add_argument('-o', '--out_mask',
                    help='Output nifti mask that is the combination of --regions.'
                    'E.g. voxels that equal 1001 or 1002 or 1003 = 1 (otherwise = 0)',
                    required=True)
args = parser.parse_args()

# load image
input_img = nib.load(args.input_file)

# create mask of voxels that equal values provided - output as int type
combined_mask = np.isin(input_img.get_fdata(), args.regions).astype(int)

# create image for combined region
combined_mask_img = nib.Nifti1Image(combined_mask, input_img.affine, dtype='uint8')

# save new image
nib.save(combined_mask_img, args.out_mask)
print('Combined mask saved: ', args.out_mask)
