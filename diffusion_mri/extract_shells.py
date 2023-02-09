from argparse import ArgumentParser, RawDescriptionHelpFormatter
import os
import nibabel as nib
import numpy as np
from dipy.io.gradients import read_bvals_bvecs


# Arguments
__description__ = '''

Author: Tom Veale
Email: tom.veale@ucl.ac.uk

This script extracts volumes with specified shells and writes new DWI volume and associated bvals and bvecs.
Currently written to work with FSL's .bval and .bvec convention files.

For example, to extract b=0 and b=1000 shells where the bvalues vary +/- 15:

    python /path/to/extract_shells.py -i dwi.nii.gz -b dwi.bval -r dwi.bvec -s 0 1000 -t 15

CAUTION!
Please check your outputs are what you expect! This has not been extensively tested.

'''

# collect inputs
parser = ArgumentParser(formatter_class=RawDescriptionHelpFormatter,
                        description=__description__)


parser.add_argument('-i', '--dwi',
                    help='Diffusion weighted image for volumes to be removed from',
                    required=True)
parser.add_argument('-b', '--bval',
                    help='File with bvalues in (.bval)',
                    required=True)
parser.add_argument('-r', '--bvec',
                    help='File with bvecs in (.bvec)',
                    required=True)
parser.add_argument('-s', '--shells',
                    help='Shells to extract separated by space (e.g. 0 1000)',
                    nargs='+',
                    required=True,
                    type=int)
parser.add_argument('-t', '--threshold',
                    help='b-value threshold. Shells will be extracted with +/- this threshold (e.g. 10). Default=0.',
                    required=False,
                    default=0,
                    type=int)
parser.add_argument('-o', '--out_dir',
                    help='Output directory for new files. If not specified, new files will be saved in input DWI location.',
                    required=False)

args = parser.parse_args()

# get bvals and bvecs
bvals, bvecs = read_bvals_bvecs(args.bval, args.bvec)

# load dwi image
dwi_img = nib.load(args.dwi)
data = dwi_img.get_fdata()

# create masks
masks = np.empty((bvals.shape[0], len(args.shells)))

# for each shell value specified, get indices of shell-values that fall within threshold
# create a 2D mask structure to store this
for ishell in range(0, len(args.shells)):
    masks[:, ishell] = (bvals >= args.shells[ishell]-args.threshold) & (bvals <= args.shells[ishell]+args.threshold)

# check data found for each specified shell
# To Do: Make check able to tell which shell is missing data
n_vols = np.sum(masks, axis=0)
if any(n_vols == 0):
    raise ValueError('Data not found for at least one shell!')

# add across columns of masks to create a mask that includes all specified shells
final_mask = np.sum(masks, axis=1)

# check shells werent extracted twice
if np.sum(final_mask > 1) != 0:
    print(np.where(final_mask > 1))
    raise ValueError('Some volumes have been extracted twice! Check your b-values and thresholds for these indices')

# make mask bool
final_mask = final_mask.astype(bool)

# use mask to select bvals, bvecs and dwi vols for specified shells
keep_bvals = bvals[final_mask]
keep_bvecs = bvecs[final_mask, :]
keep_dwi = data[..., final_mask]

# choose output directory
if args.out_dir:
    out_file_dir = args.out_dir
    if not os.path.exists(out_file_dir):
        print("Creating new directory: ", out_file_dir)
        os.makedirs(out_file_dir)
else:
    out_file_dir = os.path.dirname(args.dwi)

# create string with shells
out_file_shells = '_'.join('b' + str(x) for x in args.shells)

# create new filename with specified b-value shells included
out_dwi_file = os.path.basename(args.dwi).split('.nii.gz')[0] + '_' + str(out_file_shells) + '.nii.gz'
out_bval_file = os.path.basename(args.bval).split('.bval')[0] + '_' + str(out_file_shells) + '.bval'
out_bvec_file = os.path.basename(args.bvec).split('.bvec')[0] + '_' + str(out_file_shells) + '.bvec'

# save new dwis
keep_dwi_img = nib.Nifti1Image(keep_dwi, dwi_img.affine)
nib.save(keep_dwi_img, os.path.join(out_file_dir, out_dwi_file))
print('DWI output with', out_file_shells, 'shells saved: ', os.path.join(out_file_dir, out_dwi_file))

# save new bvals
np.savetxt(os.path.join(out_file_dir, out_bval_file), [keep_bvals.astype(int)], fmt='%i')
print('bvalues output with', out_file_shells, 'shells saved: ', os.path.join(out_file_dir, out_bval_file))

# save bvecs
# each row of bvec file is a coordinate: x,y,z so try saving bvec file 1 column at a time
np.savetxt(os.path.join(out_file_dir, out_bvec_file), (keep_bvecs[:, 0], keep_bvecs[:, 1], keep_bvecs[:, 2]), fmt='%f')
print('bvector output with', out_file_shells, 'shells saved: ', os.path.join(out_file_dir, out_bvec_file))


