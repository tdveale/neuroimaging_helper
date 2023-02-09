import os
import time
from argparse import ArgumentParser, RawDescriptionHelpFormatter
import nibabel as nib
from dipy.io.gradients import read_bvals_bvecs
from dipy.core.gradients import gradient_table
import dipy.reconst.fwdti as fwdti
from dipy.segment.mask import median_otsu

# Arguments
__description__ = '''
This script takes preprocessed diffusion data and fits the adapted free water elimination DTI model (fwDTI)
proposed by Hoy et al. (2014) and algorithm written by Henriques et al. (2017). Outputs are fwFA, fwMD
and free water volume itself (fw) in the directory of the input diffusion data.

This is essentially a wrapper around the DIPY tutorial that makes the code executable as a function from the command line.

Please also see the DIPY tutorial: 

https://dipy.org/documentation/1.0.0./examples_built/reconst_fwdti/

Please cite these below if using this model:

Pasternak, O., Sochen, N., Gur, Y., Intrator, N., Assaf, Y., 2009. 
    Free water elimination and mapping from diffusion MRI. 
    Magn. Reson. Med. 62(3): 717-30. doi: 10.1002/mrm.22055.
Hoy, A.R., Koay, C.G., Kecskemeti, S.R., Alexander, A.L., 2014.
    Optimization of a free water elimination two-compartmental model for diffusion tensor imaging.
    NeuroImage 103, 323-333. doi: 10.1016/j.neuroimage.2014.09.053
Henriques, R.N., Rokem, A., Garyfallidis, E., St-Jean, S., Peterson E.T., Correia, M.M., 2017.
    [Re] Optimization of a free water elimination two-compartment model for diffusion tensor imaging.
    ReScience volume 3, issue 1, article number 2

'''

# collect inputs
parser = ArgumentParser(formatter_class=RawDescriptionHelpFormatter,
                        description=__description__)

parser.add_argument('-i', '--dwi',
                    help='Preprocessed diffusion weighted images (i.e. after TOPUP and eddy)',
                    required=True)
parser.add_argument('-bval', '--bval',
                    help='File with bvalues in (.bval)',
                    required=True)
parser.add_argument('-bvec', '--bvec',
                    help='File with bvecs in (.bvec)',
                    required=True)
parser.add_argument('-m', '--mask',
                    help='File with mask to fit model within. If not specified, simple dipy brain extraction'
                         'masking will be done.',
                    required=False)
args = parser.parse_args()

# get bvals
print("Loading data...")
bvals, bvecs = read_bvals_bvecs(args.bval, args.bvec)
gtab = gradient_table(bvals, bvecs)

# load data
img = nib.load(args.dwi)
data = img.get_fdata()
affine = img.affine

# if mask specified load it
if args.mask:
    print("Mask specified, loading mask: " + args.mask)
    mask_img = nib.load(args.mask)
    mask_data = mask_img.get_fdata()
    mask_affine = mask_img.affine
    # create boolean mask
    mask = mask_data != 0

# if mask not specified, use basic brain extraction used in dipy
else:
    print("Mask not specified, running dipy brain extraction...")
    maskdata, mask = median_otsu(data, vol_idx=[0, 1], median_radius=4, numpass=2, autocrop=False, dilate=1)

# initialise free water DTI model
print("Initialising fwDTI model...")
fwdtimodel = fwdti.FreeWaterTensorModel(gtab)

# fit the fwDTI model
print("Fitting fwDTI model...")
t0 = time.time()
fwdtifit = fwdtimodel.fit(data, mask=mask)
t1 = time.time()
print("Fitting took: " + str((t1-t0)/60) + " minutes")

# extract fwFA, fwMD and FW from the fwDTI model
FA = fwdtifit.fa
MD = fwdtifit.md
FW = fwdtifit.f

# save outputs
FA_nifti = nib.Nifti1Image(FA, affine)
FA_filename = os.path.join(os.path.dirname(args.dwi), os.path.basename(args.dwi.split('.')[0])) + '_fwDTI_FA.nii.gz'
nib.save(FA_nifti, FA_filename)
print("fwFA saved to: " + FA_filename)

MD_nifti = nib.Nifti1Image(MD, affine)
MD_filename = os.path.join(os.path.dirname(args.dwi), os.path.basename(args.dwi.split('.')[0])) + '_fwDTI_MD.nii.gz'
nib.save(MD_nifti, MD_filename)
print("fwMD saved to: " + MD_filename)

FW_nifti = nib.Nifti1Image(FW, affine)
FW_filename = os.path.join(os.path.dirname(args.dwi), os.path.basename(args.dwi.split('.')[0])) + '_fwDTI_FW.nii.gz'
nib.save(FW_nifti, FW_filename)
print("FW saved to: " + FW_filename)
