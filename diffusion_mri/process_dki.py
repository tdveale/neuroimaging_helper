import os
import time
from argparse import ArgumentParser, RawDescriptionHelpFormatter
import nibabel as nib
import numpy as np
from dipy.io.gradients import read_bvals_bvecs
from dipy.core.gradients import gradient_table
import dipy.reconst.dki as dki
from dipy.segment.mask import median_otsu
from scipy.ndimage import gaussian_filter

# Arguments
__description__ = '''
DKI model is fit using the constrained weighted least squares (CWLS) method.
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
parser.add_argument('-s', '--smooth',
                    help='Smooth the data with a gaussian filter with FWHM of 1.25mm. Default is False.',
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

# if smooth specified, smooth the data
# often recommended to smooth data before fitting DKI
if args.smooth:
    fwhm = 1.25
    gauss_std = fwhm / np.sqrt(8 * np.log(2))  # converting fwhm to Gaussian std
    data_input = np.zeros(data.shape)
    for v in range(data.shape[-1]):
        data_input[..., v] = gaussian_filter(data[..., v], sigma=gauss_std)
else:
    data_input = data

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
    maskdata, mask = median_otsu(data_input, vol_idx=[0, 1], median_radius=4, numpass=2, autocrop=False, dilate=1)

# initialise DKI model
print("Initialising DKI model...")
dkimodel = dki.DiffusionKurtosisModel(gtab, fit_method='CLS')

# fit the DKI model
print("Fitting DKI model...")
t0 = time.time()
dkifit = dkimodel.fit(data_input, mask=mask)
t1 = time.time()
print("Fitting took: " + str((t1-t0)/60) + " minutes")

# extract mean kurtosis, axial kurtosis and radial kurtosis from the dkimodel
MK = dkifit.mk
AK = dkifit.ak
RK = dkifit.rk

# save outputs
MK_nifti = nib.Nifti1Image(MK, affine)
MK_filename = os.path.join(os.path.dirname(args.dwi), os.path.basename(args.dwi.split('.')[0])) + '_DKI_MK.nii.gz'
nib.save(MK_nifti, MK_filename)
print("MK saved to: " + MK_filename)

AK_nifti = nib.Nifti1Image(AK, affine)
AK_filename = os.path.join(os.path.dirname(args.dwi), os.path.basename(args.dwi.split('.')[0])) + '_DKI_AK.nii.gz'
nib.save(AK_nifti, AK_filename)
print("AK saved to: " + AK_filename)

RK_nifti = nib.Nifti1Image(RK, affine)
RK_filename = os.path.join(os.path.dirname(args.dwi), os.path.basename(args.dwi.split('.')[0])) + '_DKI_RK.nii.gz'
nib.save(RK_nifti, RK_filename)
print("RK saved to: " + RK_filename)
