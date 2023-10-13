import os
import argparse as ap
from collections import OrderedDict
import xml.etree.ElementTree as et
import csv

"""
Extracts GIF volumes from xml files and puts them into a spreadsheet.

Author: Tom Veale - adapted from Dave Cash's code

"""

parser = ap.ArgumentParser(description='Read GIF Parcellations')
parser.add_argument('indir', type=str,
                    help='GIF output directory where xml files stored')
parser.add_argument('outfile', type=str,
                    help='Output CSV file for collected data')
parser.add_argument('--giftype', type=str,
                    help='prob for volumeProb or cat for volumeCat')
args=parser.parse_args()


# Set which gif measure to use depending on user input
if not args.giftype:
    gif_measure = 'volumeProb'
    print('No GIF measurement type selected - Using volumeProb as default')
elif args.giftype == 'prob':
    gif_measure = 'volumeProb'
elif args.giftype == 'cat':
    gif_measure = 'volumeCat'

# walk through all directories in indir and get list of xml files
xml_files = []
for root, dirs, files in os.walk(args.indir):
    for ifile in files:
        if ifile.endswith('.xml'):
            print(os.path.join(root, ifile))
            xml_files.append(os.path.join(root, ifile))

#xml_files = glob.glob(args.indir + '*.xml')
gif_list = []

# loop through each file and get subject's labels and associated volumes
for filename in xml_files:
    subj_xml = filename.split('/')[-1]
    print subj_xml

    # Parse GIF labels and tissue vols from xml
    parc_tree = et.parse(filename)
    parc_root = parc_tree.getroot()
    parc_labels = parc_root.find('labels')
    rois = parc_labels.findall('item')
    # add meta-tissues for TIV here
    parc_tiss = parc_root.find('tissues')
    tiss = parc_tiss.findall('item')
    [rois.append(x) for x in tiss[1:]]      # miss Non-Brain Outer Tissue for now (label conflict)

    # create subject dictionary
    roi_dict = OrderedDict()
    roi_dict['XML'] = subj_xml
    roi_dict['Filename'] = filename

    # for each GIF ROI for this subject
    for r in rois:
        roi_id = r.find('number')
        roi_name = r.find('name')
        roi_vol = r.find(gif_measure)
        roi_label = str(roi_id.text) + ' - ' + roi_name.text
        roi_dict[roi_label] = roi_vol.text
    gif_list.append(roi_dict)

# Create pandas dataframe and write to csv
# total_gif = pd.DataFrame(gif_list)
# total_gif.to_csv(args.outfile, index=False)

# write list of GIF ROIs dictionaries to csv
with open(args.outfile, 'wb') as output_file:
    dict_writer = csv.DictWriter(output_file, gif_list[0].keys())
    dict_writer.writeheader()
    dict_writer.writerows(gif_list)

print('GIF volumes saved to ' + args.outfile)
