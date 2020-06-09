'''
To install spikeextractors and spiketoolkit (from master)

For installing different spikesorters: https://spikeinterface.readthedocs.io/en/latest/sortersinfo.html

I recommend using a subset of these three sorters 

1. Ironclust (Pros: accurate, fast, widely used for Neuropixels, Cons: new, lots of adjustable parameters, runs with GPU)
2. Kilosort2 (Pros: accurate, fast, widely used for Neuropixels, Cons: new, lots of false positive units, runs with GPU)
3. Herdingspikes (Pros: fast, doesn't require GPU, still fairly accurate, Matthias can help, Cons: slightly less accurate, less widely used) 

It is actually best if you can use multiple sorters and compute consensus scores (coming soon)!
'''

# If you want to use Kilosort2, Kilosort, or Ironclust (Matlab sorters)
import os
HOME = os.path.expanduser('~')
os.environ["KILOSORT2_PATH"] = HOME + "/duguidlab/shared_lab_resources/Neuropixels/git/Kilosort2"
os.environ["NPY_MATLAB_PATH"] = HOME + "/duguidlab/shared_lab_resources/Neuropixels/git/npy-matlab"
os.environ["KILOSORT_PATH"] = HOME + "/duguidlab/shared_lab_resources/Neuropixels/git/KiloSort"
os.environ["IRONCLUST_PATH"] = HOME + "/duguidlab/shared_lab_resources/Neuropixels/git/ironclust"

import spikeextractors as se
import spiketoolkit as st
import spikesorters as ss
import spikesorters as sc

# Loading the dataset into an extractor (should be memmaped)
file_path = '/home/jdacre/duguidlab/thalamus_paper/Npx_data/raw/Kampff_C1/c1_npx_raw.bin'
recording = se.SpikeGLXRecordingExtractor(file_path=file_path)

# preprocessing (optional)
recording = st.preprocessing.bandpass_filter(recording, freq_min=300, freq_max=6000)
recording = st.preprocessing.common_reference(recording, reference='median')

# Check available and installed spike sorters
#print('Available sorters', ss.available_sorters())
#print('Installed sorters', ss.installed_sorter_list)

## Check default params of sorters (klusta and ms4 in this example)
#print(ss.get_default_params('kilosort2'))
#print(ss.get_default_params('ironclust'))
#print(ss.get_default_params('herdingspikes'))

# Change params and run sorters
#ks2_params = ss.get_default_params('kilosort2')
#ks2_params['detect_threshold'] = 5
#sorting_ks2 = ss.run_kilosort2(recording=recording, **ks2_params)

ic_params = ss.get_default_params('ironclust')
#ic_params['detect_threshold'] = 4
sorting_ic = ss.run_ironclust(recording=recording, **ic_params)

#hs_params = ss.get_default_params('herdingspikes')
#hs_params['detection_threshold'] = 14
#sorting_hs = ss.run_herdingspikes(recording=recording, **hs_params)

# Inspect the results of the sorters
#print('Units found by KS2:', sorting_ks2.get_unit_ids())
print('Units found by IC:', sorting_ic.get_unit_ids())
#print('Units found by HS:', sorting_hs.get_unit_ids())

# Compute SNRs for each found unit
#snrs_ks2 = st.validation.compute_snrs(sorting_ks2, recording)
#snrs_ic  = st.validation.compute_snrs(sorting_ic, recording)
#snrs_hs = st.validation.compute_snrs(sorting_hs, recording)

# Export to phy for manual curation (optional)
#st.postprocessing.export_to_phy(recording, sorting_ks2, output_folder='phy')
out_path = '/home/jdacre/duguidlab/thalamus_paper/Npx_data/processed/Kampff_C1/c1.spikes'
st.postprocessing.export_to_phy(recording, sorting_ic, output_folder=out_path)
#st.postprocessing.export_to_phy(recording, sorting_hs, output_folder='phy')
