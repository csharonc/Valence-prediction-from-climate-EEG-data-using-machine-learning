from glob import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mne
import os   #interaction with operating system, can create directory
from os.path import isfile, join

def main(): #summation of different functions used, not relevant per se
    respondent = "1"
    res = import_data("001_1000384.csv", "Sensor_data_eerste_20_respondenten", "./")
    info, sampling_freq, n_channels, ch_names, ch_types = metadata()
    only_electrodes = ch_names
    res = stim_channels(res)
    stimuli_dict, res = change_stimuli_names(res)
    info, channel_names = make_mne_object(info, sampling_freq, n_channels, only_electrodes, ch_types)
    res = remove_nas(res, only_electrodes)

    check_numeric(res, only_electrodes)

    raw = convertto_raw(res, channel_names, info)
    raw_filter = filter_raw(raw) 
    events, baseline_list, fixcross_list, stimuli_list = extract_events(raw_filter)
    epochs = create_epochs(raw_filter, events, "no")
    low_cutoff, high_cutoff, notch_freq, artifact_threshold, n_fft, n_overlap = parameters(1.0, 0.1)

    FAA3dim, spectrum = calculate_faa_3dim(epochs, n_overlap, n_fft, 256)
    FAA2dim = calculate_faa_2dim(spectrum)

    visualization(spectrum, events, epochs, n_fft, n_overlap, 256, "no")

    create_baseline_epochs(raw_filter, events, fixcross_list)
    create_stimuli_epochs(raw_filter, events, stimuli_list)
    baseline_features, baseline_epochs = baseline_features_FAA(raw_filter, events, baseline_list, sampling_freq, respondent)
    stimuli_features, stimuli_epochs = stimuli_features_FAA(raw_filter, events, stimuli_list, sampling_freq, respondent)
    baseline_features = create_baseline_features_df(events, baseline_epochs, only_electrodes, respondent)
    stimuli_features = create_stimuli_features_df(events, stimuli_epochs, only_electrodes, respondent)
    stimuli_dict = match_bl_to_stimuli(stimuli_dict)
    check_names_in_dict(events, stimuli_dict, "int")




def import_data(filename, folder, path):
    res = pd.read_csv(join(path, folder, filename), skiprows=26, delimiter=',', low_memory=False)
    return res

def metadata(): #to get some info from the data, useful for settings in later functions
    sampling_freq = 256
    n_channels = 20
    ch_names = ["Fp1", "Fp2", "F3", "F4", "F7", "F8", "Fz" ,"Cz", "C3", "C4", "T3", "T4", "T5" , "T6",
                    "P3", "P4", "Pz", "POz", "O1", "O2" ]    #isolate channel columns
    ch_types = ["eeg"] * 20

    info = mne.create_info(ch_names, ch_types=ch_types, sfreq=sampling_freq)   #create metadata
    info.set_montage('standard_1020');    #montage = 10-20 system (spatial channel placement)
    return info, sampling_freq, n_channels, ch_names, ch_types

def stim_channels(res): #We use the beginning of Media instead of ESUTimestamp dus to missing data (and therefore missing cues)
    res['cue'] = "" 
    # res['cue'] = res.apply(lambda row: row['SourceStimuliName'] if row['ESUTimestamp'] > 0 else row['cue'], axis=1) #original code with cue
    res.loc[(res['SlideEvent'] == 'StartMedia') & (res['CollectionPhase'] == 'StimuliDisplay'), 'cue'] = res['SourceStimuliName'] #look for the start of each Media
    res["cue"] = res["cue"].shift(-1) #use one row earlier as the starting point since the Media starts will be filtered out later on
    return res

def change_stimuli_names(res): #make codes for the stimuli numbers based on what they are: stimuli/baseline/fixcross/anything else
    # Name code: xxxxxx (eg 70103). First number = type (stimulus + valence/baseline/fixcross), third number = block (0/1/2), last two numbers = which stimulus (1-6)
    stim_array = res["SourceStimuliName"].unique()

    stimuli_triggers = pd.DataFrame({"StimuliNames":stim_array})
    stimuli_triggers["Value"] = range(10001, 10001 + len(stimuli_triggers))
    stimuli_triggers[""] = 0
    stimuli_triggers[" "] = 0

    stimuli_dict = dict(zip(stimuli_triggers["StimuliNames"], stimuli_triggers["Value"]))
    repeat = "0"

    bl_count = 00
    fc_count = 00
    for key, val in stimuli_dict.items():
        value_str = str(val)
        if "Copy" in key: 
            repeat = "1"
        if "Copy (2)" in key:
            repeat = "2"
        if "baseline_" in key:
            new = "2" + value_str[1:]
            stimuli_dict[key] = int(new)
            if "baseline_open" in key:
                value_str = str(val)
                new2 = new[0] + "1" + new[2:]
                stimuli_dict[key] = int(new2)
            if "baseline_closed" in key:
                value_str = str(val)
                new2 = new[0] + "2" + new[2:]
                stimuli_dict[key] = int(new2)
        if key.startswith("grijsscherm"): #baseline
            bl_count += 1
            new = str(30000 + bl_count)              
            stimuli_dict[key] = int(new)              
        if key.startswith("fixcross"):
            fc_count += 1
            new = str(40000 + bl_count)              
            stimuli_dict[key] = int(new)  
        if key.startswith("Pos"):
            new = value_str.replace(value_str[0], "5")
            if " 1" in key:
                new2 = new[:2] + str(repeat) + "01"
            if " 2" in key:
                new2 = new[:2] + str(repeat) + "02"
            if " 3" in key:
                new2 = new[:2] + str(repeat) + "03"
            if " 4" in key:
                new2 = new[:2] + str(repeat) + "04"
            if " 5" in key:
                new2 = new[:2] + str(repeat) + "05"
            if " 6" in key:
                new2 = new[:2] + str(repeat) + "06"       
            stimuli_dict[key] = int(new2)
        if key.startswith("Neutral"):
            new = value_str.replace(value_str[0], "6")
            if " 1" in key:
                new2 = new[:2] + repeat + "01"
            if " 2" in key:
                new2 = new[:2] + repeat + "02"
            if " 3" in key:
                new2 = new[:2] + repeat + "03"
            if " 4" in key:
                new2 = new[:2] + repeat + "04"
            if " 5" in key:
                new2 = new[:2] + repeat + "05"
            if " 6" in key:
                new2 = new[:2] + repeat + "06"  
            stimuli_dict[key] = int(new2)
        if key.startswith("Neg"):
            new = value_str.replace(value_str[0], "7")
            if " 1" in key:
                new2 = new[:2] + repeat + "01"
            if " 2" in key:
                new2 = new[:2] + repeat + "02"
            if " 3" in key:
                new2 = new[:2] + repeat + "03"
            if " 4" in key:
                new2 = new[:2] + repeat + "04"
            if " 5" in key:
                new2 = new[:2] + repeat + "05"
            if " 6" in key:
                new2 = new[:2] + repeat + "06"  
            stimuli_dict[key] = int(new2)
    # stimuli_dict
    # for key, value in stimuli_dict.items():
    #     print(key, value)
            
    res['cue'] = res['cue'].replace(stimuli_dict)
    res["cue"] = res["cue"].replace({"":0, np.nan: 0, " ":0})
    return stimuli_dict, res

def flag_missing_events(res, only_electrodes, stimuli_dict): #some data was missing so we're going to flag events with missing data by adding a number to their code
    #I'm going to filter out events if they contain ANY missing data, so I can just use the whole segment instead of using timestamps

    start_media = []

    for row in np.arange(1,len(res)-1,1):
        if res["cue"][row] != 0.0: #look only for the cues that are the start of an event, so not 0
            cue_value = res["cue"].iloc[row] #get the name of the cue
            timestamp = res["Timestamp"].iloc[row] #get the timestamp of this cue
            index = row #this is the row number of the cue
            start_media.append([cue_value, timestamp, index]) #save the cue, its index row and the timestamp (this is just in case)

    # media now contains all the stimuli and their start and end timestamp.
    media = pd.DataFrame(start_media, columns=["stim", "start", "row"])

    res2 = res.copy()
    res2["missing"] = ""
    for i in range(len(res2)): #iterate through the whole df
        if str(res2["cue"][i]).startswith("4"):
            cue_value = res2["cue"][i]
            cue_row = media[media["stim"] == cue_value].index[0]
            start_index = media.loc[cue_row, "row"] - int(0.5 * 256) #TIME WINDOW OF THE BASELINE
            
            data = res2.loc[start_index:i, only_electrodes] #look only at important data
            for col in data: #check all electrode columns for missing data
                if (data[col] == 50).any() or (data[col] == -50).any():
                    res2["missing"][i] = "yes"
                    break #when found, don't look further
                else:
                    res2["missing"][i] = "no"
                    break
        if str(res2["cue"][i]).startswith(("5", "6", "7")):
            cue_value = res2["cue"][i]
            cue_row = media[media["stim"] == cue_value].index[0]
            start_index = media.loc[cue_row, "row"] 
            end_index = media.loc[cue_row, "row"] + int(2 * 256) # TIME WINDOW of stimuli (2s = 2*256 measures)
            
            data = res2.loc[start_index:end_index, only_electrodes]
            for col in data:
                if (data[col] == 50).any() or (data[col] == -50).any():
                    res2["missing"][i] = "yes"
                    break
                else:
                    res2["missing"][i] = "no"
                    break
    
    missing_dict = {}
    for i in range(len(res2)):
        cue = str(res["cue"][i])
        if cue not in missing_dict:
            missing_dict[cue] = res2["missing"][i]

    return res2, media, stimuli_dict, missing_dict

def make_mne_object(info, sampling_freq, n_channels, ch_names, ch_types): #use MNE package to process EEG data
    ch_types = ["eeg"] * n_channels + ["stim"]
    channel_names = ch_names + ["cue"]

    info = mne.create_info(channel_names, ch_types=ch_types, sfreq=sampling_freq) #we already did this but okay
    info.set_montage('standard_1020');
    return info, channel_names

def remove_nas(res, ch_names): #remove any rows that are labeled NA (metadata stuff in the raw data)
    res = res[res['EventSource.1'].notna()] 
    len_after_nav = len(res)
    # print(len_after_na)
    # print(f"because of the filtering {len_before_na - len_after_na} rows are lost")

    res[ch_names] = res[ch_names].interpolate() #Maybe don't do this? to keep the NAs and then filter out the whole epoch
    return res

def check_numeric(res, ch_names): #check if any columns contain NAs
    are_numeric = res[ch_names].map(lambda x: pd.to_numeric(x, errors='coerce')).notna().all().all()

    if are_numeric:
        print("All values in the specified columns are numeric.")
    else:
        print("There are non-numeric values in the specified columns.")
    return

def convertto_raw(res, channel_names, info): #convert MNE object to raw MNE object
    res_eeg = pd.concat([res[channel_names]/1000000], axis = 1) #from microvolts to volts?
    res_eeg["cue"] = res_eeg["cue"]*1000000 #since it has been divided before
    res_eeg["cue"] = res_eeg["cue"].astype(int)
    res_eeg["cue"].apply(type)

    #res_eeg["cue"].unique()

    res_ch = np.transpose(res_eeg[channel_names])  #switch rows (time points) and columns (channels)
    raw = mne.io.RawArray(res_ch, info) #create raw MNE file = .fif (file information format)
    
    # raw.plot_sensors(ch_type='eeg');

    # raw.plot();
    return raw

def filter_raw(raw): #filter the raw file based on frequency (notch and bandpass)
    raw_no_nans = raw.copy()
    #raw_no_nans.plot();

    raw_notch = raw_no_nans.notch_filter(freqs = 50, verbose=False)
    #raw_notch.plot();
    # Filter high & low > alles tussen 0.1 en 80 krijg je nu terug.
    raw_filter = raw_notch.filter(l_freq = 0.1, h_freq = 80, verbose=False)
    #raw_filter.plot();

    # Bekijk als tabel
    #raw_filter.to_data_frame()
    return raw_filter

def extract_events(raw_filter): # get events from filtered raw file
    # Extract events
    events = mne.find_events(raw_filter, stim_channel='cue')
    #print(events[:5])
    # Add a column for the baseline (usually 0)
    events_ids = events[:, 2]

    conditiona = lambda x: str(x).startswith(('3')) #choose the right starting number of the stimuli name
    baseline_list = [event for event in events_ids if conditiona(event)] #make a list including only the event names which start with that number

    conditionb = lambda x: str(x).startswith(('4'))
    fixcross_list= [event for event in events_ids if conditionb(event)]

    conditionc = lambda x: str(x).startswith(('5', "6", "7"))
    stimuli_list= [event for event in events_ids if conditionc(event)]

    return events, baseline_list, fixcross_list, stimuli_list
#changed events to filtered_events, will this work?
def create_epochs(raw_filter, events, plot): #make epochs of the stimuli, we don't use this later on
    epochs_filter = mne.Epochs(
    raw_filter,
    events,
    tmin= 0,  # Set your desired tmin
    tmax= 4.0,  # Set your desired tmax
    baseline=None,  # You can adjust the baseline as needed
    reject = dict(eeg = 120)  # You can set rejection criteria as needed
    )
    # You can adjust the baseline as needed
    #reject = dict(eeg=120)  # You can set rejection criteria as needed

    # Plot verdeling frequencies
    epochs_filter.plot_psd(fmin=1, fmax=40, average= False, spatial_colors= True) #LET erop dat je de epochs de juiste fmax geeft
    if plot == "yes":
        # Plot ruw
        epochs_filter.plot();

        epochs_filter.average().plot()
    else:
        pass
    return epochs_filter

def calculate_faa_3dim(epochs, n_overlap, n_fft, n_per_seg): #we don't use this, we'll do that later
    spectrum = epochs.compute_psd(method ="welch", fmin=8., fmax=12.) #, n_fft= n_fft, n_overlap= n_overlap, n_per_seg = n_per_seg
    # bandpower by integrating on the frequency dimension, e.g. alpha band
    F3 = spectrum.get_data(picks = "F3")
    F3 = F3.mean(0).mean(1) #average over frequencies AND welch pieces

    F4 = spectrum.get_data(picks = "F4")
    F4 = F4.mean(0).mean(1)
    
    FAA = np.log(F4) - np.log(F3)
    return FAA, spectrum

def parameters(time_window, overlap):
    sampling_freq = 256

    low_cutoff = 0.5  # Low cutoff frequency for band-pass filter
    high_cutoff = 100.0  # High cutoff frequency for band-pass filter

    notch_freq = 50.0  # Notch frequency for notch filter
    artifact_threshold = 120 # Artifact threshold in micro volts

    time_window = time_window  # Time window for Welch method in seconds
    overlap = overlap  # Overlap for Welch method

    n_fft = int(time_window * sampling_freq)
    n_overlap = int(overlap * sampling_freq)
    return low_cutoff, high_cutoff, notch_freq, artifact_threshold, n_fft, n_overlap

def calculate_faa_2dim(spectrum): #we don't use this, we'll do that later
    # bandpower by integrating on the frequency dimension, e.g. alpha band
    F3 = spectrum.get_data(fmin=8, fmax=12, picks = "F3")
    F3 = F3.mean(1) #average over frequencies

    F4 = spectrum.get_data(fmin=8, fmax=12, picks = "F4")
    F4 = F4.mean(1)
    
    FAA = np.log(F4) - np.log(F3)
    return(FAA)

def visualization(epochs,plot): #visualization
    spectrum = epochs.compute_psd(method ="welch", fmin=4, fmax=80, n_fft= 256, n_overlap=int(0.25 * 256), n_per_seg=int(0.5 * 256)) 
    bands = {"10 Hz": 10, "15 Hz": 15, "20 Hz": 20, "10-20 Hz": (10, 20)}

    if plot == "yes":
        # spectrum.plot_topomap()
        spectrum.plot_topomap(bands=bands, vlim="joint")
        epochs.plot();
        epochs.plot_psd(fmin=4, fmax=80, average= False, spatial_colors= True)

        epochs.average().plot()
    else:
        pass

    #Visualize events (is still ugly)
    #mne.viz.plot_events(events)
    return

def create_baseline_epochs(raw_filter, events, fixcross_list): #these are the epochs we'll use
# Create baseline epochs
    baseline_epochs = mne.Epochs(
        raw_filter,
        events,
        event_id={str(ev): ev for ev in fixcross_list},  # Convert event IDs to strings in the dict
        tmin=-0.5, # The baseline epoch will be 500 ms long this way
        tmax=0,  # This function should return a scalar value
        baseline=None,
        reject=dict(eeg=120),
        preload=True
    )
    return baseline_epochs

def create_stimuli_epochs(raw_filter, events, stimuli_list): #these are the epochs we'll use
    # Create stimuli epochs
    stimuli_epochs = mne.Epochs(
        raw_filter,
        events,
        event_id= {str(ev): ev for ev in stimuli_list},  # create a dict with stimuli event IDs
        tmin=0, # CHANGED TIME WINDOW FROM 0.3-0.7 TO 0-2.0
        tmax=2,
        baseline=None,
        reject=dict(eeg=120),
        preload=True
    )
    return stimuli_epochs

def baseline_features_FAA(raw_filter, events, fixcross_list, sampling_freq, respondent): #generate baseline FAA
    baseline_epochs = create_baseline_epochs(raw_filter, events, fixcross_list) #LET OP: WAS EERST BASELINE_LIST, NU FIXCROSS_LIST
    baseline_features = pd.DataFrame(columns=["Type", "ID"])

    FAA3_list_bl = []
    ID_list_bl = []
    baseline_str_list = []
    res = []
    for ep in events:
        ep_name = ep[2]
        if str(ep_name).startswith("4"): #baseline LET OP: IS NU ALLEEN FIXCROSSES IPV BASELINE, WANT DE BASELINES ZIJN GENOMEN VANAF DE FIXCROSS
            ep_namestr = str(ep_name)
            res.append(respondent)
            ID_list_bl.append(ep_namestr)
            # print(f"this is baseline {ep_namestr}")
            # calculate_faa_3dim(baseline_epochs[ep_namestr], int(0.1 * sampling_freq), 256, 256)

            spectrum_unique = baseline_epochs[ep_namestr].compute_psd(method ="welch", fmin=4, fmax=20) #, n_fft= 256, n_overlap= int(0.1 * sampling_freq), n_per_seg = 256
            F3 = spectrum_unique.get_data(fmin=8, fmax=12, picks = "F3")
            F3 = F3.mean(0).mean(1)

            F4 = spectrum_unique.get_data(fmin=8, fmax=12, picks = "F4")
            F4 = F4.mean(0).mean(1)
            
            FAA3 = str(np.log(F4) - np.log(F3))
            FAAa = FAA3.replace("[", "")
            FAA = float(FAAa.replace("]", ""))
            FAA3_list_bl.append(FAA)
            baseline_str_list.append("baseline")

    baseline_features["ID"] = ID_list_bl
    baseline_features["FAA"] = FAA3_list_bl
    baseline_features["Type"] = baseline_str_list
    baseline_features["Respondent"] = res
    return baseline_features, baseline_epochs

def stimuli_features_FAA(raw_filter, events, stimuli_list, sampling_freq, respondent): #generate stimuli FAA
    stimuli_epochs = create_stimuli_epochs(raw_filter, events, stimuli_list)
    stimuli_features = pd.DataFrame(columns=["Type", "ID"])

    FAA3_list_st = []
    ID_list_st = []
    type_list_st = []
    res = []
    for ep in events:
        ep_name = ep[2]
        if str(ep_name).startswith(("5", "6", "7")): #stimuli
            ep_namestr = str(ep_name)
            res.append(respondent)
            ID_list_st.append(ep_namestr)
            # print(f"this is baseline {ep_namestr}")
            # calculate_faa_3dim(baseline_epochs[ep_namestr], int(0.1 * sampling_freq), 256, 256)

            spectrum_unique = stimuli_epochs[ep_namestr].compute_psd(method ="welch", fmin=4, fmax=20, n_fft= int(2* 256), n_overlap=int(0.5 * 256), n_per_seg=int(256)) #MAYBE CHANGE THIS AS WELL IF STIMULUS IS 2000 ms, base it off parameters below
            F3 = spectrum_unique.get_data(fmin=8, fmax=12, picks = "F3")
            F3 = F3.mean(0).mean(1)

            F4 = spectrum_unique.get_data(fmin=8, fmax=12, picks = "F4")
            F4 = F4.mean(0).mean(1)
            
            FAA3 = str(np.log(F4) - np.log(F3))
            FAAa = FAA3.replace("[", "")
            FAA = float(FAAa.replace("]", ""))
            FAA3_list_st.append(FAA)
            
            if ep_namestr.startswith("5"):
                ty = "pos"
            elif ep_namestr.startswith("6"):
                ty = "neu"
            elif ep_namestr.startswith("7"):
                ty = "neg"
            type_list_st.append(ty)


    stimuli_features["ID"] = ID_list_st
    stimuli_features["FAA"] = FAA3_list_st
    stimuli_features["Type"] = type_list_st
    stimuli_features["Respondent"] = res
    return stimuli_features, stimuli_epochs

def create_baseline_features_df(events, baseline_epochs, only_electrodes, respondent, res, missing_dict): #generate baseline PSD
    baseline_features = pd.DataFrame()
    resp = []
    baseline_list = []
    baseline_ID = []


    for ep in events:
        ep_name = str(ep[2])
        if ep_name.startswith("4"): #baseline LET OP: IS NU ALLEEN FIXCROSSES IPV BASELINE, WANT DE BASELINES ZIJN GENOMEN VANAF DE FIXCROSS
            
            resp.append(respondent)
            baseline_list.append("baseline")
            baseline_ID.append(str(ep_name))

            ep_namestr = str(ep_name)
            spectrum_unique = baseline_epochs[ep_namestr].compute_psd(method ="welch", fmin=4, fmax=80, n_fft= 256, n_overlap=int(0.25 * 256), n_per_seg=int(0.5 * 256)) #ADJUST BASED ON TIME WINDOW, this just stays like this for 500 ms
            
            # baseline_features3["ID"] = ep_namestr
            for ch in only_electrodes:
                for i in range(4, 81):
                    hz = spectrum_unique.get_data(fmin=i, fmax=i, picks = ch)
                    hz_unpacked = hz[0][0][0]
                    column = ch + "_" + str(i) + "hz"
                    baseline_features.loc[ep_namestr, column]  = float(hz_unpacked)
                
    
    baseline_features["Type"] = baseline_list
    baseline_features["ID"] = baseline_ID
    baseline_features["Respondent"] = resp
    baseline_features = baseline_features[["Respondent", "ID", "Type"] + [col for col in baseline_features.columns if col not in ["Respondent", "ID", "Type"]]]

    baseline_features["Missing"] = ""
    for i in range(len(baseline_features)):
        ID = baseline_features["ID"][i]
        new = ID + ".0" #The ID as in the dictionary has a bit of a different format, so im just adding .0 to get the IDs the exact same
        if new in missing_dict.keys():
            baseline_features["Missing"][i] = missing_dict[new]

    return baseline_features

def create_stimuli_features_df(events, stimuli_epochs, only_electrodes, respondent, res, missing_dict):  #generate stimuli PSD
    stimuli_features = pd.DataFrame()

    stimuli_str_list_PSD = []
    resp = []
    stimuli_ID = []

    for ep in events:
        ep_name = ep[2]
        if str(ep_name).startswith(("5", "6", "7")): #stimuli
            ep_namestr = str(ep_name)
            resp.append(respondent)
            stimuli_ID.append(str(ep_name))

            if ep_namestr.startswith("5"):
                ty = "pos"
            elif ep_namestr.startswith("6"):
                ty = "neu"
            elif ep_namestr.startswith("7"):
                ty = "neg"
            stimuli_str_list_PSD.append(ty)

            spectrum_unique = stimuli_epochs[ep_namestr].compute_psd(method ="welch", fmin=4, fmax=80, n_fft= int(2* 256), n_overlap=int(0.5 * 256), n_per_seg=int(256)) #ADJUST BASED ON TIME WINDOW (either 400 ms or 2000 ms)

            for ch in only_electrodes:
                for i in range(4, 81):
                    hz = spectrum_unique.get_data(fmin=i, fmax=i, picks = ch)
                    hz_unpacked = hz[0][0][0]
                    column = ch + "_" + str(i) + "hz"
                    stimuli_features.loc[ep_namestr, column]  = float(hz_unpacked)

    stimuli_features["Type"] = stimuli_str_list_PSD
    stimuli_features["ID"] = stimuli_ID
    stimuli_features["Respondent"] = resp
    stimuli_features = stimuli_features[["Respondent", "ID", "Type"] + [col for col in stimuli_features.columns if col not in ["Respondent", "ID", "Type"]]]
        
    stimuli_features["Missing"] = ""
    for i in range(len(stimuli_features)):
        ID = stimuli_features["ID"][i]
        new = ID + ".0"
        if new in missing_dict.keys():
            stimuli_features["Missing"][i] = missing_dict[new]
    return stimuli_features

def match_bl_to_stimuli(stimuli_dict): #LET OP: AANGEPAST NAAR FIXCROSS
    base_to_stimulus = {}
    for key, value in stimuli_dict.items():
        if str(value).startswith("4"): #Dit is nu 4 ipv 3
            current_index = list(stimuli_dict.values()).index(value)
            if (current_index + 2) < len(stimuli_dict):
                stimulus_val = list(stimuli_dict.values())[current_index + 1] #Dit is nu +1 ipv +2
                if str(stimulus_val).startswith(("5", "6", "7")):
                    base_to_stimulus[value] = stimulus_val

    return base_to_stimulus


def check_names_in_dict(events, dictionary, ty): #check if all the right events are in the dict used to do the baseline correction
    for e in events:
        if ty == "int":
            ep = int(e[2])
        if ty == "str":
            ep = str(e[2])
        if str(ep).startswith(("5", "6", "7")) and ep not in dictionary.values():
            print(f"{ep} is not in the dict")
        if str(ep).startswith("3") and ep not in dictionary.keys():
            print(f"{ep} is not in the dict")
    return

def baseline_correction(baseline_features, stimuli_features): #baseline correction of all features
    features = stimuli_features.copy()
    skip_columns = ["Respondent", "ID", "Type", "Missing"] #only keep columns with actual data: FAA and PSD

    for i in range(len(features)):
        for col in features.columns:
            if col not in skip_columns:
                features[col][i] = stimuli_features[col][i] - baseline_features[col][i]
    return features

def import_survey_df():
    survey_raw = pd.read_csv("Survey_ratings.csv", delimiter = ',', low_memory = False)
    survey = survey_raw[(survey_raw["question"] == "valence")].drop("Unnamed: 0", axis = 1)
    return survey

def get_ratings(stimuli_list, survey):
    ratings_dict = {}
    for stimulus in stimuli_list:
        stim = str(stimulus)
        if stim.startswith("5"):
            condition = survey["image"].str.startswith("Pos") & survey["image"].astype(str).str.endswith(stim[4])
        elif stim.startswith("6"):
            condition = survey["image"].str.startswith("Neu") & survey["image"].astype(str).str.endswith(stim[4])
        elif stim.startswith("7"):
            condition = survey["image"].str.startswith("Neg") & survey["image"].astype(str).str.endswith(stim[4])
        filtered = survey[condition]
        all_values = filtered["LABEL.VALUE"].mean()
        ratings_dict[stim] = all_values
    return ratings_dict

def get_individual_ratings(features, survey):
    survey = import_survey_df()
    #features = pd.read_csv("features.csv")

    #Make a copy to work in and make new columns for the shortened participant number and the image number
    features1 = features.copy()
    features1["Participant_ID_short"] = features["Participant_ID"].str.replace(".csv", "").str[4:]
    features1["image_number"] = features1["ID"].astype(str).str[-1]

    #Make a dict with all individual ratings
    ratings_ind = {}
    for i in range(len(survey)):
        participant_name = survey["RESPONDENT"].astype(str).values[i]
        image_type = survey["image"].values[i][:3]

        image_number = survey["image"].values[i][-1] 
        rating = survey["LABEL.VALUE"].values[i]
        name = f"{participant_name}_{image_type}_{image_number}" 
        ratings_ind[name] =  [participant_name, image_type, image_number, rating]

    #Match the participants from features to the dict and get their individual rating
    for i in range(len(features1)):
        par = features1["Participant_ID_short"].astype(str).values[i]
        matching = [key for key in ratings_ind if par in key]
        if matching:
            for key in matching:
                par = ratings_ind[key][0]
                image_type = ratings_ind[key][1]
                image_number = ratings_ind[key][2]
                rating = ratings_ind[key][3]
                if features1["Type"].astype(str).values[i] == image_type.lower() and features1["image_number"].values[i] == image_number:
                    features1.loc[i, "individual_ratings"] = rating
    
    
    #Reorder the columns
    desired_order = ['Respondent', 'Participant_ID', 'mean_rating', 'Participant_ID_short', 'image_number', 'individual_ratings', "Missing"]

    features1 = features1[desired_order + [col for col in features1.columns if col not in desired_order]]
    return features1



if __name__ == "__main__":
    main()