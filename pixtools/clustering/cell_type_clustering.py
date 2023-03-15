"""
Functions to use 1D template (average waveform) metrics to cluster units, and
visualise the results.
"""

from sklearn.manifold import TSNE
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
from statannotations.Annotator import Annotator

from pixtools import utils

def cluster_cell_types(self):
    # get other infos from good units
    mouse_info, _ = exp.get_good_units_info()
    assert 0

    print("> getting good unit ids...\n")
    good_units = exp.select_units()

    #TODO: mar 8 do stats test on waveform features (one-way anova for within mouse, two
    #way if including all mice
    #TODO: mar 8 use statannotations to add stats results on plots
    """
    # remove old waveform metrics file
    for i, session in enumerate(selected_sessions):
        cache_metrics = session.interim/'cache/get_waveform_metrics.h5'
        if os.path.isfile(cache_metrics):
            os.remove(cache_metrics)
    """
    # NOTE: if intend to get all good units, DO NOT specify unit ids for waveform metrics
    # otherwise will mess up!
    mouse_waveform_metrics, _ = exp.get_waveform_metrics()

    # do k means on selected metrics or all
    cluster_on_all_metrics = True

    #jan 24th CONTINUE here, run this for first two mice!
    for m, mouse in enumerate(mice):
        # NOTE: mouse_waveform_metrics have less units bc units that have any nan metric
        # are dropped
        df = mouse_waveform_metrics[mouse]
        all_metrics = df.columns[1:].values
        metrics = ["trough_to_peak", "half_width"]#, "repolarisation_slope"]

        # create placeholder for saving unit_type
        mouse_info[mouse]["unit_type"] = np.nan

        if cluster_on_all_metrics:
            metrics = all_metrics

        df["area"] = mouse_info[mouse].area
        hpfs = df[df.area == "HPF"]
        v1s = df[df.area == "V1"]
        dfs = [hpfs, v1s]
        # exclude nan regions
        regions = np.setdiff1d(np.unique(df.area.values), "na")

        for r, region in enumerate(regions):
            # trough to peak for clustering
            reg_df = dfs[r]
            data = reg_df[metrics].values
            #ml_utils.screeplot(data)
            k = 3 # based on screeplot

            # cluster on trough-to-peak & half-width
            pred_label = ml_utils.k_means_clustering(
                n_clusters=k,
                data=data, # make sure to hide unit ids
            )
            reg_df["pred_group"] = pred_label

            # plot all metrics
            plt.clf()
            subplots = utils.Subplots2D(all_metrics)
            axes = subplots.axes_flat

            for t, metric in enumerate(all_metrics):
                sns.violinplot(
                    data=reg_df,
                    x="pred_group",
                    y=metric,
                    scale="count", # default is area
                    ax=axes[t],
                    legend=True,
                    #split=True, # can only work if hue=2
                )
            sub = f"\nClustered on {metrics}, from {reg_df.shape[0]} units."
            head = f"{mouse} norm {region} distribution of waveform metrics {len(metrics)}"
            note = f"\nNote waveform metrics are computed from normalised waveform."
            plt.suptitle(head + sub + note)
            #utils.save(fig_dir + head.replace(" ", "_"))
            #plt.show()
            #continue

            idx, _, counts = np.unique(
                pred_label,
                return_counts=True,
                return_index=True,
            )
            # biggest population is always rs
            rs1_label = np.argmax(counts) # regular-spiking 1
            maybe_fs = np.argmin(counts) # maybe fast-spiking
            remain_labels = np.setdiff1d(idx, rs1_label)

            # group with higher 95% quantile is fs
            q95 = np.zeros(len(remain_labels))
            for l, label in enumerate(remain_labels):
                q95[l] = reg_df[reg_df.pred_group == label].trough_to_peak.quantile(0.95)

            fs_label = remain_labels[np.argmin(q95)] # fast-spiking, lower q95
            rs2_label = np.setdiff1d(remain_labels, fs_label)[0]# regular-spiking 2
            # make sure fs is the group with lowest trough-to-peak 95% quantile
            assert maybe_fs == fs_label

            reg_df.loc[reg_df.pred_group == rs1_label, "pred_group"] = "regular-spiker1"
            reg_df.loc[reg_df.pred_group == rs2_label, "pred_group"] = "regular-spiker2"
            reg_df.loc[reg_df.pred_group == fs_label, "pred_group"] = "fast-spiker"

            ratios = reg_df.pred_group.value_counts(
                    ascending=False,
                    normalize=True)
            print(f"> From {mouse} {region}, {round(ratios[0]*100, 3)}% are {ratios.index[0]},\
            \n{round(ratios[1]*100, 3)}% are {ratios.index[1]}, and\
            \n{round(ratios[2]*100, 3)}% are {ratios.index[2]}.\n") 

            plt.clf()
            subplots = utils.Subplots2D(all_metrics)
            axes = subplots.axes_flat
            for t, metric in enumerate(all_metrics):
                sns.violinplot(
                    data=reg_df,
                    x="pred_group",
                    y=metric,
                    scale="count", # default is area
                    ax=axes[t],
                    legend=True,
                    #split=True, # can only work if hue=2
                )
            sub = f"\nClustered on {metrics}, from {reg_df.shape[0]} units."
            head = f"{mouse} norm {region} distribution of waveform metrics of fs and rs units {len(metrics)}"
            note = f"\nNote waveform metrics are computed from normalised waveform."
            plt.suptitle(head + sub + note)
            #utils.save(fig_dir + head.replace(" ", "_"))
            #plt.show()

            # visualise distribution of waveform metrics against each other 
            plt.clf()
            sns.pairplot(
                data=reg_df.drop(columns="unit"),
                hue='pred_group',
            )
            sub = f"\nClustered on {metrics}, from {reg_df.shape[0]} units."
            head = f"{mouse} norm {region} pairplot of waveform metrics {len(metrics)}"
            note = f"\nNote waveform metrics are computed from normalised waveform."
            plt.suptitle(head + sub + note)
            plt.subplots_adjust(top=0.9) # make room for title
            #utils.save(fig_dir + head.replace(" ", "_"))
            #plt.show()

            # update info file
            clusters = reg_df.loc[:,("unit","pred_group")]
            mouse_info[mouse].loc[mouse_info[mouse].area == region, 'unit_type'] = clusters.loc[:,"pred_group"]

            # export rs & fs unit ids
            for i, session in enumerate(selected_mouse_sessions[mouse]):
                clust_results = {}
                session_clusters = clusters.loc[i]
                output = session.processed / f'{region}_clustered_unit_ids.json'

                if output.exists():
                    print(f"> session {session.name} already have clustered {region} unit ids.\
                            \nNext session.\n")
                    results = json.load(open(output, mode="r"))
                    # output for zahid
                    output_zp = f'/home/amz/w/arthur/npx/unit_clustering/{session.name}_{region}clustered_unit_ids.csv'
                    df = pd.DataFrame(dict([(key, pd.Series(val)) for key, val in results.items()]))
                    df.to_csv(
                        output_zp,
                        index=False,
                        sep='\t',
                    )
                    continue

                # save updated info file
                #NOTE: some units do not have unit_type label bc at least one of their
                #waveform metrics is nan
                info_dir = session.interim / 'good_units_info.tsv'
                mouse_info[mouse].loc[i].to_csv(
                    info_dir,
                    index=False,
                    sep='\t',
                )

                for u, unit_type in enumerate(np.unique(clusters.pred_group.values)):
                    try:
                        clust_results[unit_type] = session_clusters.groupby(
                            "pred_group").get_group(unit_type).unit.values.tolist()
                    except:
                        clust_results[unit_type] = None
                        print(f"> Session {session.name} {region} does not have {unit_type}.\n")

                # save output
                az_utils.write_json(
                    clust_results,
                    output,
                )

                # output for zahid
                output_zp = f'/home/amz/w/arthur/npx/unit_clustering/{session.name}_{region}clustered_unit_ids.csv'
                df = pd.DataFrame(dict([(key, pd.Series(val)) for key, val in clust_results.items()]))
                df.to_csv(
                    output_zp,
                    index=False,
                    sep='\t',
                )

    assert 0
    #NOTE: indexing into all stuff from the first level in hierarchical table:
    #df.loc[first_idx, :]

    #TODO
    #fancy t-SNE with multi-channel waveform later...
    #read https://journals.physiology.org/doi/full/10.1152/jn.00680.2018;
    #https://www.jmlr.org/papers/volume9/vandermaaten08a/vandermaaten08a.pdf

    assert 0
    ###################################################
    """
    From here on is the attempt to use spikeinterface for generating template waveforms
    and cell type clustering.
    Jan 24th 2023: Currently there are issues from the si end, thus attempt aborted.
    """
    #TODO: use spikeinterface method to get template metrics?
    for i, session in enumerate(selected_sessions):
        output = session.processed/'clustered_unit_ids.json'
        try:
            ids = json.load(open(output, mode="r"))
            print(f"> session {session.name} already have clustered unit ids.\
                    \nNext session.\n")
            continue
        except:
            break
        #else:
        #finally:

        #TODO: how to save clustering results (ids) with informative names and can be
        #used to load into pixels pipeline?
        assert 0
        v1_py = session.get_spike_waveforms()

        assert 0
        session.get_spike_waveforms(method="spikeinterface")
        # also output csv file for zahid
        output_zp = f'/home/amz/w/arthur/npx/unit_clustering/{session.name}_clustered_unit_ids.csv'
        good_unit_ids_zp = f'/home/amz/w/arthur/npx/unit_clustering/{session.name}_good_unit_ids.csv'

        V1_units = az_utils.get_V1_units(session)
        HPF_units = az_utils.get_HPF_units(session)
        print(f"> Overlapps between V1 & HPF units selection:\
              \n{set(HPF_units) & set(V1_units)}.\n")

        """
        #sorting = se.NpzSortingExtractor(session.ks_output / 'saved_si_sorting_obj/sorting_cached.npz')
        # make sure to delete `cluster_info.csv` before loading the sorting object
        sorting = se.read_kilosort(session.ks_output)
        #TODO: load recording object
        recording = si.load_extractor(session.interim / 'cache/recording.json')

        print("> Re-extracting waveforms after manual curation...\n")
        waveforms_dir = session.interim / 'cache'
        waveforms = si.WaveformExtractor.load_from_folder(waveforms_dir)

        print(f"> extracting\
            \n{spost.get_template_metric_names()}\
            \nfrom waveforms...\n")

        # define sparsity
        sparsity_best = spost.get_template_channel_sparsity(
            waveforms,
            method="best_channels",
            num_channels=8,
        )
        #TODO: make sparsity work
        #sparsity_best[sorting.unit_ids[unit_id]]

        template_metrics = spost.compute_template_metrics(waveforms)
        #TODO: only saves template metrics of good units.
        print(f"> Template matrics saved in {waveforms_dir/'template_metrics'}.\n")
        #sw.plot_template_metrics(waveforms)
        #plt.show()

        print("> saving template metrics for good units...\n")
        metrics = pd.read_csv(
            waveforms_dir/'template_metrics/metrics.csv',
            #index_col=[0],
        )
        # give unit_id column name
        metrics.columns = ["unit_id", *metrics.columns[1:]]

        # getting template metrics for good units
        good_metrics = metrics[metrics.unit_id.isin(good_units[i])]
        good_metrics.reset_index(drop=True, inplace=True)
        print(f"> good units not in metrics are\
                \n{set(good_units[i]) - set(good_metrics.unit.to_list())}.\n")
        """
        #waveform = waveforms[i]
        good_metrics = waveform_metrics[i].dropna() #TODO how to avoid nan?
        print(f"\n> good units not in metrics are\
                \n{set(good_units[i]) - set(good_metrics.unit.to_list())}.\n")
        #assert len(V1_units) + len(HPF_units) == len(good_units[i])
        #assert len(good_units[i]) == len(good_metrics.unit.to_list())

        print("> clustering units into pyramidal and interneuron...\n")
        """
        units = waveform.columns.get_level_values("unit").unique()
        cols = waveform.index.values
        wf = {}
        #TODO: run k means directly on normalised median waveform from all sessions
        #for each session, get median waveform, then concat (maybe in experiment?)
        for u, unit in enumerate(units):
            mean_waveform = waveform[unit].mean(axis=1)
            values = mean_waveform.values
            wf[unit] = values
        df = pd.DataFrame(wf).T
        df.columns = cols

        ml_utils.screeplot(df)
        pred_labels_waveform = ml_utils.k_means_clustering(
            #n_clusters=2,
            n_clusters=3, # based on screeplot
            #n_clusters=4, # based on screeplot
            data=df, # make sure to hide unit ids
        )
        good_metrics.loc[:, "pred_group_waveform"] = pred_labels_waveform
        print(np.unique(pred_labels_waveform, return_counts=True))

        #TODO: this here is spikeinterface method
        colors = ['Olive', 'Teal', 'Fuchsia']
        fig, ax = plt.subplots()

        for i, unit_id in enumerate(ids):
            waveform = waveforms.get_waveforms(unit_id)
            spiketrain = sorting.get_unit_spike_train(unit_id)
            ax.plot(wf[:, :, best_chan].T, color=color, lw=0.3)
            print("> plotting template (average waveforms) of the best channel")
            template = waveforms.get_template(unit_id)
            sw.plot_unit_templates(
                waveform,
                sparsity=sparsity_best,
            )
        plt.show()
        """


def plot_TSNE(waveform_metrics, cell_type, area, colours):
    """
    Visualise waveform metrics & clustering results using T-distributed stochastic
    neighbour embedding.

    params
    ===
    waveform_metrics: df, waveform metrics of all units from a mouse.

    cell_type: nd array, cell type clustering results.

    area: nd array, area label of each unit.

    colours: dict, colour code of each cell type.

    return
    ===
    fig
    """

    X_embedded = TSNE(
        n_components=2,
        learning_rate='auto',
        init='random',
        perplexity=30, # num of nearest neighbour
        n_jobs=-1,
    ).fit_transform(waveform_metrics)

    # create df to make plotting easier
    df = pd.DataFrame(X_embedded)
    df.columns = ['D1', 'D2']
    df['cell_type'] = cell_type
    df['area'] = area
    # exclude nan regions
    areas = np.setdiff1d(np.unique(df.area.values), "na")

    fig, axes = plt.subplots(
        ncols=areas.shape[0],
        nrows=1,
        sharex=True,
    )

    for a, area in enumerate(areas):
        sns.scatterplot(
            data=df[df.area == area],
            x='D1',
            y='D2',
            hue='cell_type',
            hue_order=colours.keys(),
            ax=axes[a],
        )
        axes[a].set_title(f'TSNE of {area}')

    return fig


def plot_isi(sessions, spike_times, area, cell_type, colours):
    """
    get inter-spike interval from spike time intervals, and plot distribution for
    each cell type.
    """
    names = ['unit', 'area', 'cell_type']

    #get waveforms and plot
    isi_dic = {}
    for i, session in enumerate(sessions):
        session_spike_times = spike_times[i]
        isi = session_spike_times.diff(axis=0)
        isi = isi.add_prefix(f'{i}_')

        # add area & cell type level
        idx = pd.MultiIndex.from_arrays(
            [isi.columns, area[i], cell_type[i]],
            names=names,
        )
        isi = isi.set_axis(idx, axis=1)
        isi_dic[i] = isi

    names.insert(0, 'session')
    isi = pd.concat(
        isi_dic, 
        axis=1,
        names=names,
    )
    # group by area & cell type
    grouped = isi.groupby(['area', 'cell_type'], axis=1)
    # get keys
    groups = list(grouped.groups.keys())
    # check each group member
    #grouped.get_group(groups[1])

    # get mean of each group
    group_mean = grouped.mean()
    regions = group_mean.columns.get_level_values('area').unique()
    fig, axes = plt.subplots(
        ncols=len(regions),
        nrows=1,
        sharey=True,
        sharex=True,
    )

    for r, region in enumerate(regions):
        data = group_mean.loc[:, region]
        sns.histplot(
            data=data,
            color=colours,
            hue_order=colours.keys(),
            ax=axes[r],
        )
        axes[r].set_title(f'{region} isi')

    plt.xlim((0, 7500))
    plt.ylim((0, 4500))
    plt.xlabel('inter-spike interval (ms)')

    return fig


def plot_fr_distribution(self, fr):

    return fig
