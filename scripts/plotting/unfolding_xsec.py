import mplhep as hep
import matplotlib as mpl
import matplotlib.pyplot as plt

import itertools
import argparse
import os
import numpy as np
import pandas as pd
import hist
import json

from utilities import boostHistHelpers as hh, logging
from utilities.styles.styles import nuisance_groupings as groupings
from wremnants import plot_tools, histselections as sel
from wremnants.datasets.datagroups import Datagroups
from utilities.io_tools import input_tools, output_tools
from utilities.io_tools.combinetf_input import get_fitresult, read_impacts_pois, select_pois

import pdb

hep.style.use(hep.style.ROOT)

parser = argparse.ArgumentParser()
parser.add_argument("infile", type=str, help="Output file of the analysis stage, containing ND boost histogrdams")
parser.add_argument("--fitresult",  type=str, help="Combine fitresult root file")
parser.add_argument("--asimov",  type=str, default=None, help="Optional combine fitresult root file from an asimov fit for comparison")
parser.add_argument("-o", "--outpath", type=str, default=os.path.expanduser("~/www/WMassAnalysis"), help="Base path for output")
parser.add_argument("-f", "--outfolder", type=str, default="./test", help="Subfolder for output")
parser.add_argument("-p", "--postfix", type=str, help="Postfix for output file name")
parser.add_argument("--cmsDecor", default="Preliminary", type=str, choices=[None,"Preliminary", "Work in progress", "Internal"], help="CMS label")
parser.add_argument("--lumi", type=float, default=16.8, help="Luminosity used in the fit, needed to get the absolute cross section")
parser.add_argument("-r", "--rrange", type=float, nargs=2, default=[0.9,1.1], help="y range for ratio plot")
parser.add_argument("--ylim", type=float, nargs=2, help="Min and max values for y axis (if not specified, range set automatically)")
parser.add_argument("--logy", action='store_true', help="Make the yscale logarithmic")
parser.add_argument("--yscale", type=float, help="Scale the upper y axis by this factor (useful when auto scaling cuts off legend)")
parser.add_argument("--debug", action='store_true', help="Print debug output")
parser.add_argument("--noData", action='store_true', help="Don't plot data")
parser.add_argument("-n", "--baseName", type=str, help="Histogram name in the file (e.g., 'nominal')", default="nominal")
parser.add_argument("--scaleleg", type=float, default=1.0, help="Scale legend text")
parser.add_argument("--plots", type=str, nargs="+", default=["xsec", "uncertainties"], choices=["xsec", "uncertainties"], help="Define which plots to make")
parser.add_argument("--genFlow", action='store_true', help="Show overflow/underflow pois")
parser.add_argument("--poi", action='store_true', help="Plot signal strength parameters (mu)")
parser.add_argument("--normalize", action='store_true', help="Plot normalized distributions")
parser.add_argument("--absolute", action='store_true', help="Plot absolute uncertainties, otherwise relative")
parser.add_argument("--plotSumPOIs", action='store_true', help="Plot xsecs from sum POI groups")
parser.add_argument("--scaleXsec", type=float, default=1.0, help="Scale xsec predictions with this number")
parser.add_argument("--grouping", type=str, default=None, help="Select nuisances by a predefined grouping", choices=groupings.keys())
parser.add_argument("-t","--translate", type=str, default=None, help="Specify .json file to translate labels")
parser.add_argument("--eoscp", action='store_true', help="Override use of xrdcp and use the mount instead")

args = parser.parse_args()

logger = logging.setup_logger("unfolding_xsec", 4 if args.debug else 3)

grouping = groupings[args.grouping] if args.grouping else None

translate_label = {}
if args.translate:
    with open(args.translate) as f:
        translate_label = json.load(f)    

outdir = output_tools.make_plot_dir(args.outpath, args.outfolder)

meta_info = input_tools.get_metadata(args.fitresult)

groups = Datagroups(args.infile)

if groups.mode in ["wmass", "lowpu_w"]:
    process = "Wenu" if groups.flavor == "e" else "Wmunu"
else:
    process = "Zee" if groups.flavor == "ee" else "Zmumu"

base_process = process[0]

gen_axes = groups.gen_axes

groups.setNominalName(args.baseName)
groups.loadHistsForDatagroups(args.baseName, syst="", procsToRead=[process])

input_subdir = args.fitresult.split("/")[-2]

xlabels = {
    "ptGen" : r"$p_{T}^{\ell}$ [GeV]",
    "absEtaGen" : r"$|\eta^{\ell}|$",
    "ptVGen" : r"$p_{T}^{PROCESS}$ [GeV]",
    "absYVGen" : r"$|Y^{PROCESS}|$",
    "qGen" : r"$q^{PROCESS}$",
}

def get_xlabel(axis, process_label):
    p = process_label.replace("$","")
    l = xlabels.get(axis, axis)
    return l.replace("PROCESS", p)

def make_yields_df(hists, procs, signal=None, per_bin=False, yield_only=False, percentage=True):
    logger.debug(f"Make yield df for {procs}")

    if per_bin:
        def sum_and_unc(h,scale=100 if percentage else 1):
            return (h.values()*scale, np.sqrt(h.variances())*scale)   
    else:
        def sum_and_unc(h,scale=100 if percentage else 1):
            return (sum(h.values())*scale, np.sqrt(sum(h.variances())*scale))

    if per_bin:
        entries = [(i, v[0], v[1]) for i,v in enumerate(zip(*sum_and_unc(hists[0])))]
        index = "Bin"
    else:
        index = "Process"
        if signal is not None:
            entries = [(signal, sum([ sum(v.values()) for k,v in zip(procs, hists) if signal in k]), np.sqrt(sum([ sum(v.variances()) for k,v in zip(procs, hists) if signal in k])))]
        else:
            entries = [(k, *sum_and_unc(v)) for k,v in zip(procs, hists)]

    if yield_only:
        entries = [(e[0], e[1]) for e in entries]
        columns = [index, *procs]
    else:
        columns = [index, "Yield", "Uncertainty"]


    return pd.DataFrame(entries, columns=columns)

def plot_xsec_unfolded(df, edges, df_asimov=None, bin_widths=None, channel=None, scale=1., normalize=False, process_label="V", axes=None,
    hist_others=[], label_others=[], color_others=[]
):
    logger.info(f"Make "+("normalized " if normalize else "")+"unfoled xsec plot"+(f" in channel {channel}" if channel else ""))

    if normalize:
        yLabel="1/$\sigma$ d$\sigma("+process_label+")$"
    else:
        yLabel="d$\sigma ("+process_label+")$ [pb]"

    hist_xsec = hist.Hist(
        hist.axis.Variable(edges, underflow=False, overflow=False), storage=hist.storage.Weight())

    if bin_widths is None:
        bin_widths = edges[1:] - edges[:-1]

    hist_xsec.view(flow=False)[...] = np.stack([df["value"].values/bin_widths, (df["err_total"].values/bin_widths)**2], axis=-1)

    unc_ratio = np.sqrt(hist_xsec.variances()) /hist_xsec.values() 

    if "err_stat" in df.keys():
        hist_xsec_stat = hist.Hist(
            hist.axis.Variable(edges, underflow=False, overflow=False), storage=hist.storage.Weight())
        hist_xsec_stat.view(flow=False)[...] = np.stack([df["value"].values/bin_widths, (df["err_stat"].values/bin_widths)**2], axis=-1)
        unc_ratio_stat = np.sqrt(hist_xsec_stat.variances()) /hist_xsec.values() 

    if data_asimov is not None:
        ha_xsec = hist.Hist(hist.axis.Variable(edges, underflow=False, overflow=False))
        ha_xsec.view(flow=False)[...] = df_asimov["value"].values/bin_widths

    # make plots
    if args.ylim is None:
        ylim = (0, 1.1 * max(hist_xsec.values() + np.sqrt(hist_xsec.variances())))
    else:
        ylim = args.ylim

    rrange = args.rrange

    xlabel = "-".join([get_xlabel(a, process_label) for a in axes])
    if len(axes) >= 2:
        xlabel = xlabel.replace("[GeV]","")
        xlabel += " Bin"

    fig, ax1, ax2 = plot_tools.figureWithRatio(hist_xsec, xlabel, yLabel, ylim, "Pred./Data", rrange, width_scale=2)

    hep.histplot(
        hist_xsec,
        yerr=np.sqrt(hist_xsec.variances()),
        histtype="errorbar",
        color="black",
        label="Unfolded data",
        ax=ax1,
        alpha=1.,
        zorder=2,
    )    

    if args.genFlow:
        ax1.fill([0,18.5, 18.5, 0,0], [ylim[0],*ylim,ylim[1],ylim[0]], color="grey", alpha=0.3)
        ax1.fill([len(edges)-17.5, len(edges)+0.5, len(edges)+0.5, len(edges)-17.5, len(edges)-17.5], [ylim[0],*ylim,ylim[1],ylim[0]], color="grey", alpha=0.3)

        ax2.fill([0,18.5, 18.5, 0,0], [rrange[0],*rrange,rrange[1],rrange[0]], color="grey", alpha=0.3)
        ax2.fill([len(edges)-17.5, len(edges)+0.5, len(edges)+0.5, len(edges)-17.5, len(edges)-17.5], [rrange[0],*rrange,rrange[1],rrange[0]], color="grey", alpha=0.3)

    centers = hist_xsec.axes.centers[0]

    ax2.bar(centers, height=2*unc_ratio, bottom=1-unc_ratio, width=edges[1:] - edges[:-1], color="silver", label="Total")
    if "err_stat" in df.keys():
        ax2.bar(centers, height=2*unc_ratio_stat, bottom=1-unc_ratio_stat, width=edges[1:] - edges[:-1], color="gold", label="Stat")

    ax2.plot([min(edges), max(edges)], [1,1], color="black", linestyle="-")

    for h, l, c in zip(hist_others, label_others, color_others):
        h_flat = hist.Hist(
            hist.axis.Variable(edges, underflow=False, overflow=False), storage=hist.storage.Weight())
        h_flat.view(flow=False)[...] = np.stack([h.values(flow=args.genFlow).flatten()/bin_widths, h.variances(flow=args.genFlow).flatten()/bin_widths**2], axis=-1)

        hep.histplot(
            h_flat,
            yerr=False,
            histtype="step",
            color=c,
            label=l,
            ax=ax1,
            alpha=1.,
            zorder=2,
        )            

        hep.histplot(
            hh.divideHists(h_flat, hist_xsec, cutoff=0, rel_unc=True),
            yerr=False,
            histtype="step",
            color=c,
            ax=ax2,
            zorder=2,
        )            

    if data_asimov is not None:
        hep.histplot(
            ha_xsec,
            yerr=False,
            histtype="step",
            color="blue",
            label="Prefit model",
            ax=ax1,
            alpha=1.,
            zorder=2,
        ) 

        hep.histplot(
            hh.divideHists(ha_xsec, hist_xsec, cutoff=0, rel_unc=True),
            yerr=False,
            histtype="step",
            color="blue",
            # label="Model",
            ax=ax2
        )

    plot_tools.addLegend(ax1, ncols=2, text_size=15*args.scaleleg)
    plot_tools.addLegend(ax2, ncols=2, text_size=15*args.scaleleg)
    plot_tools.fix_axes(ax1, ax2, yscale=args.yscale)

    scale = max(1, np.divide(*ax1.get_figure().get_size_inches())*0.3)
    hep.cms.label(ax=ax1, lumi=float(f"{args.lumi:.3g}"), fontsize=20*args.scaleleg*scale, 
        label=args.cmsDecor, data=not args.noData)

    outfile = f"{input_subdir}_unfolded_xsec"

    if normalize:
        outfile += "_normalized"

    outfile += f"_{base_process}"
    outfile += "_"+"_".join(axes)
    outfile += (f"_{channel}" if channel else "")

    outfile += (f"_{args.postfix}" if args.postfix else "")
    plot_tools.save_pdf_and_png(outdir, outfile)

    if data_asimov is not None:
        asimov_yields = make_yields_df([ha_xsec], ["Model"], per_bin=True)
        asimov_yields["Uncertainty"] *= 0 # artificially set uncertainty on model hard coded to 0
    data_yields = make_yields_df([hist_xsec], ["Data"], per_bin=True)
    plot_tools.write_index_and_log(outdir, outfile, nround=4 if normalize else 2,
        yield_tables={"Data" : data_yields, "Model": asimov_yields} if df_asimov else {"Data" : data_yields},
        analysis_meta_info={args.infile : groups.getMetaInfo(), args.fitresult: meta_info},
        args=args,
    )
    plt.close()

def plot_uncertainties_unfolded(df, poi_type, channel=None, edges=None, scale=1., normalize=False, 
    logy=False, process_label="", axes=None, relative_uncertainty=False, percentage=True,
    error_threshold=0.001,   # only uncertainties are shown with a max error larger than this threshold
):
    logger.info(f"Make "+("normalized " if normalize else "")+"unfoled xsec plot"+(f" in channel {channel}" if channel else ""))

    # read nominal values and uncertainties from fit result and fill histograms
    logger.debug(f"Produce histograms")

    if poi_type == "mu":
        yLabel="$\mu\ ("+process_label+")$"
    elif normalize:
        yLabel="1/$\sigma$ d$\sigma("+process_label+")$"
    else:
        yLabel="d$\sigma ("+process_label+")$ [pb]"

    if relative_uncertainty:
        yLabel = "$\delta$ "+ yLabel
        yLabel = yLabel.replace(" [pb]","")
        if percentage:
            yLabel += " [%]"
    else:
        yLabel = "$\Delta$ "+ yLabel
    
    if poi_type == "mu":
        bin_widths = 1.0
    else:
        # central values
        bin_widths = edges[1:] - edges[:-1]

    values = df["value"].values/bin_widths

    hist_xsec = hist.Hist(hist.axis.Variable(edges, underflow=False, overflow=False))

    errors = df["err_total"].values/bin_widths
    if relative_uncertainty:
        errors /= values
        if percentage:
            errors *= 100

    hist_xsec.view(flow=False)[...] = errors

    # make plots
    if args.ylim is None:
        if logy:
            ylim = (max(errors)/10000., 1000 * max(errors))
        else:
            ylim = (0, 2 * max(errors))
    else:
        ylim = args.ylim

    xlabel = "-".join([get_xlabel(a, process_label) for a in axes])
    if len(axes) >= 2:
        xlabel = xlabel.replace("[GeV]","")
        xlabel += "Bin"

    fig, ax1 = plot_tools.figure(hist_xsec, xlabel, yLabel, ylim, logy=logy, width_scale=2)

    hep.histplot(
        hist_xsec,
        yerr=False,
        histtype="step",
        color="black",
        label="Total",
        ax=ax1,
        alpha=1.,
        zorder=2,
    )
    uncertainties = make_yields_df([hist_xsec], ["Total"], per_bin=True, yield_only=True, percentage=percentage)
    
    if args.genFlow:
        ax1.fill([0,18.5, 18.5, 0,0], [ylim[0],*ylim,ylim[1],ylim[0]], color="grey", alpha=0.3)
        ax1.fill([len(values)-17.5, len(values)+0.5, len(values)+0.5, len(values)-17.5, len(values)-17.5], [ylim[0],*ylim,ylim[1],ylim[0]], color="grey", alpha=0.3)

    if grouping:
        sources = ["err_"+g for g in grouping]
    else:
        sources =["err_stat"]
        sources += list(sorted([s for s in filter(lambda x: x.startswith("err"), df.keys()) 
            if s.replace("err_","") not in ["stat", "total"] ])) # total and stat are added first

    NUM_COLORS = len(sources)-1
    cm = mpl.colormaps["gist_rainbow"]
    # add each source of uncertainty
    i=0
    for source in sources:

        name = source.replace("err_","")

        name = translate_label.get(name,name)

        if source =="err_stat":
            color = "grey"
        else:
            color = cm(1.*i/NUM_COLORS)
            i += 1

        if i%3 == 0:
            linestype = "-" 
        elif i%3 == 1:
            linestype = "--" 
        else:
            linestype = ":" 

        hist_unc = hist.Hist(hist.axis.Variable(edges, underflow=False, overflow=False))

        if source not in df:
            logger.warning(f"Source {source} not found in dataframe")
            continue

        errors = df[source].values/bin_widths
        if relative_uncertainty:
            errors /= values

        if max(errors) < error_threshold:
            logger.debug(f"Source {source} smaller than threshold of {error_threshold}")
            continue

        logger.debug(f"Plot source {source}")

        if percentage:
            errors *= 100
        
        hist_unc.view(flow=False)[...] = errors

        hep.histplot(
            hist_unc,
            yerr=False,
            histtype="step",
            color=color,
            linestyle=linestype,
            label=name,
            ax=ax1,
            alpha=1.,
            zorder=2,
        )

        unc_df = make_yields_df([hist_unc], [name], per_bin=True, yield_only=True, percentage=True)
        uncertainties[name] = unc_df[name]

    scale = max(1, np.divide(*ax1.get_figure().get_size_inches())*0.3)

    plot_tools.addLegend(ax1, ncols=4, text_size=15*args.scaleleg*scale)

    if args.yscale:
        ymin, ymax = ax1.get_ylim()
        ax1.set_ylim(ymin, ymax*args.yscale)

    if not logy:
        plot_tools.redo_axis_ticks(ax1, "y")
    plot_tools.redo_axis_ticks(ax1, "x", no_labels=len(axes) >= 2)

    hep.cms.label(ax=ax1, lumi=float(f"{args.lumi:.3g}"), fontsize=20*args.scaleleg*scale, 
        label=args.cmsDecor, data=not args.noData)

    outfile = f"{input_subdir}_unfolded_uncertainties"

    if poi_type == "mu":
        outfile += "_mu"
    if relative_uncertainty:
        outfile += "_relative"   
    if normalize:
        outfile += "_normalized"
    if logy:
        outfile += "_log"
    outfile += f"_{base_process}"
    outfile += "_"+"_".join(axes)
    outfile += (f"_{channel}" if channel else "")

    outfile += (f"_{args.postfix}" if args.postfix else "")
    plot_tools.save_pdf_and_png(outdir, outfile)

    plot_tools.write_index_and_log(outdir, outfile, nround=4 if normalize else 2,
        yield_tables={"Unfolded data uncertainty [%]": uncertainties},
        analysis_meta_info={"AnalysisOutput" : groups.getMetaInfo(), 'CardmakerOutput': meta_info},
        args=args,
    )

    plt.close()

if args.poi:
    poi_types = ["mu",]
    scale = 1
else:
    poi_types = ["pmaskedexpnorm",] if args.normalize else ["pmaskedexp",]
    if args.plotSumPOIs:
        poi_types += ["sumpoisnorm",] if args.normalize else ["sumpois",]
    scale = 1 if args.normalize else args.lumi * 1000

fitresult = get_fitresult(args.fitresult)
if args.asimov:
    fitresult_asimov = get_fitresult(args.asimov)
    
for poi_type in poi_types:

    data = read_impacts_pois(fitresult, poi_type=poi_type, scale=scale)
    data_asimov = read_impacts_pois(fitresult_asimov, poi_type=poi_type, scale=scale) if args.asimov else None

    if poi_type.startswith("sum"):
        # make all possible lower dimensional gen axes combinations; wmass only combinations including qGen
        gen_axes_permutations = [list(k) for n in range(1, len(gen_axes)) for k in itertools.combinations(gen_axes, n)]
    else:
        # gen_axes = ["ptGen", ]
        gen_axes_permutations = [gen_axes[:],]

    for axes in gen_axes_permutations:
        logger.info(f"Make plots for process {base_process} and gen axes {axes}")

        if groups.mode in ["wmass", "lowpu_w"]:
            axes.append("qGen")

        channels = ["plus", "minus"] if "qGen" in axes else ["all"]

        for channel in channels:
            logger.info(f"Now at channel {channel}")
            if channel == "minus":
                channel_sel = {"qGen":0}
                channel_axes = [a for a in axes if a != "qGen"]
                process_label = r"\mathrm{W}^{-}" if base_process == "W" else r"\mathrm{Z}"
            elif channel == "plus":
                channel_sel = {"qGen":1}
                channel_axes = [a for a in axes if a != "qGen"]
                process_label = r"\mathrm{W}^{+}" if base_process == "W" else r"\mathrm{Z}"
            else:
                channel_sel = {}
                process_label = r"\mathrm{W}" if base_process == "W" else r"\mathrm{Z}"
                channel_axes = axes[:]

            data_c = select_pois(data, axes, selections=channel_sel, base_processes=base_process, flow=args.genFlow)
            data_c_asimov = select_pois(data_group_asimov, axes, selections=channel_sel, base_processes=base_process, flow=args.genFlow) if data_asimov else None

            if len(data_c) == 0:
                logger.info(f"No entries found for axes {axes} in channel {channel}, skip!")
                continue
            
            # find bin widths
            def get_histo(name):
                h = sum([groups.results[m.name]["output"][name].get() for m in groups.groups[process].members 
                    if not m.name.startswith("Bkg") and (base_process=="Z" or channel=="all" or channel in m.name)])
                h = h.project(*channel_axes)
                # for wlike the sample is randomly split in two based on reco charge
                this_scale = 2*scale if groups.mode == "wlike" else scale
                if "xnorm" in name:
                    this_scale /= args.scaleXsec
                h = hh.scaleHist(h, 1./this_scale)
                return h

            histo = get_histo(args.baseName)
            hxnorm = get_histo("xnorm")
            hMiNNLO = get_histo("xnorm_uncorr")

            if args.genFlow:
                edges = plot_tools.extendEdgesByFlow(histo)
            else:
                edges = histo.axes.edges

            binwidths = None
            if len(channel_axes) == 1:
                if channel_axes[0] == "qGen":
                    edges = np.array([-2,0,2])
                else:
                    edges = np.array(edges[0])
            elif len(channel_axes) == 2:
                xbins, ybins = edges
                xbinwidths = np.diff(xbins.flatten())
                ybinwidths = np.diff(ybins.flatten())
                binwidths = np.outer(xbinwidths, ybinwidths).flatten()
                edges = np.arange(0.5, len(binwidths)+1.5, 1.0)
            else:
                bins = np.product([len(e) for e in edges])
                edges = np.arange(0.5, bins+1.5, 1.0)

            if poi_type == "poi":
                binwidths = None

            if "xsec" in args.plots:
                plot_xsec_unfolded(data_c, edges, data_c_asimov, bin_widths=binwidths, channel=channel, scale=scale, normalize=args.normalize, axes=channel_axes, 
                    process_label=process_label, hist_others=[hxnorm, hMiNNLO], label_others=[r"MiNNLO $\times$ SCETlib+DYTurbo", "MiNNLO"], color_others=["blue", "red"]
                )

            if "uncertainties" in args.plots:
                plot_uncertainties_unfolded(data_c, poi_type, edges=edges, channel=channel, scale=scale, 
                    normalize=args.normalize, relative_uncertainty=not args.absolute, logy=args.logy, process_label = process_label, axes=channel_axes)

if output_tools.is_eosuser_path(args.outpath) and args.eoscp:
    output_tools.copy_to_eos(args.outpath, args.outfolder)
