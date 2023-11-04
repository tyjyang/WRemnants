import narf
from utilities import logging
import subprocess
import sys
import os
import glob
import random
import pathlib
import socket
#set the debug level for logging incase of full printout 
from wremnants.datasets.datasetDict_v9 import dataDictV9
from wremnants.datasets.datasetDict_gen import genDataDict
from wremnants.datasets.datasetDict_lowPU import dataDictLowPU
import ROOT
import XRootD.client

logger = logging.child_logger(__name__)

default_nfiles = {
    'WminusmunuPostVFP' : 1700,
    'WplusmunuPostVFP' : 2000,
    'WminustaunuPostVFP' : 400,
    'WplustaunuPostVFP' : 500,
    'ZmumuPostVFP' : 900,
    'ZtautauPostVFP' : 1200,
}

def buildFileListPosix(path):
    outfiles = []
    for root, dirs, fnames in os.walk(path):
        for fname in fnames:
            if fname.lower().endswith(".root"):
                outfiles.append(f"{root}/{fname}")

    return outfiles

def appendFilesXrd(filelist, xrdfs, path, suffixes = [".root"], recurse = False, num_clients = 16):
    status, dirlist = xrdfs.dirlist(path, flags = XRootD.client.flags.DirListFlags.STAT)

    if not status.ok:
        if status.code == 400 and status.errno == 3011:
            logger.warning(f"XRootD directory not found: {path}")
        else:
            raise RuntimeError(f"Error in XRootD.client.FileSystem.dirlist: {status.message}, {status.code}, {status.errno}")

        return

    for diritem in dirlist:
        is_dir = diritem.statinfo.flags & XRootD.client.flags.StatInfoFlags.IS_DIR
        is_other = diritem.statinfo.flags & XRootD.client.flags.StatInfoFlags.OTHER
        is_file = not (is_dir or is_other)

        if is_dir and recurse:
            childpath = f"{path}/{diritem.name}"
            appendFilesXrd(filelist, xrdfs, childpath, suffixes=suffixes, recurse=recurse, num_clients=num_clients)
        elif is_file:
            lowername = diritem.name.lower()
            matchsuffix = False
            for suffix in suffixes:
                if lowername.endswith(suffix):
                    matchsuffix = True
                    break

            if matchsuffix:
                if num_clients > 0:
                    # construct client string if necessary to force multiple xrootd connections
                    # (needed for good performance when a single or small number of xrootd servers is used)
                    client = f"user_{random.randrange(num_clients)}"
                    outname = f"{xrdfs.url.protocol}://{client}@{xrdfs.url.hostname}:{xrdfs.url.port}/{path}/{diritem.name}"
                else:
                    outname = f"{xrdfs.url.protocol}://{xrdfs.url.hostid}/{path}/{diritem.name}"

                filelist.append(outname)

def buildFileListXrd(path, num_clients = 16):
    xrdurl =  XRootD.client.URL(path)

    if not xrdurl.is_valid():
        raise ValueError(f"Invalid xrootd path {path}")

    xrdfs = XRootD.client.FileSystem(xrdurl.hostid)
    xrdpath = xrdurl.path

    outfiles = []
    appendFilesXrd(outfiles, xrdfs, xrdpath, recurse=True, num_clients=num_clients)

    return outfiles

def buildFileList(path):
    xrdprefix = "root://"
    return buildFileListXrd(path) if path.startswith(xrdprefix) else buildFileListPosix(path)

#TODO add the rest of the samples!
def makeFilelist(paths, maxFiles=-1, base_path=None, nano_prod_tags=None, is_data=False, oneMCfileEveryN=None):
    filelist = []
    for orig_path in paths:
        if maxFiles > 0 and len(filelist) >= maxFiles:
            break
        # try each tag in order until files are found
        fallback = False
        for prod_tag in nano_prod_tags:
            format_args=dict(BASE_PATH=base_path, NANO_PROD_TAG=prod_tag)

            path = orig_path.format(**format_args)
            logger.debug(f"Reading files from path {path}")

            files = buildFileList(path)

            if len(files) == 0:
                fallback = True
                logger.warning(f"Did not find any files for tag {prod_tag} matching path {path}!")
            else:
                if fallback:
                    logger.warning(f"Falling back to tag {prod_tag} with path {path}")
                break

        filelist.extend(files)

    if oneMCfileEveryN != None and not is_data:
        tmplist = []
        for i,f in enumerate(filelist):
            if i % oneMCfileEveryN == 0:
                tmplist.append(f)
        logger.warning(f"Using {len(tmplist)} files instead of {len(filelist)}")
        filelist = tmplist

    return filelist if maxFiles < 0 or len(filelist) < maxFiles else random.Random(1).sample(filelist, maxFiles)

def selectProc(selection, datasets):
    if any(selection == x.group for x in datasets):
        # if the selection matches any of the group names in the given dataset, the selection is applied to groups
        return list(filter(lambda x, s=selection: x.group is not None and x.group == s, datasets))
    else:
        # otherwise, the selection is applied to sample names
        return list(filter(lambda x, s=selection: s in x.name, datasets))

def selectProcs(selections, datasets):
    new_datasets = []
    for selection in selections:
        new_datasets += selectProc(selection, datasets)

    # remove duplicates selected by multiple filters
    new_datasets = list(set(new_datasets))
    return new_datasets

def filterProcs(filters, datasets):
    if filters:
        if isinstance(filters, list):
            new_datasets = selectProcs(filters, datasets)
        elif isinstance(filters, str):
            new_datasets = selectProc(filters, datasets)
        else:
            new_datasets = list(filter(filters, datasets))
    else:
        return datasets

    if len(new_datasets) == 0:
        logger.warning("Try to filter processes/groups but didn't find any match. Continue without filtering.")
        return datasets

    return new_datasets

def excludeProcs(excludes, datasets):
    if excludes:
        if isinstance(excludes, list):
            # remove selected datasets
            return list(filter(lambda x: x not in selectProcs(excludes, datasets), datasets))
        elif isinstance(excludes, str):
            # remove selected datasets
            return list(filter(lambda x: x not in selectProc(excludes, datasets), datasets))
        else:
            return list(filter(excludes, datasets))
    else:
        return datasets

def getDataPath(mode=None):
    import socket
    hostname = socket.gethostname()

    if hostname.endswith(".cern.ch"):
        if mode and "lowpu" in mode:
            base_path = "root://eoscms.cern.ch//store/cmst3/group/wmass/LowPU"
        else:
            base_path = "root://eoscms.cern.ch//store/cmst3/group/wmass/w-mass-13TeV/NanoAOD"
    elif hostname.endswith(".mit.edu"):
        if mode and "lowpu" in mode:
            base_path = "/scratch/submit/cms/wmass/NanoAOD/LowPU"
        else:
            base_path = "/scratch/submit/cms/wmass/NanoAOD"
    elif hostname == "cmsanalysis.pi.infn.it":
        # NOTE: If anyone wants to run lowpu analysis at Pisa they'd probably want a different path
        base_path = "/scratchnvme/wmass/NANOV9/postVFP"

    return base_path

def is_zombie(file_path):
    # Try opening the ROOT file and check if it's a zombie file
    file = ROOT.TFile.Open(file_path)
    if not file or file.IsZombie():
        logger.warning(f"Found zombie file: {file_path}")
        return True
    file.Close()
    return False

def getDatasets(maxFiles=default_nfiles, filt=None, excl=None, mode=None, base_path=None, nanoVersion="v9",
                data_tags=["TrackFitV722_NanoProdv3", "TrackFitV722_NanoProdv2"],
                mc_tags=["TrackFitV722_NanoProdv3", "TrackFitV718_NanoProdv1"], oneMCfileEveryN=None, checkFileForZombie=False):
    if maxFiles is None:
        maxFiles=default_nfiles

    if not base_path:
        base_path = getDataPath(mode)
    logger.info(f"Loading samples from {base_path}.")

    if nanoVersion == "v9":
        dataDict = dataDictV9
    else:
        raise ValueError("Only NanoAODv9 is supported")

    if mode == "gen":
        dataDict.update(genDataDict)     
    elif mode and "lowpu" in mode:
        dataDict = dataDictLowPU

    narf_datasets = []
    for sample,info in dataDict.items():
        if sample in genDataDict:
            base_path = base_path.replace("NanoAOD", "NanoGen")

        is_data = info.get("group","") == "Data"

        prod_tags = data_tags if is_data else mc_tags
        nfiles = maxFiles
        if type(maxFiles) == dict:
            nfiles = maxFiles[sample] if sample in maxFiles else -1
        paths = makeFilelist(info["filepaths"], nfiles, base_path=base_path, nano_prod_tags=prod_tags, is_data=is_data, oneMCfileEveryN=oneMCfileEveryN)
            
        if checkFileForZombie:
            paths = [p for p in paths if not is_zombie(p)]

        if not paths:
            logger.warning(f"Failed to find any files for dataset {sample}. Looking at {info['filepaths']}. Skipping!")
            continue

        narf_info = dict(
            name=sample,
            filepaths=paths,
        )

        if is_data:
            if mode == "gen":
                continue
            narf_info.update(dict(
                is_data=True,
                lumi_csv=info["lumicsv"],
                lumi_json=info["lumijson"],
                group=info["group"] if "group" in info else None,
            ))
        else:
            narf_info.update(dict(
                xsec=info["xsec"],
                group=info["group"] if "group" in info else None,
                )
            )
        narf_datasets.append(narf.Dataset(**narf_info))
    
    narf_datasets = filterProcs(filt, narf_datasets)
    narf_datasets = excludeProcs(excl, narf_datasets)

    for sample in narf_datasets:
        if not sample.filepaths:
            logger.warning(f"Failed to find any files for sample {sample.name}!")

    return narf_datasets
