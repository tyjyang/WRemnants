from utilities import boostHistHelpers as hh, common, output_tools
import hist
import math

parser,initargs = common.common_parser()

import narf
import wremnants
from wremnants import theory_tools,syst_tools

parser.add_argument("--skipAngularCoeffs", action='store_true', help="Skip the conversion of helicity moments to angular coeff fractions")
parser.add_argument("--singleLeptonHists", action='store_true', help="Also store single lepton kinematics")
args = parser.parse_args()

filt = lambda x,filts=args.filterProcs: any([f in x.name for f in filts])
datasets = wremnants.datasets2016.getDatasets(maxFiles=args.maxFiles, filt=filt if args.filterProcs else None, 
    nanoVersion="v8" if args.v8 else "v9")

axis_massWgen = hist.axis.Variable([0., math.inf], name="massVgen")

axis_massZgen = hist.axis.Regular(12, 60., 120., name="massVgen")

axis_absYVgen = hist.axis.Variable(
    [0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3, 3.25, 3.5, 3.75, 4, 5, 10], name = "absYVgen"
)
axis_ptVgen = hist.axis.Variable(
#    [0, 2, 3, 4, 4.75, 5.5, 6.5, 8, 9, 10, 12, 14, 16, 18, 20, 23, 27, 32, 40, 55, 100], name = "ptVgen"
    range(0,121), name = "ptVgen"
)


axis_chargeWgen = hist.axis.Regular(
    2, -2, 2, name="chargeVgen", underflow=False, overflow=False
)

axis_chargeZgen = hist.axis.Integer(
    0, 1, name="chargeVgen", underflow=False, overflow=False
)

axis_l_eta_gen = hist.axis.Regular(48, -2.4, 2.4, name = "prefsr_lepton_eta_gen")
axis_l_pt_gen = hist.axis.Regular(29, 26., 55., name = "prefsr_lepton_pt_gen")
axes_l_gen = [axis_l_eta_gen, axis_l_pt_gen]

wprocs = [
    "WplusmunuPostVFP", 
    "WminusmunuPostVFP",
    "WminustaunuPostVFP",
    "WplustaunuPostVFP"
]
zprocs = ["ZmumuPostVFP", "ZtautauPostVFP"]

def build_graph(df, dataset):
    print("build graph")
    print(dataset.name)
    results = []
    
    if dataset.is_data:
        raise RuntimeError("Running GEN analysis over data is not supported")

    df = df.Define("nominal_pdf_cen", theory_tools.pdf_central_weight(dataset.name, args.pdfs[0]))
    weight_expr = "std::copysign(1.0, genWeight)"
    weight_expr = f"{weight_expr}*nominal_pdf_cen"

    if "reweight_h2" in dataset.name:
        weight_expr = f"{weight_expr}*H2BugFixWeight[0]"

    df = df.Define("nominal_weight", weight_expr)
    weightsum = df.SumAndCount("nominal_weight")

    df = theory_tools.define_scale_tensor(df)

    if dataset.name in zprocs:
        nominal_axes = [axis_massZgen, axis_absYVgen, axis_ptVgen, axis_chargeZgen]
    else:
        nominal_axes = [axis_massWgen, axis_absYVgen, axis_ptVgen, axis_chargeWgen]

    nominal_cols = ["massVgen", "absYVgen", "ptVgen", "chargeVgen"]
    df = wremnants.define_prefsr_vars(df)

    if args.singleLeptonHists:
        if dataset.name == 'WplusmunuPostVFP':
            df = df.Define('ptPrefsrMuon', 'genlanti.pt()')
            df = df.Define('etaPrefsrMuon', 'genlanti.eta()')
        elif dataset.name == 'WminusmunuPostVFP' or 'ZmumuPostVFP' in dataset.name:
            df = df.Define('ptPrefsrMuon', 'genl.pt()')
            df = df.Define('etaPrefsrMuon', 'genl.eta()')
        if dataset.name in ['WplusmunuPostVFP', 'WminusmunuPostVFP'] or 'ZmumuPostVFP' in dataset.name:
            nominal_cols = [*nominal_cols, 'etaPrefsrMuon', 'ptPrefsrMuon']
            nominal_axes = [*nominal_axes, axis_l_eta_gen, axis_l_pt_gen]

    nominal_gen = df.HistoBoost("nominal_gen", nominal_axes, [*nominal_cols, "nominal_weight"])
    results.append(nominal_gen)

    results.append(theory_tools.make_scale_hist(df, nominal_axes, nominal_cols))

    for pdf in args.pdfs:
        results.extend(theory_tools.define_and_make_pdf_hists(df, nominal_axes, nominal_cols, dataset.name, pdfset=pdf))

    if dataset.name != "Zmumu_bugfix":
        isW = dataset.name in wprocs

        df, masswhist = syst_tools.define_mass_weights(df, isW, nominal_axes, nominal_cols)
        if masswhist:
            results.append(masswhist)

    df = df.Define("helicity_moments_scale_tensor", "wrem::makeHelicityMomentScaleTensor(csSineCosThetaPhi, scaleWeights_tensor, nominal_weight)")
    helicity_moments_scale = df.HistoBoost("helicity_moments_scale", nominal_axes, [*nominal_cols, "helicity_moments_scale_tensor"], tensor_axes = [wremnants.axis_helicity, *wremnants.scale_tensor_axes])
    results.append(helicity_moments_scale)

    return results, weightsum

resultdict = narf.build_and_run(datasets, build_graph)

output_tools.write_analysis_output(resultdict, "w_z_gen_dists.pkl.lz4", args.postfix)

print("computing angular coefficients")

z_moments = None
w_moments = None

for dataset in datasets:
    # For now the tau samples have a different pt spectrum
    name = dataset.name
    if "tau" in name:
        continue
    moments = resultdict[name]["output"]["helicity_moments_scale"]
    if name in zprocs:
        if z_moments is None:
            z_moments = moments
        else:
            z_moments += moments
    elif name in wprocs:
        if w_moments is None:
            w_moments = moments
        else:
            w_moments += moments

z_coeffs = {"Z" : wremnants.moments_to_angular_coeffs(hh.rebinHist(z_moments, axis_ptVgen.name, common.ptV_binning))}
w_coeffs = {"W" : wremnants.moments_to_angular_coeffs(hh.rebinHist(w_moments, axis_ptVgen.name, common.ptV_binning))}

output_tools.write_analysis_output(z_coeffs, "z_coeffs.pkl.lz4", args.postfix)
output_tools.write_analysis_output(w_coeffs, "w_coeffs.pkl.lz4", args.postfix)
