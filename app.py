import os
import streamlit as st
import pandas as pd
import joblib
import optuna
import numpy as np
import openai
from catboost import CatBoostClassifier


from google import genai
from google.genai import types



# keep a reference to the real number_input
_original_number_input = st.number_input

def safe_number_input(label, **kwargs):
    """
    Clamp `value` into [min_value, max_value] and warn if we had to adjust.
    Then call the real st.number_input with the clamped default.
    """
    min_value = kwargs.get("min_value", float("-inf"))
    max_value = kwargs.get("max_value", float("inf"))
    value     = kwargs.get("value", min_value)
    # clamp
    clamped = min(max(value, min_value), max_value)
    if clamped != value:
        st.warning(
            f"⚠️ Default for “{label}” ({value}) was outside "
            f"[{min_value}, {max_value}]; using {clamped} instead."
        )
    kwargs["value"] = clamped
    return _original_number_input(label, **kwargs)

# override streamlit's number_input globally
st.number_input = safe_number_input



# directly use your Gemini key
gemini_key = os.getenv("GEMINI_API_KEY")
if not gemini_key:
    st.error("🚨 Gemini API key not found. Please set GEMINI_API_KEY in your Space settings.")
    st.stop()

client = genai.Client(api_key=gemini_key)


# ─── Scaffold Quality Function ───────────────────────────────────────────────────
def scaffold_quality_combined(printability, cell_response,
                              weight_printability=0.3, weight_cell_response=0.7):
    if printability == 0:
        return 0.0
    if cell_response == 1:
        return 100 * (printability / 3.0)
    norm_p = printability / 3.0
    norm_c = (cell_response - 1) / 4.0
    hm = (weight_printability + weight_cell_response) / (
        (weight_printability / norm_p) +
        (weight_cell_response / norm_c)
    )
    mc = (norm_p**weight_printability) * (norm_c**weight_cell_response)
    return 100 * ((hm + mc) / 2.0)

# ─── Constants ────────────────────────────────────────────────────────────────────
BIOMATERIAL_OPTIONS = [
    "Alginate (%w/v)",
    "PVA-HA (%w/v)",
    "CaSO4 (%w/v)",
    "Na2HPO4 (%w/v)",
    "Gelatin (%w/v)",
    "GelMA (%w/v)",
    "laponite (%w/v)",
    "graphene oxide (%w/v)",
    "hydroxyapatite (%w/v)",
    "Hyaluronic_Acid (%w/v)",
    "hyaluronan metacrylate (%w/v)",
    "NorHA (%w/v)",
    "Fibroin/Fibrinogen (%w/v)",
    "Pluronic P-123 (%w/v)",
    "Collagen (%w/v)",
    "Chitosan (%w/v)",
    "CS-AEMA (%w/v)",
    "RGD (mM)",
    "TCP (%w/v)",
    "Gellan (%w/v)",
    "bioactive glass (%w/v)",
    "Nano/Methycellulose (%w/v)",
    "PEGTA (%w/v)",
    "PEGMA (%w/v)",
    "PEGDA (%w/v)",
    "Agarose (%w/v)",
    " hyaluronic acid+ Ph moieties (%w/v)",
    "matrigel (%w/v)",
    "CaCl2(mM)",
    "NaCl(mM)",
    "BaCl2(mM)",
    "SrCl2(mM)",
    "CaCO3 (mM)",
    "Genipin (%w/v)",
    "PVA (%wt)",
    "trans-glutaminase (%w/v)",
    "alginate lyase (U/ml)",
    "D-glucose (%w/v)",
    "PLGA (%w/v)",
    "vascular tissued-derived dECM (%w/v)",
    "PEG-8-SH (mM)",
    "Alginate dialdehyde (%w/v)",
    "Alginate sulfate (%w/v)",
    "RGD-modified alginate (%w/v)",
    "poly(N-isopropylacrylamide) grafted hyaluronan (%w/v)",
    "chondroitin sulfate methacrylate (%w/v)",
    "PCL (%w/v)",
    "alginate methacrylate (%w/v)",
    "HRP (U/ml)",
    "Pluronic F127 (%w/v)/Lutrol F127 (%w/v)",
    "Irgacure 2959 (%w/v)",
    "Eosin Y (%w/v)",
    "Ruthenium (mM)",
    "sodium persulfate (SPS) (mM)",
    "HEPES (mM)",
    "LAP (%w/v)",
    "glutaraldehyde (%w/v)",
    "PBS  (M)",
    "glycerol (%w/v)",
    "cECM (%w/v)",
    "gel-fu(%w/v)",
    "Rose Bengal (%w/v)",
    "Vitamin B2(%w/v)",
    "VEGF(%w/v)",
    "Polypyrrole:PSS(%w/v)",
    "boratebioactiveglass(%w/v)",
    "astaxanthin(%w/v)",
    "PRP (%v/v)",
    "methacrylated collagen (%w/v)",
    "α-Toc (µM)",
    "ascorbic acid (mM)",
    "Liver dECM(%w/v)",
    "galactosylated alginate (%w/v)",
    "SC-PEG(%w/v)",
    "SFMA-L(%w/v)",
    "SFMA-M(%w/v)",
    "SFMA-H(%w/v)",
    "KdECMMA(%w/v)",
    "BA silk fibronin (%w/v)",
    "Carrageenan(%v)",
    "Carbopol ETD 2020 NF (%w/v)",
    "Carbopol  Ultrez 10 NF(%w/v)",
    "Carbopol  NF-980(%w/v)",
    "FBS (%v/v)",
    "MeTro (%w/v)",
    "Triethanolamine (%v/v)",
    "PEG-Fibrinogen (%w/v)",
    "polyethylene glycol dimethacrylate (%w/v)",
    "aprotinin (µg/ml)",
    "gold nanorod (mg/mL)",
    "egg white (w/v)",
    "1-Vinyl-2-Pyrrolidione (v/v)",
    "carboxyl functionalized carbon nanotubes  (%w/v)",
    "polyHIPE  (%w/v)",
    "β-D galactose (mM)",
    "hydrogen peroxide (H2O2) (%v/v)",
    "lactic acid v/v",
    "NorCol (%w/v)",
    "DDT (%w/v)",
    "ammonium persulfate (mM)",
    "diTyr-RGD (mM)",
    "PHEG-Tyr (%w/v)",
    "MMP2-degradable peptide (%w/v)",
    "KdECM (%w/v)",
    "EDC (mg)",
    "NHS (mg)",
    "VA086 (%w/v)",
    "PGS (%w/v)",
    "thiolated HA (%w/v)",
    "boron nitride nanotubes (%w/v)",
    "PEDOT:PSS (ul)",
    "KCl (mM)",
    "skeletal muscle ECM methacrylate (%w/v)",
    "PEO (%w/v)",
    "Carbon dots (mg/ml)",
    "Laminin (ug/ml)",
    "DF-PEG (%w/v)",
    "omenta ECM (%w/v)",
    "thrombin (unit/ml)",
    "Carbon nanotube (CNT) (w/v)",
    "Phytagel(%v)",
    "Laponite-XLG  (%w/w)"
]

CELL_LINE_OPTIONS = [
    "chondrocyteyte",
    "HepG2",
    "bMSCs",
    "HUVECs",
    "NIH3T3",
    "MESCs",
    "hiPSC-CMs /ATCCs",
    "CPCs",
    "L929",
    "Myoblast cells",
    "hiPSCs",
    "HepaRG",
    "hESCs",
    "10T1/2",
    "Cardiac progenitor cells",
    "NSCLC PDX",
    "RAMECs",
    "hASCs",
    "HAVIC",
    "Primary mouse hepatocyte",
    "PTECs",
    "human nasoseptal chondrocytes",
    "PDX",
    "HPFs",
    "U87-MG",
    "ESCs",
    "HASSMC",
    "dermal fibroblasts",
    "MC3T3-E1",
    "Schwann cells",
    "hiPSC-CMs and HS-27A",
    "Saos-2 ",
    "SU3",
    "hTMSCs",
    "HACs",
    "HADMSCs",
    "HeLa",
    "human primary kidney cells",
    "myoblasts",
    "MSCs",
    "human primary kidneycells",
    "293FT",
    "HEK 293FT",
    "Wnt3a-293FT",
    "RSC96/HUVECs",
    "human adipogenic mesenchymal stem cells",
    "HEPG2/ECs",
    "HUVECs/MSCs",
    "RHECs",
    "Human non-small cell lung cancer line Calu-3 (Calu-3)",
    "HL-1",
    "mouse cardiac cells",
    "IMR-90",
    "EPCs",
    "MRC5",
    "rMSC",
    "basil plant cell",
    "hNCs",
    "A549",
    "human induced pluripotent stem cell-derived cardiomyocytes",
    "bMSCs/hACs",
    "EA.hy 926 cells",
    "HepG2/C3A",
    "human epithelial lung carcinoma cells",
    "Human cardiac fibroblasts",
    "hTERT-MSC",
    "cardiomyocytes",
    "Huh7",
    "NRCMs",
    "HCF",
    "Wnt reporter-293FT",
    "neonatal rat ventricular CFs",
    "human coronary artery endothelial cells",
    "hiPSC-CM / fibroblasts",
    "primary mouse hepatocyte",
    "NIH3T3/ HUVECs",
    "murine macrophage-like cell line",
    "Endothelial cells",
    "Human aortic VIC",
    "sADSC",
    "HUVECs/H9C2",
    "neonatal rat ventricular cardiomyocytes",
    "MG-63",
    "Neonatal mouse cardiomyocytes (NMVCMs)",
    "human hepatic stellate cell line",
    "HEK-293",
    "aHSC",
    "MFCs",
    "fibroblasts",
    "HNDF",
    "cardiomyocyte/MSCs",
    "ADSCs",
    "HCASMCs",
    "cardiomyocyte",
    "hCPCs",
    "Human CM /adult human fibroblasts ",
    "primary rat hepatocyte",
    "human cardiac progenitor cells",
    "SMC",
    "Human MSCs",
    "ACPCs",
    "Huh7/HepaRG",
    "Human umbilical vein endothelial cells",
    "ATDC5",
    "hESC-derived HLCs",
    "NIH 3T3",
    "n neonatal mouse ventricular cardiomyocytes",
    "CFs/CMs/HUVECs",
    "MRC-5",
    "VIC",
    "eHep",
    "hUVECs/NIH3T3",
    "MC3T3",
    "HLC",
    "hepatoma",
    "FB",
    "A549 GFP+",
    "HPAAF",
    "PMHs",
    "HUVSMCs",
    "Human CPCs",
    "Fibroblasts/THP-1",
    "rat ventricular cardiomyocytes",
    "iCMs/iCFs/iECs",
    "HUVEC/HHSC",
    "Human CPCs / MSCs",
    "HepaRG/LX-2 ",
    "iCMs/iCFs/iECs/iCMFs",
    "LX-2 ",
    "SMMC-7721",
    "Hepatoblast- single cell/iESC/iMSC",
    "iCMFs",
    "Hepatoblast- spheroid/iESC/iMSC",
    "hBM-MSCs",
    "BMSCs",
    "HUVECs and HHSCs",
    "Intrahepatic cholangiocarcinoma (ICC)",
    "VICs",
    "CMs/CFs",
    "human neonatal dermal fi broblasts",
    "10T1/2 fibroblast-laden cells",
    "human cardiac fibroblasts",
    "neonatal rat ventricular CMs",
    "HUVECs and hiPSC-CS",
    "iPSC-derived CM",
    "Human Umbilical Vein Endothelial Cells + iPSC-derived CM",
    "rabbit bone marrow mesenchymal stem cells",
    "Neonatal rat cardiomyocytes",
    "NIH 3T3 mouse fibroblasts",
    "Human CPCs & MSCs",
    "NSCLC PDX/CAFs",
    "hiPSCs-derived HLCs",
    "U87",
    "75% hepatoblast cells, 20% iEC and 5% iMSC",
    "SMCs",
    "A549/95-D cells",
    "NCI-H441",
    "pancreatic cancer cell",
    "prostate cancer stem cell",
    "human primary parathyroid cells ",
    "primary human hepatocytes",
    "CPCs / MSCs",
    "CPCs ",
    "cardiac fibroblasts",
    "iPSCs-derived cardiomyocytes",
    "cardiomyocytes/ fibroblasts",
    "hiPSC-CM",
    "iPSCs/HUVECs",
    "iPSC-CMs",
    "iPSCs",
    "H1395",
    "PC9",
    "H1650",
    "HULEC-5a",
    "NCI-H1703"
]

PRINT_PARAM_NAMES = [
    "Physical Crosslinking Duration (s)",
    "Photo Crosslinking Duration (s)",
    "Extrusion Pressure (kPa)",
    "Nozzle Movement Speed (mm/s)",
    "Nozzle Diameter (µm)",
    "Syringe Temperature (°C)",
    "Substrate Temperature (°C)",
]

# ─── Load Encoder, Scalers, Models ────────────────────────────────────────────────
@st.cache_resource
def load_encoder():
    return joblib.load('cell_line_encoder.joblib')

@st.cache_resource
def load_scalers():
    return (
        joblib.load('scaler_printability.joblib'),
        joblib.load('scaler_cell_response.joblib')
    )

@st.cache_resource
def load_models():
    m_p = CatBoostClassifier(); m_p.load_model('catboost_printability.cbm')
    m_c = CatBoostClassifier(); m_c.load_model('catboost_cell_response.cbm')
    return m_p, m_c

encoder          = load_encoder()
scaler_print, scaler_cell = load_scalers()
model_print,   model_cell = load_models()

feature_order_print = list(scaler_print.feature_names_in_)
feature_order_cell  = list(scaler_cell.feature_names_in_)

# ─── Session State Initialization ────────────────────────────────────────────────
if 'bio_rows' not in st.session_state:
    st.session_state.bio_rows = [{
        'mat': BIOMATERIAL_OPTIONS[0],
        'min': 0.0, 'max': 10.0, 'step': 0.1
    }]

if 'density_range' not in st.session_state:
    st.session_state.density_range = {'min': 0.0, 'max': 10.0, 'step': 0.1}

if 'pp_ranges' not in st.session_state:
    st.session_state.pp_ranges = {
        name: {'min': 0.0, 'max': 10.0, 'step': 3.0}
        for name in PRINT_PARAM_NAMES
    }

# ─── App Layout ──────────────────────────────────────────────────────────────────
st.title("🧬 MLATE: Machine Learning Applications in Tissue Engieering")
st.markdown(
    "<p style='font-size:0.8em; color:grey;'>"
    "An integrated platform for predicting the quality of 3D (bio)printed scaffolds "
    "in tissue engineering. For more details, please refer to and cite our paper: "
    "<a href='https://doi.org/xxx' target='_blank'>https://doi.org/xxx</a>"
    "</p>",
    unsafe_allow_html=True
)

# ─── Biomaterials Section ────────────────────────────────────────────────────────
st.subheader("Biomaterials (enter range for each)")
if st.button("➕ Add Biomaterial"):
    used = {r['mat'] for r in st.session_state.bio_rows}
    available = [m for m in BIOMATERIAL_OPTIONS if m not in used]
    if available:
        st.session_state.bio_rows.append({
            'mat': available[0], 'min': 0.0, 'max': 10.0, 'step': 0.1
        })
    st.rerun()

for i, row in enumerate(st.session_state.bio_rows):
    used_except_current = {
        r['mat'] for idx, r in enumerate(st.session_state.bio_rows) if idx != i
    }
    options = [m for m in BIOMATERIAL_OPTIONS if m not in used_except_current]

    c1, c2, c3, c4, c5 = st.columns([2, 1, 1, 1, 0.3])
    mat = c1.selectbox(
        "", options,
        index=options.index(row['mat']) if row['mat'] in options else 0,
        key=f"bio_mat_{i}"
    )
    st.session_state.bio_rows[i]['mat'] = mat

    mn = c2.number_input(
        "Min", min_value=0.0, max_value=row['max'],
        value=row['min'], step=row['step'], key=f"bio_min_{i}"
    )
    # ← min_value for Max is now the step; default value at least step
    mx = c3.number_input(
        "Max", min_value=row['step'], max_value=100.0,
        value=max(row['max'], row['step']), step=row['step'],
        key=f"bio_max_{i}"
    )
    st.session_state.bio_rows[i].update(min=mn, max=mx)

    st.session_state.bio_rows[i]['step'] = c4.number_input(
        "Step", min_value=0.0,
        max_value=(mx - mn) if mx > mn else 0.1,
        value=row['step'], step=0.1, key=f"bio_step_{i}"
    )

    if c5.button("❌", key=f"rem_{i}"):
        st.session_state.bio_rows.pop(i)
        st.rerun()

st.markdown("---")

# ─── Cell Line & Density Section ────────────────────────────────────────────────
st.subheader("Cell Line & Density")
col1, col2, col3, col4 = st.columns([2,1,1,1])
cell_line = col1.selectbox("Cell Line", CELL_LINE_OPTIONS)

dr = st.session_state.density_range
dmin = col2.number_input(
    "Min Density", min_value=0.0, max_value=dr['max'],
    value=dr['min'], step=dr['step'], key="cd_min"
)
# ← ensure Max Density cannot go below the step
dmax = col3.number_input(
    "Max Density", min_value=dr['step'], max_value=1000.0,
    value=max(dr['max'], dr['step']), step=dr['step'], key="cd_max"
)
dr.update(min=dmin, max=dmax)

dr['step'] = col4.number_input(
    "Step", min_value=0.0,
    max_value=(dmax - dmin) if dmax > dmin else 0.1,
    value=dr['step'], step=0.1, key="cd_step"
)

st.markdown("---")

# ─── Printing Parameters Section ────────────────────────────────────────────────
st.subheader("Printing Parameters (enter range)")
for name in PRINT_PARAM_NAMES:
    pmin  = st.session_state.pp_ranges[name]['min']
    pmax  = st.session_state.pp_ranges[name]['max']
    pstep = st.session_state.pp_ranges[name]['step']

    c1, c2, c3, c4 = st.columns([2,1,1,1])
    c1.write(name)
    pmin = c2.number_input(
        "Min", min_value=0.0, max_value=pmax,
        value=pmin, step=pstep, key=f"pp_min_{name}"
    )
    # ← enforce Max ≥ step
    pmax = c3.number_input(
        "Max", min_value=pstep, max_value=10000.0,
        value=max(pmax, pstep), step=pstep, key=f"pp_max_{name}"
    )
    pstep = c4.number_input(
        "Step", min_value=0.0,
        max_value=(pmax - pmin) if pmax > pmin else 1.0,
        value=pstep, step=max(1e-3, pstep/10),
        key=f"pp_step_{name}"
    )
    st.session_state.pp_ranges[name].update(min=pmin, max=pmax, step=pstep)

st.markdown("---")

# ─── Optuna Optimize & Display ──────────────────────────────────────────────────
if st.button("🛠️ Optimize WSSQ"):
    with st.spinner("Running Optuna…"):
        def objective(trial):
            bi_vals = {
                r['mat']: trial.suggest_float(
                    f"bio__{r['mat']}", r['min'], r['max'], step=r['step']
                )
                for r in st.session_state.bio_rows
            }
            for m in BIOMATERIAL_OPTIONS:
                bi_vals.setdefault(m, 0.0)

            cd = 0.0 if cell_line=="NoCellCultured" else trial.suggest_float(
                "cell_density", dr['min'], dr['max'], step=dr['step']
            )

            pp_vals = {
                name: trial.suggest_float(
                    f"pp__{name}",
                    st.session_state.pp_ranges[name]['min'],
                    st.session_state.pp_ranges[name]['max'],
                    step=st.session_state.pp_ranges[name]['step']
                )
                for name in PRINT_PARAM_NAMES
            }

            feat = {**bi_vals, **pp_vals}
            feat["Cell Density (cells/mL)"] = cd
            feat.update(
                encoder.transform(pd.DataFrame({"Cell Line":[cell_line]}))
                       .iloc[0].to_dict()
            )

            X = pd.DataFrame([feat])
            Xp = X.reindex(columns=feature_order_print, fill_value=0.0)
            Xc = X.reindex(columns=feature_order_cell,  fill_value=0.0)

            p_proba = model_print.predict_proba(scaler_print.transform(Xp))[0]
            c_proba = model_cell .predict_proba(scaler_cell .transform(Xc))[0]
            exp_p = float(np.dot(p_proba, model_print.classes_.astype(float)))
            exp_c = float(np.dot(c_proba, model_cell .classes_.astype(float)))

            return scaffold_quality_combined(exp_p, exp_c)

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=50)
        best = study.best_trial

    st.success(f"🏆 Best WSSQ: **{best.value:.3f}**")

    best_df = pd.Series(best.params, name="value") \
                .rename_axis("parameter") \
                .to_frame()
    st.table(best_df)

    # ─── Fabrication Procedure via GPT ───────────────────────────────────────────────
    st.markdown("## 🏭 Fabrication Procedure")
    with st.spinner("Generating fabrication procedure…"):
        density = best.params.get("cell_density", 0.0)
        prompt = (
            "Based on the following optimized scaffold parameters AND your selected cell line:\n\n"
            f"• Cell line: {cell_line}\n"
            "Optimized scaffold parameters:\n"
            f"{best.params}\n\n"
            "Write a detailed, step-by-step fabrication procedure for 3D (bio)printing "
            "this scaffold — including materials, equipment, printing settings, and post‑processing."
        )

        resp = client.models.generate_content(
            model="gemini-2.0-flash-001",
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=(
                    "You are a biomedical engineering professor with tissue engineering experties. "
                    "you know how to print 3d (bio)printed scaffolds in labs."
                )
            )
        )

        procedure = resp.text
        st.markdown(procedure)