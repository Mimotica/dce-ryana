import pandas as pd
import numpy as np
import re
from statsmodels.discrete.conditional_models import ConditionalLogit

# EXCEL BESTAND (zet locatie goed)
xlsx_path = r"/Users/pim/Downloads/ryana/cleaned_responses.xlsx"
df = pd.read_excel(xlsx_path)

# Identify DCE columns
dce_cols = [c for c in df.columns if "Please select the option you prefer" in c]

ratio_map = {"p_50":0.5, "p_80":0.8, "p_110":1.1, "p_100":1.0}

def extract_options(question_text):
    matches = list(re.finditer(r"Option\s+([123])", question_text))
    blocks = {}
    for i, m in enumerate(matches):
        opt = int(m.group(1))
        start = m.start()
        end = matches[i+1].start() if i+1 < len(matches) else len(question_text)
        blocks[opt] = question_text[start:end]
    return blocks

def parse_attributes(block):
    comfort = None
    attenuation = None
    speech = None
    appearance = None
    price_var = None

    m = re.search(r"Wearing comfort:\s*\*?(\d)\s*HOUR", block, re.IGNORECASE)
    if m:
        comfort = int(m.group(1))
    m = re.search(r"Sound attenuation:\s*\*?(PARTIAL|COMPLETE)", block, re.IGNORECASE)
    if m:
        attenuation = m.group(1).upper()
    m = re.search(r"Speech understanding:\s*\*?([A-Z ]+)", block, re.IGNORECASE)
    if m:
        speech = m.group(1).replace("*","").strip().upper()
    m = re.search(r"Appearance:\s*\*?\s*\*?(VISIBLE|INVISIBLE)", block, re.IGNORECASE)
    if m:
        appearance = m.group(1).upper()
    m = re.search(r"\{\{var:(p_\d+)\}\}", block, re.IGNORECASE)
    if m:
        price_var = m.group(1).lower()
    return comfort, attenuation, speech, appearance, price_var

# Build long format
rows = []
for i, r in df.iterrows():
    resp = str(r["#"]) if "#" in df.columns else str(i+1)
    anchor = r["wtp"]  # individual anchor used by the survey logic
    for t, col in enumerate(dce_cols, start=1):
        chosen_opt = r[col]  # "Option 1" / "Option 2" / "Option 3"
        opts = extract_options(col)
        for opt in [1,2,3]:
            comfort, att, speech, app, pvar = parse_attributes(opts[opt])
            purchase = 1 if opt in [1,2] else 0

            if pvar is not None and pvar in df.columns:
                price_eur = float(r[pvar])
                rel_price = float(price_eur / anchor) if (pd.notna(anchor) and float(anchor)!=0) else np.nan
            else:
                price_eur = 0.0
                rel_price = 0.0

            chosen = 1 if chosen_opt == f"Option {opt}" else 0

            rows.append({
                "respondent": resp,
                "task": t,
                "alt": opt,
                "group": f"{resp}_{t}",
                "chosen": chosen,
                "purchase": purchase,
                "price_eur": price_eur,
                "rel_price": rel_price,
                "comfort": comfort,
                "att": att,
                "speech": speech,
                "app": app
            })

long = pd.DataFrame(rows).fillna(0)

# Dummies (reference levels: comfort=1h; attenuation=PARTIAL; speech=EASY; appearance=VISIBLE)
long["comfort_2"] = (long["comfort"]==2).astype(int)
long["comfort_4"] = (long["comfort"]==4).astype(int)
long["att_complete"] = (long["att"]=="COMPLETE").astype(int)
long["speech_moderate"] = (long["speech"]=="MODERATE").astype(int)
long["speech_more_effort"] = (long["speech"]=="MORE EFFORT").astype(int)
long["app_invisible"] = (long["app"]=="INVISIBLE").astype(int)

X = long[["rel_price","comfort_2","comfort_4","att_complete","speech_moderate","speech_more_effort","app_invisible","purchase"]]
y = long["chosen"].astype(int)
groups = long["group"]

model = ConditionalLogit(y, X, groups=groups)
res = model.fit()
print(res.summary())

# WTP in % of anchor: WTP = -beta_feature / beta_price
beta_p = res.params["rel_price"]
for f in ["comfort_2","comfort_4","att_complete","speech_moderate","speech_more_effort","app_invisible"]:
    wtp_pct = (-res.params[f] / beta_p) * 100
    print(f, wtp_pct)
