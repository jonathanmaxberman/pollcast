"""Outside spending in Senate races, from the FEC's independent-expenditure filings.

Independent expenditures (super PACs, party committees, other groups) must be
reported within 24-48 hours, and each row carries the date the FEC received
it, so a backtest can count only what had been reported by its forecast date.

Each cycle's bulk file (fec.gov/files/bulk-downloads/<year>/) is filtered to
Senate rows when fetched, since the full files run about 16 MB a year; the
candidate master file (cn<yy>.zip) supplies party affiliation, which the
expenditure rows often leave blank.

Spending counts for the Democratic side when it supports a Democrat or
opposes a Republican, and for the Republican side the other way round.
Independent candidates count on the non-Republican side.
"""

from __future__ import annotations

import io
import zipfile

import pandas as pd

from pollcast.sources._538_format import STATE_NAMES
from pollcast.sources.base import Source
from pollcast.storage import Snapshot, SnapshotWriter

BULK = "https://www.fec.gov/files/bulk-downloads/{year}/{file}"
CN_COLUMNS = ["CAND_ID", "CAND_NAME", "CAND_PTY_AFFILIATION", "CAND_ELECTION_YR", "CAND_OFFICE_ST",
              "CAND_OFFICE", "CAND_OFFICE_DISTRICT", "CAND_ICI", "CAND_STATUS", "CAND_PCC", "CAND_ST1",
              "CAND_ST2", "CAND_CITY", "CAND_ST", "CAND_ZIP"]


class FECIndependentExpenditures(Source):
    name = "fec_ie"
    table = "spending"
    frozen = False
    attribution = "Outside spending: Federal Election Commission independent-expenditure filings."

    def fetch(self, writer: SnapshotWriter) -> None:
        for year in self.settings.get("cycles", list(range(2010, 2028, 2))):
            ie = self.get(BULK.format(year=year, file=f"independent_expenditure_{year}.csv"), pause=10).content
            df = pd.read_csv(io.BytesIO(ie), dtype=str, keep_default_na=False)
            df = df[df["can_office"] == "S"]
            writer.write(f"ie_senate_{year}.csv", df.to_csv(index=False).encode(),
                         url=BULK.format(year=year, file=f"independent_expenditure_{year}.csv"),
                         note="filtered to Senate rows")
            cn_zip = self.get(BULK.format(year=year, file=f"cn{str(year)[2:]}.zip"), pause=10).content
            with zipfile.ZipFile(io.BytesIO(cn_zip)) as z:
                cn = pd.read_csv(z.open(z.namelist()[0]), sep="|", header=None, names=CN_COLUMNS, dtype=str)
            cn = cn[cn["CAND_OFFICE"] == "S"][["CAND_ID", "CAND_NAME", "CAND_PTY_AFFILIATION", "CAND_OFFICE_ST"]]
            writer.write(f"cn_senate_{year}.csv", cn.to_csv(index=False).encode(),
                         url=BULK.format(year=year, file=f"cn{str(year)[2:]}.zip"))

    def normalize(self, snapshot: Snapshot) -> pd.DataFrame:
        return spending_rows(snapshot)


def _last_name(name: str) -> str:
    return str(name).split(",")[0].strip().upper()


def spending_rows(snapshot: Snapshot) -> pd.DataFrame:
    """One row per expenditure (latest version of each), with side and received date."""
    frames = []
    for p in sorted(snapshot.path.glob("ie_senate_*.csv")):
        year = int(p.stem.split("_")[-1])
        ie = pd.read_csv(p, dtype=str, keep_default_na=False)
        cn = pd.read_csv(snapshot.path / f"cn_senate_{year}.csv", dtype=str, keep_default_na=False)
        party_by_id = dict(zip(cn["CAND_ID"], cn["CAND_PTY_AFFILIATION"]))
        cn["key"] = cn["CAND_OFFICE_ST"] + "|" + cn["CAND_NAME"].map(_last_name)
        party_by_name = cn.drop_duplicates("key").set_index("key")["CAND_PTY_AFFILIATION"].to_dict()
        party = ie["cand_id"].map(party_by_id)
        by_name = (ie["can_office_state"] + "|" + ie["cand_name"].map(_last_name)).map(party_by_name)
        party = party.fillna(by_name).fillna(ie["cand_pty_aff"].replace("", pd.NA))
        party = party.map(lambda x: "REP" if x == "REP" else ("DEM" if x in ("DEM", "DFL") else
                                                             ("IND" if isinstance(x, str) and x else None)))
        sup = ie["sup_opp"].str.upper().str[0]
        rep_side = ((party == "REP") & (sup == "S")) | ((party != "REP") & party.notna() & (sup == "O"))
        out = pd.DataFrame({
            "cycle": year, "state": ie["can_office_state"].map(STATE_NAMES),
            "election_type": ie["ele_type"].str.upper().str[0],
            "spender": ie["spe_id"], "tran_id": ie["tran_id"],
            "file_num": ie["file_num"], "prev_file_num": ie["prev_file_num"],
            "amount": pd.to_numeric(ie["exp_amo"], errors="coerce"),
            "received": pd.to_datetime(ie["receipt_dat"], format="%d-%b-%y", errors="coerce"),
            "side": pd.Series(pd.NA, index=ie.index, dtype="object"),
        })
        out.loc[party.notna(), "side"] = "DEM"
        out.loc[rep_side, "side"] = "REP"
        frames.append(out[out["side"].notna() & out["amount"].notna() & out["received"].notna()
                          & (out["election_type"].isin(["G", ""]))])
    return pd.concat(frames, ignore_index=True)


def spending_as_of(rows: pd.DataFrame, cycle: int, as_of) -> pd.DataFrame:
    """Democratic- and Republican-side outside spending per state, counting only
    reports received by `as_of`.

    An amended report restates the whole filing it replaces (named in
    prev_file_num), so a filing's rows are dropped once an amendment of it had
    been received. Transaction IDs alone can't be used: spenders reuse them
    across separate reports.
    """
    r = rows[(rows["cycle"] == cycle) & (rows["received"] <= pd.Timestamp(as_of))]
    superseded = set(r["prev_file_num"].replace("", pd.NA).dropna())
    r = r[~r["file_num"].isin(superseded)]
    s = r.groupby(["state", "side"])["amount"].sum().unstack(fill_value=0.0)
    for side in ("DEM", "REP"):
        if side not in s:
            s[side] = 0.0
    return s[["DEM", "REP"]]
