"""Bundle a Russian-propaganda domain blocklist from multiple sources.

Motivation: any single curated list covers only a sliver of SWARM's search-
surfaced domains, so we union several propaganda-domain lists. This union is the
source-based baseline reported in the paper.

LOCAL (always included, already in the repo):
  * wiki_disinfo   data/russian_prop_wiki.csv   (Wikipedia "List of political
                   disinformation website campaigns in Russia", scraped)

CITED / EXTERNAL (pass with --extra LABEL=PATH; SWARM cites these source-based
resources, whose outlet lists are exactly the "weak source labels" we critique):
  * proppy         Proppy corpus outlet list (Barron-Cedeno 2019; MBFC-derived)
  * rashkin        Rashkin 2017 source-type list (propaganda/hoax sources)
  * euvsdisinfo    EUvsDisinfo pro-Kremlin article domains (Leite 2024)
This script does NOT download anything; drop the files in and point --extra at
them. Each --extra file may be a .csv (a 'domain'/'url'/'link' column is
auto-detected) or a .txt (one domain per line).

OPTIONAL (off by default): --include-low-quality also unions domains flagged
only by a generic news-reliability score (data/news_quality_ratings.csv,
pc1 < --quality-thr). This is NOT part of the paper's blocklist, because low
news reliability is a quality signal, not evidence of Russian propaganda.

Matching: domains are normalised to registered domain (eTLD+1), lower-cased,
de-fanged ([.]->.), www./m. stripped, so subdomain variants
(news.tsargrad.tv -> tsargrad.tv) collapse together.

Output: data/propaganda_domains_bundle.csv  (domain, sources, n_sources)
Also prints coverage of the bundle against the current eval items.
"""
import argparse
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
WIKI = ROOT / "data" / "russian_prop_wiki.csv"
QUALITY = ROOT / "data" / "news_quality_ratings.csv"
ITEMS = ROOT / "data" / "ra_annotated_package" / "items.csv"
OUT = ROOT / "data" / "propaganda_domains_bundle.csv"

# registered-domain reduction without the full Public Suffix List: a compact set
# of common multi-label public suffixes covering the TLDs seen in this data.
MULTI_TLD = {
    "co.uk", "org.uk", "gov.uk", "ac.uk", "com.au", "org.au", "net.au",
    "co.nz", "com.br", "com.ua", "co.in", "org.in", "com.tr", "com.pl",
    "com.ru", "org.ru", "net.ru", "co.il", "com.es", "com.pt", "co.za",
}


def reg_domain(value: str) -> str:
    """Lower-cased eTLD+1 from a URL or bare domain; '' if unparseable."""
    if value is None:
        return ""
    s = str(value).strip().lower().replace("[.]", ".").replace("(dot)", ".")
    if not s:
        return ""
    if "//" in s or "/" in s:                      # looks like a URL
        s = urlparse(s if "//" in s else "//" + s).netloc or s.split("/")[0]
    s = s.split("@")[-1].split(":")[0]             # strip creds / port
    for pre in ("www.", "m.", "amp."):
        if s.startswith(pre):
            s = s[len(pre):]
    labels = [l for l in s.split(".") if l]
    if len(labels) < 2:
        return s
    last2 = ".".join(labels[-2:])
    last3 = ".".join(labels[-3:]) if len(labels) >= 3 else ""
    if last2 in MULTI_TLD and len(labels) >= 3:
        return last3
    return last2


def domains_from_csv(path: Path) -> set:
    df = pd.read_csv(path)
    for col in ("clean_domain", "domain", "url", "link", "Domain", "website"):
        if col in df.columns:
            return {d for d in (reg_domain(v) for v in df[col].dropna()) if d}
    # fall back to the first column
    first = df.columns[0]
    return {d for d in (reg_domain(v) for v in df[first].dropna()) if d}


def domains_from_txt(path: Path) -> set:
    return {d for d in (reg_domain(l) for l in Path(path).read_text().splitlines()) if d}


def load_source(path: Path) -> set:
    return domains_from_txt(path) if path.suffix.lower() == ".txt" else domains_from_csv(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--include-low-quality", action="store_true",
                    help="also union generic low-news-reliability domains "
                         "(pc1 < --quality-thr); NOT part of the paper's blocklist")
    ap.add_argument("--quality-thr", type=float, default=0.5,
                    help="pc1 below this counts a domain as low-reliability "
                         "(only used with --include-low-quality)")
    ap.add_argument("--extra", nargs="*", default=[], metavar="LABEL=PATH",
                    help="additional list files, e.g. proppy=data/proppy_outlets.csv")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    sources = {}  # label -> set(domains)

    # local: Wikipedia disinfo campaigns
    sources["wiki_disinfo"] = load_source(WIKI)
    # optional: generic low-reliability domains (off by default; not in the paper)
    if args.include_low_quality:
        q = pd.read_csv(QUALITY)
        lowq = q.loc[q["pc1"] < args.quality_thr, "domain"]
        sources["low_quality"] = {d for d in (reg_domain(v) for v in lowq.dropna()) if d}

    # cited / external, user-supplied
    for spec in args.extra:
        if "=" not in spec:
            raise SystemExit(f"--extra expects LABEL=PATH, got {spec!r}")
        label, path = spec.split("=", 1)
        sources[label] = load_source(Path(path))
        print(f"  loaded {label}: {len(sources[label])} domains from {path}")

    # union with provenance
    prov = {}
    for label, doms in sources.items():
        for d in doms:
            prov.setdefault(d, set()).add(label)
    bundle = pd.DataFrame(
        [{"domain": d, "sources": ";".join(sorted(s)), "n_sources": len(s)}
         for d, s in sorted(prov.items())]
    )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    bundle.to_csv(args.out, index=False)

    print("\n=== bundle composition ===")
    for label, doms in sources.items():
        print(f"  {label:14s}: {len(doms):6d} domains")
    print(f"  {'UNION':14s}: {len(bundle):6d} unique registered domains")
    print(f"  overlap (>=2 sources): {(bundle['n_sources'] >= 2).sum()}")

    # coverage on the eval items
    if ITEMS.exists():
        items = pd.read_csv(ITEMS, usecols=["original_id", "link"])
        items["rd"] = items["link"].map(reg_domain)
        flagged = set(bundle["domain"])
        hit = items["rd"].isin(flagged)
        print("\n=== coverage on eval items ===")
        print(f"  {int(hit.sum())}/{len(items)} items map to a bundled domain "
              f"({100 * hit.mean():.1f}%)")
    print(f"\nWritten: {args.out}")


if __name__ == "__main__":
    main()
