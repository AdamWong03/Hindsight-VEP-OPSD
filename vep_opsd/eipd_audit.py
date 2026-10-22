"""Create reproducible stratified EIPD review sheets and an editable quality-audit CSV."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import textwrap
from collections import defaultdict
from pathlib import Path

def bridge_bin(record: dict) -> str:
    gap = int(record["future_native_indices"][0]) - int(record["past_native_indices"][-1])
    return "short" if gap <= 20 else ("medium" if gap <= 90 else "long")

def deterministic_pick(rows: list[dict], per_stratum: int, seed: int) -> list[dict]:
    bins=defaultdict(list)
    for row in rows:
        bins[(row["category"],bridge_bin(row))].append(row)
    picked=[]
    for key in sorted(bins):
        group=sorted(bins[key],key=lambda r:hashlib.sha256(
            f"{seed}:{r['sample_id']}".encode()).digest())
        picked.extend(group[:per_stratum])
    return picked

def create(args):
    from PIL import Image, ImageDraw, ImageFont, ImageOps
    root=Path(args.data_root).expanduser().resolve()
    manifest=root/f"{args.split}.jsonl"
    rows=[json.loads(s) for s in manifest.read_text(encoding="utf-8").splitlines() if s.strip()]
    selected=deterministic_pick(rows,args.per_stratum,args.seed)
    out=Path(args.output).resolve()
    out.mkdir(parents=True,exist_ok=True)
    annotations=[]
    for index,row in enumerate(selected):
        imgs=[]
        for name,paths in (("PAST",row["past_files"]),("BRIDGE",[row["bridge_file"]]),
                           ("FUTURE",row["future_files"])):
            for ix,rel in enumerate(paths):
                file=(root/rel).resolve()
                if not file.is_relative_to(root):
                    raise ValueError(f"Unsafe frame: {rel}")
                with Image.open(file) as src:
                    im=ImageOps.contain(src.convert("RGB"),(300,195))
                    tile=Image.new("RGB",(320,235),"white")
                    tile.paste(im,((320-im.width)//2,10))
                    ImageDraw.Draw(tile).text((10,211),f"{name} {ix+1}",fill="black")
                    imgs.append(tile)
        cols=min(4,len(imgs))
        rows_n=math.ceil(len(imgs)/cols)
        canvas=Image.new("RGB",(cols*320,max(1,rows_n)*235+190),"white")
        for j,img in enumerate(imgs):
            canvas.paste(img,((j%cols)*320,(j//cols)*235))
        dr=ImageDraw.Draw(canvas)
        start=rows_n*235+8
        text=f"ID {row['sample_id']} | {row['category']} | {bridge_bin(row)}\nQuestion: {row['question']}\nAnswer: {row['answer']}"
        lines=[]
        for raw in text.splitlines():
            lines.extend(textwrap.wrap(raw,width=115) or [""])
        for j,line in enumerate(lines[:9]):
            dr.text((12,start+j*18),line,fill="black")
        filename=f"{index:05d}_{row['sample_id']}.jpg"
        canvas.save(out/filename,quality=92)
        annotations.append({
            "sample_id":row["sample_id"],"source_id":row["source_id"],
            "category":row["category"],"gap_bin":bridge_bin(row),
            "contact_sheet":filename,
            "answer_supported_by_future":"", "question_is_future_event":"",
            "bridge_already_reveals_answer":"","visual_alignment_correct":"",
            "reviewer_notes":""
        })
    fields=list(annotations[0]) if annotations else []
    with (out/"review.csv").open("w",encoding="utf-8",newline="") as f:
        if fields:
            w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(annotations)
    summary={"population":len(rows),"review_count":len(selected),
             "review_by_category_bin":{
                 f"{cat}/{bin}":sum(1 for r in selected if r["category"]==cat and bridge_bin(r)==bin)
                 for cat,bin in sorted({(r["category"],bridge_bin(r)) for r in selected})},
             "seed":args.seed,"per_stratum":args.per_stratum,
             "instruction":"Fill yes/no in review.csv after checking sheets. This review does not certify all examples."}
    (out/"selection.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    return summary

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root",required=True)
    p.add_argument("--split",choices=("train","val"),default="train")
    p.add_argument("--output",required=True)
    p.add_argument("--per-stratum",type=int,default=50,
                   help="50 per category x gap bin yields up to 450 reviews")
    p.add_argument("--seed",type=int,default=2026)
    args=p.parse_args()
    if args.per_stratum<=0:
        p.error("per-stratum must be positive")
    print(json.dumps(create(args),indent=2))

if __name__=="__main__":
    main()
