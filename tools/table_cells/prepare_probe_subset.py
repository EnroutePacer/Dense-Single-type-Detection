"""Stream large COCO files into fixed train/val subsets; no training RNG use."""
import argparse
import hashlib
import heapq
import json
from pathlib import Path


def sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024), b''): digest.update(block)
    return digest.hexdigest()


def subset(path, count, seed, excluded=()):
    try:
        import ijson
    except ImportError as exc:
        raise RuntimeError(
            'Subset preparation needs ijson to stream the 6 GB annotation file; '
            'select an approved existing environment and do not install into a shared server') from exc
    def items(prefix):
        with Path(path).open('rb') as f:
            yield from ijson.items(f,prefix,use_float=True)
    def order(image):
        return (hashlib.sha256(f'{seed}:{image["file_name"]}'.encode()).hexdigest(),image['id'])
    images=heapq.nsmallest(count,(im for im in items('images.item') if im['file_name'] not in excluded),key=order)
    if len(images)!=count: raise ValueError('Not enough non-overlapping images for requested subset')
    ids={im['id'] for im in images}
    return {'images':images,'annotations':[a for a in items('annotations.item') if a['image_id'] in ids],
            'categories':list(items('categories.item'))}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--train-ann',type=Path,required=True);p.add_argument('--val-ann',type=Path,required=True)
    p.add_argument('--images',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--images-per-split',type=int,default=128);p.add_argument('--seed',type=int,default=20261014)
    args=p.parse_args()
    if args.images_per_split<1: p.error('images-per-split must be positive')
    args.output.mkdir(parents=True,exist_ok=False)
    val=subset(args.val_ann,args.images_per_split,args.seed)
    train=subset(args.train_ann,args.images_per_split,args.seed,{im['file_name'] for im in val['images']})
    manifest={'schema':'sgqs_fixed_probe_v1','seed':args.seed,'images_root':str(args.images.resolve()),
              'selection':'sha256(seed:file_name), no model-based selection; fixed size is a pilot', 'splits':{}}
    split_hashes={}
    for name,source,data in [('train',args.train_ann,train),('val',args.val_ann,val)]:
        hashes={im['file_name']:sha256(args.images/im['file_name']) for im in data['images']}
        split_hashes[name]=set(hashes.values())
        out=args.output/f'{name}.json';out.write_text(json.dumps(data,ensure_ascii=False),encoding='utf-8')
        manifest['splits'][name]={'annotation':out.name,'annotation_sha256':sha256(out),
            'source_annotation':str(source.resolve()),'source_sha256':sha256(source),
            'image_sha256':hashes,'images':len(data['images']),'annotations':len(data['annotations'])}
    if split_hashes['train'] & split_hashes['val']:
        raise ValueError('Identical image bytes in selected train/val; resolve split leakage before testing overfitting')
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(args.output/'manifest.json')


if __name__=='__main__':main()
