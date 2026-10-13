"""Fixed-subset raw/EMA COCO evaluation and per-GT ranking replay, offline."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import torch
from PIL import Image
from torchvision.transforms import v2
from faster_coco_eval import COCO, COCOeval_faster
from engine.core import YAMLConfig
from tools.table_cells import supervision_probe  # register optional config wrapper
from tools.table_cells.supervision_probe import ranking_rows, overlap
from tools.table_cells.prepare_probe_subset import sha256


class CaptureCandidates:
    """Observe actual selector inputs; decode dense boxes only for diagnostics."""
    def __init__(self,decoder):
        self.decoder=decoder;self.original=decoder._select_topk;self.data=None

    def __enter__(self):
        def capture(memory,logits,anchors,k):
            result=self.original(memory,logits,anchors,k)
            self.data={'boxes':(self.decoder.enc_bbox_head(memory)+anchors).sigmoid().detach(),
                       'selected_boxes':(self.decoder.enc_bbox_head(result[0])+result[2]).sigmoid().detach(),
                       'logits':logits.detach(), 'k':k}
            return result
        self.decoder._select_topk=capture
        return self

    def __exit__(self,*args):self.decoder._select_topk=self.original


def coco_metrics(coco,predictions,budget):
    if not predictions: raise ValueError('No predictions; refusing to report a fabricated metric')
    dt=coco.loadRes(predictions)
    ev=COCOeval_faster(coco,dt,iouType='bbox',print_function=lambda *a:None,separate_eval=True)
    ev.params.maxDets=[1,10,budget]
    ev.evaluate();ev.accumulate();ev.summarize()
    return ev.stats.tolist()


@torch.no_grad()
def replay(model,coco,images_root,device,row_file):
    if len(coco.getCatIds())!=1:raise ValueError('This protocol is single-class only')
    category=coco.getCatIds()[0]
    transform=v2.Compose([v2.Resize([640,640]),v2.ToImage(),v2.ToDtype(torch.float32,scale=True)])
    predictions=[]
    counts={'small':0,'medium':0,'large':0}
    with CaptureCandidates(model.decoder) as capture, row_file.open('w',encoding='utf-8') as rows:
        for image_id in sorted(coco.getImgIds()):
            info=coco.imgs[image_id]
            with Image.open(images_root/info['file_name']) as image:
                image=image.convert('RGB')
                if image.size!=(info['width'],info['height']):raise ValueError('Image/annotation dimensions disagree')
                inputs=transform(image).unsqueeze(0).to(device)
            output=model(inputs)
            pred=output['pred_boxes'][0].float();scores=output['pred_logits'][0,:,0].sigmoid().float()
            width,height=info['width'],info['height']
            for box,score in zip(pred.cpu().tolist(),scores.cpu().tolist()):
                cx,cy,bw,bh=box
                predictions.append({'image_id':image_id,'category_id':category,'score':score,
                                    'bbox':[(cx-bw/2)*width,(cy-bh/2)*height,bw*width,bh*height]})
            anns=[a for a in coco.loadAnns(coco.getAnnIds(imgIds=[image_id])) if not a.get('iscrowd',0)]
            if not anns:continue
            gt=torch.tensor([[(a['bbox'][0]+a['bbox'][2]/2)/width,
                              (a['bbox'][1]+a['bbox'][3]/2)/height,a['bbox'][2]/width,a['bbox'][3]/height]
                             for a in anns],device=device)
            data=capture.data
            valid=torch.isfinite(data['logits'][0,:,0])
            enc_boxes=data['boxes'][0,valid];enc_scores=data['logits'][0,valid,0]
            dense=ranking_rows(enc_boxes,enc_scores,gt,budgets=(data['k'],))
            # Preserve the actual selector's membership, including score ties.
            dense[f'best_iou_top{data["k"]}']=overlap(data['selected_boxes'][0],gt).max(0).values
            final=ranking_rows(pred,scores,gt,budgets=(300,len(pred)))
            for g,ann in enumerate(anns):
                area=ann['area'];bucket='small' if area<32**2 else 'medium' if area<96**2 else 'large'
                counts[bucket]+=1
                record={'image_id':image_id,'annotation_id':ann['id'],'coco_area':area,'coco_area_group':bucket,
                        'input_short_px':float(torch.minimum(gt[g,2],gt[g,3])*640),
                        'encoder_valid_candidates':len(enc_boxes),'decoder_queries':len(pred)}
                for prefix,values in [('encoder_dense',dense),('decoder',final)]:
                    record.update({prefix+'_'+k:float(v[g]) for k,v in values.items()})
                rows.write(json.dumps(record,allow_nan=False)+'\n')
    return {'coco_maxdets300':coco_metrics(coco,predictions,300),
            'coco_maxdets640':coco_metrics(coco,predictions,640),'gt_counts':counts,
            'images':len(coco.imgs), 'coverage_note':'per-GT best-IoU rows are coverage, not one-to-one COCO recall'}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cuda');p.add_argument('--states',nargs='+',choices=['raw','ema'],default=['raw','ema'])
    args=p.parse_args()
    manifest=json.loads(args.manifest.read_text(encoding='utf-8'))
    images_root=Path(manifest['images_root'])
    if manifest['schema']!='sgqs_fixed_probe_v1':raise ValueError('Unknown subset protocol')
    args.output.mkdir(parents=True,exist_ok=False)
    config=YAMLConfig(str(args.config));config.yaml_cfg['HGNetv2']['pretrained']=False
    if config.yaml_cfg.get('eval_spatial_size')!=[640,640]:raise ValueError('Replay requires fixed 640')
    model=config.model.to(args.device).eval()
    if model.decoder.num_queries!=640 or model.decoder.num_classes!=1:raise ValueError('Expected single-class, 640 queries')
    state=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    weights={}
    for name in args.states:
        weights[name]=state['model'] if name=='raw' else state['ema']['module']
    summary={'checkpoint':str(args.checkpoint.resolve()),'checkpoint_sha256':sha256(args.checkpoint),
             'last_epoch':state.get('last_epoch'),'subset_sha256':sha256(args.manifest),
             'config':config.yaml_cfg,'results':{}}
    for split,meta in manifest['splits'].items():
        annotation=args.manifest.parent/meta['annotation']
        if sha256(annotation)!=meta['annotation_sha256']:raise ValueError('Subset annotations changed')
        for filename,expected in meta['image_sha256'].items():
            if sha256(images_root/filename)!=expected:raise ValueError(f'Fixed image changed: {filename}')
        coco=COCO(str(annotation))
        for name in args.states:
            model.load_state_dict(weights[name],strict=True)
            summary['results'][f'{split}/{name}']=replay(model,coco,images_root,args.device,
                                                       args.output/f'{split}.{name}.gt.jsonl')
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2,default=str,allow_nan=False),encoding='utf-8')
    print(args.output/'summary.json')


if __name__=='__main__':main()
