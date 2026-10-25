"""CPU regression checks for COCO mappings, supervision and evaluation."""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import hashlib, json, unittest
import numpy as np
import torch
from PIL import Image
from research.config import (load_config, dataset_spec, validate_splits, read_ids,
                             image_path, mask_path, load_mask, validate_role_bank)
from research.metrics import confusion, metric
from research.jev_roles import role_partial_loss
from research.methods import anchored_graph

class COCODatasetTests(unittest.TestCase):
    def setUp(self):
        self.cfg=load_config('configs/coco.json')

    def test_complete_coco_lists_and_labels(self):
        report=validate_splits(self.cfg)
        self.assertEqual((report['train_images'],report['val_images'],report['num_classes']),(82081,40137,81))

    def test_incomplete_validation_rejected(self):
        with TemporaryDirectory() as tmp:
            path=Path(tmp)/'val.txt'
            path.write_text('\n'.join(read_ids(self.cfg['val_ids'],'coco')[:-1]))
            with self.assertRaises(ValueError):
                validate_splits(dict(self.cfg,val_ids=str(path)),check_labels=False)

    def test_wrong_membership_rejected_with_same_count(self):
        with TemporaryDirectory() as tmp:
            ids=read_ids(self.cfg['train_ids'],'coco');ids[0]='COCO_train2014_999999999999'
            path=Path(tmp)/'train.txt';path.write_text('\n'.join(ids))
            with self.assertRaises(ValueError):
                validate_splits(dict(self.cfg,train_ids=str(path)),check_labels=False)

    def test_image_and_mask_paths_preserve_zero_padding(self):
        name='COCO_val2014_000000000042'
        self.assertEqual(image_path(self.cfg,name).relative_to(self.cfg['data']).as_posix(),'JPEGImages/val2014/COCO_val2014_000000000042.jpg')
        self.assertEqual(mask_path(self.cfg,name).relative_to(self.cfg['data']).as_posix(),'SegmentationClass/val2014/000000000042.png')
        train='COCO_train2014_000000000009'
        self.assertEqual(image_path(self.cfg,train,'train').relative_to(self.cfg['data']).as_posix(),'JPEGImages/train2014/'+train+'.jpg')

    def test_noncontiguous_category_mapping(self):
        spec=dataset_spec(self.cfg)
        self.assertEqual(len(set(spec['category_ids'])),80)
        self.assertEqual(spec['category_ids'][11],13)
        self.assertEqual(spec['category_ids'][-1],90)
        self.assertEqual(spec['classes'][-1],'toothbrush')

    def test_indexed_masks_preserve_class_80_and_ignore(self):
        with TemporaryDirectory() as tmp:
            cfg=dict(self.cfg,data=tmp);name='COCO_val2014_000000000042'
            path=mask_path(cfg,name);path.parent.mkdir(parents=True)
            expected=np.array([[0,80,255]],dtype=np.uint8)
            Image.fromarray(expected).save(path)
            self.assertTrue(np.array_equal(load_mask(cfg,name),expected))
            Image.fromarray(np.array([[0,90,255]],dtype=np.uint8)).save(path)
            with self.assertRaises(ValueError):load_mask(cfg,name)

    def test_rejects_voc_cache_for_coco(self):
        voc=load_config('configs/voc.json')
        cfg=dict(self.cfg,jev_role_bank=voc['jev_role_bank'],jev_role_bank_sha256=voc['jev_role_bank_sha256'])
        with self.assertRaisesRegex(ValueError,'class order'):validate_role_bank(cfg)

    def test_missing_cache_has_actionable_error(self):
        with TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(FileNotFoundError,'--role-bank'):
                validate_role_bank(dict(self.cfg,jev_role_bank=str(Path(tmp)/'absent.json')))

    def test_cache_schema_checksum_and_class_order(self):
        # Synthetic roles are test fixtures only, never a released semantic cache.
        with TemporaryDirectory() as tmp:
            path=Path(tmp)/'fixture.json'
            bank=dict(classes=dataset_spec(self.cfg)['classes'].copy(),descriptors=['test descriptor'],shared_weights=np.zeros((81*81,1)).tolist(),unique_weights=np.zeros((81*81,1)).tolist())
            path.write_text(json.dumps(bank))
            cfg=dict(self.cfg,jev_role_bank=str(path),jev_role_bank_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            self.assertEqual(len(validate_role_bank(cfg)['classes']),80)
            with self.assertRaisesRegex(ValueError,'checksum'):validate_role_bank(dict(cfg,jev_role_bank_sha256='invalid'))
            bank['classes'][0],bank['classes'][1]=bank['classes'][1],bank['classes'][0]
            path.write_text(json.dumps(bank));cfg['jev_role_bank_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
            with self.assertRaisesRegex(ValueError,'class order'):validate_role_bank(cfg)

class COCOSupervisionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(18)
        self.cfg=load_config('configs/coco.json')
        self.logits=torch.randn(1,81,8,8,requires_grad=True)
        self.pseudo=torch.full((1,8,8),79,dtype=torch.long)
        self.proto=torch.zeros(1,81,8,8);self.proto[:,79]=.4;self.proto[:,80]=.6
        self.tags=torch.zeros(1,80);self.tags[:,78:]=1
        self.cues=torch.ones(1,3,8,8);self.cams=torch.ones(1,80,8,8)
        self.bank=SimpleNamespace(shared=torch.zeros(81*81,3),unique=torch.zeros(81*81,3))

    def loss(self,**changes):
        args=dict(logits=self.logits,pseudo=self.pseudo,prototype=self.proto,labels=self.tags,cues=self.cues,refined_cams=self.cams,bank=self.bank,cfg=self.cfg)
        args.update(changes);return role_partial_loss(**args)

    def test_shared_roles_admit_class_80_with_gradients(self):
        self.bank.shared[81*79+80,0]=1
        loss,aux,stats=self.loss()
        self.assertGreater(stats['jev_added_pair_fraction'],0)
        self.assertEqual(int((aux==255).sum()),9)
        loss.backward();self.assertTrue(torch.isfinite(self.logits.grad).all())
        self.assertGreater(float(self.logits.grad[:,80].abs().sum()),0)

    def test_specific_roles_preserve_high_index_anchor(self):
        self.proto[:,79]=.2;self.proto[:,80]=.8;self.bank.unique[81*79+80,0]=1
        _,aux,stats=self.loss()
        self.assertEqual(stats['jev_hard_anchor_fraction'],1)
        self.assertEqual(stats['partial_fraction'],0)
        self.assertTrue(torch.equal(aux,self.pseudo))

    def test_absent_class_80_cannot_be_admitted(self):
        self.tags[:,79]=0;self.bank.shared[81*79+80,0]=1
        _,aux,stats=self.loss();self.assertEqual(stats['partial_fraction'],0)
        self.assertTrue(torch.equal(aux,self.pseudo))

    def test_ignored_targets_have_zero_gradient(self):
        loss,_,_=self.loss(pseudo=torch.full_like(self.pseudo,255));loss.backward()
        self.assertEqual(float(loss.detach()),0);self.assertEqual(float(self.logits.grad.abs().sum()),0)

    def test_coco_prototypes_respect_class_presence(self):
        _,proto=anchored_graph(torch.rand(1,80,4,4),torch.randn(1,32,4,4),self.tags)
        self.assertEqual(tuple(proto.shape),(1,81,4,4));self.assertEqual(float(proto[:,1:79].abs().max()),0)
        self.assertTrue(torch.allclose(proto.sum(1),torch.ones(1,4,4),atol=1e-6))

    def test_confusion_counts_last_class_and_ignores_255(self):
        gt=np.array([[0,80,255],[79,80,0]],dtype=np.uint8)
        pred=np.array([[0,79,80],[79,80,0]],dtype=np.uint8)
        hist=confusion(gt,pred,81)
        self.assertEqual(hist.shape,(81,81));self.assertEqual(hist.sum(),5)
        self.assertEqual(hist[80,79],1);self.assertEqual(hist[80,80],1)
        self.assertEqual(metric(np.eye(81,dtype=np.int64))['miou_percent'],100)

    def test_confusion_rejects_raw_category_ids(self):
        with self.assertRaises(ValueError):confusion(np.array([[90]]),np.array([[80]]),81)

if __name__=='__main__':unittest.main()
