"""CPU checks for full VOC coverage and the candidate/mask supervision contract."""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
import torch

from research.config import ROOT, load_config, validate_voc_splits
from research.methods import partial_label_loss, anchored_graph
from research.jev_roles import role_partial_loss


class VOCSplitTests(unittest.TestCase):
    def setUp(self):
        self.cfg = load_config('configs/voc.json')

    def test_complete_training_and_validation_coverage(self):
        report = validate_voc_splits(self.cfg)
        self.assertEqual(report['train_images'], 10582)
        self.assertEqual(report['val_images'], 1449)
        self.assertEqual(report['overlapping_images'], 0)

    def test_rejects_incomplete_training_list(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / 'train.txt'
            ids = Path(self.cfg['train_ids']).read_text().split()
            path.write_text('\n'.join(ids[:-1])+'\n')
            with self.assertRaises(ValueError):
                validate_voc_splits(dict(self.cfg, train_ids=str(path)))

    def test_checks_membership_not_just_count(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / 'train.txt'
            ids = Path(self.cfg['train_ids']).read_text().split()
            ids[0] = Path(self.cfg['val_ids']).read_text().split()[0]
            path.write_text('\n'.join(ids)+'\n')
            with self.assertRaises(ValueError):
                validate_voc_splits(dict(self.cfg, train_ids=str(path)))

    def test_rejects_duplicate_ids(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / 'train.txt'
            ids = Path(self.cfg['train_ids']).read_text().split()
            ids[0] = ids[1]
            path.write_text('\n'.join(ids)+'\n')
            with self.assertRaises(ValueError):
                validate_voc_splits(dict(self.cfg, train_ids=str(path)))


class RoleSupervisionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        self.cfg = load_config('configs/voc.json')
        self.logits = torch.randn(1,21,8,8,requires_grad=True)
        self.pseudo = torch.zeros(1,8,8,dtype=torch.long)
        self.proto = torch.zeros(1,21,8,8)
        self.proto[:,0] = .4
        self.proto[:,1] = .6
        self.tags = torch.zeros(1,20)
        self.tags[:,0] = 1
        self.cues = torch.ones(1,400,8,8)
        self.cams = torch.ones(1,20,8,8)
        self.bank = SimpleNamespace(shared=torch.zeros(441,400),unique=torch.zeros(441,400))

    def loss(self, **changes):
        values = dict(logits=self.logits,pseudo=self.pseudo,prototype=self.proto,labels=self.tags,
                      cues=self.cues,refined_cams=self.cams,bank=self.bank,cfg=self.cfg)
        values.update(changes)
        return role_partial_loss(**values)

    def test_no_role_evidence_matches_visual_set_loss(self):
        expected,_ = partial_label_loss(self.logits,self.pseudo,self.proto,self.tags)
        actual,aux,_ = self.loss()
        self.assertTrue(torch.allclose(actual,expected,atol=1e-7))
        self.assertTrue(torch.equal(aux,self.pseudo))

    def test_shared_roles_add_pairs_below_visual_threshold(self):
        self.bank.shared[1,0] = 1
        loss,aux,stats = self.loss()
        self.assertGreater(stats['jev_added_pair_fraction'],0)
        self.assertEqual(int((aux==255).sum()),int(.15*self.pseudo.numel()))
        self.assertAlmostEqual(stats['partial_fraction'],stats['jev_auxiliary_ignore_fraction'])
        loss.backward()
        self.assertTrue(torch.isfinite(self.logits.grad).all())
        masked = aux==255
        self.assertGreater(float(self.logits.grad.permute(0,2,3,1)[masked].abs().sum()),0)

    def test_visual_only_pairs_keep_hard_auxiliary_targets(self):
        self.proto[:,0] = .2
        self.proto[:,1] = .8
        _,aux,stats = self.loss()
        self.assertEqual(stats['partial_fraction'],1)
        self.assertEqual(stats['jev_auxiliary_ignore_fraction'],0)
        self.assertTrue(torch.equal(aux,self.pseudo))

    def test_absent_image_class_cannot_be_admitted(self):
        self.bank.shared[1,0] = 1
        _,aux,stats = self.loss(labels=torch.zeros_like(self.tags))
        self.assertEqual(stats['partial_fraction'],0)
        self.assertTrue(torch.equal(aux,self.pseudo))

    def test_specific_evidence_vetoes_alternative(self):
        pseudo = torch.ones_like(self.pseudo)
        proto = torch.zeros_like(self.proto)
        proto[:,1] = .2
        proto[:,2] = .8
        tags = self.tags.clone(); tags[:,1] = 1
        self.bank.unique[23,0] = 1
        _,aux,stats = self.loss(pseudo=pseudo,prototype=proto,labels=tags)
        self.assertEqual(stats['jev_hard_anchor_fraction'],1)
        self.assertEqual(stats['partial_fraction'],0)
        self.assertTrue(torch.equal(aux,pseudo))

    def test_ignored_pixels_have_zero_finite_gradient(self):
        loss,_,_ = self.loss(pseudo=torch.full_like(self.pseudo,255))
        self.assertEqual(float(loss.detach()),0)
        loss.backward()
        self.assertTrue(torch.isfinite(self.logits.grad).all())
        self.assertEqual(float(self.logits.grad.abs().sum()),0)

    def test_prototypes_exclude_absent_classes(self):
        cams = torch.rand(1,20,4,4)
        dense = torch.randn(1,32,4,4)
        _,proto = anchored_graph(cams,dense,self.tags)
        self.assertEqual(float(proto[:,2:].abs().max()),0)
        self.assertTrue(torch.allclose(proto.sum(1),torch.ones(1,4,4),atol=1e-6))


if __name__ == '__main__':
    unittest.main()
