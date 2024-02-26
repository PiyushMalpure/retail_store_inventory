#! /usr/bin/env python3
from pathlib import Path
import argparse
import random
import numpy as np
import matplotlib.cm as cm
import torch
import sys
from loguru import logger
import os

sys.path.append('../utils')

from utils.SuperGlue.models.matching import Matching
from utils.SuperGlue.models.utils import (make_matching_plot,
                                                           AverageTimer, read_image)

torch.set_grad_enabled(False)

class ImageMatching:
    def __init__(self, 
                 input_pairs : str,
                 input_dir : str,
                 output_dir : str,
                 resize_float: bool = True,
                 max_length : int = -1,
                 resize : list = [640, 480],
                 superglue : str = "indoor",
                 max_keypoints : int = 1024,
                 keypoint_threshold : float = 0.003,
                 nms_radius: int = 10,
                 sinkhorn_iterations: int = 20,
                 match_threshold: float = 0.3,
                 viz : bool = True,
                 eval : bool = False,
                 fast_viz : bool = False,
                 cache : bool = False,
                 show_keypoints : bool = False,
                 viz_extension: str = 'png',
                 opencv_display : bool = False,
                 shuffle : bool = False,
                 force_cpu : bool = False) -> None:
        """
        Initialize the ImageMatching class
        Args:
            input_pairs: str - path to the list of image pairs
            input_dir: str - path to the directory that contains the images
            output_dir: str - path to the directory in which the .npz results and optionally, the visualization images are written
            max_length: int - maximum number of pairs to evaluate
            resize: list - resize the input image before running inference. If two numbers, resize to the exact dimensions, if one number, resize the max dimension, if -1, do not resize
            resize_float: bool - resize the image after casting uint8 to float
            superglue: str - SuperGlue weights
            max_keypoints: int - maximum number of keypoints detected by Superpoint ('-1' keeps all keypoints)
            keypoint_threshold: float - SuperPoint keypoint detector confidence threshold
            nms_radius: int - SuperPoint Non Maximum Suppression (NMS) radius (Must be positive)
            sinkhorn_iterations: int - number of Sinkhorn iterations performed by SuperGlue
            match_threshold: float - SuperGlue match threshold
            viz: bool - visualize the matches and dump the plots
            eval: bool - perform the evaluation (requires ground truth pose and intrinsics)
            fast_viz: bool - use faster image visualization with OpenCV instead of Matplotlib
            cache: bool - skip the pair if output .npz files are already found
            show_keypoints: bool - plot the keypoints in addition to the matches
            viz_extension: str - visualization file extension. Use pdf for highest-quality
            opencv_display: bool - visualize via OpenCV before saving output images
            shuffle: bool - shuffle ordering of pairs before processing
            force_cpu: bool - force pytorch to run in CPU mode
        """
        self.input_pairs = input_pairs
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.max_length = max_length
        self.resize = resize
        self.resize_float = resize_float
        self.superglue = superglue
        self.max_keypoints = max_keypoints
        self.keypoint_threshold = keypoint_threshold
        self.nms_radius = nms_radius
        self.sinkhorn_iterations = sinkhorn_iterations
        self.match_threshold = match_threshold
        self.viz = viz
        self.eval = eval
        self.fast_viz = fast_viz
        self.cache = cache
        self.show_keypoints = show_keypoints
        self.viz_extension = viz_extension
        self.opencv_display = opencv_display
        self.shuffle = shuffle
        self.force_cpu = force_cpu
        self.device = 'cuda' if torch.cuda.is_available() and not self.force_cpu else 'cpu'
        self.config = {
            'superpoint': {
                'nms_radius': self.nms_radius,
                'keypoint_threshold': self.keypoint_threshold,
                'max_keypoints': self.max_keypoints
            },
            'superglue': {
                'weights': self.superglue,
                'sinkhorn_iterations': self.sinkhorn_iterations,
                'match_threshold': self.match_threshold,
            }
        }
        self.matching = Matching(self.config).eval().to(self.device)
        self.input_dir = Path(self.input_dir)
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.timer = AverageTimer(newline=True)
        self.best_match = {}
        self.best_match_resutls = {}
        self.pairs = []
        with open(self.input_pairs, 'r') as f:
            self.pairs = [l.split() for l in f.readlines()]
        if self.max_length > -1:
            self.pairs = self.pairs[0:np.min([len(self.pairs), self.max_length])]
        if self.shuffle:
            random.Random(0).shuffle(self.pairs)
        if self.eval:

            if not all([len(p) == 38 for p in self.pairs]):
                raise ValueError(
                    'All pairs should have ground truth info for evaluation.'
                    'File \"{}\" needs 38 valid entries per row'.format(self.input_pairs))
        
        self.run_inference()
        self.save_best_matches()

    def run_inference(self):
        for i, pair in enumerate(self.pairs):
            name0, name1 = pair[:2]
            stem0, stem1 = Path(name0).stem, Path(name1).stem
            matches_path = self.output_dir / '{}_{}_matches.npz'.format(stem0, stem1)
            eval_path = self.output_dir / '{}_{}_evaluation.npz'.format(stem0, stem1)
            viz_path = self.output_dir / '{}_{}_matches.{}'.format(stem0, stem1, self.viz_extension)
            viz_eval_path = self.output_dir / \
                '{}_{}_evaluation.{}'.format(stem0, stem1, self.viz_extension)
            
            if name0 not in self.best_match.keys():
                self.best_match[name0] = None
                self.best_match_resutls[name0] = None
            
            do_match = True
            do_eval = self.eval
            do_viz = self.viz
            do_viz_eval = self.eval and self.viz
            if self.cache:
                if matches_path.exists():
                    try:
                        results = np.load(matches_path)
                    except:
                        raise IOError('Cannot load matches .npz file: %s' %
                                    matches_path)

                    kpts0, kpts1 = results['keypoints0'], results['keypoints1']
                    matches, conf = results['matches'], results['match_confidence']
                    do_match = False
                
                if self.viz and viz_path.exists():
                    do_viz = False
                
                self.timer.update('load_cache')

            if not (do_match or do_eval or do_viz or do_viz_eval):
                self.timer.print('Finished pair {:5} of {:5}'.format(i, len(self.pairs)))
                continue

            if len(pair) >= 5:
                rot0, rot1 = int(pair[2]), int(pair[3])
            else:
                rot0, rot1 = 0, 0

            image0, inp0, scales0 = read_image(
                self.input_dir / name0, self.device, self.resize, rot0, self.resize_float)
            image1, inp1, scales1 = read_image(
                self.input_dir / name1, self.device, self.resize, rot1, self.resize_float)
            if image0 is None or image1 is None:
                logger.error(f'Problem reading image pair: {self.input_dir/name0}, {self.input_dir/name1}')
                exit(1)
            self.timer.update('load_image')

            if do_match:
                pred = self.matching({'image0': inp0, 'image1': inp1})
                pred = {k: v[0].cpu().numpy() for k, v in pred.items()}
                kpts0, kpts1 = pred['keypoints0'], pred['keypoints1']
                matches, conf = pred['matches0'], pred['matching_scores0']
                self.timer.update('matcher')

            valid = matches > -1
            mkpts0 = kpts0[valid]
            mkpts1 = kpts1[matches[valid]]
            mconf = conf[valid]

            if self.best_match[name0] is None:
                self.best_match[name0] = name1
                self.best_match_resutls[name0] = {'valid': valid, 'mkpts0': mkpts0, 'mkpts1': mkpts1, 'mconf': mconf}

            elif len(mkpts0) > len(self.best_match_resutls[name0]['mkpts0']):
                self.best_match[name0] = name1
                self.best_match_resutls[name0] = {'valid': valid, 'mkpts0': mkpts0, 'mkpts1': mkpts1, 'mconf': mconf}

            if do_viz:
                color = cm.jet(mconf)
                text = [
                    'SuperGlue',
                    'Keypoints: {}:{}'.format(len(kpts0), len(kpts1)),
                    'Matches: {}'.format(len(mkpts0)),
                ]
                if rot0 != 0 or rot1 != 0:
                    text.append('Rotation: {}:{}'.format(rot0, rot1))

                k_thresh = self.matching.superpoint.config['keypoint_threshold']
                m_thresh = self.matching.superglue.config['match_threshold']
                small_text = [
                    'Keypoint Threshold: {:.4f}'.format(k_thresh),
                    'Match Threshold: {:.2f}'.format(m_thresh),
                    'Image Pair: {}:{}'.format(stem0, stem1),
                ]

                make_matching_plot(
                    image0, image1, kpts0, kpts1, mkpts0, mkpts1, color,
                    text, viz_path, self.show_keypoints,
                    self.fast_viz, self.opencv_display, 'Matches', small_text)

                self.timer.update('viz_match')

            self.timer.print('Finished pair {:5} of {:5}'.format(i, len(self.pairs)))

        # print('Best match for each image:', self.best_match)

    def get_best_matches(self):
        return self.best_match
    
    def save_best_matches(self, save_path: str = None):
        if not save_path:
            save_path = os.path.join(self.output_dir, 'best_matches.txt')
        with open(save_path, 'w') as f:
            for k, v in self.best_match.items():
                f.write(f"{k} {v}\n")


if __name__ == '__main__':
    ImageMatching(input_pairs = '../data/pairs.txt',
                  input_dir = '../data',
                  output_dir= '../results',
                  viz=True
                  )
