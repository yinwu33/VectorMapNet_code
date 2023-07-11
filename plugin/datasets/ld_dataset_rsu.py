import mmcv
import numpy as np

import tempfile
import warnings
from os import path as osp

import os
import os.path as osp
import time

from IPython import embed

from torch.utils.data import Dataset
from mmdet3d.datasets.utils import extract_result_dict, get_loading_pipeline
from mmdet3d.core.points import BasePoints

from mmdet.datasets import DATASETS

from mmdet3d.datasets.pipelines import Compose

from .evaluation.precision_recall.average_precision_gen import eval_chamfer


@DATASETS.register_module()
class LDDatasetRSU(Dataset):
    def __init__(self,
                 data_root,
                 ann_file,
                 roi_size,
                 cat2id,
                 pipeline=None,
                 eval_cfg: dict = dict(),
                 interval=1,
                 coord_dim=3,
                 work_dir=None,
                 modality=dict(
                     use_camera=False,
                     use_lidar=True,
                     use_radar=False,
                     use_map=False,
                     use_external=False,
                 ),
                 **kwargs,
                 ):
        super().__init__(        )
        
        self.modality = modality
        self.pipeline = Compose(pipeline) if pipeline is not None else None
        self.cat2id = cat2id
        self.interval = interval
        
        self.roi_size = roi_size
        self.coord_dim = coord_dim
        self.eval_cfg = eval_cfg
        
        # lidar files
        lidar_folder_path = osp.join(data_root)
        self.samples = []
        for file in os.listdir(lidar_folder_path):
            self.samples.append(osp.join(lidar_folder_path, file))
            
        with open(ann_file, "rb") as f:
            import pickle
            self.map_info = pickle.load(f)

        # dummy flag to fit with mmdet
        self.flag = np.zeros(len(self), dtype=np.uint8)
        # self.map_extractor = NuscMapExtractor(data_root, self.roi_size)
        self.work_dir = work_dir
        

    def get_sample(self, idx):
        lidar_bin_file = self.samples[idx]
        
        points = self._load_points(lidar_bin_file).reshape(-1, 4)
        
        input_dict = dict(
            points = BasePoints(points, 4),
            map_info = self.map_info
        )
        
        return input_dict

    def prepare_data(self, index):
        """Prepare data for testing.

        Args:
            index (int): Index for accessing the target data.

        Returns:
            dict: Testing data dict of the corresponding index.
        """
        input_dict = self.get_sample(index)
        example = self.pipeline(input_dict)
        return example
    
    
    def format_results(self, results, name, prefix=None, patch_size=(60, 30), origin=(0, 0)):

        meta = self.modality
        submissions = {
            'meta': meta,
            'results': {},
            "groundTruth": {},  # for validation
        }
        patch_size = np.array(patch_size)
        origin = np.array(origin)

        for case in mmcv.track_iter_progress(results):
            '''
                vectorized_line {
                    "pts":               List[<float, 2>]  -- Ordered points to define the vectorized line.
                    "pts_num":           <int>,            -- Number of points in this line.
                    "type":              <0, 1, 2>         -- Type of the line: 0: ped; 1: divider; 2: boundary
                    "confidence_level":  <float>           -- Confidence level for prediction (used by Average Precision)
                }
            '''

            if case is None:
                continue

            vector_lines = []
            for i in range(case['nline']):
                vector = case['lines'][i] * patch_size + origin
                bboxes = case['bboxes'][i]  # (x1, y1, x2, y2)
                vector_lines.append({
                    'pts': vector,
                    'pts_num': len(case['lines'][i]),
                    'type': case['labels'][i],
                    'bbox': bboxes,
                    'confidence_level': case['scores'][i],
                })
                # submissions['results'][case['token']]['vectors'] = vector_lines
            submissions['results']['vectors'] = vector_lines

            if 'groundTruth' in case:

                # submissions['groundTruth'][case['token']] = {}
                vector_lines = []
                for i in range(case['groundTruth']['nline']):
                    line = case['groundTruth']['lines'][i] * \
                        patch_size + origin

                    vector_lines.append({
                        'pts': line,
                        'pts_num': len(case['groundTruth']['lines'][i]),
                        'type': case['groundTruth']['labels'][i],
                        'confidence_level': 1.,
                    })
                # submissions['groundTruth'][case['token']
                #                            ]['vectors'] = vector_lines
                submissions['groundTruth']['vectors'] = vector_lines

        # Use pickle format to minimize submission file size.
        print('Done!')
        mmcv.mkdir_or_exist(prefix)
        res_path = os.path.join(prefix, '{}.pkl'.format(name))
        mmcv.dump(submissions, res_path)

        return res_path


    def evaluate(self,
                 results,
                 logger=None,
                 name=None,
                 **kwargs):
        '''
        Args:
            results (list[Tensor]): List of results.
            eval_cfg (Dict): Config of test dataset.
            output_format (str): Model output format, should be either 'raster' or 'vector'.

        Returns:
            dict: Evaluation results.
        '''

        print('len of the results', len(results))
        name = 'result_ld' if name is None else name
        result_path = self.format_results(
            results, name, prefix=self.work_dir, patch_size=self.eval_cfg.patch_size, origin=self.eval_cfg.origin)

        self.eval_cfg.evaluation_cfg['result_path'] = result_path
        # self.eval_cfg.evaluation_cfg['ann_file'] = self.ann_file

        # mean_ap = eval_chamfer(
        #     self.eval_cfg.evaluation_cfg, update=True, logger=logger)

        result_dict = {
            'mAP': 0.0,
        }

        print('VectormapNet Evaluation Results:')
        print(result_dict)

        return result_dict
    
    
    def __len__(self):
        """Return the length of data infos.

        Returns:
            int: Length of data infos.
        """
        return len(self.samples)
        
    def _rand_another(self, idx):
        """Randomly get another item.

        Returns:
            int: Another index of item.
        """
        return np.random.choice(self.__len__)
    
    
    def _load_points(self, pts_filename: str):
        """Private function to load point clouds data.

        Args:
            pts_filename (str): Filename of point clouds data.

        Returns:
            np.ndarray: An array containing point clouds data.
        """
        file_client = mmcv.FileClient(**dict(backend='disk'))
        try:
            pts_bytes = file_client.get(pts_filename)
            points = np.frombuffer(pts_bytes, dtype=np.float32)
        except ConnectionError:
            mmcv.check_file_exist(pts_filename)
            if pts_filename.endswith('.npy'):
                points = np.load(pts_filename)
            else:
                points = np.fromfile(pts_filename, dtype=np.float32)
        return points

    def __getitem__(self, idx):
        """Get item from infos according to the given index.

        Returns:
            dict: Data dictionary of the corresponding index.
        """
        data = self.prepare_data(idx)

        return data