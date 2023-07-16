import mmcv
import numpy as np
from mmdet.datasets.builder import PIPELINES
from nuscenes.eval.common.utils import Quaternion, quaternion_yaw
from nuscenes.map_expansion.map_api import NuScenesMap
from shapely import affinity, ops
from shapely.geometry import (LineString, MultiLineString, MultiPolygon, Point,
                              Polygon, box, polygon)

try:
    from ..nuscences_utils.map_api import CNuScenesMapExplorer
except:
    from nuscences_utils.map_api import CNuScenesMapExplorer

import warnings

import networkx as nx
from shapely.strtree import STRtree

warnings.filterwarnings("ignore")


@PIPELINES.register_module(force=True)
class VectorizeLocalMapLDCity(object):  # ! customized

    def __init__(self,
                 patch_size,
                 line_classes,
                 ped_crossing_classes,
                 contour_classes,
                 centerline_class,
                 sample_dist,
                 num_samples,
                 padding,
                 max_len,
                 normalize,
                 fixed_num,
                 sample_pts,
                 class2label, 
                 **kwargs):
        '''
        Args:
            fixed_num = -1 : no fixed num
        '''
        super().__init__()
        
        self.line_classes = line_classes
        self.ped_crossing_classes = ped_crossing_classes  # []
        self.contour_classes = contour_classes  # []
        self.centerline_class = centerline_class  # []


        self.class2label = class2label

        self.layer2class = {
            'solid_lane': 'divider',
            'dash_lane': 'divider',
            'stop_line': 'contours',
            'road_boundary': 'contours',  # TODO: divider before
        }


        self.process_func = {
            'ped_crossing': self.ped_geoms_to_vectors,
            'divider': self.line_geoms_to_vectors,
            'contours': self.line_geoms_to_vectors,  # TODO, here was polygon before
            'centerline': self.line_geoms_to_vectors,
        }

        self.colors = {
            # 'ped_crossing': 'blue',
            'ped_crossing': 'royalblue',
            'divider': 'orange',
            'contours': 'green',
            # origin type
            'lane': 'orange',  # we need
            'road_boundary': 'orange',  # we need
            'road_segment': 'green',
            'lane': 'green',
        }

        self.sample_pts = sample_pts

        self.patch_size = patch_size
        self.sample_dist = sample_dist
        self.num_samples = num_samples
        self.padding = padding
        self.max_len = max_len
        self.normalize = normalize
        self.fixed_num = fixed_num
        self.size = np.array([self.patch_size[1], self.patch_size[0]])  # + 2


    def retrive_geom(self):
        '''
            Get the geometric data.
            Returns: dict
        '''
        geoms_dict = {}

        layers = \
            self.line_classes + self.ped_crossing_classes + \
            self.contour_classes

        layers = set(layers)
        for layer_name in layers:  # layers = ['road_boundary', 'lane']
            # retrieve from self.map_info
            if layer_name in self.map_info:
                geoms_array = self.map_info[layer_name]  # geoms = [np.ndarray, np.ndarray, ...]
                
                geoms = []
                
                for geom in geoms_array:
                    if geom.shape[0] < 2:
                        continue
                    geoms.append(LineString(geom))
            else:
                raise ValueError(f"Layer {layer_name} not found in map_info.")
            
            if geoms is None:
                continue

            # change every geoms set to list
            if not isinstance(geoms, list):
                geoms = [geoms, ]

            geoms_dict[layer_name] = geoms

        return geoms_dict

    def union_geoms(self, geoms_dict):

        customized_geoms_dict = {}

        # contour
        roads = []
        lanes = []
        union_roads = ops.unary_union(roads)
        union_lanes = ops.unary_union(lanes)
        union_segments = ops.unary_union([union_roads, union_lanes])
        if union_segments.geom_type != 'MultiPolygon':
            union_segments = MultiPolygon([union_segments])

        customized_geoms_dict['contours'] = ('contours', [union_segments, ])

        # ped
        geoms_dict['ped_crossing'] = self.union_ped([])

        for layer_name, custom_class in self.layer2class.items():

            # if custom_class == 'contours':
            #     continue

            customized_geoms_dict[layer_name] = (
                custom_class, geoms_dict[layer_name])

        return customized_geoms_dict

    def union_ped(self, ped_geoms):

        def get_rec_direction(geom):
            rect = geom.minimum_rotated_rectangle
            rect_v_p = np.array(rect.exterior.coords)[:3]
            rect_v = rect_v_p[1:]-rect_v_p[:-1]
            v_len = np.linalg.norm(rect_v, axis=-1)
            longest_v_i = v_len.argmax()

            return rect_v[longest_v_i], v_len[longest_v_i]

        tree = STRtree(ped_geoms)
        index_by_id = dict((id(pt), i) for i, pt in enumerate(ped_geoms))

        final_pgeom = []
        remain_idx = [i for i in range(len(ped_geoms))]
        for i, pgeom in enumerate(ped_geoms):

            if i not in remain_idx:
                continue
            # update
            remain_idx.pop(remain_idx.index(i))
            pgeom_v, pgeom_v_norm = get_rec_direction(pgeom)
            final_pgeom.append(pgeom)

            for o in tree.query(pgeom):
                o_idx = index_by_id[id(o)]
                if o_idx not in remain_idx:
                    continue

                o_v, o_v_norm = get_rec_direction(o)
                cos = pgeom_v.dot(o_v)/(pgeom_v_norm*o_v_norm)
                if 1 - np.abs(cos) < 0.01:  # theta < 8 degrees.
                    final_pgeom[-1] =\
                        final_pgeom[-1].union(o)
                    # update
                    remain_idx.pop(remain_idx.index(o_idx))

        for i in range(len(final_pgeom)):
            if final_pgeom[i].geom_type != 'MultiPolygon':
                final_pgeom[i] = MultiPolygon([final_pgeom[i]]) 

        return final_pgeom

    def convert2vec(self, geoms_dict: dict, sample_pts=False, override_veclen: int = None):

        vector_dict = {}
        for layer_name, (customized_class, geoms) in geoms_dict.items():

            line_strings = self.process_func[customized_class](geoms)

            vector_len = -1
            if override_veclen is not None:
                vector_len = override_veclen

            vectors = self._geom_to_vectors(
                line_strings, customized_class, vector_len, sample_pts)
            vector_dict.update({layer_name: (customized_class, vectors)})

        return vector_dict

    def _geom_to_vectors(self, line_geom, label, vector_len, sample_pts=False):
        '''
            transfrom the geo type 2 line vectors
        '''
        line_vectors = {'vectors': [], 'length': []}
        for line in line_geom:
            if not line.is_empty:
                if line.geom_type == 'MultiLineString':
                    for l in line:
                        if sample_pts:
                            v, nl = self._sample_pts_from_line(
                                l, label, vector_len)
                        else:
                            v, nl = self._geoms2pts(l, label, vector_len)
                        line_vectors['vectors'].append(v.astype(np.float))
                        line_vectors['length'].append(nl)
                elif line.geom_type == 'LineString':
                    if sample_pts:
                        v, nl = self._sample_pts_from_line(
                            line, label, vector_len)
                    else:
                        v, nl = self._geoms2pts(line, label, vector_len)
                    line_vectors['vectors'].append(v.astype(np.float))
                    line_vectors['length'].append(nl)
                else:
                    raise NotImplementedError

        return line_vectors

    def poly_geoms_to_vectors(self, polygon_geoms: list):

        results = []
        max_x = self.patch_size[1] / 2
        max_y = self.patch_size[0] / 2
        local_patch = box(-max_x + 0.2, -max_y + 0.2, max_x - 0.2, max_y - 0.2)
        exteriors = []
        interiors = []

        for geom in polygon_geoms:
            for poly in geom:
                exteriors.append(poly.exterior)
                for inter in poly.interiors:
                    interiors.append(inter)

        results = []
        for ext in exteriors:
            if ext.is_ccw:
                ext.coords = list(ext.coords)[::-1]
            lines = ext.intersection(local_patch)
            # since the start and end will disjoint
            # after applying the intersection.
            if lines.type != 'LineString':
                lines = ops.linemerge(lines)
            results.append(lines)

        for inter in interiors:
            if not inter.is_ccw:
                inter.coords = list(inter.coords)[::-1]
            lines = inter.intersection(local_patch)
            if lines.type != 'LineString':
                lines = ops.linemerge(lines)
            results.append(lines)

        return results

    def ped_geoms_to_vectors(self, geoms: list):

        max_x = self.patch_size[1] / 2
        max_y = self.patch_size[0] / 2
        local_patch = box(-max_x + 0.2, -max_y + 0.2, max_x - 0.2, max_y - 0.2)
        results = []
        for geom in geoms:
            for ped_poly in geom:
                # rect = ped_poly.minimum_rotated_rectangle
                ext = ped_poly.exterior
                if not ext.is_ccw:
                    ext.coords = list(ext.coords)[::-1]
                lines = ext.intersection(local_patch)

                if lines.type != 'LineString':
                    lines = ops.linemerge(lines)

                # same instance but not connected.
                if lines.type != 'LineString':
                    ls = []
                    for l in lines.geoms:
                        ls.append(np.array(l.coords))

                    lines = np.concatenate(ls, axis=0)
                    lines = LineString(lines)

                results.append(lines)

        return results

    def line_geoms_to_vectors(self, geom):
        # XXX
        return geom

    def _geoms2pts(self, line, label, fixed_point_num):

        # if we still use the fix point
        if fixed_point_num > 0:
            remain_points = fixed_point_num - np.asarray(line.coords).shape[0]
            if remain_points < 0:

                tolerance = 0.4
                while np.asarray(line.coords).shape[0] > fixed_point_num:
                    line = line.simplify(tolerance, preserve_topology=True)
                    tolerance += 0.2

                remain_points = fixed_point_num - \
                    np.asarray(line.coords).shape[0]
                if remain_points > 0:
                    line = self.pad_line_with_interpolated_line(
                        line, remain_points)

            elif remain_points > 0:

                line = self.pad_line_with_interpolated_line(
                    line, remain_points)

            v = line
            if not isinstance(v, np.ndarray):
                v = np.asarray(line.coords)

            valid_len = v.shape[0]

        elif self.padding:  # dynamic points

            if self.max_len < np.asarray(line.coords).shape[0]:

                tolerance = 0.4
                while np.asarray(line.coords).shape[0] > self.max_len:
                    line = line.simplify(tolerance, preserve_topology=True)
                    tolerance += 0.2

            v = np.asarray(line.coords)
            valid_len = v.shape[0]

            pad_len = self.max_len - valid_len
            v = np.pad(v, ((0, pad_len), (0, 0)), 'constant')

        else:
            # dynamic points without padding
            line = line.simplify(0.2, preserve_topology=True)
            v = np.array(line.coords)
            valid_len = len(v)

        if self.normalize:
            v = self.normalize_line(v)

        return v, valid_len

    def pad_line_with_interpolated_line(self, line: LineString, remain_points):
        ''' pad variable line with the interploated points'''

        origin_line = line
        line_length = line.length
        v = np.array(origin_line.coords)
        line_size = v.shape[0]

        interval = np.linalg.norm(v[1:]-v[:-1], axis=-1).cumsum()
        edges = np.hstack((np.array([0]), interval))/line_length

        # padding points
        interpolated_distances = np.linspace(
            0, 1, remain_points+2)[1:-1]  # get rid of start and end
        sampled_points = np.array([list(origin_line.interpolate(distance, normalized=True).coords)
                                   for distance in interpolated_distances]).reshape(-1, 2)

        # merge two line
        insert_idx = np.searchsorted(edges, interpolated_distances) - 1

        last_idx = 0
        new_line = []
        inserted_pos = np.unique(insert_idx)

        for i, idx in enumerate(inserted_pos):
            new_line += [v[last_idx:idx+1], sampled_points[insert_idx == idx]]
            last_idx = idx+1
        # for the remain points
        if last_idx <= line_size-1:
            new_line += [v[last_idx:], ]

        merged_line = np.concatenate(new_line, 0)

        return merged_line

    def _sample_pts_from_line(self, line, label, fixed_point_num):

        if fixed_point_num < 0:
            distances = list(np.arange(self.sample_dist,
                             line.length, self.sample_dist))
            distances = [0, ] + distances + [line.length, ]
            sampled_points = np.array([list(line.interpolate(distance).coords)
                                       for distance in distances]).reshape(-1, 2)
        else:
            # fixed number of points, so distance is line.length / self.fixed_num

            distances = np.linspace(0, line.length, fixed_point_num)
            sampled_points = np.array([list(line.interpolate(distance).coords)
                                       for distance in distances]).reshape(-1, 2)

        num_valid = len(sampled_points)

        # padding
        if fixed_point_num < 0 and self.padding:

            # fixed distance sampling need padding!
            if num_valid < self.num_samples:
                padding = np.zeros((self.num_samples - len(sampled_points), 2))
                sampled_points = np.concatenate(
                    [sampled_points, padding], axis=0)
            else:
                sampled_points = sampled_points[:self.num_samples, :]
                num_valid = self.num_samples

        if self.normalize:
            sampled_points = self.normalize_line(sampled_points)

        return sampled_points, num_valid

    def normalize_line(self, line):
        '''
            prevent extrime pts such as 0 or 1. 
        '''

        origin = -np.array([self.patch_size[1]/2, self.patch_size[0]/2])

        # for better learning
        # line_ret = (line - origin) / self.size

        half_x = self.size[0] // 2
        half_y = self.size[1] // 2

        assert line[:, 0].max() <= half_x and line[:, 0].min() >= -half_x
        assert line[:, 1].max() <= half_y and line[:, 1].min() >= -half_y

        line_ret = line / self.size * 2

        return line_ret


    def vectorization(self, input_dict: dict):

        # Retrive geo from map_info.pkl
        geoms_dict = self.retrive_geom()
        # self.debug_vis(patch_params, geoms_dict=geoms_dict, orgin=False)

        # Optional union the data and convert customized labels
        geoms_dict = self.union_geoms(geoms_dict)
        # self.debug_vis(patch_params, geoms_dict=geoms_dict, origin=False, token=input_dict['token'])

        # Convert Geo 2 vec
        vectors_dict = self.convert2vec(geoms_dict, self.sample_pts)
        # self.debug_vis(patch_params, vectors_dict=vectors_dict,
        #                origin=False, token=input_dict['token'])

        # format the outputs list
        vectors = []
        for k, (custom_class, v) in vectors_dict.items():

            label = self.class2label.get(custom_class, -1)
            # filter out -1
            if label == -1:
                continue

            for vec, l in zip(v['vectors'], v['length']):

                vectors.append((vec, l, label))

        input_dict['vectors'] = vectors

        return input_dict

    def __call__(self, input_dict: dict):
        
        self.map_info = input_dict['map_info']

        input_dict = self.vectorization(input_dict)

        return input_dict


def get_start_name(i):
    return str(i)+'_start'


def get_end_name(i):
    return str(i)+'_end'
