_base_ = [
    './_base_/default_runtime.py'
]

type = 'Mapper'
plugin = True
plugin_dir = 'plugin/'

point_cloud_range = [0, -25, -1, 100, 25, 3.]
voxel_size = [0.2, 0.2, 4]

# img_norm_cfg = dict(
#     mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)

# img_size = (128, 352)

class2label = {
    'ped_crossing': 0,
    'solid_lane': 1,
    'dash_lane': 2,
    "road_boundary": 3,
    "stop_line": 4,
    "shadow_area": 5,
    # 'others': -1,
    # 'centerline': 3,
}

# rasterize params
roi_size = (200, 50)  # unit: meter, annotation range, means 100m left, 100m right, 100m front, 100m back
canvas_size = (800, 200)  # unit: pixel
thickness = 3

# vectorize params
coords_dim = 2
sample_dist = 1.0
sample_num = -1

input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False)

head_dim = 256
norm_cfg = dict(type='BN2d')
num_class = max(list(class2label.values()))+1
num_points = 30



model = dict(
    type='VectorMapNetLD',
    backbone_cfg=dict(
        type='IPMEncoder',
        img_backbone=dict(
            type='ResNet',
            with_cp=False,
            pretrained='open-mmlab://detectron2/resnet50_caffe',
            depth=50,
            num_stages=4,
            out_indices=(0, 1, 2, 3),
            frozen_stages=-1,
            norm_cfg=norm_cfg,
            norm_eval=True,
            style='caffe',
            dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False),
            stage_with_dcn=(False, False, True, True),
            ),
        img_neck=dict(
            type='FPN',
            in_channels=[256, 512, 1024, 2048],
            out_channels=128,
            start_level=0,
            add_extra_convs=True,
            # extra_convs_on_inputs=False,  # use P5
            num_outs=4,
            norm_cfg=norm_cfg,
            relu_before_extra_convs=True),
        upsample=dict(
            zoom_size=(1, 2, 4, 8),
            in_channels=128,
            out_channels=128,),
        xbound=[-30.0, 30.0, 0.3],
        ybound=[-15.0, 15.0, 0.3],
        heights=[-1.1, 0, 0.5, 1.1],
        out_channels=128,
        pretrained=None,
        num_cam=6,
        use_image=False,
        use_lidar=True,
        ),
    head_cfg=dict(
        type='DGHead',
        augmentation=True,
        augmentation_kwargs=dict(
            p=0.3,scale=0.01,
            bbox_type='xyxy',
            ),
        det_net_cfg=dict(
            type='MapElementDetector',
            num_query=120,
            max_lines=70,
            bbox_size=2,
            canvas_size=canvas_size,
            separate_detect=False,
            discrete_output=False,
            num_classes=num_class,
            in_channels=128,
            score_thre=0.1,
            num_reg_fcs=2,
            num_points=4,
            iterative=False,
            pc_range=[-15, -30, -5.0, 15, 30, 3.0],
            sync_cls_avg_factor=True,
            transformer=dict(
                type='DeformableDetrTransformer_',
                encoder=dict(
                    type='PlaceHolderEncoder',
                    embed_dims=head_dim,
                ),
                decoder=dict(
                    type='DeformableDetrTransformerDecoder_',
                    num_layers=6,
                    return_intermediate=True,
                    transformerlayers=dict(
                        type='DetrTransformerDecoderLayer',
                        attn_cfgs=[
                            dict(
                                type='MultiheadAttention',
                                embed_dims=head_dim,
                                num_heads=8,
                                attn_drop=0.1,
                                proj_drop=0.1,
                                dropout_layer=dict(type='Dropout', drop_prob=0.1),),
                            dict(
                                type='MultiScaleDeformableAttention',
                                embed_dims=head_dim,
                                num_heads=8,
                                num_levels=1,
                                ),
                        ],
                        ffn_cfgs=dict(
                            type='FFN',
                            embed_dims=head_dim,
                            feedforward_channels=head_dim*2,
                            num_fcs=2,
                            ffn_drop=0.1,
                            act_cfg=dict(type='ReLU', inplace=True),        
                        ),
                        feedforward_channels=head_dim*2,
                        ffn_dropout=0.1,
                        operation_order=('norm', 'self_attn', 'norm', 'cross_attn',
                                        'norm', 'ffn',)))
                ),
            positional_encoding=dict(
                type='SinePositionalEncoding',
                num_feats=head_dim//2,
                normalize=True,
                offset=-0.5),
            loss_cls=dict(
                type='FocalLoss',
                use_sigmoid=True,
                gamma=2.0,
                alpha=0.25,
                loss_weight=2.0),
            loss_reg=dict(
                type='LinesLoss',
                loss_weight=0.1),
            train_cfg=dict(
                assigner=dict(
                    type='HungarianLinesAssigner',
                    cost=dict(
                        type='MapQueriesCost',
                        cls_cost=dict(type='FocalLossCost', weight=2.0),
                        reg_cost=dict(type='BBoxCostC', weight=0.1), # continues
                        iou_cost=dict(type='IoUCostC', weight=1,box_format='xyxy'), # continues
                        ),
                    ),
                ),
        ),
        gen_net_cfg=dict(
            type='PolylineGenerator',
            in_channels=128,
            encoder_config=None,
            decoder_config={
                    'layer_config': {
                        'd_model': 512,
                        'nhead': 8,
                        'dim_feedforward': 512,
                        'dropout': 0.2,
                        'norm_first': True,
                        're_zero': True,
                    },
                    'num_layers': 6,
                },
            class_conditional=True,
            num_classes=num_class,
            canvas_size=canvas_size, #xy
            max_seq_length=500,
            decoder_cross_attention=False,
            use_discrete_vertex_embeddings=True,
        ),
    max_num_vertices=80,
    top_p_gen_model=0.9,
    sync_cls_avg_factor=True,
    joint_training=True,
    ),  
    with_auxiliary_head=False,
    model_name='VectorMapNet'
)

train_pipeline = [
    # dict(type='LoadMultiViewImagesFromFiles'),
    # dict(type='ResizeMultiViewImages',
    #      size = (int(128*2), int((16/9*128)*2)), # H, W
    #      change_intrinsics=True,
    #      ),
    dict(
        type='VectorizeLocalMapLDCity',  # ! customized
        patch_size=(roi_size[1],roi_size[0]),
        line_classes=['road_boundary', 'solid_lane', 'dash_lane', 'stop_line'],
        polygon_classes=['ped_crossing', 'shadow_area'],
        sample_dist=0.7,
        num_samples=150,
        sample_pts=False,
        padding=False,
        max_len=num_points,
        normalize=True,
        fixed_num={
            'ped_crossing': -1,
            'solid_lane': -1,
            'dash_lane': -1,
            'road_boundary': -1,
            'stop_line': -1,
            'shadow_area': -1,
        },
        class2label=class2label,
        centerline=False,
    ),
    dict(
        type='PolygonizeLocalMapBbox',
        canvas_size=canvas_size,  # xy
        coord_dim=2,
        num_class=num_class,
        mode='xyxy',
        test_mode=True,
        threshold=4/200,
        flatten=False,
    ),
    # dict(type='Normalize3D', **img_norm_cfg),
    # dict(type='PadMultiViewImages', size_divisor=32, change_intrinsics=True),
    dict(type='FormatBundleMap', collect=False),
    dict(type='Collect3D', keys=['polys', 'points'], meta_keys=('sample_idx',))
]


eval_cfg = dict(
    patch_size=roi_size,
    origin=(-roi_size[0]//2, -roi_size[1]//2),  # origin=(-30,-15),
    evaluation_cfg=dict(
        result_path='./',
        dataroot='TODO',
        # will be overwirte in code
        ann_file='TODO',
        num_class=num_class,
        class_name=['ped_crossing','divider','contours'],
    )
)

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type='LDDatasetMini',
        data_root='/home/yinwu/Data/Mapping/lidar',
        ann_file='/home/yinwu/Data/Mapping/map_infos.pkl',
        roi_size=roi_size,
        cat2id=class2label,
        pipeline=train_pipeline,
        eval_cfg=eval_cfg,
        interval=1,
    ),
    val=dict(
        type='LDDatasetMini',
        data_root='/home/yinwu/Data/Mapping/lidar_one',
        ann_file='/home/yinwu/Data/Mapping/map_infos.pkl',
        roi_size=roi_size,
        cat2id=class2label,
        pipeline=train_pipeline,
        eval_cfg=eval_cfg,
        interval=3,
    ),
    test=dict(
        type='LDDatasetMini',
        data_root='/home/yinwu/Data/Mapping/lidar_one',
        ann_file='/home/yinwu/Data/Mapping/map_infos.pkl',
        roi_size=roi_size,
        cat2id=class2label,
        pipeline=train_pipeline,
        eval_cfg=eval_cfg,
        interval=3,
        samples_per_gpu=2,
    ),
)

optimizer = dict(
    type='AdamW',
    lr=1e-3,
    paramwise_cfg=dict(
    custom_keys={
        'backbone': dict(lr_mult=0.1),
    }),
    weight_decay=0.01)
optimizer_config = dict(grad_clip=dict(max_norm=.5, norm_type=2))


# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=400,
    warmup_ratio=0.1,
    step=[100, 120])
checkpoint_config = dict(interval=3)

# kwargs for dataset evaluation
eval_kwargs = dict()
evaluation = dict(
    interval=1, 
    **eval_kwargs)

total_epochs = 500
runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)

find_unused_parameters = True
log_config = dict(
    interval=5,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
