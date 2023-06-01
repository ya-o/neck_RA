dataset_info = dict(
    dataset_name='coco',
    paper_info=dict(
        author='Lin, Tsung-Yi and Maire, Michael and '
        'Belongie, Serge and Hays, James and '
        'Perona, Pietro and Ramanan, Deva and '
        r'Doll{\'a}r, Piotr and Zitnick, C Lawrence',
        title='Microsoft coco: Common objects in context',
        container='European conference on computer vision',
        year='2014',
        homepage='http://cocodataset.org/',
    ),
    keypoint_info={
        0:
        dict(name='point_1', id=0, color=[255, 0, 0], type='upper', swap=''),
        1:
        dict(
            name='point_2',
            id=1,
            color=[160, 32, 240],
            type='upper',
            swap=''),
        2:
        dict(
            name='point_3',
            id=2,
            color=[0, 255, 0],
            type='upper',
            swap=''),
        3:
        dict(
            name='point_4',
            id=3,
            color=[255, 255, 0],
            type='upper',
            swap=''),
    },
    skeleton_info={
        0:
        dict(link=('point_1', 'point_2'), id=0, color=[0, 255, 0]),
        1:
        dict(link=('point_2', 'point_3'), id=1, color=[0, 255, 0]),
        2:
        dict(link=('point_3', 'point_4'), id=2, color=[0, 255, 0]),
        3:
        dict(link=('point_4', 'point_1'), id=3, color=[0, 255, 0])
    }  ,
    joint_weights=[
        1., 1., 1., 1.,
    ],
    sigmas=[
        0.025, 0.025, 0.025, 0.025,
    ])
