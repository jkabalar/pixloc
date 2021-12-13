import pickle

from . import set_logging_debug, logger
from .localization import RetrievalLocalizer, PoseLocalizer
from .utils.data import Paths, create_argparser, parse_paths, parse_conf
from .utils.io import write_pose_results
from .utils.eval import evaluate

default_paths = Paths(
    reference_images='../kapture-localization/pipeline/examples/datasets/RIO10/scene01/mapping/sensors/records_data/',
    query_images='../kapture-localization/pipeline/examples/datasets/RIO10/scene01/validation/sensors/records_data/',
    reference_sfm='../kapture-localization/pipeline/examples/datasets/RIO10/scene01/sfm-localize/sfm-localize/sfm_superpoint_max_superglue/',
    query_list='../kapture-localization/pipeline/examples/datasets/RIO10/scene01/sfm-localize/sfm-localize/image_list_with_intrinsics.txt',
    global_descriptors='../kapture-localization/pipeline/examples/datasets/RIO10/scene01/sfm-localize/sfm-localize/feats-superpoint-n4096-rmax1600.h5',
    retrieval_pairs='../kapture-localization/pipeline/examples/datasets/RIO10/scene01/sfm-localize/sfm-localize/pairs_localization_20.txt',
    results='pixloc_RIO10_{scene}.txt',
)

experiment = 'pixloc_rio10'

default_confs = {
    'from_retrieval': {
        'experiment': experiment,
        'features': {},
        'optimizer': {
            'num_iters': 100,
            'pad': 2,  # to 1?
        },
        'refinement': {
            'num_dbs': 5,
            'multiscale': [4, 1],
            'point_selection': 'all',
            'normalize_descriptors': True,
            'average_observations': True,
            'filter_covisibility': False,
            'do_pose_approximation': False,
        },
    },
    'from_poses': {
        'experiment': experiment,
        'features': {},
        'optimizer': {
            'num_iters': 100,
            'pad': 2,
        },
        'refinement': {
            'num_dbs': 5,
            'min_points_opt': 100,
            'point_selection': 'inliers',
            'normalize_descriptors': True,
            'average_observations': True,
            'layer_indices': [0, 1],
        },
    },
}

SCENES = ['scene01']


def main():
    parser = create_argparser('RIO10')
    parser.add_argument('--root', default="../kapture-localization/pipeline/examples/datasets/RIO10")
    parser.add_argument('--scenes', default=SCENES, choices=SCENES, nargs='+')
    parser.add_argument('--eval_only', action='store_true')
    args = parser.parse_args()

    set_logging_debug(args.verbose)
    #paths = parse_paths(args, default_paths)
    paths = default_paths
    conf = parse_conf(args, default_confs)

    all_poses = {}
    for scene in args.scenes:
        logger.info('Working on scene %s.', scene)
        paths_scene = paths.interpolate(scene=scene)
        if args.eval_only and paths_scene.results.exists():
            all_poses[scene] = paths_scene.results
            continue

        if args.from_poses:
            localizer = PoseLocalizer(paths_scene, conf)
        else:
            localizer = RetrievalLocalizer(paths_scene, conf)
        poses, logs = localizer.run_batched(skip=args.skip)
        write_pose_results(poses, paths_scene.results,
                           prepend_camera_name=True)
        with open(f'{paths_scene.results}_logs.pkl', 'wb') as f:
            pickle.dump(logs, f)
        all_poses[scene] = poses


if __name__ == '__main__':
    main()
